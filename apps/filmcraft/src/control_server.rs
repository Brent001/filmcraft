//! Localhost JSON-lines control server (one request per line, one reply per line). Loopback only.
//!
//! The server only ever reads requests: a line that is not a JSON object with a string `method`
//! (an HTTP request line from a browser's cross-origin `fetch`, a stray `nc`, binary junk) gets
//! one error reply and the connection is closed, so nothing after it is executed. Lines are
//! bounded ([`MAX_LINE`]) and so is the number of open connections ([`MAX_CONNECTIONS`]); junk
//! never reaches the UI thread. Every request must carry a private 64-hex-character token.
//! Explicit CLI launches read `FILMCRAFT_CONTROL_TOKEN`; settings-driven autostart uses a private
//! per-user token file. The listener remains closed without a valid credential.
//! See `docs/control-protocol.md`.

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

use filmcraft_ui_egui::ControlRequest;
use serde_json::{Value, json};
use subtle::ConstantTimeEq;

/// Longest accepted request line (bytes, without the newline). Requests are small JSON objects;
/// anything longer is refused and the connection closed.
pub const MAX_LINE: usize = 4 * 1024 * 1024;
/// Connections served at once; further ones get an error line and are closed.
pub const MAX_CONNECTIONS: usize = 16;
/// A peer that stops sending or reading must eventually give back its connection slot.
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(120);

fn valid_token(token: &str) -> bool {
    token.len() == 64 && token.bytes().all(|b| b.is_ascii_hexdigit())
}

fn read_token_file(path: &Path) -> Result<String, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("cannot read control token file {}: {e}", path.display()))?;
    let token = text.trim().to_string();
    if !valid_token(&token) {
        return Err(format!("control token file {} is invalid; remove it and restart FilmCraft to generate a new one", path.display()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .map_err(|e| format!("cannot restrict permissions on control token file {}: {e}", path.display()))?;
    }
    Ok(token)
}

/// Return the control token used by settings-based autostart. A valid environment token keeps
/// precedence and is also stored privately so a separately launched MCP bridge can find it.
/// Otherwise a random token is created once in the platform/portable data directory. The token is
/// never logged or exposed through Settings.
pub fn token_for_autostart(data_dir: Option<&Path>) -> Result<String, String> {
    let preferred = std::env::var("FILMCRAFT_CONTROL_TOKEN").ok();
    if preferred.as_deref().is_some_and(|token| !valid_token(token)) {
        return Err("FILMCRAFT_CONTROL_TOKEN must contain exactly 64 hexadecimal characters".into());
    }
    let dir = data_dir.ok_or("FilmCraft has no writable data directory for the private control token")?;
    ensure_token_file_with_preference(dir, preferred.as_deref())
}

#[cfg(test)]
fn ensure_token_file(dir: &Path) -> Result<String, String> {
    ensure_token_file_with_preference(dir, None)
}

fn ensure_token_file_with_preference(dir: &Path, preferred: Option<&str>) -> Result<String, String> {
    fs::create_dir_all(dir).map_err(|e| format!("cannot create FilmCraft data directory {}: {e}", dir.display()))?;
    let path = dir.join(filmcraft_engine::autosave::CONTROL_TOKEN_FILE);
    match fs::read_to_string(&path) {
        Ok(_) => matching_token(read_token_file(&path)?, preferred),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let token = match preferred {
                Some(token) => token.to_string(),
                None => {
                    let mut random = [0_u8; 32];
                    getrandom::fill(&mut random).map_err(|e| format!("cannot generate a secure MCP control token: {e}"))?;
                    let hex = b"0123456789abcdef";
                    let mut token = String::with_capacity(64);
                    for byte in random {
                        token.push(char::from(hex[usize::from(byte >> 4)]));
                        token.push(char::from(hex[usize::from(byte & 0x0f)]));
                    }
                    token
                }
            };
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(mut file) => {
                    file.write_all(token.as_bytes()).map_err(|e| format!("cannot write private control token {}: {e}", path.display()))?;
                    file.sync_all().map_err(|e| format!("cannot sync private control token {}: {e}", path.display()))?;
                    Ok(token)
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => matching_token(read_token_file(&path)?, preferred),
                Err(e) => Err(format!("cannot create private control token {}: {e}", path.display())),
            }
        }
        Err(e) => Err(format!("cannot inspect private control token {}: {e}", path.display())),
    }
}

fn matching_token(saved: String, preferred: Option<&str>) -> Result<String, String> {
    if preferred.is_some_and(|token| token != saved) {
        return Err(
            "FILMCRAFT_CONTROL_TOKEN differs from the saved private control token; unset the environment variable or remove the control-token file and restart FilmCraft".into(),
        );
    }
    Ok(saved)
}

/// Start the server from Settings using an explicitly obtained private credential. Unlike the
/// legacy CLI path below, bind/token errors are returned so the desktop can show them in its status.
pub fn start_managed(port: u16, ctx: egui::Context, token: String) -> Result<Receiver<ControlRequest>, String> {
    if !valid_token(&token) {
        return Err("the MCP control token is invalid".into());
    }
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| format!("cannot listen on 127.0.0.1:{port}: {e}"))?;
    let (tx, rx) = channel::<ControlRequest>();
    log::info!("FilmCraft MCP / Control Server listening on 127.0.0.1:{port}");
    std::thread::spawn(move || accept_loop(listener, tx, ctx, token));
    Ok(rx)
}

pub fn start(port: u16, ctx: egui::Context) -> Receiver<ControlRequest> {
    start_with_token(port, ctx, std::env::var("FILMCRAFT_CONTROL_TOKEN").ok())
}

fn start_with_token(port: u16, ctx: egui::Context, token: Option<String>) -> Receiver<ControlRequest> {
    let (tx, rx) = channel::<ControlRequest>();
    // An inherited environment variable shares the credential with the owner's bridge process
    // without exposing it in command-line arguments, logs, or a shared discovery file.
    let Some(token) = token.filter(|t| t.len() == 64 && t.bytes().all(|b| b.is_ascii_hexdigit())) else {
        eprintln!("filmcraft: control server disabled: FILMCRAFT_CONTROL_TOKEN must contain 64 random hex characters (see docs/control-protocol.md)");
        return rx;
    };
    let listener = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("filmcraft: control server failed to bind 127.0.0.1:{port}: {e}");
            return rx;
        }
    };
    eprintln!("filmcraft: control server listening on 127.0.0.1:{port}");
    std::thread::spawn(move || accept_loop(listener, tx, ctx, token));
    rx
}

fn accept_loop(listener: TcpListener, tx: Sender<ControlRequest>, ctx: egui::Context, token: String) {
    let open = Arc::new(AtomicUsize::new(0));
    for mut stream in listener.incoming().flatten() {
        if open.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
            open.fetch_sub(1, Ordering::SeqCst);
            let _ = writeln!(stream, "{}", json!({"ok": false, "error": "too many control connections"}));
            continue;
        }
        let tx = tx.clone();
        let ctx = ctx.clone();
        let token = token.clone();
        // The slot is given back when the guard drops: after `serve` returns, if it panics, or if
        // the thread can't be started (the closure, and with it the guard, is dropped).
        let slot = Slot(Arc::clone(&open));
        let spawned = std::thread::Builder::new().name("control-conn".into()).spawn(move || {
            let _slot = slot;
            serve(stream, &tx, &ctx, &token);
        });
        if let Err(e) = spawned {
            eprintln!("filmcraft: control server could not start a connection thread: {e}");
        }
    }
}

/// One counted connection; decrements the open-connection count when dropped.
struct Slot(Arc<AtomicUsize>);

impl Drop for Slot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

fn serve(stream: TcpStream, tx: &Sender<ControlRequest>, ctx: &egui::Context, token: &str) {
    if let Err(e) = serve_with_timeout(stream, tx, ctx, token, CONNECTION_TIMEOUT) {
        eprintln!("filmcraft: could not configure control connection: {e}");
    }
}

fn serve_with_timeout(stream: TcpStream, tx: &Sender<ControlRequest>, ctx: &egui::Context, token: &str, timeout: Duration) -> std::io::Result<()> {
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let read = stream.try_clone()?;
    serve_lines(read, stream, token, |method, params| {
        let (req, rrx) = ControlRequest::new(method, params);
        tx.send(req).ok()?;
        ctx.request_repaint();
        Some(rrx.recv_timeout(Duration::from_secs(60)).unwrap_or_else(|_| json!({"ok": false, "error": "timeout"})))
    });
    Ok(())
}

/// One parsed line.
enum Line {
    Request {
        id: Value,
        method: String,
        params: Value,
    },
    Blank,
    /// Not a request: reply with this error and close the connection.
    Reject(String),
}

fn parse_line(bytes: &[u8], token: &str) -> Line {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return Line::Reject("not a control request (invalid UTF-8); closing the connection".into());
    };
    let text = text.trim();
    if text.is_empty() {
        return Line::Blank;
    }
    let msg = match serde_json::from_str::<Value>(text) {
        Ok(v) => v,
        Err(e) => return Line::Reject(format!("bad JSON ({e}); expected one JSON request object per line; closing the connection")),
    };
    let Some(obj) = msg.as_object() else {
        return Line::Reject("not a control request (expected a JSON object); closing the connection".into());
    };
    let Some(method) = obj.get("method").and_then(Value::as_str) else {
        return Line::Reject("not a control request (missing string `method`); closing the connection".into());
    };
    // Authenticate every request before constructing or enqueueing a ControlRequest. A valid
    // earlier request must not authorize a later one with a missing or incorrect credential.
    let authenticated = obj.get("token").and_then(Value::as_str).is_some_and(|supplied| bool::from(supplied.as_bytes().ct_eq(token.as_bytes())));
    if !authenticated {
        return Line::Reject("control authentication failed; closing the connection".into());
    }
    Line::Request { id: obj.get("id").cloned().unwrap_or(Value::Null), method: method.to_string(), params: obj.get("params").cloned().unwrap_or(json!({})) }
}

#[derive(Debug, PartialEq, Eq)]
enum ReadFailure {
    Timeout,
    Rejected(String),
}

/// Read one `\n`-terminated line of at most `max` bytes into `buf` (newline stripped).
/// `Ok(false)` at end of stream; `Err` when the line is too long or the read fails.
fn read_bounded_line(reader: &mut impl BufRead, buf: &mut Vec<u8>, max: usize) -> Result<bool, ReadFailure> {
    buf.clear();
    // `take` bounds the bytes buffered for one line: a peer can't make us allocate more than `max`.
    let limit = u64::try_from(max).unwrap_or(u64::MAX).saturating_add(1);
    let n = reader.take(limit).read_until(b'\n', buf).map_err(|e| match e.kind() {
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock => ReadFailure::Timeout,
        _ => ReadFailure::Rejected(e.to_string()),
    })?;
    if n == 0 {
        return Ok(false);
    }
    if buf.last() == Some(&b'\n') {
        buf.pop();
        if buf.last() == Some(&b'\r') {
            buf.pop();
        }
        return Ok(true);
    }
    if buf.len() > max {
        return Err(ReadFailure::Rejected(format!("request line longer than {max} bytes; closing the connection")));
    }
    // Last line without a newline, then end of stream.
    Ok(true)
}

/// Serve requests from `read`, answering on `out`, until the peer closes, sends something that is
/// not a request, or `handle` returns `None` (app gone). `handle` gets `(method, params)` and
/// returns the reply object (the request's `id` is added here).
fn serve_lines<R: Read, W: Write>(read: R, mut out: W, token: &str, mut handle: impl FnMut(String, Value) -> Option<Value>) {
    let mut reader = BufReader::new(read);
    let mut buf = Vec::new();
    loop {
        let reject = match read_bounded_line(&mut reader, &mut buf, MAX_LINE) {
            Ok(false) => return,
            // Expiration is not a request: a pooled client must see EOF and reconnect, rather
            // than read a stale error reply. A partial line must not be parsed on timeout.
            Err(ReadFailure::Timeout) => return,
            Err(ReadFailure::Rejected(e)) => e,
            Ok(true) => match parse_line(&buf, token) {
                Line::Blank => continue,
                Line::Reject(e) => e,
                Line::Request { id, method, params } => {
                    let Some(mut r) = handle(method, params) else { return };
                    if let Some(o) = r.as_object_mut() {
                        o.insert("id".into(), id);
                    }
                    if writeln!(out, "{r}").and_then(|()| out.flush()).is_err() {
                        return;
                    }
                    continue;
                }
            },
        };
        let _ = writeln!(out, "{}", json!({"ok": false, "error": reject})).and_then(|()| out.flush());
        return;
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod autostart_token_tests {
    use super::{ensure_token_file, ensure_token_file_with_preference, valid_token};
    use std::path::PathBuf;

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("filmcraft-control-token-{}-{name}", std::process::id()))
    }

    #[test]
    fn settings_autostart_creates_and_reuses_a_private_random_token() {
        let dir = temp_dir("valid");
        let _ = std::fs::remove_dir_all(&dir);
        let token = ensure_token_file(&dir).expect("a data directory should allow token creation");
        assert!(valid_token(&token));
        assert_eq!(token.len(), 64);

        let file = dir.join(filmcraft_engine::autosave::CONTROL_TOKEN_FILE);
        let saved = std::fs::read_to_string(&file).expect("token file was written");
        assert_eq!(saved, token);
        assert_eq!(ensure_token_file(&dir).expect("existing token should be reused"), token);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&file).expect("token metadata").permissions().mode() & 0o777, 0o600);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn environment_token_is_shared_with_the_mcp_bridge_token_file() {
        let dir = temp_dir("environment");
        let _ = std::fs::remove_dir_all(&dir);
        let expected = "ab".repeat(32);
        assert_eq!(ensure_token_file_with_preference(&dir, Some(&expected)).expect("valid environment token should be saved"), expected);
        let file = dir.join(filmcraft_engine::autosave::CONTROL_TOKEN_FILE);
        assert_eq!(std::fs::read_to_string(&file).expect("saved token"), expected);
        assert_eq!(ensure_token_file_with_preference(&dir, Some(&expected)).expect("matching environment token"), expected);
        assert!(ensure_token_file_with_preference(&dir, Some(&"cd".repeat(32))).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_malformed_private_token_is_not_silently_replaced() {
        let dir = temp_dir("invalid");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create test data directory");
        let file = dir.join(filmcraft_engine::autosave::CONTROL_TOKEN_FILE);
        std::fs::write(&file, "not-a-token").expect("write malformed test token");
        let error = ensure_token_file(&dir).expect_err("invalid stored credentials must fail closed");
        assert!(error.contains("invalid"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
