// FilmCraft GPU effect stage: the standard effects of `filmcraft_render::gpufx::FxOp`, one
// compute pass each, reading the working image (`src`, linear premultiplied RGBA f32) and writing
// the next one (`dst`). The math mirrors `FxOp::apply` (the CPU reference) operation by
// operation; parameters arrive evaluated (keyframes, defaults and clamps applied on the CPU).

struct U {
    i0: vec4<u32>,  // op, width, height, unused
    i1: vec4<u32>,  // integer parameters (box radius, repeat, steps, channel, rect…)
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    p3: vec4<f32>,
    p4: vec4<f32>,
    p5: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba32float, write>;
// Original input kept alive for compound effects (Unsharp Mask, Drop Shadow).
@group(0) @binding(3) var aux: texture_2d<f32>;

const OP_BRIGHTNESS_CONTRAST: u32 = 1u;
const OP_PROC_AMP: u32 = 2u;
const OP_TINT: u32 = 3u;
const OP_BLACK_WHITE: u32 = 4u;
const OP_COLOR_BALANCE: u32 = 5u;
const OP_LEAVE_COLOR: u32 = 6u;
const OP_CHANGE_TO_COLOR: u32 = 7u;
const OP_COLOR_PASS: u32 = 8u;
const OP_GAMMA: u32 = 9u;
const OP_LEVELS: u32 = 10u;
const OP_EXTRACT: u32 = 11u;
const OP_INVERT: u32 = 12u;
const OP_INVERT_ALPHA: u32 = 13u;
const OP_POSTERIZE: u32 = 14u;
const OP_ASC_CDL: u32 = 15u;
const OP_CHANNEL_MIX: u32 = 16u;
const OP_COLOR_REPLACE: u32 = 17u;
const OP_ALPHA_ADJUST: u32 = 18u;
const OP_VIGNETTE: u32 = 19u;
const OP_BOX: u32 = 20u;
const OP_DIRECTIONAL: u32 = 22u;
const OP_UNSHARP: u32 = 23u;
const OP_CROP: u32 = 24u;
const OP_RESAMPLE: u32 = 25u;
const OP_HFLIP: u32 = 26u;
const OP_VFLIP: u32 = 27u;
const OP_MIRROR: u32 = 28u;
const OP_OFFSET: u32 = 29u;
const OP_VIDEO_LIMITER: u32 = 30u;
const OP_LUMETRI: u32 = 31u;
const OP_MOSAIC: u32 = 32u;
const OP_FIND_EDGES: u32 = 33u;
const OP_EMBOSS: u32 = 34u;
const OP_REPLICATE: u32 = 35u;
const OP_STROBE: u32 = 36u;
const OP_LUMA_KEY: u32 = 37u;
const OP_FOUR_COLOR_GRADIENT: u32 = 38u;
const OP_RAMP: u32 = 39u;
const OP_CIRCLE: u32 = 40u;
const OP_GRID: u32 = 41u;
const OP_LENS_FLARE: u32 = 42u;
const OP_TWIRL: u32 = 43u;
const OP_WAVE_WARP: u32 = 44u;
const OP_LENS_DISTORTION: u32 = 45u;
const OP_BASIC_3D: u32 = 46u;
const OP_BEVEL_ALPHA: u32 = 47u;
const OP_NOISE: u32 = 48u;
const OP_KEY: u32 = 49u;
const OP_SHADOW_SEED: u32 = 50u;
const OP_SHADOW_COMBINE: u32 = 51u;

fn size() -> vec2<i32> {
    return vec2<i32>(i32(u.i0.y), i32(u.i0.z));
}

fn ld(p: vec2<i32>) -> vec4<f32> {
    return textureLoad(src, p, 0);
}

// `Image::get_or_clear`
fn ld_or_clear(p: vec2<i32>) -> vec4<f32> {
    let s = size();
    if p.x < 0 || p.y < 0 || p.x >= s.x || p.y >= s.y {
        return vec4(0.0);
    }
    return textureLoad(src, p, 0);
}

fn ld_clamped(p: vec2<i32>) -> vec4<f32> {
    let s = size();
    return textureLoad(src, clamp(p, vec2<i32>(0, 0), s - vec2<i32>(1, 1)), 0);
}

// `Image::sample_bilinear`: pixel centres at +0.5, transparent outside.
fn sample_bilinear(x: f32, y: f32) -> vec4<f32> {
    let fx = x - 0.5;
    let fy = y - 0.5;
    let x0 = floor(fx);
    let y0 = floor(fy);
    let tx = fx - x0;
    let ty = fy - y0;
    let i = vec2<i32>(i32(x0), i32(y0));
    let a = ld_or_clear(i);
    let b = ld_or_clear(i + vec2(1, 0));
    let c = ld_or_clear(i + vec2(0, 1));
    let d = ld_or_clear(i + vec2(1, 1));
    let top = a + (b - a) * tx;
    let bot = c + (d - c) * tx;
    return top + (bot - top) * ty;
}

// `Image::sample_bilinear_clamped`
fn sample_clamped(x: f32, y: f32) -> vec4<f32> {
    let s = size();
    let fx = clamp(x - 0.5, 0.0, f32(s.x) - 1.0);
    let fy = clamp(y - 0.5, 0.0, f32(s.y) - 1.0);
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let x1 = min(x0 + 1, s.x - 1);
    let y1 = min(y0 + 1, s.y - 1);
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    let a = ld(vec2(x0, y0));
    let b = ld(vec2(x1, y0));
    let c = ld(vec2(x0, y1));
    let d = ld(vec2(x1, y1));
    let top = a + (b - a) * tx;
    let bot = c + (d - c) * tx;
    return top + (bot - top) * ty;
}

// ---- colour helpers (`filmcraft_color`, `effects::enc` / `dec`)

fn linear_to_srgb1(v: f32) -> f32 {
    if v <= 0.0031308 {
        return v * 12.92;
    }
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

fn srgb_to_linear1(v: f32) -> f32 {
    if v <= 0.04045 {
        return v / 12.92;
    }
    return pow((v + 0.055) / 1.055, 2.4);
}

fn enc(c: vec3<f32>) -> vec3<f32> {
    let m = max(c, vec3(0.0));
    return vec3(linear_to_srgb1(m.x), linear_to_srgb1(m.y), linear_to_srgb1(m.z));
}

fn dec(c: vec3<f32>) -> vec3<f32> {
    let m = clamp(c, vec3(0.0), vec3(1.0));
    return vec3(srgb_to_linear1(m.x), srgb_to_linear1(m.y), srgb_to_linear1(m.z));
}

fn luma709(c: vec3<f32>) -> f32 {
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

fn edge_luma(p: vec2<i32>) -> f32 {
    return luma709(ld_clamped(p).rgb);
}

// u64 is represented as (low32, high32), allowing the deterministic CPU noise hash to
// run identically on WGSL implementations that do not expose a native 64-bit integer type.
fn mul_hi_u32(a: u32, b: u32) -> u32 {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let w0 = a0 * b0;
    let t = a1 * b0 + (w0 >> 16u);
    let mid1 = t & 0xffffu;
    let mid2 = a0 * b1 + mid1;
    return a1 * b1 + (t >> 16u) + (mid2 >> 16u);
}

fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x * b.x, mul_hi_u32(a.x, b.x) + a.x * b.y + a.y * b.x);
}

fn shr64(v: vec2<u32>, n: u32) -> vec2<u32> {
    if n == 0u {
        return v;
    }
    if n < 32u {
        return vec2<u32>((v.x >> n) | (v.y << (32u - n)), v.y >> n);
    }
    return vec2<u32>(v.y >> (n - 32u), 0u);
}

fn add_seed64(seed: vec2<u32>, amount: u32) -> vec2<u32> {
    let lo = seed.x + amount;
    let carry = select(0u, 1u, lo < seed.x);
    return vec2<u32>(lo, seed.y + carry);
}

fn noise_hash64(x: u32, y: u32, seed: vec2<u32>) -> f32 {
    let x64 = vec2<u32>(x, 0u);
    let y64 = vec2<u32>(y, 0u);
    var h = mul64(x64, vec2<u32>(0x7F4A7C15u, 0x9E3779B9u))
          ^ mul64(y64, vec2<u32>(0x27D4EB4Fu, 0xC2B2AE3Du))
          ^ mul64(seed, vec2<u32>(0x9E3779F9u, 0x165667B1u));
    h = h ^ shr64(h, 31u);
    h = mul64(h, vec2<u32>(0x1CE4E5B9u, 0xBF58476Du));
    h = h ^ shr64(h, 29u);
    return f32(h.y >> 8u) / 16777216.0;
}

// `f32::powf` for x >= 0 (WGSL leaves pow(0, y) undefined).
fn powf(x: f32, y: f32) -> f32 {
    if x == 0.0 {
        if y == 0.0 {
            return 1.0;
        }
        if y > 0.0 {
            return 0.0;
        }
        return 3.0e38;
    }
    return pow(x, y);
}

// `f32::round`: halves away from zero (WGSL `round` rounds them to even).
fn round_away(x: f32) -> f32 {
    let a = abs(x);
    let f = floor(a);
    let r = select(f, f + 1.0, a - f >= 0.5);
    return select(r, -r, x < 0.0);
}

// `rem_euclid(1.0)`
fn fract_euclid(x: f32) -> f32 {
    return x - floor(x);
}

// `vfx::smoothstep` (not WGSL's: a degenerate edge pair divides by 1e-6)
fn smoothstep_fx(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn rgb_to_hsl(c: vec3<f32>) -> vec3<f32> {
    let r = c.x; let g = c.y; let b = c.z;
    let mx = max(max(r, g), b);
    let mn = min(min(r, g), b);
    let l = (mx + mn) / 2.0;
    if abs(mx - mn) < 1e-7 {
        return vec3(0.0, 0.0, l);
    }
    let d = mx - mn;
    var s: f32;
    if l > 0.5 {
        s = d / (2.0 - mx - mn);
    } else {
        s = d / (mx + mn);
    }
    var h: f32;
    if mx == r {
        h = (g - b) / d + select(0.0, 6.0, g < b);
    } else if mx == g {
        h = (b - r) / d + 2.0;
    } else {
        h = (r - g) / d + 4.0;
    }
    return vec3(h / 6.0, s, l);
}

fn hsl_channel(p: f32, q: f32, t0: f32) -> f32 {
    let t = fract_euclid(t0);
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 0.5 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    return p;
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    if s <= 0.0 {
        return vec3(l);
    }
    var q: f32;
    if l < 0.5 {
        q = l * (1.0 + s);
    } else {
        q = l + s - l * s;
    }
    let p = 2.0 * l - q;
    return vec3(hsl_channel(p, q, h + 1.0 / 3.0), hsl_channel(p, q, h), hsl_channel(p, q, h - 1.0 / 3.0));
}

fn linear_to_enc_limiter(v: f32) -> f32 {
    if v <= 1.0 {
        return linear_to_srgb1(max(v, 0.0));
    }
    return 1.0 + (v - 1.0) / 2.4;
}

fn enc_to_linear_limiter(v: f32) -> f32 {
    if v <= 1.0 {
        return srgb_to_linear1(max(v, 0.0));
    }
    return 1.0 + (v - 1.0) * 2.4;
}

fn knee_limiter(v: f32, max_val: f32, comp: f32) -> f32 {
    let k = max_val * (1.0 - comp);
    if comp <= 0.0 || v <= k {
        return min(v, max_val);
    }
    let r = max_val - k;
    return k + r * (1.0 - exp(-(v - k) / r));
}

fn to_rgb_709(y: f32, cb: f32, cr: f32) -> vec3<f32> {
    let kr = 0.2126;
    let kb = 0.0722;
    let kg = 1.0 - kr - kb;
    let r = y + 2.0 * (1.0 - kr) * cr;
    let b = y + 2.0 * (1.0 - kb) * cb;
    let g = (y - kr * r - kb * b) / kg;
    return vec3(r, g, b);
}

fn limit_video(v: vec3<f32>, max_val: f32, comp: f32, axis: u32) -> vec3<f32> {
    let kr = 0.2126;
    let kb = 0.0722;
    let kg = 1.0 - kr - kb;
    var y = kr * v.x + kg * v.y + kb * v.z;
    var cb = (v.z - y) / (2.0 * (1.0 - kb));
    var cr = (v.x - y) / (2.0 * (1.0 - kr));

    if axis != 1u {
        y = knee_limiter(max(y, 0.0), max_val, comp);
    }

    if axis != 0u {
        let rgb = to_rgb_709(y, cb, cr);
        var s = 1.0;
        let d = rgb - vec3(y);

        if rgb.x > max_val && d.x > 1e-6 {
            s = min(s, max((max_val - y) / d.x, 0.0));
        }
        if rgb.x < 0.0 && d.x < -1e-6 {
            s = min(s, max(y / (-d.x), 0.0));
        }

        if rgb.y > max_val && d.y > 1e-6 {
            s = min(s, max((max_val - y) / d.y, 0.0));
        }
        if rgb.y < 0.0 && d.y < -1e-6 {
            s = min(s, max(y / (-d.y), 0.0));
        }

        if rgb.z > max_val && d.z > 1e-6 {
            s = min(s, max((max_val - y) / d.z, 0.0));
        }
        if rgb.z < 0.0 && d.z < -1e-6 {
            s = min(s, max(y / (-d.z), 0.0));
        }

        if axis == 3u && s < 1.0 - 1e-4 {
            s = min(knee_limiter(s, 1.0, max(comp, 0.03)), 1.0);
        }
        cb *= s;
        cr *= s;
    }

    let out_rgb = to_rgb_709(y, cb, cr);
    return clamp(out_rgb, vec3(0.0), vec3(max_val));
}

// ---- per-pixel colour effects on straight colour (`Image::map_rgb`)

fn color_op(op: u32, c: vec3<f32>) -> vec3<f32> {
    switch op {
        case OP_BRIGHTNESS_CONTRAST: {
            let br = u.p0.x; let co = u.p0.y;
            return dec((enc(c) - 0.5) * co + 0.5 + br);
        }
        case OP_PROC_AMP: {
            let br = u.p0.x; let co = u.p0.y; let hue = u.p0.z; let sat = u.p0.w;
            var hsl = rgb_to_hsl(enc(c));
            hsl.x = fract_euclid(hsl.x + hue);
            hsl.y = clamp(hsl.y * sat, 0.0, 1.0);
            return dec((hsl_to_rgb(hsl.x, hsl.y, hsl.z) - 0.5) * co + 0.5 + br);
        }
        case OP_TINT: {
            let bl = u.p0.xyz; let wh = u.p1.xyz; let amt = u.p0.w;
            let l = linear_to_srgb1(max(luma709(c), 0.0));
            let t = bl + (wh - bl) * l;
            let e = enc(c);
            return dec(e + (t - e) * amt);
        }
        case OP_BLACK_WHITE: {
            return vec3(luma709(c));
        }
        case OP_COLOR_BALANCE: {
            let e = enc(c);
            let l = 0.2126 * e.x + 0.7152 * e.y + 0.0722 * e.z;
            let ws = (1.0 - l) * (1.0 - l);
            let wh = l * l;
            let wm = 1.0 - ws - wh;
            var o = e + u.p0.xyz * ws + u.p1.xyz * max(wm, 0.0) + u.p2.xyz * wh;
            if u.i1.x != 0u {
                let l2 = 0.2126 * o.x + 0.7152 * o.y + 0.0722 * o.z;
                o = o + (l - l2);
            }
            return dec(o);
        }
        case OP_LEAVE_COLOR: {
            let amt = u.p0.x; let kh = u.p0.y; let tol = u.p0.z; let soft = u.p0.w;
            let h = rgb_to_hsl(enc(c)).x;
            let d = min(abs(h - kh), 1.0 - abs(h - kh)) * 2.0;
            let keep = 1.0 - clamp((d - tol) / soft, 0.0, 1.0);
            let l = luma709(c);
            let k = amt * (1.0 - keep);
            return c + (l - c) * k;
        }
        case OP_CHANGE_TO_COLOR: {
            let fh = u.p0.x; let th = u.p0.y; let tol = u.p0.z; let soft = u.p0.w;
            var hsl = rgb_to_hsl(enc(c));
            let d = min(abs(hsl.x - fh), 1.0 - abs(hsl.x - fh));
            let w = 1.0 - clamp((d - tol) / soft, 0.0, 1.0);
            hsl.x = fract_euclid(hsl.x + (th - fh) * w);
            return dec(hsl_to_rgb(hsl.x, hsl.y, hsl.z));
        }
        case OP_COLOR_PASS: {
            let e = enc(c);
            let k = e - u.p0.xyz;
            let d = sqrt(k.x * k.x + k.y * k.y + k.z * k.z);
            let passes = (d <= u.p0.w * 1.2) != (u.i1.x != 0u);
            if passes {
                return c;
            }
            return vec3(luma709(c));
        }
        case OP_GAMMA: {
            let e = enc(c);
            let g = u.p0.x;
            return dec(vec3(powf(max(e.x, 0.0), g), powf(max(e.y, 0.0), g), powf(max(e.z, 0.0), g)));
        }
        case OP_LEVELS: {
            let ib = u.p0.x; let iw = u.p0.y; let ob = u.p0.z; let ow = u.p0.w; let g = u.p1.x;
            let v = clamp((enc(c) - ib) / (iw - ib), vec3(0.0), vec3(1.0));
            return dec(ob + vec3(powf(v.x, g), powf(v.y, g), powf(v.z, g)) * (ow - ob));
        }
        case OP_EXTRACT: {
            let lo = u.p0.x; let hi = u.p0.y; let soft = u.p0.z;
            let l = linear_to_srgb1(max(luma709(c), 0.0));
            // Hard cutoffs avoid both division by zero and an artificial narrow fade
            // that amplifies transfer-function rounding relative to the CPU reference.
            var inside = select(0.0, 1.0, l > lo && l < hi);
            if soft > 0.0 {
                inside = min(clamp((l - lo) / soft, 0.0, 1.0), clamp((hi - l) / soft, 0.0, 1.0));
            }
            return vec3(select(inside, 1.0 - inside, u.i1.x != 0u));
        }
        case OP_INVERT: {
            let ch = u.i1.x;
            let blend = u.p0.x;
            let e = enc(c);
            var o = e;
            if ch == 0u || ch == 1u { o.x = 1.0 - e.x; }
            if ch == 0u || ch == 2u { o.y = 1.0 - e.y; }
            if ch == 0u || ch == 3u { o.z = 1.0 - e.z; }
            return dec(o + (e - o) * blend);
        }
        case OP_POSTERIZE: {
            let n = u.p0.x;
            let e = enc(c) * n;
            return dec(vec3(round_away(e.x), round_away(e.y), round_away(e.z)) / n);
        }
        case OP_ASC_CDL: {
            let e = enc(c);
            let s = u.p0.xyz; let o = u.p1.xyz; let pw = max(u.p2.xyz, vec3(0.0)); let sat = u.p0.w;
            let q = clamp(e * s + o, vec3(0.0), vec3(1.0));
            let v = vec3(powf(q.x, pw.x), powf(q.y, pw.y), powf(q.z, pw.z));
            let l = luma709(v);
            return dec(clamp(l + sat * (v - l), vec3(0.0), vec3(1.0)));
        }
        case OP_CHANNEL_MIX: {
            let v = enc(c);
            let r = u.p0; let g = u.p1; let b = u.p2;
            return dec(clamp(vec3(r.x * v.x + r.y * v.y + r.z * v.z + r.w, g.x * v.x + g.y * v.y + g.z * v.z + g.w, b.x * v.x + b.y * v.y + b.z * v.z + b.w), vec3(0.0), vec3(1.0)));
        }
        case OP_COLOR_REPLACE: {
            let v = enc(c);
            let t = u.p0.xyz; let sim = u.p0.w;
            let dd = v - t;
            let d = sqrt(dd.x * dd.x + dd.y * dd.y + dd.z * dd.z);
            let k = 1.0 - smoothstep_fx(sim * 0.85, max(sim, 1e-4), d);
            if k <= 0.0 {
                return c;
            }
            var repl = u.p1.xyz;
            if u.i1.x == 0u {
                let rh = u.p2.xyz;
                repl = hsl_to_rgb(rh.x, rh.y, rgb_to_hsl(v).z);
            }
            return dec(v + (repl - v) * k);
        }
        case OP_STROBE: {
            if u.i1.x == 0u {
                return c;
            }
            let lc = dec(u.p0.xyz);
            return lc + (c - lc) * u.p0.w;
        }
        default: {
            return c;
        }
    }
}

fn lumetri_op(c: vec3<f32>, p: vec2<i32>) -> vec3<f32> {
    let s = size();
    let creative_on = u.i1.x != 0u;
    let vignette_on = u.i1.y != 0u;

    // white balance + exposure in linear light
    let ge = u.p0.xyz;
    let lin = c * ge;
    var v = enc(lin);

    // whites / blacks: endpoints
    let b0 = u.p0.w;
    let w0 = u.p1.x;
    let denom = max(w0 - b0, 1e-3);
    v = (v - vec3(b0)) / denom;

    // highlights / shadows: luma-weighted lift/compress, hue preserving
    let sh_k = u.p1.y;
    let hl_k = u.p1.z;
    let l = luma709(v);
    let ws_val = clamp(1.0 - l, 0.0, 1.0);
    let ws = ws_val * ws_val * ws_val;
    let whl_val = clamp(l, 0.0, 1.0);
    let whl = whl_val * whl_val * whl_val;
    let nl = max(l + sh_k * ws + hl_k * whl, 0.0);
    if l > 1e-5 {
        let k = nl / l;
        v = v * k;
    }

    // contrast: smooth S-curve around mid grey
    let contrast = u.p1.w;
    if abs(contrast) > 1e-4 {
        let k = 1.0 + contrast;
        let q = clamp(v, vec3(0.0), vec3(1.0));
        let sc = q * q * (vec3(3.0) - 2.0 * q);
        if k >= 1.0 {
            v = q + (sc - q) * (k - 1.0);
        } else {
            v = vec3(0.5) + (q - vec3(0.5)) * k;
        }
    }

    // faded film: lift blacks and compress
    let faded = u.p2.x;
    if faded > 0.0 {
        v = v * (1.0 - 0.25 * faded) + vec3(0.12 * faded);
    }

    // split tone
    if creative_on {
        let st_k = u.p4.xyz;
        let ht_k = u.p5.xyz;
        let l2 = clamp(luma709(v), 0.0, 1.0);
        v = v + st_k * (1.0 - l2) + ht_k * l2;
    }

    // saturation & vibrance
    let sat = u.p2.y;
    let vib = u.p2.z;
    let l3 = luma709(v);
    let cur_sat = max(max(v.x, v.y), v.z) - min(min(v.x, v.y), v.z);
    let sat_mult = sat * (1.0 + vib * (1.0 - clamp(cur_sat, 0.0, 1.0)));
    v = vec3(l3) + (v - vec3(l3)) * sat_mult;

    // vignette
    let va = u.p2.w;
    if vignette_on && abs(va) > 1e-4 {
        let vmid = u.p3.x;
        let vfeather = u.p3.y;
        let vround_aspect = u.p3.z;
        let w = f32(s.x);
        let h = f32(s.y);
        let nx = (f32(p.x) / w - 0.5) * 2.0 * vround_aspect;
        let ny = (f32(p.y) / h - 0.5) * 2.0;
        let d = sqrt(nx * nx + ny * ny) / 1.4142135623730951;
        let edge = clamp((d - vmid * 0.9) / (max(vfeather, 0.01) * 0.9), 0.0, 1.0);
        let e2 = edge * edge * (3.0 - 2.0 * edge);
        let k = 1.0 + va * 0.2 * e2;
        if va < 0.0 {
            v = v * max(k, 0.0);
        } else {
            v = v + (vec3(1.0) - v) * (k - 1.0);
        }
    }

    return dec(v);
}

// One output pixel of every op but the running-sum box blur.
fn pixel(op: u32, p: vec2<i32>) -> vec4<f32> {
    let s = size();
    let pc = vec2<f32>(p) + 0.5;
    switch op {
        case OP_INVERT_ALPHA: {
            let o = ld(p);
            let a = o.a;
            let na = 1.0 - a;
            // (the CPU's threshold; `na / a` is only used when a > 1e-6)
            var k = 0.0;
            if a > 1e-6 {
                k = na / a;
            }
            let blend = u.p0.x;
            return vec4(o.rgb * k, na * (1.0 - blend) + a * blend);
        }
        case OP_ALPHA_ADJUST: {
            let o = ld(p);
            let c = select(o.rgb / o.a, vec3(0.0), o.a <= 1e-6);
            var a = select(o.a, 1.0, u.i1.x != 0u);
            if u.i1.y != 0u {
                a = 1.0 - a;
            }
            a = clamp(a * u.p0.x, 0.0, 1.0);
            if u.i1.z != 0u {
                return vec4(vec3(srgb_to_linear1(a)), 1.0);
            }
            return vec4(c * a, a);
        }
        case OP_BOX: {
            // one box pass along x (i1.z = 0) or y (1): mean over 2r+1 pixels, clamped to the
            // edge (repeat) or skipping pixels outside, as `effects::box_rows`
            let r = i32(u.i1.x);
            let repeat = u.i1.y != 0u;
            let vertical = u.i1.z != 0u;
            let n = select(s.x, s.y, vertical);
            let x = select(p.x, p.y, vertical);
            var acc = vec4(0.0);
            for (var i = x - r; i <= x + r; i++) {
                if repeat || (i >= 0 && i < n) {
                    let j = clamp(i, 0, n - 1);
                    acc += ld(select(vec2(j, p.y), vec2(p.x, j), vertical));
                }
            }
            return acc * (1.0 / f32(2 * r + 1));
        }
        case OP_DIRECTIONAL: {
            let steps = u.i1.x;
            let dx = u.p0.x; let dy = u.p0.y;
            var acc = vec4(0.0);
            for (var i = 0u; i < steps; i++) {
                let t = f32(i) / f32(steps - 1u) - 0.5;
                acc += sample_clamped(pc.x + dx * t, pc.y + dy * t);
            }
            return acc / f32(steps);
        }
        case OP_MOSAIC: {
            let bx = max(f32(max(u.i1.x, 1u)), 1.0);
            let by = max(f32(max(u.i1.y, 1u)), 1.0);
            let bw = max(f32(s.x) / bx, 1.0);
            let bh = max(f32(s.y) / by, 1.0);
            let cx = (floor(f32(p.x) / bw) + 0.5) * bw;
            let cy = (floor(f32(p.y) / bh) + 0.5) * bh;
            return sample_clamped(cx, cy);
        }
        case OP_FIND_EDGES: {
            let gx = edge_luma(p + vec2<i32>(1, -1)) + 2.0 * edge_luma(p + vec2<i32>(1, 0)) + edge_luma(p + vec2<i32>(1, 1))
                   - edge_luma(p + vec2<i32>(-1, -1)) - 2.0 * edge_luma(p + vec2<i32>(-1, 0)) - edge_luma(p + vec2<i32>(-1, 1));
            let gy = edge_luma(p + vec2<i32>(-1, 1)) + 2.0 * edge_luma(p + vec2<i32>(0, 1)) + edge_luma(p + vec2<i32>(1, 1))
                   - edge_luma(p + vec2<i32>(-1, -1)) - 2.0 * edge_luma(p + vec2<i32>(0, -1)) - edge_luma(p + vec2<i32>(1, -1));
            var m = min(sqrt(gx * gx + gy * gy), 1.0);
            if u.i1.x == 0u {
                m = 1.0 - m;
            }
            let o = ld(p);
            let rgb = vec3<f32>(m * o.a * (1.0 - u.p0.x)) + o.rgb * u.p0.x;
            return vec4<f32>(rgb, o.a);
        }
        case OP_EMBOSS: {
            let dx = u.p0.x;
            let dy = u.p0.y;
            let contrast = u.p0.z;
            let blend = u.p0.w;
            let a = sample_clamped(pc.x + dx, pc.y + dy);
            let bq = sample_clamped(pc.x - dx, pc.y - dy);
            let v = srgb_to_linear1(clamp(0.5 + (luma709(a.rgb) - luma709(bq.rgb)) * contrast * 2.0, 0.0, 1.0));
            let o = ld(p);
            return vec4<f32>(vec3<f32>(v * o.a * (1.0 - blend)) + o.rgb * blend, o.a);
        }
        case OP_REPLICATE: {
            let n = max(i32(u.i1.x), 1);
            let ux = (p.x * n) % s.x;
            let vy = (p.y * n) % s.y;
            let sx = f32(ux) + 0.5 + (f32(n) - 1.0) * 0.5 / f32(n);
            let sy = f32(vy) + 0.5;
            return sample_clamped(sx, sy);
        }
        case OP_LUMA_KEY: {
            let o = ld(p);
            if o.a <= 0.0 {
                return o;
            }
            let c = o.rgb / o.a;
            let l = linear_to_srgb1(max(luma709(c), 0.0));
            let threshold = u.p0.x;
            let cutoff = u.p0.y;
            var k = 0.0;
            if l <= cutoff {
                k = 0.0;
            } else if l >= max(threshold, cutoff + 1e-3) {
                k = 1.0;
            } else {
                k = (l - cutoff) / max(threshold - cutoff, 1e-3);
            }
            return o * k;
        }
        case OP_NOISE: {
            let o = ld(p);
            if o.a <= 0.0 {
                return o;
            }
            let seed = vec2<u32>(u.i1.z, u.i1.w);
            let x = u32(p.x);
            let y = u32(p.y);
            let n0 = noise_hash64(x, y, seed) - 0.5;
            var noise = vec3<f32>(n0);
            if u.i1.x != 0u {
                noise.y = noise_hash64(x, y, add_seed64(seed, 7777u)) - 0.5;
                noise.z = noise_hash64(x, y, add_seed64(seed, 99991u)) - 0.5;
            }
            var rgb = enc(o.rgb / o.a) + noise * u.p0.x;
            if u.i1.y != 0u {
                rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
            }
            let out_rgb = dec(rgb);
            return vec4<f32>(out_rgb * o.a, o.a);
        }
        case OP_KEY: {
            let o = ld(p);
            if o.a <= 0.0 {
                return o;
            }
            let c = enc(o.rgb / o.a);
            let y = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
            let cb = (c.z - y) / (2.0 * (1.0 - 0.0722));
            let cr = (c.x - y) / (2.0 * (1.0 - 0.2126));
            let delta = vec3<f32>(y, cb, cr) - u.p0.xyz;
            let d = sqrt(delta.y * delta.y + delta.z * delta.z) + abs(delta.x) * 0.15;
            let alpha = clamp((d - u.p0.w) / u.p1.x, 0.0, 1.0);
            var out_rgb = c;
            let spill = u.p1.y;
            let dom = u.i1.x;
            if spill > 0.0 {
                if dom == 0u {
                    let others = (out_rgb.y + out_rgb.z) / 2.0;
                    if out_rgb.x > others { out_rgb.x -= (out_rgb.x - others) * spill; }
                } else if dom == 1u {
                    let others = (out_rgb.x + out_rgb.z) / 2.0;
                    if out_rgb.y > others { out_rgb.y -= (out_rgb.y - others) * spill; }
                } else {
                    let others = (out_rgb.x + out_rgb.y) / 2.0;
                    if out_rgb.z > others { out_rgb.z -= (out_rgb.z - others) * spill; }
                }
            }
            if u.i1.y == 1u {
                return vec4<f32>(vec3<f32>(alpha * o.a), o.a);
            }
            let a = o.a * alpha;
            return vec4<f32>(dec(out_rgb) * a, a);
        }
        case OP_FOUR_COLOR_GRADIENT: {
            let c1 = vec3<f32>(u.p0.x, u.p0.y, u.p0.z);
            let c2 = vec3<f32>(u.p0.w, u.p1.x, u.p1.y);
            let c3 = vec3<f32>(u.p1.z, u.p1.w, u.p2.x);
            let c4 = vec3<f32>(u.p2.y, u.p2.z, u.p2.w);
            let expn = u.p3.y;
            let a = max(length(pc - vec2<f32>(f32(s.x) * 0.25, f32(s.y) * 0.25)), 1.0);
            let b = max(length(pc - vec2<f32>(f32(s.x) * 0.75, f32(s.y) * 0.25)), 1.0);
            let c = max(length(pc - vec2<f32>(f32(s.x) * 0.25, f32(s.y) * 0.75)), 1.0);
            let d = max(length(pc - vec2<f32>(f32(s.x) * 0.75, f32(s.y) * 0.75)), 1.0);
            let wa = 1.0 / powf(a, expn);
            let wb = 1.0 / powf(b, expn);
            let wc = 1.0 / powf(c, expn);
            let wd = 1.0 / powf(d, expn);
            let rgb = dec((c1 * wa + c2 * wb + c3 * wc + c4 * wd) / (wa + wb + wc + wd));
            let op = u.p3.x;
            let o = ld(p);
            return vec4<f32>(rgb * op + o.rgb * (1.0 - op), op + o.a * (1.0 - op));
        }
        case OP_RAMP: {
            let start = u.p0.xy;
            let end = u.p0.zw;
            let sc = u.p1.xyz;
            let ec = vec3<f32>(u.p1.w, u.p2.x, u.p2.y);
            let blend = u.p2.z;
            let v = end - start;
            let len2 = max(dot(v, v), 1e-9);
            var t = 0.0;
            if u.i1.x == 1u {
                t = length(pc - start) / sqrt(len2);
            } else {
                t = dot(pc - start, v) / len2;
            }
            t = clamp(t, 0.0, 1.0);
            let rgb = dec(sc + (ec - sc) * t);
            let op = 1.0 - blend;
            let o = ld(p);
            return vec4<f32>(rgb * op + o.rgb * (1.0 - op), op + o.a * (1.0 - op));
        }
        case OP_CIRCLE: {
            let center = u.p0.xy;
            let radius = u.p0.z;
            let col = dec(vec3<f32>(u.p0.w, u.p1.x, u.p1.y));
            let op = u.p1.z;
            let a = clamp(radius - length(pc - center) + 0.5, 0.0, 1.0) * op;
            let o = ld(p);
            if a <= 0.0 {
                return o;
            }
            return vec4<f32>(col * a + o.rgb * (1.0 - a), a + o.a * (1.0 - a));
        }
        case OP_GRID: {
            let size = max(u.p0.x, 1.0);
            let border = u.p0.y;
            let col = dec(vec3<f32>(u.p0.z, u.p0.w, u.p1.x));
            let op = u.p1.y;
            let ox = f32(s.x) / 2.0;
            let oy = f32(s.y) / 2.0;
            let rx = f32(p.x) - ox;
            let ry = f32(p.y) - oy;
            let mx = rx - floor(rx / size) * size;
            let my = ry - floor(ry / size) * size;
            let gx = min(mx, size - mx);
            let gy = min(my, size - my);
            let a = clamp(border * 0.5 - min(gx, gy) + 0.5, 0.0, 1.0) * op;
            let o = ld(p);
            if a <= 0.0 {
                return o;
            }
            return vec4<f32>(col * a + o.rgb * (1.0 - a), a + o.a * (1.0 - a));
        }
        case OP_LENS_FLARE: {
            let center = u.p0.xy;
            let brightness = u.p0.z;
            let blend = u.p0.w;
            let scale = max(f32(max(s.x, s.y)), 1.0);
            let d = length(pc - center) / scale;
            var add = vec3<f32>(1.0, 0.9, 0.7) * (0.35 * exp(-d * 18.0) + 0.08 * exp(-d * 3.0));
            let mx = f32(s.x) / 2.0;
            let my = f32(s.y) / 2.0;
            let g0 = center + (vec2<f32>(mx, my) - center) * (2.0 * 0.5);
            let g1 = center + (vec2<f32>(mx, my) - center) * (2.0 * 1.3);
            let g2 = center + (vec2<f32>(mx, my) - center) * (2.0 * 1.7);
            let d0 = length(pc - g0) / scale;
            let d1 = length(pc - g1) / scale;
            let d2 = length(pc - g2) / scale;
            let r0 = max(1.0 - abs((d0 - 0.05) / 0.01), 0.0) * 0.15 + select(0.0, 0.06, d0 < 0.05);
            let r1 = max(1.0 - abs((d1 - 0.03) / 0.01), 0.0) * 0.15 + select(0.0, 0.06, d1 < 0.03);
            let r2 = max(1.0 - abs((d2 - 0.08) / 0.01), 0.0) * 0.15 + select(0.0, 0.06, d2 < 0.08);
            add += vec3<f32>(0.4, 0.6, 1.0) * r0;
            add += vec3<f32>(1.0, 0.5, 0.3) * r1;
            add += vec3<f32>(0.4, 1.0, 0.5) * r2;
            let o = ld(p);
            return vec4<f32>(o.rgb + add * brightness * (1.0 - blend) * max(o.a, 0.0), o.a);
        }
        case OP_TWIRL: {
            let center = u.p0.xy;
            let angle = u.p0.z;
            let radius = u.p0.w;
            let delta = pc - center;
            let d = length(delta);
            if d >= radius {
                return ld(p);
            }
            let k = 1.0 - d / radius;
            let a = angle * k * k;
            let sn = sin(a);
            let cs = cos(a);
            let uv = center + vec2<f32>(delta.x * cs - delta.y * sn, delta.x * sn + delta.y * cs);
            return sample_bilinear(uv.x, uv.y);
        }
        case OP_WAVE_WARP: {
            let height = u.p0.x;
            let width = max(u.p0.y, 1.0);
            let ux = u.p0.z;
            let uy = u.p0.w;
            let phase = u.p1.x;
            let along = pc.x * ux + pc.y * uy;
            let off = height * sin(along / width * 6.283185307179586 - phase);
            return sample_bilinear(pc.x - uy * off, pc.y + ux * off);
        }
        case OP_LENS_DISTORTION: {
            let curvature = u.p0.x;
            let center = u.p0.yz;
            let norm = max(u.p0.w, 1e-6);
            let delta = (pc - center) / norm;
            let r2 = dot(delta, delta);
            let k = 1.0 - curvature * r2;
            let uv = center + delta * k * norm;
            return sample_bilinear(uv.x, uv.y);
        }
        case OP_BASIC_3D: {
            let sw = u.p0.x;
            let tl = u.p0.y;
            let dist = u.p0.z;
            let w = f32(s.x);
            let h = f32(s.y);
            let focal = max(w, h) * 1.2;
            let px = pc.x - w / 2.0;
            let py = pc.y - h / 2.0;
            let ss = sin(sw);
            let cs = cos(sw);
            let st = sin(tl);
            let ct = cos(tl);
            let normal = vec3<f32>(ss * ct, -st, cs * ct);
            let basis_u = vec3<f32>(cs, 0.0, -ss);
            let basis_v = vec3<f32>(ss * st, ct, cs * st);
            let z0 = focal + dist * 10.0;
            let dir = vec3<f32>(px, py, focal);
            let denom = dot(normal, dir);
            if abs(denom) < 1e-9 {
                return vec4<f32>(0.0);
            }
            let tt = (normal.z * z0) / denom;
            if tt <= 0.0 {
                return vec4<f32>(0.0);
            }
            let hit = dir * tt - vec3<f32>(0.0, 0.0, z0);
            let su = dot(hit, basis_u);
            let sv = dot(hit, basis_v);
            return sample_bilinear(su + w / 2.0, sv + h / 2.0);
        }
        case OP_BEVEL_ALPHA: {
            let thickness = u.p0.x;
            let lx = u.p0.y;
            let ly = u.p0.z;
            let intensity = u.p0.w;
            let gx = sample_clamped(pc.x + thickness, pc.y).a - sample_clamped(pc.x - thickness, pc.y).a;
            let gy = sample_clamped(pc.x, pc.y + thickness).a - sample_clamped(pc.x, pc.y - thickness).a;
            let o = ld(p);
            let shade = -(gx * lx + gy * ly) * intensity;
            return vec4<f32>(clamp(o.rgb + vec3<f32>(shade * o.a), vec3<f32>(0.0), vec3<f32>(o.a)), o.a);
        }
        case OP_SHADOW_SEED: {
            let dx = u.p0.x;
            let dy = u.p0.y;
            let color = dec(vec3<f32>(u.p0.z, u.p0.w, u.p1.x));
            let opacity = u.p1.y;
            let source_alpha = sample_bilinear(pc.x - dx, pc.y - dy).a;
            let a = source_alpha * opacity;
            return vec4<f32>(color * a, a);
        }
        case OP_SHADOW_COMBINE: {
            let shadow = ld(p);
            let original = textureLoad(aux, p, 0);
            // The CPU reference places the original over the shadow using Normal compositing.
            return original + shadow * (1.0 - original.a);
        }
        case OP_UNSHARP: {
            // src: the blurred image, aux: the original. Each colour channel on its own, as
            // `effects::unsharp_boxes`, written as vector math: FXC (DX12 without dxcompiler.dll)
            // aborts on a write to `o[k]` inside a loop ("internal error: compilation aborted
            // unexpectedly"), which took the whole `fx` pipeline down with it.
            let o = textureLoad(aux, p, 0);
            let bq = ld(p);
            let amount = u.p0.x; let th = u.p0.y;
            let d = o.rgb - bq.rgb;
            let sharpened = max(o.rgb + d * amount, vec3(0.0));
            return vec4(select(o.rgb, sharpened, abs(d) >= vec3(th * o.a)), o.a);
        }
        case OP_CROP: {
            let x0 = u.p0.x; let x1 = u.p0.y; let y0 = u.p0.z; let y1 = u.p0.w; let fe = u.p1.x;
            let d = min(min(pc.x - x0, x1 - pc.x), min(pc.y - y0, y1 - pc.y));
            var a: f32;
            if fe > 0.0 {
                a = clamp(d / fe, 0.0, 1.0);
            } else {
                a = clamp(d + 0.5, 0.0, 1.0);
            }
            let o = ld(p);
            return select(o, o * a, a < 1.0);
        }
        case OP_RESAMPLE: {
            // inverse map p0 = (a b c d), p1 = (e f opacity valid); rect i1 = x0 x1 y0 y1
            if u.p1.w == 0.0 || u32(p.x) < u.i1.x || u32(p.x) >= u.i1.y || u32(p.y) < u.i1.z || u32(p.y) >= u.i1.w {
                return vec4(0.0);
            }
            let m0 = u.p0; let m1 = u.p1;
            let uu = m0.x * pc.x + m0.z * pc.y + m1.x;
            let vv = m0.y * pc.x + m0.w * pc.y + m1.y;
            if uu < -1.0 || vv < -1.0 || uu > f32(s.x) + 1.0 || vv > f32(s.y) + 1.0 {
                return vec4(0.0);
            }
            return sample_bilinear(uu, vv) * m1.z;
        }
        case OP_HFLIP: {
            return ld(vec2(s.x - 1 - p.x, p.y));
        }
        case OP_VFLIP: {
            return ld(vec2(p.x, s.y - 1 - p.y));
        }
        case OP_MIRROR: {
            let c = u.p0.xy; let nrm = u.p0.zw;
            let d = (pc.x - c.x) * nrm.x + (pc.y - c.y) * nrm.y;
            if d > 0.0 {
                return sample_bilinear(pc.x - 2.0 * d * nrm.x, pc.y - 2.0 * d * nrm.y);
            }
            return ld(p);
        }
        case OP_OFFSET: {
            let w = f32(s.x); let h = f32(s.y);
            // dx, dy in 0..w / 0..h, so the shifted coordinate is within (-w, w + 0.5)
            var uu = pc.x - u.p0.x;
            var vv = pc.y - u.p0.y;
            if uu < 0.0 { uu += w; }
            if uu >= w { uu -= w; }
            if vv < 0.0 { vv += h; }
            if vv >= h { vv -= h; }
            let q = sample_clamped(uu, vv);
            let o = ld(p);
            return q + (o - q) * u.p0.z;
        }
        case OP_VIGNETTE: {
            let o = ld(p);
            let a = o.a;
            if a <= 1e-6 {
                return o;
            }
            let amt = u.p0.x;
            if abs(amt) < 1e-5 {
                return o;
            }
            let mid = u.p0.y;
            let round = u.p0.z;
            let feather = u.p0.w;
            let tgt = u.p1.xyz;
            let w = f32(s.x);
            let h = f32(s.y);
            let aspect = w / h;
            var nx = pc.x / w * 2.0 - 1.0;
            let ny = pc.y / h * 2.0 - 1.0;
            if round > 0.0 {
                nx *= 1.0 + (aspect - 1.0) * round;
            }
            let p_exp = select(2.0, 2.0 + (-round) * 6.0, round < 0.0);
            let d = powf(powf(abs(nx), p_exp) + powf(abs(ny), p_exp), 1.0 / p_exp) / powf(2.0, 1.0 / p_exp);
            let edge = smoothstep_fx(mid - feather * 0.5, mid + feather * 0.5, d);
            let c = o.rgb / a;
            let ec = enc(c);
            let res = dec(ec + (tgt - ec) * (edge * abs(amt)));
            return vec4(res * a, a);
        }
        case OP_VIDEO_LIMITER: {
            let o = ld(p);
            let a = o.a;
            if a <= 1e-6 {
                return o;
            }
            let c = o.rgb / a;
            let v = vec3(linear_to_enc_limiter(c.x), linear_to_enc_limiter(c.y), linear_to_enc_limiter(c.z));
            let max_val = u.p0.x;
            let comp = u.p0.y;
            let axis = u.i1.x;
            let warn = u.i1.y != 0u;
            let wc = u.p0.zw;
            let wc_b = u.p1.x;
            let warning_col = vec3(wc.x, wc.y, wc_b);
            let out_enc = limit_video(v, max_val, comp, axis);
            let diff = abs(v - out_enc);
            if warn && (diff.x > 1e-4 || diff.y > 1e-4 || diff.z > 1e-4) {
                return vec4(warning_col * a, a);
            }
            let res = vec3(enc_to_linear_limiter(out_enc.x), enc_to_linear_limiter(out_enc.y), enc_to_linear_limiter(out_enc.z));
            return vec4(res * a, a);
        }
        case OP_LUMETRI: {
            let o = ld(p);
            let a = o.a;
            if a <= 1e-6 {
                return o;
            }
            let c = o.rgb / a;
            let res = lumetri_op(c, p);
            return vec4(res * a, a);
        }
        default: {
            let o = ld(p);
            let a = o.a;
            if a <= 1e-6 {
                return o;
            }
            return vec4(color_op(op, o.rgb / a) * a, a);
        }
    }
}

@compute @workgroup_size(16, 16)
fn fx_px(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u.i0.y || id.y >= u.i0.z {
        return;
    }
    let p = vec2<i32>(id.xy);
    textureStore(dst, p, pixel(u.i0.x, p));
}

// A box pass as a running sum along one row (i1.z = 0) or column (1) per invocation: O(1) per
// pixel for any radius, the arithmetic of `effects::box_rows`.
@compute @workgroup_size(64)
fn fx_run(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = size();
    let vertical = u.i1.z != 0u;
    let n = select(s.x, s.y, vertical);
    let lines = select(s.y, s.x, vertical);
    let line = i32(id.x);
    if line >= lines {
        return;
    }
    let r = i32(u.i1.x);
    let repeat = u.i1.y != 0u;
    let inv = 1.0 / f32(2 * r + 1);
    var acc = vec4(0.0);
    for (var i = -r; i <= r; i++) {
        if repeat || (i >= 0 && i < n) {
            let j = clamp(i, 0, n - 1);
            acc += ld(select(vec2(j, line), vec2(line, j), vertical));
        }
    }
    for (var x = 0; x < n; x++) {
        textureStore(dst, select(vec2(x, line), vec2(line, x), vertical), acc * inv);
        let out_i = x - r;
        let in_i = x + r + 1;
        if repeat || out_i >= 0 {
            let j = clamp(out_i, 0, n - 1);
            acc -= ld(select(vec2(j, line), vec2(line, j), vertical));
        }
        if repeat || in_i < n {
            let j = clamp(in_i, 0, n - 1);
            acc += ld(select(vec2(j, line), vec2(line, j), vertical));
        }
    }
}
