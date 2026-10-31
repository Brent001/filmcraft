use super::{FilmcraftApp, FrameKey, GpuState, frames, gpu_compositor_unsupported, plan_side};

impl FilmcraftApp {
    /// Enable the GPU compositor on the eframe wgpu device.
    pub fn set_wgpu(&mut self, rs: eframe::egui_wgpu::RenderState) {
        if let Some(why) = gpu_compositor_unsupported(&rs.adapter) {
            log::warn!("GPU compositor disabled ({why}); compositing on the CPU");
            self.ui.status = format!("{} ({why})", tl!("GPU acceleration is unavailable; preview renders on the CPU"));
            return;
        }

        // wgpu's default reaction to a validation error or lost device is to panic. Log instead
        // and fall back to the CPU compositor.
        let broken = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = broken.clone();
        rs.device.on_uncaptured_error(std::sync::Arc::new(move |e: eframe::wgpu::Error| {
            log::error!("GPU error, switching to the CPU compositor: {e}");
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }));

        let made = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            filmcraft_gpu::GpuCompositor::new(&rs.device, &rs.queue)
        }));
        match made {
            Ok(mut compositor) if !broken.load(std::sync::atomic::Ordering::Relaxed) => {
                compositor.set_upload_cache_limit(self.session.prefs.gpu.upload_cache_entries as usize);
                let max_texture = rs.device.limits().max_texture_dimension_2d;
                let adapter_name = rs.adapter.get_info().name;
                if compositor.effects_accelerated() {
                    log::info!("GPU compositor and standard-effect compute pipeline enabled on {adapter_name}; advanced VFX may still use the CPU fallback");
                    self.ui.status = format!("GPU compositing and standard effects active on {adapter_name}; advanced effects may use CPU");
                } else {
                    log::warn!("GPU compositor enabled on {adapter_name}, but its compute effect stage is unavailable; effect layers still render on the CPU");
                    self.ui.status = format!("GPU compositor active on {adapter_name}; GPU effect kernels unavailable, effects render on CPU");
                }
                self.gpu = Some(GpuState { render_state: rs, compositor, outputs: std::collections::HashMap::new(), last_ms: 0.0, broken, max_texture });
            }
            _ => log::error!("GPU compositor failed to initialise; compositing on the CPU"),
        }
    }

    /// Drop the GPU compositor after a GPU error; the monitors composite on the CPU from then on.
    fn disable_gpu(&mut self, why: &str) {
        log::error!("GPU compositor disabled: {why}");
        if let Some(g) = self.gpu.take() {
            let mut renderer = g.render_state.renderer.write();
            for output in g.outputs.values() {
                if let Some(id) = output.texture {
                    renderer.free_texture(&id);
                }
            }
        }
        self.ui.status = tl!("Graphics problem: switched to software compositing").into();
    }

    /// Composite a plan for the Program monitor (slot 0).
    pub fn gpu_present(&mut self, key: FrameKey, plan: &frames::GpuPlan) -> Option<(egui::TextureId, (u32, u32))> {
        self.gpu_present_for_slot(key, plan, 0)
    }

    /// Composite into a monitor-specific retained texture. Program and Source can render in the
    /// same egui frame, so they must not share the compositor's transient target or native ID.
    pub fn gpu_present_for_slot(&mut self, key: FrameKey, plan: &frames::GpuPlan, slot: u64) -> Option<(egui::TextureId, (u32, u32))> {
        if self.gpu.as_ref()?.broken.load(std::sync::atomic::Ordering::Relaxed) {
            self.disable_gpu("device error");
            return None;
        }

        let g = self.gpu.as_mut()?;
        if let Some(output) = g.outputs.get(&slot)
            && output.last_key == Some(key)
            && let Some(id) = output.texture
        {
            return Some((id, output.size));
        }

        if plan_side(&plan.plan) > g.max_texture as usize {
            self.disable_gpu("frame larger than the GPU's maximum texture size");
            return None;
        }

        let t0 = web_time::Instant::now();
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (view, size) = g.compositor.composite_prepared_for_slot(&plan.plan, Some(&plan.prepared), slot);
            (view.clone(), size)
        }));
        let Ok((view, size)) = ran else {
            self.disable_gpu("compositor panicked");
            return None;
        };

        let g = self.gpu.as_mut()?;
        g.last_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let GpuState { render_state, outputs, .. } = g;
        let output = outputs.entry(slot).or_default();
        let mut renderer = render_state.renderer.write();
        let id = match output.texture {
            Some(id) => {
                renderer.update_egui_texture_from_wgpu_texture(&render_state.device, &view, eframe::wgpu::FilterMode::Linear, id);
                id
            }
            None => renderer.register_native_texture(&render_state.device, &view, eframe::wgpu::FilterMode::Linear),
        };
        output.texture = Some(id);
        output.last_key = Some(key);
        output.size = size;
        Some((id, size))
    }
}
