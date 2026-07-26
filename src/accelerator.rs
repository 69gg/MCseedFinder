#[cfg(any(feature = "cuda", feature = "rocm"))]
use std::ffi::{CStr, c_char, c_int};
#[cfg(any(feature = "cuda", feature = "rocm"))]
use std::ptr::NonNull;

use anyhow::{Context, Result, anyhow, bail};
use clap::ValueEnum;
use serde::Deserialize;

use crate::config::{Anchor, CompiledCondition, CompiledFilter};
use crate::native::{self, GpuCandidate, GpuPredicate, GpuStructureConfig};

#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("cuda 与 rocm feature 不能同时启用");

#[cfg(any(feature = "cuda", feature = "rocm"))]
const ERROR_CAPACITY: usize = 512;
#[cfg(any(feature = "cuda", feature = "rocm"))]
const DEVICE_NAME_CAPACITY: usize = 256;

const GPU_ANCHOR_ORIGIN: i32 = 0;
const GPU_ANCHOR_SPAWN: i32 = 1;
const GPU_ANCHOR_NETHER_SPAWN: i32 = 2;
const GPU_ANCHOR_COORDINATES: i32 = 3;
const GPU_DENSITY_SCALE: u128 = 1_u128 << 64;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorKind {
    /// 使用构建时包含且运行时可用的 GPU，否则安全回退 CPU
    #[default]
    Auto,
    /// 仅使用多线程 CPU
    Cpu,
    /// NVIDIA CUDA；构建时需要 `--features cuda`
    Cuda,
    /// AMD ROCm/HIP；构建时需要 `--features rocm`
    #[serde(alias = "hip")]
    #[value(alias = "hip")]
    Rocm,
}

impl AcceleratorKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AcceleratorInfo {
    pub backend: &'static str,
    pub device: Option<u32>,
    pub device_name: Option<String>,
    pub note: Option<String>,
    pub predicate_count: usize,
}

impl AcceleratorInfo {
    pub fn description(&self) -> String {
        let description = match self.device {
            Some(device) => {
                let name = self
                    .device_name
                    .as_deref()
                    .map(|name| format!(" ({name})"))
                    .unwrap_or_default();
                format!(
                    "{}:{device}{name}，{} 个保守预筛选条件",
                    self.backend, self.predicate_count
                )
            }
            None => self.backend.to_owned(),
        };
        match &self.note {
            Some(note) => format!("{description}（{note}）"),
            None => description,
        }
    }
}

#[derive(Debug)]
pub struct GpuPlan {
    pub(crate) configs: Vec<GpuStructureConfig>,
    pub(crate) predicates: Vec<GpuPredicate>,
    coarse_predicates: Option<Vec<GpuPredicate>>,
    needs_spawn: bool,
}

impl GpuPlan {
    fn compile(filter: &CompiledFilter) -> Result<Self> {
        let mut plan = Self {
            configs: Vec::new(),
            predicates: Vec::new(),
            coarse_predicates: None,
            needs_spawn: false,
        };
        plan.collect_required(&filter.root)?;
        plan.optimize_predicate_order();
        plan.prepare_coarse_predicates()?;
        Ok(plan)
    }

    fn collect_required(&mut self, condition: &CompiledCondition) -> Result<()> {
        match condition {
            CompiledCondition::All(children) => {
                for child in children {
                    self.collect_required(child)?;
                }
            }
            CompiledCondition::StructureNear {
                targets,
                anchor,
                radius,
                min_count,
                ..
            } if *min_count > 0 => {
                let configs = targets
                    .iter()
                    .map(|target| native::gpu_structure_config(target.id))
                    .collect::<Result<Vec<_>>>()?;
                if configs.iter().all(Option::is_some) {
                    self.push_predicate(
                        configs.into_iter().flatten(),
                        *anchor,
                        *radius,
                        *min_count,
                    )?;
                }
            }
            CompiledCondition::StructurePieceNear {
                parent,
                anchor,
                radius,
                min_count,
                ..
            } if *min_count > 0 => {
                if let Some(config) = native::gpu_structure_config(parent.id)? {
                    // 一个或多个匹配部件都必然要求至少有一个父结构候选。
                    self.push_predicate([config].into_iter(), *anchor, *radius, 1)?;
                }
            }
            // Any/Not 的单个子条件不是整体成立的必要条件，不能安全下推。
            _ => {}
        }
        Ok(())
    }

    fn push_predicate(
        &mut self,
        configs: impl Iterator<Item = GpuStructureConfig>,
        anchor: Anchor,
        radius: u32,
        minimum: u64,
    ) -> Result<()> {
        let offset = u32::try_from(self.configs.len()).context("GPU 结构配置数量超出 u32")?;
        self.configs.extend(configs);
        let end = u32::try_from(self.configs.len()).context("GPU 结构配置数量超出 u32")?;
        let count = end
            .checked_sub(offset)
            .ok_or_else(|| anyhow!("GPU 结构配置范围溢出"))?;
        if count == 0 {
            bail!("内部错误：GPU 预筛选条件没有结构配置");
        }
        let (anchor_kind, anchor_x, anchor_z) = match anchor {
            Anchor::Origin => (GPU_ANCHOR_ORIGIN, 0, 0),
            Anchor::Spawn => {
                self.needs_spawn = true;
                (GPU_ANCHOR_SPAWN, 0, 0)
            }
            Anchor::NetherSpawn => {
                self.needs_spawn = true;
                (GPU_ANCHOR_NETHER_SPAWN, 0, 0)
            }
            Anchor::Coordinates { x, z } => (GPU_ANCHOR_COORDINATES, x, z),
        };
        self.predicates.push(GpuPredicate {
            config_offset: offset,
            config_count: count,
            radius,
            anchor_kind,
            anchor_x,
            anchor_z,
            minimum,
        });
        Ok(())
    }

    fn prepare_coarse_predicates(&mut self) -> Result<()> {
        if !self.needs_spawn || self.predicates.is_empty() {
            return Ok(());
        }
        let Some(spawn_radius) = native::spawn_refinement_radius()? else {
            return Ok(());
        };
        // floor(x / 8) 在每个轴上最多再引入不足 1 格的取整误差；
        // 用 2 格覆盖二维误差的欧氏长度（sqrt(2)）。
        let nether_radius = spawn_radius
            .div_ceil(8)
            .checked_add(2)
            .ok_or_else(|| anyhow!("下界出生点细化半径溢出"))?;
        let mut coarse = self.predicates.clone();
        for predicate in &mut coarse {
            let expansion = match predicate.anchor_kind {
                GPU_ANCHOR_SPAWN => spawn_radius,
                GPU_ANCHOR_NETHER_SPAWN => nether_radius,
                _ => 0,
            };
            predicate.radius = predicate
                .radius
                .checked_add(expansion)
                .ok_or_else(|| anyhow!("GPU 粗筛半径溢出"))?;
        }
        coarse.sort_by_key(|predicate| predicate_density_score(&self.configs, predicate));
        self.coarse_predicates = Some(coarse);
        Ok(())
    }

    fn optimize_predicate_order(&mut self) {
        let configs = &self.configs;
        self.predicates
            .sort_by_key(|predicate| predicate_density_score(configs, predicate));
    }

    pub const fn needs_spawn(&self) -> bool {
        self.needs_spawn
    }

    pub const fn has_coarse_stage(&self) -> bool {
        self.coarse_predicates.is_some()
    }
}

fn predicate_density_score(configs: &[GpuStructureConfig], predicate: &GpuPredicate) -> u128 {
    // 随机分区放置在圆形范围内的期望候选数与 radius² / region_area 成正比。
    // 分数仅用于 AND 短路顺序；使用定点比例避免浮点排序的不确定性。
    let start = predicate.config_offset as usize;
    let end = start + predicate.config_count as usize;
    let density = configs[start..end].iter().fold(0_u128, |total, config| {
        let region_span = config.region_size as u128 * 16;
        let region_area = region_span * region_span;
        total.saturating_add(GPU_DENSITY_SCALE.div_ceil(region_area))
    });
    let radius = u128::from(predicate.radius);
    let minimum = u128::from(predicate.minimum.max(1));
    radius.saturating_mul(radius).saturating_mul(density) / minimum
}

pub enum SearchAccelerator {
    Cpu(AcceleratorInfo),
    Gpu {
        info: AcceleratorInfo,
        plan: GpuPlan,
        backend: BackendContext,
        coarse_backend: Option<BackendContext>,
    },
}

impl SearchAccelerator {
    pub fn prepare(
        requested: AcceleratorKind,
        requested_device: Option<u32>,
        filter: &CompiledFilter,
    ) -> Result<Self> {
        if requested == AcceleratorKind::Cpu {
            if requested_device.is_some() {
                bail!("--gpu-device 不能与 --accelerator cpu 同时使用");
            }
            return Ok(Self::cpu(None));
        }

        let plan = GpuPlan::compile(filter)?;
        if plan.predicates.is_empty() {
            let reason = "当前条件没有可安全下推的结构放置筛选";
            return if requested == AcceleratorKind::Auto {
                Ok(Self::cpu(Some(reason.to_owned())))
            } else {
                bail!("无法启用 {}：{reason}", requested.label())
            };
        }

        let Some(compiled) = compiled_backend() else {
            if requested != AcceleratorKind::Auto {
                bail!(
                    "当前二进制未包含 GPU 后端；CUDA 请使用 `cargo build --release --features cuda`，ROCm/HIP 请使用 `cargo build --release --features rocm`"
                );
            }
            return Ok(Self::cpu(Some(
                "当前二进制未编译 CUDA/ROCm 后端".to_owned(),
            )));
        };
        if requested != AcceleratorKind::Auto && requested != compiled {
            bail!(
                "当前二进制包含 {} 后端，不能使用 --accelerator {}",
                compiled.label(),
                requested.label()
            );
        }

        let count = match BackendContext::device_count() {
            Ok(count) => count,
            Err(error) if requested == AcceleratorKind::Auto => {
                return Ok(Self::cpu(Some(format!(
                    "{} 运行时不可用：{error:#}",
                    compiled.label()
                ))));
            }
            Err(error) => return Err(error),
        };
        if count == 0 {
            let reason = format!("没有检测到 {} 设备", compiled.label());
            return if requested == AcceleratorKind::Auto {
                Ok(Self::cpu(Some(reason)))
            } else {
                bail!("{reason}")
            };
        }
        let device = requested_device.unwrap_or(0);
        if device >= count {
            bail!(
                "GPU 设备索引 {device} 超出范围；{} 后端检测到 {count} 个设备",
                compiled.label()
            );
        }
        let (device_name, device_note) = match BackendContext::device_name(device) {
            Ok(name) => (Some(name), None),
            Err(error) => (
                None,
                Some(format!("无法读取设备名称，但设备初始化成功：{error:#}")),
            ),
        };
        let backends = match (|| -> Result<_> {
            let backend = BackendContext::create(device, &plan.configs, &plan.predicates)?;
            let coarse_backend = plan
                .coarse_predicates
                .as_ref()
                .map(|predicates| BackendContext::create(device, &plan.configs, predicates))
                .transpose()?;
            Ok((backend, coarse_backend))
        })() {
            Ok(backends) => backends,
            Err(error) if requested == AcceleratorKind::Auto => {
                return Ok(Self::cpu(Some(format!(
                    "{} 初始化失败：{error:#}",
                    compiled.label()
                ))));
            }
            Err(error) => return Err(error),
        };
        Ok(Self::Gpu {
            info: AcceleratorInfo {
                backend: compiled.label(),
                device: Some(device),
                device_name,
                note: device_note,
                predicate_count: plan.predicates.len(),
            },
            plan,
            backend: backends.0,
            coarse_backend: backends.1,
        })
    }

    fn cpu(note: Option<String>) -> Self {
        Self::Cpu(AcceleratorInfo {
            backend: "cpu",
            device: None,
            device_name: None,
            note,
            predicate_count: 0,
        })
    }

    pub const fn info(&self) -> &AcceleratorInfo {
        match self {
            Self::Cpu(info) | Self::Gpu { info, .. } => info,
        }
    }

    pub const fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu { .. })
    }

    pub const fn needs_spawn(&self) -> bool {
        match self {
            Self::Gpu { plan, .. } => plan.needs_spawn(),
            Self::Cpu(_) => false,
        }
    }

    pub const fn has_coarse_stage(&self) -> bool {
        match self {
            Self::Gpu { plan, .. } => plan.has_coarse_stage(),
            Self::Cpu(_) => false,
        }
    }

    pub fn filter_coarse(&mut self, candidates: &[GpuCandidate]) -> Result<Vec<u8>> {
        match self {
            Self::Gpu {
                coarse_backend: Some(backend),
                ..
            } => backend.filter(candidates),
            Self::Gpu { .. } => bail!("内部错误：当前 GPU 计划没有出生点粗筛阶段"),
            Self::Cpu(_) => bail!("内部错误：CPU 搜索调用了 GPU 出生点粗筛"),
        }
    }

    pub fn filter(&mut self, candidates: &[GpuCandidate]) -> Result<Vec<u8>> {
        match self {
            Self::Gpu { backend, .. } => backend.filter(candidates),
            Self::Cpu(_) => bail!("内部错误：CPU 搜索调用了 GPU 预筛选"),
        }
    }
}

const fn compiled_backend() -> Option<AcceleratorKind> {
    #[cfg(feature = "cuda")]
    {
        return Some(AcceleratorKind::Cuda);
    }
    #[cfg(feature = "rocm")]
    {
        return Some(AcceleratorKind::Rocm);
    }
    #[allow(unreachable_code)]
    None
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
#[repr(C)]
struct RawBackendContext {
    _private: [u8; 0],
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
unsafe extern "C" {
    fn mcseed_gpu_device_count(
        count: *mut c_int,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn mcseed_gpu_device_name(
        device: c_int,
        name: *mut c_char,
        name_capacity: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn mcseed_gpu_context_create(
        device: c_int,
        configs: *const GpuStructureConfig,
        config_count: usize,
        predicates: *const GpuPredicate,
        predicate_count: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> *mut RawBackendContext;
    fn mcseed_gpu_context_destroy(context: *mut RawBackendContext);
    fn mcseed_gpu_filter(
        context: *mut RawBackendContext,
        candidates: *const GpuCandidate,
        candidate_count: usize,
        matches: *mut u8,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) struct BackendContext {
    raw: NonNull<RawBackendContext>,
}

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
pub(crate) struct BackendContext;

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl BackendContext {
    fn device_count() -> Result<u32> {
        let mut count = 0;
        let mut error = error_buffer();
        let status =
            unsafe { mcseed_gpu_device_count(&mut count, error.as_mut_ptr(), error.len()) };
        ensure_backend_status(status, &error, "检测 GPU 设备")?;
        u32::try_from(count).context("GPU 后端返回了负设备数量")
    }

    fn device_name(device: u32) -> Result<String> {
        let device = c_int::try_from(device).context("GPU 设备索引超出 i32")?;
        let mut name = [0 as c_char; DEVICE_NAME_CAPACITY];
        let mut error = error_buffer();
        let status = unsafe {
            mcseed_gpu_device_name(
                device,
                name.as_mut_ptr(),
                name.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure_backend_status(status, &error, "读取 GPU 设备名称")?;
        Ok(unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    fn create(
        device: u32,
        configs: &[GpuStructureConfig],
        predicates: &[GpuPredicate],
    ) -> Result<Self> {
        let device = c_int::try_from(device).context("GPU 设备索引超出 i32")?;
        let mut error = error_buffer();
        let raw = unsafe {
            mcseed_gpu_context_create(
                device,
                configs.as_ptr(),
                configs.len(),
                predicates.as_ptr(),
                predicates.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let raw = NonNull::new(raw)
            .ok_or_else(|| anyhow!("创建 GPU 上下文失败：{}", error_message(&error)))?;
        Ok(Self { raw })
    }

    fn filter(&mut self, candidates: &[GpuCandidate]) -> Result<Vec<u8>> {
        let mut matches = vec![0; candidates.len()];
        let mut error = error_buffer();
        let status = unsafe {
            mcseed_gpu_filter(
                self.raw.as_ptr(),
                candidates.as_ptr(),
                candidates.len(),
                matches.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure_backend_status(status, &error, "执行 GPU 预筛选")?;
        if matches.iter().any(|value| *value > 1) {
            bail!("GPU 预筛选返回了无效布尔值");
        }
        Ok(matches)
    }
}

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
impl BackendContext {
    fn device_count() -> Result<u32> {
        bail!("当前二进制未包含 GPU 后端")
    }

    fn device_name(_device: u32) -> Result<String> {
        bail!("当前二进制未包含 GPU 后端")
    }

    fn create(
        _device: u32,
        _configs: &[GpuStructureConfig],
        _predicates: &[GpuPredicate],
    ) -> Result<Self> {
        bail!("当前二进制未包含 GPU 后端")
    }

    fn filter(&mut self, _candidates: &[GpuCandidate]) -> Result<Vec<u8>> {
        bail!("当前二进制未包含 GPU 后端")
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl Drop for BackendContext {
    fn drop(&mut self) {
        unsafe { mcseed_gpu_context_destroy(self.raw.as_ptr()) };
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn error_buffer() -> [c_char; ERROR_CAPACITY] {
    [0; ERROR_CAPACITY]
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn error_message(buffer: &[c_char]) -> String {
    if buffer.is_empty() || buffer[0] == 0 {
        return "GPU 后端未提供错误详情".to_owned();
    }
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn ensure_backend_status(status: i32, error: &[c_char], operation: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        bail!(
            "{operation}失败（错误码 {status}）：{}",
            error_message(error)
        )
    }
}

#[cfg(test)]
mod tests {
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    use super::BackendContext;
    use super::{
        AcceleratorInfo, AcceleratorKind, GpuPlan, SearchAccelerator, predicate_density_score,
    };
    use crate::config::{CompiledFilter, conditions_from_flags};
    use crate::native::{
        GpuCandidate, NativeContext, gpu_reference_filter, gpu_structure_config, structure_by_name,
        structures,
    };

    fn filter_from_structure(specification: &str) -> CompiledFilter {
        let specifications = conditions_from_flags(&[], &[], &[specification.to_owned()], &[], &[])
            .expect("结构条件应可解析");
        CompiledFilter::compile(specifications).expect("结构条件应可编译")
    }

    #[test]
    fn accelerator_description_keeps_device_identity_when_the_name_is_unavailable() {
        let info = AcceleratorInfo {
            backend: "rocm",
            device: Some(2),
            device_name: None,
            note: Some("无法读取设备名称".to_owned()),
            predicate_count: 3,
        };
        assert_eq!(
            info.description(),
            "rocm:2，3 个保守预筛选条件（无法读取设备名称）"
        );
    }

    #[test]
    fn registry_exposes_every_random_spread_family() {
        let unsupported: Vec<String> = structures()
            .expect("结构注册表")
            .into_iter()
            .filter_map(|structure| {
                gpu_structure_config(structure.id)
                    .expect("GPU 放置配置")
                    .is_none()
                    .then_some(structure.name)
            })
            .collect();
        assert_eq!(unsupported, ["mineshaft", "stronghold"]);
    }

    #[test]
    fn plan_extracts_only_logically_required_conditions() {
        let filter = filter_from_structure("village:1024");
        let plan = GpuPlan::compile(&filter).expect("GPU 计划");
        assert_eq!(plan.predicates.len(), 1);
        assert_eq!(plan.configs.len(), 1);
        assert!(plan.needs_spawn());
        assert!(plan.has_coarse_stage());
        assert_eq!(
            plan.coarse_predicates.as_ref().expect("粗筛条件")[0].radius,
            1_149
        );

        let unsupported = filter_from_structure("village,stronghold:1024");
        let unsupported_plan = GpuPlan::compile(&unsupported).expect("GPU 计划");
        assert!(unsupported_plan.predicates.is_empty());
    }

    #[test]
    fn plan_extracts_all_accelerable_stages_from_a_combined_query() {
        let specifications = conditions_from_flags(
            &["plains".to_owned()],
            &[],
            &[
                "ruined_portal:100".to_owned(),
                "fortress:200".to_owned(),
                "bastion_remnant:200".to_owned(),
                "ancient_city:500".to_owned(),
            ],
            &["village:blacksmith:100".to_owned()],
            &["6..12".to_owned()],
        )
        .expect("组合条件应可解析");
        let filter = CompiledFilter::compile(specifications).expect("组合条件应可编译");
        let plan = GpuPlan::compile(&filter).expect("GPU 计划");
        assert_eq!(plan.predicates.len(), 5);
        assert_eq!(plan.configs.len(), 5);
        assert!(plan.needs_spawn());
        let coarse = plan
            .coarse_predicates
            .as_ref()
            .expect("组合查询应有粗筛阶段");
        assert_eq!(coarse.len(), 5);
        let scores = plan
            .predicates
            .iter()
            .map(|predicate| predicate_density_score(&plan.configs, predicate))
            .collect::<Vec<_>>();
        assert!(scores.windows(2).all(|window| window[0] <= window[1]));
        let coarse_scores = coarse
            .iter()
            .map(|predicate| predicate_density_score(&plan.configs, predicate))
            .collect::<Vec<_>>();
        assert!(
            coarse_scores
                .windows(2)
                .all(|window| window[0] <= window[1])
        );
        let mut coarse_radii = coarse
            .iter()
            .map(|predicate| predicate.radius)
            .collect::<Vec<_>>();
        coarse_radii.sort_unstable();
        assert_eq!(coarse_radii, [218, 218, 225, 225, 625]);
    }

    #[test]
    fn cpu_reference_prefilter_never_rejects_exact_village_hits() {
        let filter = filter_from_structure("village:1024");
        let plan = GpuPlan::compile(&filter).expect("GPU 计划");
        let village = structure_by_name("village").expect("村庄");
        let mut native = NativeContext::new().expect("原生上下文");
        let mut candidates = Vec::new();
        let mut exact = Vec::new();
        for seed in 0_i64..32 {
            native.set_seed(seed);
            let spawn = native.spawn_raw().expect("出生点");
            let scan = native
                .find_structure(
                    village.id,
                    spawn.position.x,
                    spawn.position.z,
                    1_024,
                    1,
                    false,
                )
                .expect("村庄扫描");
            candidates.push(GpuCandidate {
                seed: seed as u64,
                spawn_x: spawn.position.x,
                spawn_z: spawn.position.z,
            });
            exact.push(scan.found > 0);
        }
        let matches = gpu_reference_filter(&candidates, &plan.configs, &plan.predicates);
        assert!(exact.iter().any(|matched| *matched));
        for (index, exact_match) in exact.into_iter().enumerate() {
            if exact_match {
                assert_eq!(matches[index], 1, "seed {index} 被错误淘汰");
            }
        }
    }

    #[test]
    fn estimated_spawn_coarse_stage_contains_every_exact_placement_hit() {
        let mut native = NativeContext::new().expect("原生上下文");
        let mut estimated_candidates = Vec::new();
        let mut exact_candidates = Vec::new();
        for seed in 0_i64..128 {
            native.set_seed(seed);
            let estimated = native.estimated_spawn().expect("估计出生点");
            let spawn = native.spawn_raw().expect("最终出生点");
            estimated_candidates.push(GpuCandidate {
                seed: seed as u64,
                spawn_x: estimated.x,
                spawn_z: estimated.z,
            });
            exact_candidates.push(GpuCandidate {
                seed: seed as u64,
                spawn_x: spawn.position.x,
                spawn_z: spawn.position.z,
            });
        }
        for specification in ["village:128", "fortress:200"] {
            let filter = filter_from_structure(specification);
            let plan = GpuPlan::compile(&filter).expect("GPU 计划");
            let coarse = plan.coarse_predicates.as_ref().expect("粗筛条件");
            let coarse_matches = gpu_reference_filter(&estimated_candidates, &plan.configs, coarse);
            let exact_matches =
                gpu_reference_filter(&exact_candidates, &plan.configs, &plan.predicates);
            assert!(exact_matches.contains(&1), "{specification} 样本应有命中");
            for (index, exact) in exact_matches.into_iter().enumerate() {
                if exact == 1 {
                    assert_eq!(
                        coarse_matches[index], 1,
                        "{specification} 的 seed {index} 被粗筛错误淘汰"
                    );
                }
            }
        }
    }

    #[test]
    fn cpu_reference_is_conservative_for_all_supported_structure_families() {
        let mut native = NativeContext::new().expect("原生上下文");
        let mut exact_hits = 0;
        for structure in structures().expect("结构注册表") {
            let Some(config) = gpu_structure_config(structure.id).expect("GPU 放置配置") else {
                continue;
            };
            let radius = if structure.name == "end_city" {
                2_048
            } else {
                512
            };
            let predicate = crate::native::GpuPredicate {
                config_offset: 0,
                config_count: 1,
                radius,
                anchor_kind: 0,
                anchor_x: 0,
                anchor_z: 0,
                minimum: 1,
            };
            for seed in 0_i64..4 {
                native.set_seed(seed);
                let exact = native
                    .find_structure(structure.id, 0, 0, radius, 1, false)
                    .expect("精确结构扫描")
                    .found
                    > 0;
                let candidate = GpuCandidate {
                    seed: seed as u64,
                    spawn_x: 0,
                    spawn_z: 0,
                };
                let placement = gpu_reference_filter(&[candidate], &[config], &[predicate]);
                if exact {
                    exact_hits += 1;
                    assert_eq!(
                        placement[0], 1,
                        "结构 {} 的 seed {seed} 被保守预筛选错误淘汰",
                        structure.name
                    );
                }
            }
        }
        assert!(exact_hits > 10, "样本应覆盖足够多的精确结构命中");
    }

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    #[test]
    fn compiled_gpu_kernel_matches_all_shared_placement_algorithms_when_available() {
        let Ok(device_count) = BackendContext::device_count() else {
            return;
        };
        if device_count == 0 {
            return;
        }
        let candidates: Vec<GpuCandidate> = (-2_048_i64..2_048)
            .map(|seed| GpuCandidate {
                seed: seed as u64,
                spawn_x: 0,
                spawn_z: 0,
            })
            .collect();
        let mut saw_rejected = false;
        let mut saw_retained = false;
        for structure in structures().expect("结构注册表") {
            let Some(config) = gpu_structure_config(structure.id).expect("GPU 放置配置") else {
                continue;
            };
            let radius = if structure.name == "end_city" {
                1_536
            } else {
                256
            };
            let plan = GpuPlan {
                configs: vec![config],
                predicates: vec![crate::native::GpuPredicate {
                    config_offset: 0,
                    config_count: 1,
                    radius,
                    anchor_kind: 0,
                    anchor_x: 0,
                    anchor_z: 0,
                    minimum: 1,
                }],
                coarse_predicates: None,
                needs_spawn: false,
            };
            let expected = gpu_reference_filter(&candidates, &plan.configs, &plan.predicates);
            saw_rejected |= expected.contains(&0);
            saw_retained |= expected.contains(&1);
            let mut backend = BackendContext::create(0, &plan.configs, &plan.predicates)
                .expect("创建 GPU 上下文");
            let actual = backend.filter(&candidates).expect("执行 GPU 内核");
            assert_eq!(
                actual, expected,
                "结构 {} 的 GPU 结果不一致",
                structure.name
            );
        }
        assert!(
            saw_rejected && saw_retained,
            "设备测试必须同时覆盖保留与淘汰"
        );
    }

    #[test]
    fn cpu_selection_rejects_a_gpu_device_index() {
        let filter = filter_from_structure("village:1024");
        let error = SearchAccelerator::prepare(AcceleratorKind::Cpu, Some(0), &filter)
            .err()
            .expect("CPU 模式应拒绝 GPU 索引");
        assert!(error.to_string().contains("gpu-device"));
    }
}
