#[cfg(any(feature = "cuda", feature = "rocm"))]
use std::ffi::{CStr, c_char, c_int};
#[cfg(any(feature = "cuda", feature = "rocm"))]
use std::ptr::NonNull;

use anyhow::{Context, Result, anyhow, bail};
use clap::ValueEnum;
use serde::Deserialize;

use crate::config::{Anchor, CompiledCondition, CompiledFilter};
use crate::native::{
    self, GpuCandidate, GpuPairPredicate, GpuPredicate, GpuSpawnConfig, GpuStructureConfig,
};

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
const GPU_ANCHOR_END_GATEWAY: i32 = 4;
const GPU_DENSITY_SCALE: u128 = 1_u128 << 64;
const GPU_PAIR_GROUPED: u32 = 1 << 31;
const GPU_PAIR_ANCHOR_SPAWN: u32 = 1 << 30;
const GPU_PAIR_ANCHOR_NETHER_SPAWN: u32 = 1 << 29;
const GPU_PAIR_SHIFT_MASK: u32 = 31;
const GPU_NETHER_COORDINATE_SHIFT: u32 = 3;
const GPU_OUTER_GROUPED_PAIR_MIN_PREDICATES: usize = 2;
const GPU_GROUPED_PAIR_MIN_PREDICATES: usize = 5;

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
    outer_coarse_predicates: Option<Vec<GpuPredicate>>,
    outer_pair_predicates: Vec<GpuPairPredicate>,
    pre_spawn_predicates: Vec<GpuPredicate>,
    pair_predicates: Vec<GpuPairPredicate>,
    needs_spawn: bool,
}

impl GpuPlan {
    fn compile(filter: &CompiledFilter) -> Result<Self> {
        let mut plan = Self {
            configs: Vec::new(),
            predicates: Vec::new(),
            coarse_predicates: None,
            outer_coarse_predicates: None,
            outer_pair_predicates: Vec::new(),
            pre_spawn_predicates: Vec::new(),
            pair_predicates: Vec::new(),
            needs_spawn: false,
        };
        plan.collect_required(&filter.root)?;
        plan.optimize_predicate_order();
        plan.prepare_coarse_predicates()?;
        plan.prepare_outer_coarse_predicates()?;
        plan.prepare_pre_spawn_filter()?;
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
        let (anchor_kind, anchor_x, anchor_z, radius) = match anchor {
            Anchor::Origin => (GPU_ANCHOR_ORIGIN, 0, 0, radius),
            Anchor::Spawn => {
                self.needs_spawn = true;
                (GPU_ANCHOR_SPAWN, 0, 0, radius)
            }
            Anchor::NetherSpawn => {
                self.needs_spawn = true;
                (GPU_ANCHOR_NETHER_SPAWN, 0, 0, radius)
            }
            Anchor::EndGateway => {
                let Some(margin) = native::end_gateway_gpu_margin()? else {
                    return Ok(());
                };
                let radius = radius
                    .checked_add(margin)
                    .ok_or_else(|| anyhow!("末地折跃门 GPU 包络半径溢出"))?;
                (GPU_ANCHOR_END_GATEWAY, 0, 0, radius)
            }
            Anchor::Coordinates { x, z } => (GPU_ANCHOR_COORDINATES, x, z, radius),
        };
        let offset = u32::try_from(self.configs.len()).context("GPU 结构配置数量超出 u32")?;
        self.configs.extend(configs);
        let end = u32::try_from(self.configs.len()).context("GPU 结构配置数量超出 u32")?;
        let count = end
            .checked_sub(offset)
            .ok_or_else(|| anyhow!("GPU 结构配置范围溢出"))?;
        if count == 0 {
            bail!("内部错误：GPU 预筛选条件没有结构配置");
        }
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
        expand_spawn_predicates(&mut coarse, spawn_radius, nether_radius, "GPU 粗筛")?;
        coarse.sort_by_key(|predicate| predicate_order_key(&self.configs, predicate));
        self.coarse_predicates = Some(coarse);
        Ok(())
    }

    fn prepare_outer_coarse_predicates(&mut self) -> Result<()> {
        if self.coarse_predicates.is_none() {
            return Ok(());
        }
        let Some(config) = native::gpu_spawn_config()? else {
            return Ok(());
        };
        let Some(refinement_radius) = native::spawn_refinement_radius()? else {
            return Ok(());
        };
        let spawn_radius = config
            .inner_radius
            .checked_add(config.output_rounding_radius)
            .and_then(|radius| radius.checked_add(refinement_radius))
            .ok_or_else(|| anyhow!("GPU 外层出生点包络半径溢出"))?;
        let nether_radius = spawn_radius
            .div_ceil(8)
            .checked_add(2)
            .ok_or_else(|| anyhow!("GPU 外层下界出生点包络半径溢出"))?;
        let mut predicates = self.predicates.clone();
        expand_spawn_predicates(&mut predicates, spawn_radius, nether_radius, "GPU 外层粗筛")?;
        predicates.sort_by_key(|predicate| predicate_order_key(&self.configs, predicate));
        self.outer_pair_predicates = build_same_anchor_pairs(
            &self.configs,
            &self.predicates,
            spawn_radius,
            nether_radius,
            true,
        );
        append_grouped_cross_dimension_pairs(
            &mut self.outer_pair_predicates,
            &self.configs,
            &self.predicates,
            spawn_radius,
            GPU_OUTER_GROUPED_PAIR_MIN_PREDICATES,
            true,
        );
        self.outer_coarse_predicates = Some(predicates);
        Ok(())
    }

    fn optimize_predicate_order(&mut self) {
        let configs = &self.configs;
        self.predicates
            .sort_by_key(|predicate| predicate_order_key(configs, predicate));
    }

    fn prepare_pre_spawn_filter(&mut self) -> Result<()> {
        if self.coarse_predicates.is_none() {
            return Ok(());
        }
        let Some(spawn_radius) = native::spawn_origin_radius()? else {
            return Ok(());
        };
        // floor(x / 8) 在每个轴上引入不足一格的误差；两格覆盖二维误差。
        let nether_radius = spawn_radius
            .div_ceil(8)
            .checked_add(2)
            .ok_or_else(|| anyhow!("下界出生点原点半径溢出"))?;

        self.pre_spawn_predicates = self.predicates.clone();
        for predicate in &mut self.pre_spawn_predicates {
            match predicate.anchor_kind {
                GPU_ANCHOR_SPAWN => {
                    predicate.anchor_kind = GPU_ANCHOR_ORIGIN;
                    predicate.radius = predicate
                        .radius
                        .checked_add(spawn_radius)
                        .ok_or_else(|| anyhow!("出生点前主世界结构包络半径溢出"))?;
                }
                GPU_ANCHOR_NETHER_SPAWN => {
                    predicate.anchor_kind = GPU_ANCHOR_ORIGIN;
                    predicate.radius = predicate
                        .radius
                        .checked_add(nether_radius)
                        .ok_or_else(|| anyhow!("出生点前下界结构包络半径溢出"))?;
                }
                _ => {}
            }
        }
        self.pre_spawn_predicates
            .sort_by_key(|predicate| predicate_order_key(&self.configs, predicate));

        self.pair_predicates = build_same_anchor_pairs(
            &self.configs,
            &self.predicates,
            spawn_radius,
            nether_radius,
            false,
        );

        append_grouped_cross_dimension_pairs(
            &mut self.pair_predicates,
            &self.configs,
            &self.predicates,
            spawn_radius,
            GPU_GROUPED_PAIR_MIN_PREDICATES,
            false,
        );
        Ok(())
    }

    pub const fn needs_spawn(&self) -> bool {
        self.needs_spawn
    }

    pub const fn has_coarse_stage(&self) -> bool {
        self.coarse_predicates.is_some()
    }

    pub const fn has_outer_coarse_stage(&self) -> bool {
        self.outer_coarse_predicates.is_some()
    }

    pub const fn has_pre_spawn_stage(&self) -> bool {
        !self.pre_spawn_predicates.is_empty() || !self.pair_predicates.is_empty()
    }
}

fn expand_spawn_predicates(
    predicates: &mut [GpuPredicate],
    spawn_radius: u32,
    nether_radius: u32,
    stage: &str,
) -> Result<()> {
    for predicate in predicates {
        let expansion = match predicate.anchor_kind {
            GPU_ANCHOR_SPAWN => spawn_radius,
            GPU_ANCHOR_NETHER_SPAWN => nether_radius,
            _ => 0,
        };
        predicate.radius = predicate
            .radius
            .checked_add(expansion)
            .ok_or_else(|| anyhow!("{stage}半径溢出"))?;
    }
    Ok(())
}

fn build_same_anchor_pairs(
    configs: &[GpuStructureConfig],
    predicates: &[GpuPredicate],
    spawn_radius: u32,
    nether_radius: u32,
    candidate_relative: bool,
) -> Vec<GpuPairPredicate> {
    let mut pairs = Vec::new();
    for (anchor_kind, anchor_radius, candidate_anchor_flag) in [
        (GPU_ANCHOR_SPAWN, spawn_radius, GPU_PAIR_ANCHOR_SPAWN),
        (
            GPU_ANCHOR_NETHER_SPAWN,
            nether_radius,
            GPU_PAIR_ANCHOR_NETHER_SPAWN,
        ),
    ] {
        let anchored = predicates
            .iter()
            .filter(|predicate| {
                predicate.anchor_kind == anchor_kind
                    && predicate_supports_pair_filter(configs, predicate)
            })
            .collect::<Vec<_>>();
        for left_index in 0..anchored.len() {
            for right_index in left_index + 1..anchored.len() {
                let left = anchored[left_index];
                let right = anchored[right_index];
                let mut pair = GpuPairPredicate {
                    left_config_offset: left.config_offset,
                    left_config_count: left.config_count,
                    right_config_offset: right.config_offset,
                    right_config_count: right.config_count,
                    left_radius: left.radius,
                    right_radius: right.radius,
                    anchor_radius,
                    left_coordinate_shift: if candidate_relative {
                        candidate_anchor_flag
                    } else {
                        0
                    },
                };
                if pair_left_scan_score(configs, &pair, false)
                    > pair_left_scan_score(configs, &pair, true)
                {
                    std::mem::swap(&mut pair.left_config_offset, &mut pair.right_config_offset);
                    std::mem::swap(&mut pair.left_config_count, &mut pair.right_config_count);
                    std::mem::swap(&mut pair.left_radius, &mut pair.right_radius);
                }
                pairs.push(pair);
            }
        }
    }
    pairs.sort_by_key(|predicate| pair_density_score(configs, predicate));
    pairs
}

fn append_grouped_cross_dimension_pairs(
    pairs: &mut Vec<GpuPairPredicate>,
    configs: &[GpuStructureConfig],
    predicates: &[GpuPredicate],
    spawn_radius: u32,
    minimum_predicates: usize,
    candidate_relative: bool,
) {
    let overworld = predicates
        .iter()
        .filter(|predicate| {
            predicate.anchor_kind == GPU_ANCHOR_SPAWN
                && predicate_supports_pair_filter(configs, predicate)
        })
        .collect::<Vec<_>>();
    let nether = predicates
        .iter()
        .filter(|predicate| {
            predicate.anchor_kind == GPU_ANCHOR_NETHER_SPAWN
                && predicate_supports_pair_filter(configs, predicate)
        })
        .collect::<Vec<_>>();
    if overworld.is_empty()
        || nether.is_empty()
        || overworld.len() + nether.len() < minimum_predicates
    {
        return;
    }
    let Some(pivot) = overworld
        .iter()
        .min_by_key(|predicate| predicate_anchor_scan_score(configs, predicate, spawn_radius))
        .copied()
    else {
        return;
    };
    let flags = GPU_PAIR_GROUPED
        | if candidate_relative {
            GPU_PAIR_ANCHOR_SPAWN
        } else {
            0
        };
    for right in overworld
        .iter()
        .copied()
        .filter(|predicate| !std::ptr::eq(*predicate, pivot))
    {
        pairs.push(GpuPairPredicate {
            left_config_offset: pivot.config_offset,
            left_config_count: pivot.config_count,
            right_config_offset: right.config_offset,
            right_config_count: right.config_count,
            left_radius: pivot.radius,
            right_radius: right.radius,
            anchor_radius: spawn_radius,
            left_coordinate_shift: flags,
        });
    }
    for right in nether.iter().copied() {
        pairs.push(GpuPairPredicate {
            left_config_offset: pivot.config_offset,
            left_config_count: pivot.config_count,
            right_config_offset: right.config_offset,
            right_config_count: right.config_count,
            left_radius: pivot.radius,
            right_radius: right.radius,
            anchor_radius: spawn_radius,
            left_coordinate_shift: flags | GPU_NETHER_COORDINATE_SHIFT,
        });
    }
}

fn config_density(configs: &[GpuStructureConfig], offset: u32, count: u32) -> u128 {
    let start = offset as usize;
    let end = start + count as usize;
    configs[start..end].iter().fold(0_u128, |total, config| {
        let region_span = config.region_size as u128 * 16;
        let region_area = region_span * region_span;
        total.saturating_add(GPU_DENSITY_SCALE.div_ceil(region_area))
    })
}

fn predicate_density_score(configs: &[GpuStructureConfig], predicate: &GpuPredicate) -> u128 {
    // 随机分区放置在圆形范围内的期望候选数与 radius² / region_area 成正比。
    // 分数仅用于 AND 短路顺序；使用定点比例避免浮点排序的不确定性。
    let density = config_density(configs, predicate.config_offset, predicate.config_count);
    let radius = u128::from(predicate.radius);
    let minimum = u128::from(predicate.minimum.max(1));
    radius.saturating_mul(radius).saturating_mul(density) / minimum
}

fn predicate_supports_pair_filter(
    configs: &[GpuStructureConfig],
    predicate: &GpuPredicate,
) -> bool {
    let start = predicate.config_offset as usize;
    let end = start + predicate.config_count as usize;
    configs[start..end]
        .iter()
        .all(|config| config.kind != native::GPU_PLACEMENT_STRONGHOLD)
}

fn predicate_order_key(configs: &[GpuStructureConfig], predicate: &GpuPredicate) -> (bool, u128) {
    // Concentric rings require sequential trigonometry per seed. Evaluate
    // cheaper random-spread predicates first so most candidates short-circuit.
    (
        !predicate_supports_pair_filter(configs, predicate),
        predicate_density_score(configs, predicate),
    )
}

fn pair_left_scan_score(
    configs: &[GpuStructureConfig],
    predicate: &GpuPairPredicate,
    swapped: bool,
) -> u128 {
    let (offset, count, radius) = if swapped {
        (
            predicate.right_config_offset,
            predicate.right_config_count,
            predicate.right_radius,
        )
    } else {
        (
            predicate.left_config_offset,
            predicate.left_config_count,
            predicate.left_radius,
        )
    };
    let envelope = u128::from(predicate.anchor_radius) + u128::from(radius);
    envelope
        .saturating_mul(envelope)
        .saturating_mul(config_density(configs, offset, count))
}

fn predicate_anchor_scan_score(
    configs: &[GpuStructureConfig],
    predicate: &GpuPredicate,
    anchor_radius: u32,
) -> u128 {
    let envelope = u128::from(anchor_radius) + u128::from(predicate.radius);
    envelope
        .saturating_mul(envelope)
        .saturating_mul(config_density(
            configs,
            predicate.config_offset,
            predicate.config_count,
        ))
}

fn pair_density_score(configs: &[GpuStructureConfig], predicate: &GpuPairPredicate) -> u128 {
    // 期望共址对数量与左侧原点包络面积、右侧相对搜索面积及两侧密度的乘积成正比。
    let left_envelope = u128::from(predicate.anchor_radius) + u128::from(predicate.left_radius);
    let shift = predicate.left_coordinate_shift & GPU_PAIR_SHIFT_MASK;
    let scale = 1_u128 << shift;
    let pair_radius = if shift == 0 {
        u128::from(predicate.left_radius) + u128::from(predicate.right_radius)
    } else {
        u128::from(predicate.left_radius).div_ceil(scale) + u128::from(predicate.right_radius) + 2
    };
    let left_candidates = left_envelope
        .saturating_mul(left_envelope)
        .saturating_mul(config_density(
            configs,
            predicate.left_config_offset,
            predicate.left_config_count,
        ))
        .div_ceil(GPU_DENSITY_SCALE);
    let right_candidates = pair_radius
        .saturating_mul(pair_radius)
        .saturating_mul(config_density(
            configs,
            predicate.right_config_offset,
            predicate.right_config_count,
        ))
        .div_ceil(GPU_DENSITY_SCALE);
    left_candidates.saturating_mul(right_candidates)
}

pub enum SearchAccelerator {
    Cpu(AcceleratorInfo),
    Gpu {
        info: AcceleratorInfo,
        plan: Box<GpuPlan>,
        backend: BackendContext,
        coarse_backend: Option<BackendContext>,
        outer_coarse_backend: Option<BackendContext>,
        pre_spawn_backend: Option<BackendContext>,
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
        let spawn_config = plan
            .needs_spawn()
            .then(native::gpu_spawn_config)
            .transpose()?
            .flatten();
        let backends = match (|| -> Result<_> {
            let backend = BackendContext::create_with_spawn(
                device,
                &plan.configs,
                &plan.predicates,
                &[],
                spawn_config.as_ref(),
            )?;
            let coarse_backend = plan
                .coarse_predicates
                .as_ref()
                .map(|predicates| BackendContext::create(device, &plan.configs, predicates, &[]))
                .transpose()?;
            let outer_coarse_backend = plan
                .outer_coarse_predicates
                .as_ref()
                .map(|predicates| {
                    BackendContext::create(
                        device,
                        &plan.configs,
                        predicates,
                        &plan.outer_pair_predicates,
                    )
                })
                .transpose()?;
            let pre_spawn_backend = plan
                .has_pre_spawn_stage()
                .then(|| {
                    BackendContext::create(
                        device,
                        &plan.configs,
                        &plan.pre_spawn_predicates,
                        &plan.pair_predicates,
                    )
                })
                .transpose()?;
            Ok((
                backend,
                coarse_backend,
                outer_coarse_backend,
                pre_spawn_backend,
            ))
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
                predicate_count: plan.predicates.len() + plan.pair_predicates.len(),
            },
            plan: Box::new(plan),
            backend: backends.0,
            coarse_backend: backends.1,
            outer_coarse_backend: backends.2,
            pre_spawn_backend: backends.3,
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

    pub const fn has_pre_spawn_stage(&self) -> bool {
        match self {
            Self::Gpu { plan, .. } => plan.has_pre_spawn_stage(),
            Self::Cpu(_) => false,
        }
    }

    pub const fn has_outer_coarse_stage(&self) -> bool {
        match self {
            Self::Gpu { plan, .. } => plan.has_outer_coarse_stage(),
            Self::Cpu(_) => false,
        }
    }

    pub const fn has_gpu_spawn_estimator(&self) -> bool {
        match self {
            Self::Gpu { backend, .. } => backend.has_spawn_estimator(),
            Self::Cpu(_) => false,
        }
    }

    pub fn estimate_spawns(&mut self, candidates: &[GpuCandidate]) -> Result<Vec<GpuCandidate>> {
        match self {
            Self::Gpu { backend, .. } => backend.estimate_spawns(candidates),
            Self::Cpu(_) => bail!("内部错误：CPU 搜索调用了 GPU 出生点估算"),
        }
    }

    pub fn estimate_spawn_outer(
        &mut self,
        candidates: &[GpuCandidate],
    ) -> Result<Vec<GpuCandidate>> {
        match self {
            Self::Gpu { backend, .. } => backend.estimate_spawn_outer(candidates),
            Self::Cpu(_) => bail!("内部错误：CPU 搜索调用了 GPU 外层出生点估算"),
        }
    }

    pub fn refine_spawn_estimates(
        &mut self,
        candidates: &[GpuCandidate],
        state_indices: &[u32],
    ) -> Result<Vec<GpuCandidate>> {
        match self {
            Self::Gpu { backend, .. } => backend.refine_spawn_estimates(candidates, state_indices),
            Self::Cpu(_) => bail!("内部错误：CPU 搜索调用了 GPU 内层出生点细化"),
        }
    }

    pub fn filter_pre_spawn(&mut self, candidates: &[GpuCandidate]) -> Result<Vec<u8>> {
        match self {
            Self::Gpu {
                pre_spawn_backend: Some(backend),
                ..
            } => backend.filter(candidates),
            Self::Gpu { .. } => bail!("内部错误：当前 GPU 计划没有出生点前共址预筛阶段"),
            Self::Cpu(_) => bail!("内部错误：CPU 搜索调用了 GPU 出生点前共址预筛"),
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

    pub fn filter_outer_coarse(&mut self, candidates: &[GpuCandidate]) -> Result<Vec<u8>> {
        match self {
            Self::Gpu {
                outer_coarse_backend: Some(backend),
                ..
            } => backend.filter(candidates),
            Self::Gpu { .. } => bail!("内部错误：当前 GPU 计划没有外层出生点粗筛阶段"),
            Self::Cpu(_) => bail!("内部错误：CPU 搜索调用了 GPU 外层出生点粗筛"),
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
        pair_predicates: *const GpuPairPredicate,
        pair_predicate_count: usize,
        spawn_config: *const GpuSpawnConfig,
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
    fn mcseed_gpu_estimate_spawns(
        context: *mut RawBackendContext,
        candidates: *const GpuCandidate,
        candidate_count: usize,
        estimates: *mut GpuCandidate,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn mcseed_gpu_estimate_spawn_outer(
        context: *mut RawBackendContext,
        candidates: *const GpuCandidate,
        candidate_count: usize,
        estimates: *mut GpuCandidate,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn mcseed_gpu_refine_cached_spawn_estimates(
        context: *mut RawBackendContext,
        candidates: *const GpuCandidate,
        state_indices: *const u32,
        candidate_count: usize,
        estimates: *mut GpuCandidate,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) struct BackendContext {
    raw: NonNull<RawBackendContext>,
    spawn_estimator: bool,
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
        pair_predicates: &[GpuPairPredicate],
    ) -> Result<Self> {
        Self::create_with_spawn(device, configs, predicates, pair_predicates, None)
    }

    fn create_with_spawn(
        device: u32,
        configs: &[GpuStructureConfig],
        predicates: &[GpuPredicate],
        pair_predicates: &[GpuPairPredicate],
        spawn_config: Option<&GpuSpawnConfig>,
    ) -> Result<Self> {
        let device = c_int::try_from(device).context("GPU 设备索引超出 i32")?;
        let mut error = error_buffer();
        let predicates_pointer = if predicates.is_empty() {
            std::ptr::null()
        } else {
            predicates.as_ptr()
        };
        let pair_predicates_pointer = if pair_predicates.is_empty() {
            std::ptr::null()
        } else {
            pair_predicates.as_ptr()
        };
        let spawn_config_pointer = spawn_config.map_or(std::ptr::null(), std::ptr::from_ref);
        let raw = unsafe {
            mcseed_gpu_context_create(
                device,
                configs.as_ptr(),
                configs.len(),
                predicates_pointer,
                predicates.len(),
                pair_predicates_pointer,
                pair_predicates.len(),
                spawn_config_pointer,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let raw = NonNull::new(raw)
            .ok_or_else(|| anyhow!("创建 GPU 上下文失败：{}", error_message(&error)))?;
        Ok(Self {
            raw,
            spawn_estimator: spawn_config.is_some(),
        })
    }

    const fn has_spawn_estimator(&self) -> bool {
        self.spawn_estimator
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

    fn estimate_spawns(&mut self, candidates: &[GpuCandidate]) -> Result<Vec<GpuCandidate>> {
        self.run_spawn_estimator(
            candidates,
            mcseed_gpu_estimate_spawns,
            "执行 GPU 出生点估算",
        )
    }

    fn estimate_spawn_outer(&mut self, candidates: &[GpuCandidate]) -> Result<Vec<GpuCandidate>> {
        self.run_spawn_estimator(
            candidates,
            mcseed_gpu_estimate_spawn_outer,
            "执行 GPU 外层出生点估算",
        )
    }

    fn refine_spawn_estimates(
        &mut self,
        candidates: &[GpuCandidate],
        state_indices: &[u32],
    ) -> Result<Vec<GpuCandidate>> {
        if !self.spawn_estimator {
            bail!("当前 GPU 上下文未启用出生点估算");
        }
        if candidates.len() != state_indices.len() {
            bail!("GPU 内层出生点候选数与缓存状态索引数不一致");
        }
        let mut estimates = candidates.to_vec();
        let mut error = error_buffer();
        let status = unsafe {
            mcseed_gpu_refine_cached_spawn_estimates(
                self.raw.as_ptr(),
                candidates.as_ptr(),
                state_indices.as_ptr(),
                candidates.len(),
                estimates.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure_backend_status(status, &error, "执行 GPU 内层出生点细化")?;
        if estimates
            .iter()
            .zip(candidates)
            .any(|(estimate, candidate)| estimate.seed != candidate.seed)
        {
            bail!("GPU 内层出生点细化改变了候选种子顺序");
        }
        Ok(estimates)
    }

    fn run_spawn_estimator(
        &mut self,
        candidates: &[GpuCandidate],
        estimator: unsafe extern "C" fn(
            *mut RawBackendContext,
            *const GpuCandidate,
            usize,
            *mut GpuCandidate,
            *mut c_char,
            usize,
        ) -> c_int,
        operation: &str,
    ) -> Result<Vec<GpuCandidate>> {
        if !self.spawn_estimator {
            bail!("当前 GPU 上下文未启用出生点估算");
        }
        let mut estimates = candidates.to_vec();
        let mut error = error_buffer();
        let status = unsafe {
            estimator(
                self.raw.as_ptr(),
                candidates.as_ptr(),
                candidates.len(),
                estimates.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure_backend_status(status, &error, operation)?;
        if estimates
            .iter()
            .zip(candidates)
            .any(|(estimate, candidate)| estimate.seed != candidate.seed)
        {
            bail!("{operation}改变了候选种子顺序");
        }
        Ok(estimates)
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
        _pair_predicates: &[GpuPairPredicate],
    ) -> Result<Self> {
        bail!("当前二进制未包含 GPU 后端")
    }

    fn create_with_spawn(
        _device: u32,
        _configs: &[GpuStructureConfig],
        _predicates: &[GpuPredicate],
        _pair_predicates: &[GpuPairPredicate],
        _spawn_config: Option<&GpuSpawnConfig>,
    ) -> Result<Self> {
        bail!("当前二进制未包含 GPU 后端")
    }

    const fn has_spawn_estimator(&self) -> bool {
        false
    }

    fn filter(&mut self, _candidates: &[GpuCandidate]) -> Result<Vec<u8>> {
        bail!("当前二进制未包含 GPU 后端")
    }

    fn estimate_spawns(&mut self, _candidates: &[GpuCandidate]) -> Result<Vec<GpuCandidate>> {
        bail!("当前二进制未包含 GPU 后端")
    }

    fn estimate_spawn_outer(&mut self, _candidates: &[GpuCandidate]) -> Result<Vec<GpuCandidate>> {
        bail!("当前二进制未包含 GPU 后端")
    }

    fn refine_spawn_estimates(
        &mut self,
        _candidates: &[GpuCandidate],
        _state_indices: &[u32],
    ) -> Result<Vec<GpuCandidate>> {
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
        AcceleratorInfo, AcceleratorKind, GPU_ANCHOR_COORDINATES, GPU_ANCHOR_END_GATEWAY,
        GPU_NETHER_COORDINATE_SHIFT, GPU_PAIR_ANCHOR_NETHER_SPAWN, GPU_PAIR_ANCHOR_SPAWN,
        GPU_PAIR_GROUPED, GPU_PAIR_SHIFT_MASK, GpuPlan, SearchAccelerator, pair_density_score,
        predicate_density_score,
    };
    use crate::config::{CompiledFilter, conditions_from_flags};
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    use crate::native::gpu_spawn_config;
    use crate::native::{
        GpuCandidate, NativeContext, gpu_reference_filter, gpu_reference_pair_filter,
        gpu_structure_config, structure_by_name, structures,
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
    fn registry_exposes_every_gpu_prefilter_family() {
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
        assert_eq!(unsupported, ["mineshaft"]);
    }

    #[test]
    fn plan_extracts_only_logically_required_conditions() {
        let filter = filter_from_structure("village:1024");
        let plan = GpuPlan::compile(&filter).expect("GPU 计划");
        assert_eq!(plan.predicates.len(), 1);
        assert_eq!(plan.configs.len(), 1);
        assert!(plan.needs_spawn());
        assert!(plan.has_coarse_stage());
        assert!(plan.has_outer_coarse_stage());
        assert_eq!(
            plan.coarse_predicates.as_ref().expect("粗筛条件")[0].radius,
            1_149
        );
        assert_eq!(
            plan.outer_coarse_predicates.as_ref().expect("外层粗筛条件")[0].radius,
            1_673
        );
        assert_eq!(plan.pre_spawn_predicates[0].radius, 3_721);

        let combined = filter_from_structure("village,stronghold:1024");
        let combined_plan = GpuPlan::compile(&combined).expect("GPU 计划");
        assert_eq!(combined_plan.predicates.len(), 1);
        assert_eq!(combined_plan.configs.len(), 2);
        assert!(
            combined_plan
                .configs
                .iter()
                .any(|config| config.kind == crate::native::GPU_PLACEMENT_STRONGHOLD)
        );
    }

    #[test]
    fn end_gateway_plan_uses_a_conservative_seed_dependent_anchor() {
        let filter = filter_from_structure("end_city:500");
        let plan = GpuPlan::compile(&filter).expect("末地折跃门 GPU 计划");
        assert_eq!(plan.predicates.len(), 1);
        assert_eq!(plan.predicates[0].anchor_kind, GPU_ANCHOR_END_GATEWAY);
        assert_eq!(plan.predicates[0].radius, 820);
        assert!(!plan.needs_spawn());
        assert!(!plan.has_coarse_stage());

        let mut native = NativeContext::new().expect("末地生成器");
        let mut exact_hits = 0;
        for seed in 0_i64..16 {
            native.set_seed(seed);
            let exit = native.first_end_gateway_exit().expect("外岛折跃门出口");
            let end_city = structure_by_name("end_city").expect("末地城注册项");
            let exact = native
                .find_structure(end_city.id, exit.x, exit.z, 500, 1, false)
                .expect("精确末地城扫描")
                .found
                > 0;
            let candidate = GpuCandidate {
                seed: seed as u64,
                spawn_x: 0,
                spawn_z: 0,
            };
            let retained = gpu_reference_filter(&[candidate], &plan.configs, &plan.predicates)[0];
            if exact {
                exact_hits += 1;
                assert_eq!(retained, 1, "seed {seed} 的末地城命中被折跃门包络淘汰");
            }
        }
        assert!(exact_hits > 0, "样本应覆盖末地城精确命中");
    }

    #[test]
    fn stronghold_prefilter_runs_last_and_stays_out_of_pair_scans() {
        let specifications = conditions_from_flags(
            &[],
            &[],
            &[
                "stronghold:500".to_owned(),
                "village:100".to_owned(),
                "ruined_portal:100".to_owned(),
            ],
            &[],
            &[],
        )
        .expect("强要塞组合条件应可解析");
        let filter = CompiledFilter::compile(specifications).expect("强要塞组合条件应可编译");
        let plan = GpuPlan::compile(&filter).expect("强要塞组合 GPU 计划");
        let is_stronghold = |predicate: &crate::native::GpuPredicate| {
            let start = predicate.config_offset as usize;
            let end = start + predicate.config_count as usize;
            plan.configs[start..end]
                .iter()
                .any(|config| config.kind == crate::native::GPU_PLACEMENT_STRONGHOLD)
        };
        assert_eq!(plan.predicates.len(), 3);
        assert!(is_stronghold(plan.predicates.last().expect("精筛条件")));
        assert!(is_stronghold(
            plan.coarse_predicates
                .as_ref()
                .expect("粗筛条件")
                .last()
                .expect("强要塞粗筛条件")
        ));
        assert_eq!(plan.pair_predicates.len(), 1);
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
        let mut outer_radii = plan
            .outer_coarse_predicates
            .as_ref()
            .expect("组合查询应有外层粗筛阶段")
            .iter()
            .map(|predicate| predicate.radius)
            .collect::<Vec<_>>();
        outer_radii.sort_unstable();
        assert_eq!(outer_radii, [284, 284, 749, 749, 1_149]);
        assert_eq!(plan.outer_pair_predicates.len(), 8);
        assert!(plan.outer_pair_predicates.iter().all(|predicate| {
            predicate.left_coordinate_shift & (GPU_PAIR_ANCHOR_SPAWN | GPU_PAIR_ANCHOR_NETHER_SPAWN)
                != 0
        }));
        assert_eq!(
            plan.outer_pair_predicates
                .iter()
                .filter(|predicate| predicate.left_coordinate_shift & GPU_PAIR_GROUPED != 0)
                .count(),
            4
        );
        let mut pre_spawn_radii = plan
            .pre_spawn_predicates
            .iter()
            .map(|predicate| predicate.radius)
            .collect::<Vec<_>>();
        pre_spawn_radii.sort_unstable();
        assert_eq!(pre_spawn_radii, [540, 540, 2_797, 2_797, 3_197]);
        assert!(plan.has_pre_spawn_stage());
        assert_eq!(plan.pair_predicates.len(), 8);
        let pair_scores = plan
            .pair_predicates
            .iter()
            .map(|predicate| pair_density_score(&plan.configs, predicate))
            .collect::<Vec<_>>();
        assert!(
            pair_scores[..4]
                .windows(2)
                .all(|window| window[0] <= window[1])
        );
        let mut anchor_radii = plan
            .pair_predicates
            .iter()
            .map(|predicate| predicate.anchor_radius)
            .collect::<Vec<_>>();
        anchor_radii.sort_unstable();
        assert_eq!(
            anchor_radii,
            [340, 2_697, 2_697, 2_697, 2_697, 2_697, 2_697, 2_697,]
        );
        assert_eq!(
            plan.pair_predicates
                .iter()
                .filter(|predicate| predicate.left_coordinate_shift & GPU_PAIR_GROUPED != 0)
                .count(),
            4
        );
        assert_eq!(
            plan.pair_predicates
                .iter()
                .filter(|predicate| {
                    predicate.left_coordinate_shift & GPU_PAIR_SHIFT_MASK
                        == GPU_NETHER_COORDINATE_SHIFT
                })
                .count(),
            2
        );
    }

    #[test]
    fn pre_spawn_pair_filter_never_rejects_exact_colocated_hits() {
        let specifications = conditions_from_flags(
            &[],
            &[],
            &["village:512".to_owned(), "ruined_portal:512".to_owned()],
            &[],
            &[],
        )
        .expect("共址条件应可解析");
        let filter = CompiledFilter::compile(specifications).expect("共址条件应可编译");
        let plan = GpuPlan::compile(&filter).expect("GPU 计划");
        assert_eq!(plan.pair_predicates.len(), 1);

        let mut native = NativeContext::new().expect("原生上下文");
        let mut exact_candidates = Vec::new();
        let mut origin_candidates = Vec::new();
        for seed in 0_i64..128 {
            native.set_seed(seed);
            let spawn = native.spawn_raw().expect("最终出生点");
            exact_candidates.push(GpuCandidate {
                seed: seed as u64,
                spawn_x: spawn.position.x,
                spawn_z: spawn.position.z,
            });
            origin_candidates.push(GpuCandidate {
                seed: seed as u64,
                spawn_x: 0,
                spawn_z: 0,
            });
        }
        let exact = gpu_reference_filter(&exact_candidates, &plan.configs, &plan.predicates);
        let envelopes = gpu_reference_filter(
            &origin_candidates,
            &plan.configs,
            &plan.pre_spawn_predicates,
        );
        let paired =
            gpu_reference_pair_filter(&origin_candidates, &plan.configs, &plan.pair_predicates);
        assert!(exact.contains(&1), "样本应覆盖精确共址命中");
        for (seed, exact_match) in exact.into_iter().enumerate() {
            if exact_match == 1 {
                assert_eq!(envelopes[seed], 1, "seed {seed} 被出生点前原点包络错误淘汰");
                assert_eq!(paired[seed], 1, "seed {seed} 被出生点前预筛错误淘汰");
            }
        }
    }

    #[test]
    fn grouped_cross_dimension_filter_never_rejects_exact_colocated_hits() {
        let specifications = conditions_from_flags(
            &[],
            &[],
            &[
                "village:1024".to_owned(),
                "ruined_portal:1024".to_owned(),
                "ancient_city:1024".to_owned(),
                "fortress:512".to_owned(),
                "bastion_remnant:512".to_owned(),
            ],
            &[],
            &[],
        )
        .expect("跨维共址条件应可解析");
        let filter = CompiledFilter::compile(specifications).expect("跨维共址条件应可编译");
        let plan = GpuPlan::compile(&filter).expect("跨维共址 GPU 计划");
        assert_eq!(plan.pair_predicates.len(), 8);
        assert!(
            plan.pair_predicates[4..]
                .iter()
                .all(|predicate| predicate.left_coordinate_shift & GPU_PAIR_GROUPED != 0)
        );

        let mut native = NativeContext::new().expect("原生上下文");
        let mut exact_candidates = Vec::new();
        let mut origin_candidates = Vec::new();
        for seed in 0_i64..256 {
            native.set_seed(seed);
            let spawn = native.spawn_raw().expect("最终出生点");
            exact_candidates.push(GpuCandidate {
                seed: seed as u64,
                spawn_x: spawn.position.x,
                spawn_z: spawn.position.z,
            });
            origin_candidates.push(GpuCandidate {
                seed: seed as u64,
                spawn_x: 0,
                spawn_z: 0,
            });
        }
        let exact = gpu_reference_filter(&exact_candidates, &plan.configs, &plan.predicates);
        let paired =
            gpu_reference_pair_filter(&origin_candidates, &plan.configs, &plan.pair_predicates);
        assert!(exact.contains(&1), "样本应覆盖跨维精确共址命中");
        for (seed, exact_match) in exact.into_iter().enumerate() {
            if exact_match == 1 {
                assert_eq!(paired[seed], 1, "seed {seed} 被跨维共址预筛错误淘汰");
            }
        }
    }

    #[test]
    fn outer_anchor_pair_filter_never_rejects_exact_colocated_hits() {
        let specifications = conditions_from_flags(
            &[],
            &[],
            &[
                "village:1024".to_owned(),
                "ruined_portal:1024".to_owned(),
                "fortress:512".to_owned(),
                "bastion_remnant:512".to_owned(),
            ],
            &[],
            &[],
        )
        .expect("外层共锚条件应可解析");
        let filter = CompiledFilter::compile(specifications).expect("外层共锚条件应可编译");
        let plan = GpuPlan::compile(&filter).expect("外层共锚 GPU 计划");
        assert_eq!(plan.outer_pair_predicates.len(), 5);

        let mut native = NativeContext::new().expect("原生上下文");
        let mut exact_candidates = Vec::new();
        let mut displaced_candidates = Vec::new();
        for seed in 0_i64..256 {
            native.set_seed(seed);
            let spawn = native.spawn_raw().expect("最终出生点");
            exact_candidates.push(GpuCandidate {
                seed: seed as u64,
                spawn_x: spawn.position.x,
                spawn_z: spawn.position.z,
            });
            // sqrt(400² + 400²) < 649，覆盖外层锚点允许的保守误差。
            displaced_candidates.push(GpuCandidate {
                seed: seed as u64,
                spawn_x: spawn.position.x + 400,
                spawn_z: spawn.position.z - 400,
            });
        }
        let exact = gpu_reference_filter(&exact_candidates, &plan.configs, &plan.predicates);
        let paired = gpu_reference_pair_filter(
            &displaced_candidates,
            &plan.configs,
            &plan.outer_pair_predicates,
        );
        assert!(exact.contains(&1), "样本应覆盖外层精确共锚命中");
        for (seed, exact_match) in exact.into_iter().enumerate() {
            if exact_match == 1 {
                assert_eq!(paired[seed], 1, "seed {seed} 被外层共锚预筛错误淘汰");
            }
        }
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

    #[test]
    fn stronghold_envelope_never_rejects_biome_adjusted_hits() {
        let stronghold = structure_by_name("stronghold").expect("强要塞注册项");
        let config = gpu_structure_config(stronghold.id)
            .expect("强要塞 GPU 配置")
            .expect("当前版本应支持强要塞 GPU 粗筛");
        assert_eq!(config.kind, crate::native::GPU_PLACEMENT_STRONGHOLD);
        assert_eq!(
            (
                config.salt,
                config.region_size,
                config.chunk_range,
                config.reserved,
            ),
            (128, 32, 3, 144)
        );
        let candidates = (-16_i64..16)
            .map(|seed| GpuCandidate {
                seed: seed as u64,
                spawn_x: 0,
                spawn_z: 0,
            })
            .collect::<Vec<_>>();
        let cases = [
            (0, 0, 3_000, 2_u64),
            (2_000, 0, 500, 1_u64),
            (-1_800, 800, 600, 1_u64),
        ];
        let mut native = NativeContext::new().expect("原生上下文");
        let mut exact_hits = 0;
        for (anchor_x, anchor_z, radius, minimum) in cases {
            let predicate = crate::native::GpuPredicate {
                config_offset: 0,
                config_count: 1,
                radius,
                anchor_kind: GPU_ANCHOR_COORDINATES,
                anchor_x,
                anchor_z,
                minimum,
            };
            let placement = gpu_reference_filter(&candidates, &[config], &[predicate]);
            for (index, candidate) in candidates.iter().enumerate() {
                native.set_seed(candidate.seed as i64);
                let exact = native
                    .find_structure(stronghold.id, anchor_x, anchor_z, radius, minimum, false)
                    .expect("精确强要塞扫描")
                    .found
                    >= minimum;
                if exact {
                    exact_hits += 1;
                    assert_eq!(
                        placement[index], 1,
                        "seed {} 的精确强要塞命中被 GPU 包络淘汰",
                        candidate.seed as i64
                    );
                }
            }
        }
        assert!(exact_hits >= 32, "样本应覆盖足够多的强要塞精确命中");
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
                outer_coarse_predicates: None,
                outer_pair_predicates: Vec::new(),
                pre_spawn_predicates: Vec::new(),
                pair_predicates: Vec::new(),
                needs_spawn: false,
            };
            let expected = gpu_reference_filter(&candidates, &plan.configs, &plan.predicates);
            saw_rejected |= expected.contains(&0);
            saw_retained |= expected.contains(&1);
            let mut backend = BackendContext::create(0, &plan.configs, &plan.predicates, &[])
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

        {
            let stronghold = structure_by_name("stronghold").expect("强要塞注册项");
            let config = gpu_structure_config(stronghold.id)
                .expect("强要塞 GPU 配置")
                .expect("当前版本应支持强要塞 GPU 粗筛");
            let predicate = crate::native::GpuPredicate {
                config_offset: 0,
                config_count: 1,
                radius: 500,
                anchor_kind: GPU_ANCHOR_COORDINATES,
                anchor_x: 2_000,
                anchor_z: 0,
                minimum: 1,
            };
            let expected = gpu_reference_filter(&candidates, &[config], &[predicate]);
            assert!(expected.contains(&0) && expected.contains(&1));
            let mut backend = BackendContext::create(0, &[config], &[predicate], &[])
                .expect("创建强要塞 GPU 粗筛上下文");
            let actual = backend.filter(&candidates).expect("执行强要塞 GPU 粗筛");
            assert_eq!(actual, expected, "强要塞 GPU 包络与 CPU 参考不一致");
        }

        {
            let plan = GpuPlan::compile(&filter_from_structure("end_city:500"))
                .expect("末地折跃门 GPU 计划");
            let expected = gpu_reference_filter(&candidates, &plan.configs, &plan.predicates);
            let mut backend = BackendContext::create(0, &plan.configs, &plan.predicates, &[])
                .expect("创建末地折跃门 GPU 粗筛上下文");
            let actual = backend
                .filter(&candidates)
                .expect("执行末地折跃门 GPU 粗筛");
            assert_eq!(actual, expected, "末地折跃门 GPU 包络与 CPU 参考不一致");
        }

        let specifications = conditions_from_flags(
            &[],
            &[],
            &["fortress:200".to_owned(), "bastion_remnant:200".to_owned()],
            &[],
            &[],
        )
        .expect("下界共址条件应可解析");
        let filter = CompiledFilter::compile(specifications).expect("下界共址条件应可编译");
        let plan = GpuPlan::compile(&filter).expect("下界共址 GPU 计划");
        let expected = gpu_reference_pair_filter(&candidates, &plan.configs, &plan.pair_predicates);
        assert!(expected.contains(&0) && expected.contains(&1));
        let mut backend = BackendContext::create(0, &plan.configs, &[], &plan.pair_predicates)
            .expect("创建 GPU 共址预筛上下文");
        let actual = backend.filter(&candidates).expect("执行 GPU 共址预筛内核");
        assert_eq!(actual, expected, "GPU 共址预筛结果与 CPU 参考实现不一致");

        let specifications = conditions_from_flags(
            &[],
            &[],
            &[
                "village:1024".to_owned(),
                "ruined_portal:1024".to_owned(),
                "ancient_city:1024".to_owned(),
                "fortress:512".to_owned(),
                "bastion_remnant:512".to_owned(),
            ],
            &[],
            &[],
        )
        .expect("跨维共址条件应可解析");
        let filter = CompiledFilter::compile(specifications).expect("跨维共址条件应可编译");
        let plan = GpuPlan::compile(&filter).expect("跨维共址 GPU 计划");
        let expected = gpu_reference_pair_filter(&candidates, &plan.configs, &plan.pair_predicates);
        assert_eq!(expected.len(), candidates.len());
        let mut backend = BackendContext::create(0, &plan.configs, &[], &plan.pair_predicates)
            .expect("创建 GPU 跨维共址预筛上下文");
        let actual = backend
            .filter(&candidates)
            .expect("执行 GPU 跨维共址预筛内核");
        assert_eq!(actual, expected, "GPU 跨维共址结果与 CPU 参考实现不一致");

        let outer_candidates = candidates
            .iter()
            .map(|candidate| GpuCandidate {
                seed: candidate.seed,
                spawn_x: (candidate.seed as i64 as i32).wrapping_mul(3),
                spawn_z: (candidate.seed as i64 as i32).wrapping_mul(-2),
            })
            .collect::<Vec<_>>();
        let expected = gpu_reference_pair_filter(
            &outer_candidates,
            &plan.configs,
            &plan.outer_pair_predicates,
        );
        assert!(expected.contains(&0) && expected.contains(&1));
        let mut backend =
            BackendContext::create(0, &plan.configs, &[], &plan.outer_pair_predicates)
                .expect("创建 GPU 外层共锚预筛上下文");
        let actual = backend
            .filter(&outer_candidates)
            .expect("执行 GPU 外层共锚预筛内核");
        assert_eq!(actual, expected, "GPU 外层共锚结果与 CPU 参考实现不一致");
    }

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    fn assert_gpu_spawn_estimator_matches_cpu(start: i64, end: i64) {
        let Ok(device_count) = BackendContext::device_count() else {
            return;
        };
        if device_count == 0 {
            return;
        }
        let filter = filter_from_structure("village:128");
        let plan = GpuPlan::compile(&filter).expect("GPU 计划");
        let spawn_config = gpu_spawn_config()
            .expect("GPU 出生点配置")
            .expect("当前版本应支持 GPU 出生点估算");
        let candidates = (start..end)
            .map(|seed| GpuCandidate {
                seed: seed as u64,
                spawn_x: 0,
                spawn_z: 0,
            })
            .collect::<Vec<_>>();
        let mut backend = BackendContext::create_with_spawn(
            0,
            &plan.configs,
            &plan.predicates,
            &[],
            Some(&spawn_config),
        )
        .expect("创建 GPU 出生点上下文");
        let gpu_started = std::time::Instant::now();
        let outer = backend
            .estimate_spawn_outer(&candidates)
            .expect("执行 GPU 外层出生点估算");
        let state_indices = (0..outer.len())
            .map(|index| u32::try_from(index).expect("GPU 出生点状态索引"))
            .collect::<Vec<_>>();
        let staged = backend
            .refine_spawn_estimates(&outer, &state_indices)
            .expect("执行 GPU 内层出生点细化");
        let actual = backend
            .estimate_spawns(&candidates)
            .expect("执行 GPU 出生点估算");
        let gpu_elapsed = gpu_started.elapsed();
        assert_eq!(staged, actual, "分级 GPU 出生点估算必须与完整内核一致");
        let mut native = NativeContext::new().expect("CPU 世界生成器");
        let cpu_started = std::time::Instant::now();
        let refinement_radius = crate::native::spawn_refinement_radius()
            .expect("出生点细化半径")
            .expect("当前版本应支持出生点细化");
        let outer_bound =
            spawn_config.inner_radius + spawn_config.output_rounding_radius + refinement_radius;
        for ((candidate, outer), estimate) in candidates.iter().zip(outer).zip(actual) {
            native.set_seed(candidate.seed as i64);
            let expected = native.estimated_spawn().expect("CPU 出生点估算");
            assert_eq!(
                (estimate.spawn_x, estimate.spawn_z),
                (expected.x, expected.z),
                "seed {} 的 GPU 出生点估算不一致",
                candidate.seed as i64
            );
            let spawn = native.spawn_raw().expect("CPU 最终出生点");
            let dx = i64::from(spawn.position.x) - i64::from(outer.spawn_x);
            let dz = i64::from(spawn.position.z) - i64::from(outer.spawn_z);
            assert!(
                dx * dx + dz * dz <= i64::from(outer_bound) * i64::from(outer_bound),
                "seed {} 的最终出生点超出 GPU 外层安全包络",
                candidate.seed as i64
            );
        }
        eprintln!(
            "出生点估算对照：{} seeds，GPU {:.3}s，CPU {:.3}s",
            candidates.len(),
            gpu_elapsed.as_secs_f64(),
            cpu_started.elapsed().as_secs_f64()
        );
    }

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    #[test]
    fn compiled_gpu_spawn_estimator_matches_cpu_when_available() {
        assert_gpu_spawn_estimator_matches_cpu(-128, 128);
    }

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    #[test]
    #[ignore = "有限规模 GPU/CPU 精确性验证；不纳入日常测试"]
    fn compiled_gpu_spawn_estimator_matches_cpu_for_ten_thousand_seeds() {
        assert_gpu_spawn_estimator_matches_cpu(0, 10_000);
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
