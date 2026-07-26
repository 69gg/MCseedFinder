use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_ulonglong};
use std::ptr::NonNull;

use anyhow::{Context, Result, anyhow, bail};

use crate::domain::{Dimension, Position, SpawnInfo};

const HIT_CAPACITY: usize = 8;

#[repr(C)]
struct RawContext {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawHit {
    x: i32,
    y: i32,
    z: i32,
    id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawPieceHit {
    x: i32,
    y: i32,
    z: i32,
    parent_x: i32,
    parent_z: i32,
    eye_mask: i32,
    name: *const c_char,
}

impl Default for RawPieceHit {
    fn default() -> Self {
        Self {
            x: 0,
            y: i32::MIN,
            z: 0,
            parent_x: 0,
            parent_z: 0,
            eye_mask: -1,
            name: std::ptr::null(),
        }
    }
}

unsafe extern "C" {
    fn mcseed_context_create() -> *mut RawContext;
    fn mcseed_context_destroy(context: *mut RawContext);
    fn mcseed_context_set_seed(context: *mut RawContext, seed: c_ulonglong);
    fn mcseed_spawn(context: *mut RawContext, spawn: *mut RawHit, biome_id: *mut c_int) -> c_int;
    fn mcseed_estimated_spawn(context: *mut RawContext, spawn: *mut RawHit) -> c_int;
    fn mcseed_spawn_refinement_radius(radius: *mut c_uint) -> c_int;

    fn mcseed_biome_count() -> c_int;
    fn mcseed_biome_name_at(index: c_int) -> *const c_char;
    fn mcseed_biome_id_at(index: c_int) -> c_int;
    fn mcseed_biome_dimension_at(index: c_int) -> c_int;
    fn mcseed_biome_id_from_name(name: *const c_char) -> c_int;
    fn mcseed_biome_name_from_id(id: c_int) -> *const c_char;

    fn mcseed_structure_count() -> c_int;
    fn mcseed_structure_name_at(index: c_int) -> *const c_char;
    fn mcseed_structure_id_at(index: c_int) -> c_int;
    fn mcseed_structure_dimension_at(index: c_int) -> c_int;
    fn mcseed_structure_accuracy_at(index: c_int) -> c_int;
    fn mcseed_structure_id_from_name(name: *const c_char) -> c_int;
    fn mcseed_structure_gpu_config(structure_id: c_int, config: *mut GpuStructureConfig) -> c_int;

    fn mcseed_piece_count() -> c_int;
    fn mcseed_piece_name_at(index: c_int) -> *const c_char;
    fn mcseed_piece_structure_id_at(index: c_int) -> c_int;
    fn mcseed_piece_accuracy_at(index: c_int) -> c_int;
    fn mcseed_piece_is_group_at(index: c_int) -> c_int;
    fn mcseed_piece_selector_valid(structure_id: c_int, name: *const c_char) -> c_int;

    fn mcseed_find_biomes(
        context: *mut RawContext,
        dimension: c_int,
        biome_ids: *const c_int,
        biome_count: usize,
        anchor_x: c_int,
        anchor_z: c_int,
        radius: c_uint,
        y_min: c_int,
        y_max: c_int,
        limit: c_ulonglong,
        hits: *mut RawHit,
        hit_capacity: usize,
        found: *mut c_ulonglong,
        limit_reached: *mut c_int,
    ) -> c_int;

    fn mcseed_find_structure(
        context: *mut RawContext,
        structure_id: c_int,
        anchor_x: c_int,
        anchor_z: c_int,
        radius: c_uint,
        limit: c_ulonglong,
        hits: *mut RawHit,
        hit_capacity: usize,
        found: *mut c_ulonglong,
        limit_reached: *mut c_int,
    ) -> c_int;

    fn mcseed_find_structure_pieces(
        context: *mut RawContext,
        structure_id: c_int,
        selectors: *const *const c_char,
        selector_count: usize,
        anchor_x: c_int,
        anchor_z: c_int,
        radius: c_uint,
        limit: c_ulonglong,
        hits: *mut RawPieceHit,
        hit_capacity: usize,
        found: *mut c_ulonglong,
        limit_reached: *mut c_int,
    ) -> c_int;

    fn mcseed_nearest_stronghold_portal(
        context: *mut RawContext,
        anchor_x: c_int,
        anchor_z: c_int,
        hit: *mut RawPieceHit,
    ) -> c_int;

    #[cfg(test)]
    fn mcseed_gpu_reference_filter(
        candidates: *const GpuCandidate,
        candidate_count: usize,
        configs: *const GpuStructureConfig,
        config_count: usize,
        predicates: *const GpuPredicate,
        predicate_count: usize,
        matches: *mut u8,
    );
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GpuStructureConfig {
    pub kind: i32,
    pub salt: i32,
    pub region_size: i32,
    pub chunk_range: i32,
    pub flags: i32,
    pub reserved: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GpuPredicate {
    pub config_offset: u32,
    pub config_count: u32,
    pub radius: u32,
    pub anchor_kind: i32,
    pub anchor_x: i32,
    pub anchor_z: i32,
    pub minimum: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GpuCandidate {
    pub seed: u64,
    pub spawn_x: i32,
    pub spawn_z: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BiomeInfo {
    pub id: i32,
    pub name: String,
    pub dimension: Dimension,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructureInfo {
    pub id: i32,
    pub name: String,
    pub dimension: Dimension,
    pub accuracy: StructureAccuracy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PieceInfo {
    pub structure_id: i32,
    pub name: String,
    pub accuracy: PieceAccuracy,
    pub is_group: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceAccuracy {
    Exact,
    ApproximateTerrain,
    Partial,
}

impl PieceAccuracy {
    pub const fn description(self) -> &'static str {
        match self {
            Self::Exact => "部件类型与水平布局精确",
            Self::ApproximateTerrain => "模板池与随机序列精确，地表高度近似",
            Self::Partial => "底层只实现该结构的稳定部件子集",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructureAccuracy {
    PlacementAndBiome,
    ApproximateTerrain,
    CandidateOnly,
}

impl StructureAccuracy {
    pub const fn description(self) -> &'static str {
        match self {
            Self::PlacementAndBiome => "放置、生物群系及可用地形校验",
            Self::ApproximateTerrain => "放置与生物群系精确，地表高度近似",
            Self::CandidateOnly => "放置与生物群系精确，内部地形可能使生成失败",
        }
    }
}

#[derive(Debug, Clone)]
pub struct NativeHit {
    pub position: Position,
    pub id: i32,
}

#[derive(Debug, Clone)]
pub struct ScanResult {
    pub found: u64,
    pub limit_reached: bool,
    pub hits: Vec<NativeHit>,
}

#[derive(Debug, Clone)]
pub struct NativePieceHit {
    pub name: String,
    pub position: Position,
    pub parent_position: Position,
    pub eye_mask: Option<u16>,
}

#[derive(Debug, Clone)]
pub struct PieceScanResult {
    pub found: u64,
    pub limit_reached: bool,
    pub hits: Vec<NativePieceHit>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NativeSpawn {
    pub position: Position,
    pub biome_id: i32,
}

impl NativeSpawn {
    pub fn into_info(self) -> Result<SpawnInfo> {
        let biome = biome_name(self.biome_id)
            .ok_or_else(|| anyhow!("出生点返回了未知生物群系 ID {}", self.biome_id))?;
        Ok(SpawnInfo {
            position: self.position,
            biome,
            biome_id: self.biome_id,
        })
    }
}

pub struct NativeContext {
    raw: NonNull<RawContext>,
}

impl NativeContext {
    pub fn new() -> Result<Self> {
        let raw = unsafe { mcseed_context_create() };
        let raw = NonNull::new(raw).ok_or_else(|| anyhow!("无法分配世界生成器"))?;
        Ok(Self { raw })
    }

    pub fn set_seed(&mut self, seed: i64) {
        unsafe { mcseed_context_set_seed(self.raw.as_ptr(), seed as u64) };
    }

    pub(crate) fn spawn_raw(&mut self) -> Result<NativeSpawn> {
        let mut raw_hit = RawHit::default();
        let mut biome_id = -1;
        let status = unsafe { mcseed_spawn(self.raw.as_ptr(), &mut raw_hit, &mut biome_id) };
        ensure_status(status, "计算出生点")?;
        Ok(NativeSpawn {
            position: Position {
                x: raw_hit.x,
                y: Some(raw_hit.y),
                z: raw_hit.z,
            },
            biome_id,
        })
    }

    pub fn spawn(&mut self) -> Result<SpawnInfo> {
        self.spawn_raw()?.into_info()
    }

    pub(crate) fn estimated_spawn(&mut self) -> Result<Position> {
        let mut raw_hit = RawHit::default();
        let status = unsafe { mcseed_estimated_spawn(self.raw.as_ptr(), &mut raw_hit) };
        ensure_status(status, "估算出生点")?;
        Ok(Position {
            x: raw_hit.x,
            y: None,
            z: raw_hit.z,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn find_biomes(
        &mut self,
        dimension: Dimension,
        biome_ids: &[i32],
        anchor_x: i32,
        anchor_z: i32,
        radius: u32,
        y_min: i32,
        y_max: i32,
        limit: u64,
        collect_hits: bool,
    ) -> Result<ScanResult> {
        let mut raw_hits = if collect_hits {
            vec![RawHit::default(); HIT_CAPACITY]
        } else {
            Vec::new()
        };
        let mut found = 0;
        let mut limit_reached = 0;
        let (hits_pointer, hit_capacity) = hit_buffer(&mut raw_hits);
        let status = unsafe {
            mcseed_find_biomes(
                self.raw.as_ptr(),
                dimension.as_raw(),
                biome_ids.as_ptr(),
                biome_ids.len(),
                anchor_x,
                anchor_z,
                radius,
                y_min,
                y_max,
                limit,
                hits_pointer,
                hit_capacity,
                &mut found,
                &mut limit_reached,
            )
        };
        ensure_status(status, "扫描生物群系")?;
        Ok(scan_result(raw_hits, found, limit_reached))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn find_structure(
        &mut self,
        structure_id: i32,
        anchor_x: i32,
        anchor_z: i32,
        radius: u32,
        limit: u64,
        collect_hits: bool,
    ) -> Result<ScanResult> {
        let mut raw_hits = if collect_hits {
            vec![RawHit::default(); HIT_CAPACITY]
        } else {
            Vec::new()
        };
        let mut found = 0;
        let mut limit_reached = 0;
        let (hits_pointer, hit_capacity) = hit_buffer(&mut raw_hits);
        let status = unsafe {
            mcseed_find_structure(
                self.raw.as_ptr(),
                structure_id,
                anchor_x,
                anchor_z,
                radius,
                limit,
                hits_pointer,
                hit_capacity,
                &mut found,
                &mut limit_reached,
            )
        };
        ensure_status(status, "扫描结构")?;
        Ok(scan_result(raw_hits, found, limit_reached))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn find_structure_pieces(
        &mut self,
        structure_id: i32,
        selectors: &[String],
        anchor_x: i32,
        anchor_z: i32,
        radius: u32,
        limit: u64,
        collect_hits: bool,
    ) -> Result<PieceScanResult> {
        if selectors.is_empty() {
            bail!("子结构选择器不能为空");
        }
        let selectors = selectors
            .iter()
            .map(|selector| CString::new(selector.as_str()).context("子结构选择器包含 NUL 字符"))
            .collect::<Result<Vec<_>>>()?;
        let selector_pointers: Vec<*const c_char> =
            selectors.iter().map(|selector| selector.as_ptr()).collect();
        let mut raw_hits = if collect_hits {
            vec![RawPieceHit::default(); HIT_CAPACITY]
        } else {
            Vec::new()
        };
        let mut found = 0;
        let mut limit_reached = 0;
        let (hits_pointer, hit_capacity) = if raw_hits.is_empty() {
            (std::ptr::null_mut(), 0)
        } else {
            (raw_hits.as_mut_ptr(), raw_hits.len())
        };
        let status = unsafe {
            mcseed_find_structure_pieces(
                self.raw.as_ptr(),
                structure_id,
                selector_pointers.as_ptr(),
                selector_pointers.len(),
                anchor_x,
                anchor_z,
                radius,
                limit,
                hits_pointer,
                hit_capacity,
                &mut found,
                &mut limit_reached,
            )
        };
        ensure_status(status, "扫描结构子部件")?;
        let stored = usize::try_from(found)
            .unwrap_or(usize::MAX)
            .min(raw_hits.len());
        let hits = raw_hits
            .into_iter()
            .take(stored)
            .map(native_piece_hit)
            .collect::<Result<Vec<_>>>()?;
        Ok(PieceScanResult {
            found,
            limit_reached: limit_reached != 0,
            hits,
        })
    }

    pub fn nearest_stronghold_portal(
        &mut self,
        anchor_x: i32,
        anchor_z: i32,
    ) -> Result<NativePieceHit> {
        let mut raw_hit = RawPieceHit::default();
        let status = unsafe {
            mcseed_nearest_stronghold_portal(self.raw.as_ptr(), anchor_x, anchor_z, &mut raw_hit)
        };
        ensure_status(status, "定位最近要塞传送门房")?;
        native_piece_hit(raw_hit)
    }
}

fn native_piece_hit(hit: RawPieceHit) -> Result<NativePieceHit> {
    let eye_mask = match hit.eye_mask {
        -1 => None,
        0..=0x0fff => Some(u16::try_from(hit.eye_mask).context("末地传送门眼框掩码无效")?),
        value => bail!("末地传送门眼框掩码超出 12 位范围：{value}"),
    };
    Ok(NativePieceHit {
        name: required_c_string(hit.name)?,
        position: Position {
            x: hit.x,
            y: (hit.y != i32::MIN).then_some(hit.y),
            z: hit.z,
        },
        parent_position: Position {
            x: hit.parent_x,
            y: None,
            z: hit.parent_z,
        },
        eye_mask,
    })
}

impl Drop for NativeContext {
    fn drop(&mut self) {
        unsafe { mcseed_context_destroy(self.raw.as_ptr()) };
    }
}

pub fn biomes() -> Result<Vec<BiomeInfo>> {
    let count = unsafe { mcseed_biome_count() };
    if count < 0 {
        bail!("读取生物群系列表失败");
    }
    (0..count)
        .map(|index| {
            let id = unsafe { mcseed_biome_id_at(index) };
            let dimension = unsafe { mcseed_biome_dimension_at(index) };
            Ok(BiomeInfo {
                id,
                name: required_c_string(unsafe { mcseed_biome_name_at(index) })?,
                dimension: Dimension::from_raw(dimension)
                    .ok_or_else(|| anyhow!("生物群系 {id} 的维度无效：{dimension}"))?,
            })
        })
        .collect()
}

pub fn structures() -> Result<Vec<StructureInfo>> {
    let count = unsafe { mcseed_structure_count() };
    if count < 0 {
        bail!("读取结构列表失败");
    }
    (0..count)
        .map(|index| {
            let id = unsafe { mcseed_structure_id_at(index) };
            let dimension = unsafe { mcseed_structure_dimension_at(index) };
            let accuracy = match unsafe { mcseed_structure_accuracy_at(index) } {
                0 => StructureAccuracy::PlacementAndBiome,
                1 => StructureAccuracy::ApproximateTerrain,
                2 => StructureAccuracy::CandidateOnly,
                value => bail!("结构 {id} 的精度等级无效：{value}"),
            };
            Ok(StructureInfo {
                id,
                name: required_c_string(unsafe { mcseed_structure_name_at(index) })?,
                dimension: Dimension::from_raw(dimension)
                    .ok_or_else(|| anyhow!("结构 {id} 的维度无效：{dimension}"))?,
                accuracy,
            })
        })
        .collect()
}

pub fn pieces() -> Result<Vec<PieceInfo>> {
    let count = unsafe { mcseed_piece_count() };
    if count < 0 {
        bail!("读取子结构列表失败");
    }
    (0..count)
        .map(|index| {
            let structure_id = unsafe { mcseed_piece_structure_id_at(index) };
            let accuracy = match unsafe { mcseed_piece_accuracy_at(index) } {
                0 => PieceAccuracy::Exact,
                1 => PieceAccuracy::ApproximateTerrain,
                2 => PieceAccuracy::Partial,
                value => bail!("子结构注册项 {index} 的精度等级无效：{value}"),
            };
            let is_group = match unsafe { mcseed_piece_is_group_at(index) } {
                0 => false,
                1 => true,
                value => bail!("子结构注册项 {index} 的分组标志无效：{value}"),
            };
            Ok(PieceInfo {
                structure_id,
                name: required_c_string(unsafe { mcseed_piece_name_at(index) })?,
                accuracy,
                is_group,
            })
        })
        .collect()
}

pub fn biome_by_name(name: &str) -> Result<BiomeInfo> {
    let normalized = normalize_resource_name(name);
    let c_name = CString::new(normalized.as_str()).context("生物群系名称包含 NUL 字符")?;
    let id = unsafe { mcseed_biome_id_from_name(c_name.as_ptr()) };
    if id < 0 {
        bail!("未知生物群系 {name:?}；运行 `mcseed-finder list biomes` 查看名称");
    }
    let info = biomes()?
        .into_iter()
        .find(|candidate| candidate.id == id)
        .ok_or_else(|| anyhow!("生物群系注册表中缺少 ID {id}"))?;
    Ok(info)
}

pub fn structure_by_name(name: &str) -> Result<StructureInfo> {
    let normalized = normalize_resource_name(name);
    let c_name = CString::new(normalized.as_str()).context("结构名称包含 NUL 字符")?;
    let id = unsafe { mcseed_structure_id_from_name(c_name.as_ptr()) };
    if id < 0 {
        bail!("未知结构 {name:?}；运行 `mcseed-finder list structures` 查看名称");
    }
    structures()?
        .into_iter()
        .find(|candidate| candidate.id == id)
        .ok_or_else(|| anyhow!("结构注册表中缺少 ID {id}"))
}

pub(crate) fn gpu_structure_config(structure_id: i32) -> Result<Option<GpuStructureConfig>> {
    let mut config = GpuStructureConfig::default();
    match unsafe { mcseed_structure_gpu_config(structure_id, &mut config) } {
        1 => {
            if config.kind <= 0 || config.region_size <= 0 || config.chunk_range <= 0 {
                bail!("结构 {structure_id} 返回了无效的 GPU 放置配置");
            }
            Ok(Some(config))
        }
        0 => Ok(None),
        status => bail!("读取结构 {structure_id} 的 GPU 放置配置失败（错误码 {status}）"),
    }
}

pub(crate) fn spawn_refinement_radius() -> Result<Option<u32>> {
    let mut radius = 0;
    match unsafe { mcseed_spawn_refinement_radius(&mut radius) } {
        1 => Ok(Some(radius)),
        0 => Ok(None),
        status => bail!("读取出生点细化半径失败（错误码 {status}）"),
    }
}

#[cfg(test)]
pub(crate) fn gpu_reference_filter(
    candidates: &[GpuCandidate],
    configs: &[GpuStructureConfig],
    predicates: &[GpuPredicate],
) -> Vec<u8> {
    let mut matches = vec![0; candidates.len()];
    unsafe {
        mcseed_gpu_reference_filter(
            candidates.as_ptr(),
            candidates.len(),
            configs.as_ptr(),
            configs.len(),
            predicates.as_ptr(),
            predicates.len(),
            matches.as_mut_ptr(),
        )
    };
    matches
}

pub fn piece_selector(structure_id: i32, name: &str) -> Result<String> {
    let normalized = normalize_resource_name(name);
    let c_name = CString::new(normalized.as_str()).context("子结构名称包含 NUL 字符")?;
    let valid = unsafe { mcseed_piece_selector_valid(structure_id, c_name.as_ptr()) };
    if valid == 0 {
        let structure = structures()?
            .into_iter()
            .find(|candidate| candidate.id == structure_id)
            .map(|candidate| candidate.name)
            .unwrap_or_else(|| structure_id.to_string());
        bail!(
            "结构 {structure} 不支持子结构选择器 {name:?}；运行 `mcseed-finder list pieces` 查看名称"
        );
    }
    Ok(normalized)
}

pub fn biome_name(id: i32) -> Option<String> {
    optional_c_string(unsafe { mcseed_biome_name_from_id(id) })
}

fn normalize_resource_name(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

fn hit_buffer(raw_hits: &mut [RawHit]) -> (*mut RawHit, usize) {
    if raw_hits.is_empty() {
        (std::ptr::null_mut(), 0)
    } else {
        (raw_hits.as_mut_ptr(), raw_hits.len())
    }
}

fn scan_result(raw_hits: Vec<RawHit>, found: u64, limit_reached: i32) -> ScanResult {
    let stored = usize::try_from(found)
        .unwrap_or(usize::MAX)
        .min(raw_hits.len());
    let hits = raw_hits
        .into_iter()
        .take(stored)
        .map(|hit| NativeHit {
            position: Position {
                x: hit.x,
                y: (hit.y != i32::MIN).then_some(hit.y),
                z: hit.z,
            },
            id: hit.id,
        })
        .collect();
    ScanResult {
        found,
        limit_reached: limit_reached != 0,
        hits,
    }
}

fn ensure_status(status: i32, operation: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        bail!("{operation}失败（原生世界生成器错误码 {status}）")
    }
}

fn required_c_string(pointer: *const c_char) -> Result<String> {
    optional_c_string(pointer).ok_or_else(|| anyhow!("原生注册表返回了空名称"))
}

fn optional_c_string(pointer: *const c_char) -> Option<String> {
    if pointer.is_null() {
        return None;
    }
    let value = unsafe { CStr::from_ptr(pointer) };
    Some(value.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::{
        GpuCandidate, GpuPredicate, GpuStructureConfig, NativeContext, PieceAccuracy,
        biome_by_name, biomes, piece_selector, pieces, spawn_refinement_radius, structure_by_name,
        structures,
    };
    use crate::domain::Dimension;

    #[test]
    fn registries_include_26_2_additions_and_all_structure_families() {
        let sulfur = biome_by_name("minecraft:sulfur-caves").expect("硫磺洞穴应受支持");
        assert_eq!(sulfur.name, "sulfur_caves");
        assert_eq!(sulfur.dimension, Dimension::Overworld);

        let fossil = structure_by_name("nether_fossil").expect("下界化石应受支持");
        assert_eq!(fossil.dimension, Dimension::Nether);
        assert_eq!(structures().expect("结构列表").len(), 22);
        assert!(biomes().expect("生物群系列表").len() > 50);
    }

    #[test]
    fn gpu_ffi_layout_matches_the_native_abi() {
        assert_eq!(std::mem::size_of::<GpuStructureConfig>(), 24);
        assert_eq!(std::mem::size_of::<GpuPredicate>(), 32);
        assert_eq!(std::mem::size_of::<GpuCandidate>(), 16);
    }

    #[test]
    fn estimated_spawn_bound_contains_the_final_spawn() {
        let radius = spawn_refinement_radius()
            .expect("出生点细化能力")
            .expect("当前版本应提供严格细化半径");
        assert_eq!(radius, 125);
        let radius_squared = u64::from(radius) * u64::from(radius);
        let mut context = NativeContext::new().expect("生成器");
        for seed in -16_i64..16 {
            context.set_seed(seed);
            let estimated = context.estimated_spawn().expect("估计出生点");
            let final_spawn = context.spawn_raw().expect("最终出生点");
            let dx = i64::from(final_spawn.position.x) - i64::from(estimated.x);
            let dz = i64::from(final_spawn.position.z) - i64::from(estimated.z);
            assert!(
                (dx * dx + dz * dz) as u64 <= radius_squared,
                "seed {seed} 的出生点细化超出严格边界"
            );
        }
    }

    #[test]
    fn structure_aliases_are_normalized() {
        let temple = structure_by_name("Jungle-Temple").expect("别名应受支持");
        assert_eq!(temple.name, "jungle_pyramid");
        let mansion = structure_by_name("woodland mansion").expect("别名应受支持");
        assert_eq!(mansion.name, "mansion");
    }

    #[test]
    fn piece_registry_exposes_groups_and_exact_templates() {
        let village = structure_by_name("village").expect("村庄注册项");
        let shipwreck = structure_by_name("shipwreck").expect("沉船注册项");
        let entries = pieces().expect("子结构注册表");
        assert!(entries.len() > 500);
        assert!(entries.iter().any(|entry| {
            entry.structure_id == village.id && entry.name == "blacksmith" && entry.is_group
        }));
        assert!(entries.iter().any(|entry| {
            entry.structure_id == village.id
                && entry.name == "village/snowy/houses/snowy_weapon_smith_1"
                && !entry.is_group
                && entry.accuracy == PieceAccuracy::ApproximateTerrain
        }));
        assert!(entries.iter().any(|entry| {
            entry.structure_id == shipwreck.id
                && entry.name == "shipwreck/with_mast"
                && entry.accuracy == PieceAccuracy::Exact
        }));
        assert_eq!(
            piece_selector(village.id, "Blacksmith").expect("分组选择器"),
            "blacksmith"
        );
        assert_eq!(
            piece_selector(village.id, "Librarian").expect("职业别名"),
            "librarian"
        );
        assert!(piece_selector(village.id, "not_a_house").is_err());
    }

    #[test]
    fn seed_zero_matches_known_26_2_regression_values() {
        let mut context = NativeContext::new().expect("生成器");
        context.set_seed(0);
        let spawn = context.spawn().expect("出生点");
        assert_eq!(
            (spawn.position.x, spawn.position.y, spawn.position.z),
            (-32, Some(65), 0)
        );
        assert_eq!(spawn.biome, "forest");

        let village = structure_by_name("village").expect("村庄注册项");
        let village_scan = context
            .find_structure(village.id, -32, 0, 1_024, 1, true)
            .expect("村庄扫描");
        assert_eq!(village_scan.found, 1);
        assert_eq!(
            (
                village_scan.hits[0].position.x,
                village_scan.hits[0].position.z
            ),
            (272, 944)
        );

        let blacksmith_scan = context
            .find_structure_pieces(
                village.id,
                &["blacksmith".to_owned()],
                -32,
                0,
                1_024,
                1,
                true,
            )
            .expect("铁匠铺扫描");
        assert_eq!(blacksmith_scan.found, 1);
        assert_eq!(
            blacksmith_scan.hits[0].name,
            "village/plains/houses/plains_weaponsmith_1"
        );
        assert_eq!(
            (
                blacksmith_scan.hits[0].position.x,
                blacksmith_scan.hits[0].position.y,
                blacksmith_scan.hits[0].position.z,
            ),
            (267, Some(71), 960)
        );
        assert_eq!(
            (
                blacksmith_scan.hits[0].parent_position.x,
                blacksmith_scan.hits[0].parent_position.z,
            ),
            (272, 944)
        );

        let portal = context
            .nearest_stronghold_portal(-32, 0)
            .expect("最近要塞传送门房");
        assert_eq!(portal.name, "stronghold/portal_room");
        assert_eq!(
            (portal.parent_position.x, portal.parent_position.z),
            (-204, -1692)
        );
        assert_eq!((portal.position.x, portal.position.z), (-196, -1728));
        assert_eq!(portal.eye_mask, Some(0x030));

        let fossil = structure_by_name("nether_fossil").expect("下界化石注册项");
        let fossil_scan = context
            .find_structure(fossil.id, -4, 0, 512, 1, true)
            .expect("下界化石扫描");
        assert_eq!(fossil_scan.found, 1);
        assert_eq!(
            (
                fossil_scan.hits[0].position.x,
                fossil_scan.hits[0].position.z
            ),
            (-32, -480)
        );

        context.set_seed(-1);
        let negative_spawn = context.spawn().expect("负种子出生点");
        assert_eq!(negative_spawn.biome, "plains");
        assert_ne!(negative_spawn.biome, "sulfur_caves");
    }
}
