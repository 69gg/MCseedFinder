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

unsafe extern "C" {
    fn mcseed_context_create() -> *mut RawContext;
    fn mcseed_context_destroy(context: *mut RawContext);
    fn mcseed_context_set_seed(context: *mut RawContext, seed: c_ulonglong);
    fn mcseed_spawn(context: *mut RawContext, spawn: *mut RawHit, biome_id: *mut c_int) -> c_int;

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

    pub fn spawn(&mut self) -> Result<SpawnInfo> {
        let mut raw_hit = RawHit::default();
        let mut biome_id = -1;
        let status = unsafe { mcseed_spawn(self.raw.as_ptr(), &mut raw_hit, &mut biome_id) };
        ensure_status(status, "计算出生点")?;
        let biome = biome_name(biome_id)
            .ok_or_else(|| anyhow!("出生点返回了未知生物群系 ID {biome_id}"))?;
        Ok(SpawnInfo {
            position: Position {
                x: raw_hit.x,
                y: Some(raw_hit.y),
                z: raw_hit.z,
            },
            biome,
            biome_id,
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
    use super::{NativeContext, biome_by_name, biomes, structure_by_name, structures};
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
    fn structure_aliases_are_normalized() {
        let temple = structure_by_name("Jungle-Temple").expect("别名应受支持");
        assert_eq!(temple.name, "jungle_pyramid");
        let mansion = structure_by_name("woodland mansion").expect("别名应受支持");
        assert_eq!(mansion.name, "mansion");
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
