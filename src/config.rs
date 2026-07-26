use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;

use crate::accelerator::AcceleratorKind;
use crate::domain::Dimension;
use crate::native::{self, BiomeInfo, StructureInfo};

pub const MINECRAFT_VERSION: &str = env!("MCSEED_MINECRAFT_VERSION");
pub const MAX_RADIUS: u32 = 30_000_000;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchFileConfig {
    pub start: Option<i128>,
    pub end: Option<i128>,
    pub random: Option<bool>,
    pub random_seed: Option<u64>,
    pub max_attempts: Option<u128>,
    pub results: Option<usize>,
    pub threads: Option<usize>,
    pub batch_size: Option<usize>,
    pub accelerator: Option<AcceleratorKind>,
    pub gpu_device: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default)]
    pub search: SearchFileConfig,
    #[serde(default)]
    pub conditions: Vec<ConditionSpec>,
}

impl Default for FileConfig {
    fn default() -> Self {
        Self {
            version: default_version(),
            search: SearchFileConfig::default(),
            conditions: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ConditionSpec {
    SpawnBiome {
        #[serde(alias = "biomes")]
        any_of: Vec<String>,
    },
    BiomeNear {
        #[serde(default)]
        dimension: Option<Dimension>,
        #[serde(alias = "biomes")]
        any_of: Vec<String>,
        #[serde(default)]
        anchor: Option<AnchorSpec>,
        radius: u32,
        #[serde(default)]
        y: Option<i32>,
        #[serde(default)]
        y_min: Option<i32>,
        #[serde(default)]
        y_max: Option<i32>,
        #[serde(default = "default_min_count")]
        min_count: u64,
        #[serde(default)]
        max_count: Option<u64>,
    },
    StructureNear {
        #[serde(default)]
        dimension: Option<Dimension>,
        #[serde(alias = "structures")]
        any_of: Vec<String>,
        #[serde(default)]
        anchor: Option<AnchorSpec>,
        radius: u32,
        #[serde(default = "default_min_count")]
        min_count: u64,
        #[serde(default)]
        max_count: Option<u64>,
    },
    #[serde(alias = "substructure_near")]
    StructurePieceNear {
        structure: String,
        #[serde(alias = "pieces")]
        any_of: Vec<String>,
        #[serde(default)]
        anchor: Option<AnchorSpec>,
        radius: u32,
        #[serde(default = "default_min_count")]
        min_count: u64,
        #[serde(default)]
        max_count: Option<u64>,
    },
    StrongholdEyes {
        #[serde(default)]
        anchor: Option<AnchorSpec>,
        #[serde(default)]
        eyes: Option<u8>,
        #[serde(default)]
        min_eyes: Option<u8>,
        #[serde(default)]
        max_eyes: Option<u8>,
    },
    All {
        conditions: Vec<ConditionSpec>,
    },
    Any {
        conditions: Vec<ConditionSpec>,
    },
    Not {
        condition: Box<ConditionSpec>,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnchorSpec {
    Named(AnchorName),
    Coordinates(CoordinateAnchor),
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorName {
    Origin,
    Spawn,
    NetherSpawn,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoordinateAnchor {
    pub x: i32,
    pub z: i32,
}

#[derive(Debug, Clone)]
pub struct CompiledFilter {
    pub root: CompiledCondition,
}

#[derive(Debug, Clone)]
pub enum CompiledCondition {
    SpawnBiome {
        targets: Vec<BiomeInfo>,
    },
    BiomeNear {
        dimension: Dimension,
        targets: Vec<BiomeInfo>,
        anchor: Anchor,
        radius: u32,
        y_min: i32,
        y_max: i32,
        min_count: u64,
        max_count: Option<u64>,
    },
    StructureNear {
        dimension: Dimension,
        targets: Vec<StructureInfo>,
        anchor: Anchor,
        radius: u32,
        min_count: u64,
        max_count: Option<u64>,
    },
    StructurePieceNear {
        parent: StructureInfo,
        selectors: Vec<String>,
        anchor: Anchor,
        radius: u32,
        min_count: u64,
        max_count: Option<u64>,
    },
    StrongholdEyes {
        anchor: Anchor,
        min_eyes: u8,
        max_eyes: u8,
    },
    All(Vec<CompiledCondition>),
    Any(Vec<CompiledCondition>),
    Not(Box<CompiledCondition>),
}

#[derive(Debug, Clone, Copy)]
pub enum Anchor {
    Origin,
    Spawn,
    NetherSpawn,
    Coordinates { x: i32, z: i32 },
}

impl Anchor {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Origin => "origin",
            Self::Spawn => "spawn",
            Self::NetherSpawn => "nether_spawn",
            Self::Coordinates { .. } => "coordinates",
        }
    }
}

pub fn load_file(path: Option<&Path>) -> Result<FileConfig> {
    let Some(path) = path else {
        return Ok(FileConfig::default());
    };
    let content =
        fs::read_to_string(path).with_context(|| format!("无法读取配置文件 {}", path.display()))?;
    let config: FileConfig = serde_json::from_str(&content)
        .with_context(|| format!("配置文件 {} 格式错误", path.display()))?;
    if config.version != MINECRAFT_VERSION {
        bail!(
            "配置要求 Minecraft {}，本程序只实现 {}",
            config.version,
            MINECRAFT_VERSION
        );
    }
    Ok(config)
}

impl CompiledFilter {
    pub fn compile(specifications: Vec<ConditionSpec>) -> Result<Self> {
        let mut conditions = specifications
            .into_iter()
            .map(compile_condition)
            .collect::<Result<Vec<_>>>()?;
        sort_by_cost(&mut conditions);
        Ok(Self {
            root: CompiledCondition::All(conditions),
        })
    }
}

pub fn conditions_from_flags(
    spawn_biomes: &[String],
    biome_near: &[String],
    structure_near: &[String],
    piece_near: &[String],
    stronghold_eyes: &[String],
) -> Result<Vec<ConditionSpec>> {
    let mut conditions = Vec::new();
    if !spawn_biomes.is_empty() {
        let any_of = split_names(spawn_biomes.iter().map(String::as_str))?;
        conditions.push(ConditionSpec::SpawnBiome { any_of });
    }
    for specification in biome_near {
        conditions.push(parse_biome_near(specification)?);
    }
    for specification in structure_near {
        conditions.push(parse_structure_near(specification)?);
    }
    for specification in piece_near {
        conditions.push(parse_piece_near(specification)?);
    }
    for specification in stronghold_eyes {
        conditions.push(parse_stronghold_eyes(specification)?);
    }
    Ok(conditions)
}

fn parse_biome_near(value: &str) -> Result<ConditionSpec> {
    let parts: Vec<&str> = value.split(':').collect();
    if !(3..=4).contains(&parts.len()) {
        bail!(
            "生物群系条件 {value:?} 格式错误；应为 DIMENSION:BIOME[,BIOME]:RADIUS[:Y|Y_MIN..Y_MAX]"
        );
    }
    let dimension = Dimension::from_str(parts[0]).map_err(|error| anyhow!(error))?;
    let any_of = split_names(std::iter::once(parts[1]))?;
    let radius = parse_radius(parts[2], value)?;
    let (y, y_min, y_max) = if parts.len() == 4 {
        if let Some((minimum, maximum)) = parts[3].split_once("..") {
            (
                None,
                Some(parse_i32(minimum, "Y_MIN", value)?),
                Some(parse_i32(maximum, "Y_MAX", value)?),
            )
        } else {
            (Some(parse_i32(parts[3], "Y", value)?), None, None)
        }
    } else {
        (None, None, None)
    };
    Ok(ConditionSpec::BiomeNear {
        dimension: Some(dimension),
        any_of,
        anchor: None,
        radius,
        y,
        y_min,
        y_max,
        min_count: default_min_count(),
        max_count: None,
    })
}

fn parse_structure_near(value: &str) -> Result<ConditionSpec> {
    let (names, radius) = value
        .rsplit_once(':')
        .ok_or_else(|| anyhow!("结构条件 {value:?} 格式错误；应为 STRUCTURE[,STRUCTURE]:RADIUS"))?;
    Ok(ConditionSpec::StructureNear {
        dimension: None,
        any_of: split_names(std::iter::once(names))?,
        anchor: None,
        radius: parse_radius(radius, value)?,
        min_count: default_min_count(),
        max_count: None,
    })
}

fn parse_piece_near(value: &str) -> Result<ConditionSpec> {
    let (head, radius) = value.rsplit_once(':').ok_or_else(|| {
        anyhow!("子结构条件 {value:?} 格式错误；应为 STRUCTURE:PIECE[,PIECE]:RADIUS")
    })?;
    let head = head.strip_prefix("minecraft:").unwrap_or(head);
    let (structure, pieces) = head.split_once(':').ok_or_else(|| {
        anyhow!("子结构条件 {value:?} 格式错误；应为 STRUCTURE:PIECE[,PIECE]:RADIUS")
    })?;
    if structure.trim().is_empty() {
        bail!("子结构条件 {value:?} 的父结构不能为空");
    }
    Ok(ConditionSpec::StructurePieceNear {
        structure: structure.trim().to_owned(),
        any_of: split_names(std::iter::once(pieces))?,
        anchor: None,
        radius: parse_radius(radius, value)?,
        min_count: default_min_count(),
        max_count: None,
    })
}

fn parse_stronghold_eyes(value: &str) -> Result<ConditionSpec> {
    let value = value.trim();
    if value.is_empty() {
        bail!("要塞眼数不能为空；应为 EYES 或 MIN..MAX");
    }
    if let Some((minimum, maximum)) = value.split_once("..") {
        if minimum.is_empty() || maximum.is_empty() {
            bail!("要塞眼数 {value:?} 格式错误；范围应为 MIN..MAX");
        }
        Ok(ConditionSpec::StrongholdEyes {
            anchor: None,
            eyes: None,
            min_eyes: Some(parse_eye_count(minimum, value)?),
            max_eyes: Some(parse_eye_count(maximum, value)?),
        })
    } else {
        Ok(ConditionSpec::StrongholdEyes {
            anchor: None,
            eyes: Some(parse_eye_count(value, value)?),
            min_eyes: None,
            max_eyes: None,
        })
    }
}

fn compile_condition(specification: ConditionSpec) -> Result<CompiledCondition> {
    match specification {
        ConditionSpec::SpawnBiome { any_of } => {
            let targets = resolve_biomes(any_of)?;
            require_dimension("出生生物群系", &targets, Dimension::Overworld)?;
            Ok(CompiledCondition::SpawnBiome { targets })
        }
        ConditionSpec::BiomeNear {
            dimension,
            any_of,
            anchor,
            radius,
            y,
            y_min,
            y_max,
            min_count,
            max_count,
        } => {
            validate_radius(radius)?;
            validate_counts(min_count, max_count)?;
            if y.is_some() && (y_min.is_some() || y_max.is_some()) {
                bail!("biome_near 不能同时设置 y 和 y_min/y_max");
            }
            let targets = resolve_biomes(any_of)?;
            let inferred_dimension = common_biome_dimension(&targets)?;
            let dimension = dimension.unwrap_or(inferred_dimension);
            require_dimension("生物群系", &targets, dimension)?;
            let (y_min, y_max) = resolve_y_range(dimension, y, y_min, y_max)?;
            Ok(CompiledCondition::BiomeNear {
                dimension,
                targets,
                anchor: compile_anchor(anchor, dimension)?,
                radius,
                y_min,
                y_max,
                min_count,
                max_count,
            })
        }
        ConditionSpec::StructureNear {
            dimension,
            any_of,
            anchor,
            radius,
            min_count,
            max_count,
        } => {
            validate_radius(radius)?;
            validate_counts(min_count, max_count)?;
            let targets = resolve_structures(any_of)?;
            let inferred_dimension = common_structure_dimension(&targets)?;
            let dimension = dimension.unwrap_or(inferred_dimension);
            for target in &targets {
                if target.dimension != dimension {
                    bail!(
                        "结构 {} 属于 {}，不能用于 {} 条件",
                        target.name,
                        target.dimension,
                        dimension
                    );
                }
            }
            Ok(CompiledCondition::StructureNear {
                dimension,
                targets,
                anchor: compile_anchor(anchor, dimension)?,
                radius,
                min_count,
                max_count,
            })
        }
        ConditionSpec::StructurePieceNear {
            structure,
            any_of,
            anchor,
            radius,
            min_count,
            max_count,
        } => {
            validate_radius(radius)?;
            validate_counts(min_count, max_count)?;
            let parent = native::structure_by_name(&structure)?;
            if any_of.is_empty() {
                bail!("子结构选择器列表不能为空");
            }
            let mut seen = HashSet::new();
            let selectors = any_of
                .into_iter()
                .map(|name| native::piece_selector(parent.id, &name))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .filter(|name| seen.insert(name.clone()))
                .collect::<Vec<_>>();
            Ok(CompiledCondition::StructurePieceNear {
                anchor: compile_anchor(anchor, parent.dimension)?,
                parent,
                selectors,
                radius,
                min_count,
                max_count,
            })
        }
        ConditionSpec::StrongholdEyes {
            anchor,
            eyes,
            min_eyes,
            max_eyes,
        } => {
            let (min_eyes, max_eyes) = resolve_eye_range(eyes, min_eyes, max_eyes)?;
            Ok(CompiledCondition::StrongholdEyes {
                anchor: compile_anchor(anchor, Dimension::Overworld)?,
                min_eyes,
                max_eyes,
            })
        }
        ConditionSpec::All { conditions } => {
            if conditions.is_empty() {
                bail!("all 条件至少要包含一个子条件");
            }
            let mut compiled = conditions
                .into_iter()
                .map(compile_condition)
                .collect::<Result<Vec<_>>>()?;
            sort_by_cost(&mut compiled);
            Ok(CompiledCondition::All(compiled))
        }
        ConditionSpec::Any { conditions } => {
            if conditions.is_empty() {
                bail!("any 条件至少要包含一个子条件");
            }
            let mut compiled = conditions
                .into_iter()
                .map(compile_condition)
                .collect::<Result<Vec<_>>>()?;
            sort_by_cost(&mut compiled);
            Ok(CompiledCondition::Any(compiled))
        }
        ConditionSpec::Not { condition } => Ok(CompiledCondition::Not(Box::new(
            compile_condition(*condition)?,
        ))),
    }
}

fn resolve_biomes(names: Vec<String>) -> Result<Vec<BiomeInfo>> {
    if names.is_empty() {
        bail!("生物群系列表不能为空");
    }
    deduplicate(
        names
            .into_iter()
            .map(|name| native::biome_by_name(&name))
            .collect::<Result<Vec<_>>>()?,
        |entry| entry.id,
    )
}

fn resolve_structures(names: Vec<String>) -> Result<Vec<StructureInfo>> {
    if names.is_empty() {
        bail!("结构列表不能为空");
    }
    deduplicate(
        names
            .into_iter()
            .map(|name| native::structure_by_name(&name))
            .collect::<Result<Vec<_>>>()?,
        |entry| entry.id,
    )
}

fn deduplicate<T, F>(values: Vec<T>, key: F) -> Result<Vec<T>>
where
    F: Fn(&T) -> i32,
{
    let mut seen = HashSet::new();
    let unique: Vec<T> = values
        .into_iter()
        .filter(|value| seen.insert(key(value)))
        .collect();
    if unique.is_empty() {
        bail!("筛选目标不能为空");
    }
    Ok(unique)
}

fn require_dimension(label: &str, targets: &[BiomeInfo], dimension: Dimension) -> Result<()> {
    for target in targets {
        if target.dimension != dimension {
            bail!(
                "{label} {} 属于 {}，不能用于 {} 条件",
                target.name,
                target.dimension,
                dimension
            );
        }
    }
    Ok(())
}

fn common_biome_dimension(targets: &[BiomeInfo]) -> Result<Dimension> {
    let dimension = targets
        .first()
        .ok_or_else(|| anyhow!("生物群系列表不能为空"))?
        .dimension;
    require_dimension("生物群系", targets, dimension)?;
    Ok(dimension)
}

fn common_structure_dimension(targets: &[StructureInfo]) -> Result<Dimension> {
    let dimension = targets
        .first()
        .ok_or_else(|| anyhow!("结构列表不能为空"))?
        .dimension;
    for target in targets {
        if target.dimension != dimension {
            bail!(
                "同一个 structure_near 中的结构必须处于同一维度；{} 属于 {}",
                target.name,
                target.dimension
            );
        }
    }
    Ok(dimension)
}

fn compile_anchor(anchor: Option<AnchorSpec>, dimension: Dimension) -> Result<Anchor> {
    let anchor = match anchor {
        Some(AnchorSpec::Named(AnchorName::Origin)) => Anchor::Origin,
        Some(AnchorSpec::Named(AnchorName::Spawn)) => Anchor::Spawn,
        Some(AnchorSpec::Named(AnchorName::NetherSpawn)) => Anchor::NetherSpawn,
        Some(AnchorSpec::Coordinates(value)) => {
            if value.x.unsigned_abs() > MAX_RADIUS || value.z.unsigned_abs() > MAX_RADIUS {
                bail!(
                    "自定义锚点必须位于世界边界 ±{MAX_RADIUS} 内，实际为 ({}, {})",
                    value.x,
                    value.z
                );
            }
            Anchor::Coordinates {
                x: value.x,
                z: value.z,
            }
        }
        None => match dimension {
            Dimension::Overworld => Anchor::Spawn,
            Dimension::Nether => Anchor::NetherSpawn,
            Dimension::End => Anchor::Origin,
        },
    };
    Ok(anchor)
}

fn resolve_y_range(
    dimension: Dimension,
    y: Option<i32>,
    y_min: Option<i32>,
    y_max: Option<i32>,
) -> Result<(i32, i32)> {
    let default_y = match dimension {
        Dimension::Overworld => 64,
        Dimension::Nether => 64,
        Dimension::End => 64,
    };
    let (minimum, maximum) = if let Some(y) = y {
        (y, y)
    } else {
        match (y_min, y_max) {
            (Some(minimum), Some(maximum)) => (minimum, maximum),
            (Some(value), None) | (None, Some(value)) => (value, value),
            (None, None) => (default_y, default_y),
        }
    };
    if minimum > maximum {
        bail!("Y 范围无效：{minimum} 大于 {maximum}");
    }
    let (world_minimum, world_maximum) = match dimension {
        Dimension::Overworld => (-64, 319),
        Dimension::Nether => (0, 127),
        Dimension::End => (0, 255),
    };
    if minimum < world_minimum || maximum > world_maximum {
        bail!(
            "{} 的 Y 范围必须在 {}..={} 内，实际为 {}..={}",
            dimension,
            world_minimum,
            world_maximum,
            minimum,
            maximum
        );
    }
    Ok((minimum, maximum))
}

fn validate_radius(radius: u32) -> Result<()> {
    if radius > MAX_RADIUS {
        bail!("半径不能超过世界边界 {MAX_RADIUS} 格，实际为 {radius}");
    }
    Ok(())
}

fn validate_counts(minimum: u64, maximum: Option<u64>) -> Result<()> {
    if let Some(maximum) = maximum {
        if maximum < minimum {
            bail!("max_count ({maximum}) 不能小于 min_count ({minimum})");
        }
    } else if minimum == 0 {
        bail!("min_count=0 且未设置 max_count 的条件没有筛选作用");
    }
    Ok(())
}

fn parse_eye_count(value: &str, specification: &str) -> Result<u8> {
    let count = value
        .parse::<u8>()
        .with_context(|| format!("要塞眼数 {specification:?} 中的 {value:?} 不是 0..12 的整数"))?;
    if count > 12 {
        bail!("要塞眼数不能超过 12，实际为 {count}");
    }
    Ok(count)
}

fn resolve_eye_range(
    eyes: Option<u8>,
    minimum: Option<u8>,
    maximum: Option<u8>,
) -> Result<(u8, u8)> {
    if let Some(eyes) = eyes {
        if minimum.is_some() || maximum.is_some() {
            bail!("stronghold_eyes 不能同时设置 eyes 和 min_eyes/max_eyes");
        }
        if eyes > 12 {
            bail!("要塞眼数不能超过 12，实际为 {eyes}");
        }
        return Ok((eyes, eyes));
    }
    let minimum = minimum.unwrap_or(0);
    let maximum = maximum.unwrap_or(12);
    if minimum > 12 || maximum > 12 {
        bail!("要塞眼数范围必须位于 0..=12，实际为 {minimum}..={maximum}");
    }
    if maximum < minimum {
        bail!("max_eyes ({maximum}) 不能小于 min_eyes ({minimum})");
    }
    Ok((minimum, maximum))
}

fn sort_by_cost(conditions: &mut [CompiledCondition]) {
    conditions.sort_by_key(condition_cost);
}

fn condition_cost(condition: &CompiledCondition) -> u8 {
    match condition {
        CompiledCondition::SpawnBiome { .. } => 1,
        CompiledCondition::StructureNear { targets, .. } => {
            if targets.iter().any(|target| target.name == "stronghold") {
                4
            } else {
                2
            }
        }
        CompiledCondition::StructurePieceNear { parent, .. } => {
            if parent.name == "stronghold" {
                5
            } else {
                4
            }
        }
        CompiledCondition::StrongholdEyes { .. } => 5,
        CompiledCondition::BiomeNear { .. } => 3,
        CompiledCondition::All(children) | CompiledCondition::Any(children) => {
            children.iter().map(condition_cost).min().unwrap_or(1)
        }
        CompiledCondition::Not(child) => condition_cost(child),
    }
}

fn split_names<'a>(values: impl Iterator<Item = &'a str>) -> Result<Vec<String>> {
    let names: Vec<String> = values
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    if names.is_empty() {
        bail!("筛选名称不能为空");
    }
    Ok(names)
}

fn parse_radius(value: &str, specification: &str) -> Result<u32> {
    let radius = value
        .parse::<u32>()
        .with_context(|| format!("条件 {specification:?} 中的半径不是非负整数"))?;
    validate_radius(radius)?;
    Ok(radius)
}

fn parse_i32(value: &str, label: &str, specification: &str) -> Result<i32> {
    value
        .parse::<i32>()
        .with_context(|| format!("条件 {specification:?} 中的 {label} 不是整数"))
}

fn default_version() -> String {
    MINECRAFT_VERSION.to_owned()
}

const fn default_min_count() -> u64 {
    1
}

#[cfg(test)]
mod tests {
    use super::{Anchor, CompiledCondition, CompiledFilter, FileConfig, conditions_from_flags};
    use crate::accelerator::AcceleratorKind;

    #[test]
    fn rejects_unknown_json_fields() {
        let error = serde_json::from_str::<FileConfig>(
            r#"{"version":"26.2","unknown":true,"conditions":[]}"#,
        )
        .expect_err("未知字段必须报错");
        assert!(error.to_string().contains("unknown field"));

        let nested_error = serde_json::from_str::<FileConfig>(
            r#"{"conditions":[{"type":"spawn_biome","any_of":["plains"],"raduis":1}]}"#,
        )
        .expect_err("条件内的未知字段必须报错");
        assert!(nested_error.to_string().contains("unknown field"));
    }

    #[test]
    fn rejects_conflicting_stronghold_eye_fields() {
        let file: FileConfig = serde_json::from_str(
            r#"{"conditions":[{"type":"stronghold_eyes","eyes":2,"min_eyes":1}]}"#,
        )
        .expect("JSON 语法有效");
        let error =
            CompiledFilter::compile(file.conditions).expect_err("精确眼数和范围不能同时出现");
        assert!(error.to_string().contains("不能同时设置"));
    }

    #[test]
    fn parses_gpu_search_settings_and_hip_alias() {
        let rocm: FileConfig =
            serde_json::from_str(r#"{"search":{"accelerator":"rocm","gpu_device":2}}"#)
                .expect("ROCm 搜索配置应有效");
        assert_eq!(rocm.search.accelerator, Some(AcceleratorKind::Rocm));
        assert_eq!(rocm.search.gpu_device, Some(2));

        let hip: FileConfig =
            serde_json::from_str(r#"{"search":{"accelerator":"hip"}}"#).expect("HIP 别名应有效");
        assert_eq!(hip.search.accelerator, Some(AcceleratorKind::Rocm));
    }

    #[test]
    fn parses_common_cli_filters() {
        let specs = conditions_from_flags(
            &["plains,forest".to_owned()],
            &["nether:warped_forest,crimson_forest:256:32..96".to_owned()],
            &["village:512".to_owned(), "fortress,bastion:256".to_owned()],
            &["village:blacksmith,library:1024".to_owned()],
            &["2..5".to_owned()],
        )
        .expect("CLI 条件应可解析");
        let filter = CompiledFilter::compile(specs).expect("条件应可编译");
        let CompiledCondition::All(conditions) = filter.root else {
            panic!("根条件应为 all");
        };
        assert_eq!(conditions.len(), 6);
        assert!(conditions.iter().any(|condition| matches!(
            condition,
            CompiledCondition::StructureNear {
                anchor: Anchor::NetherSpawn,
                ..
            }
        )));
        assert!(conditions.iter().any(|condition| matches!(
            condition,
            CompiledCondition::StrongholdEyes {
                anchor: Anchor::Spawn,
                min_eyes: 2,
                max_eyes: 5,
            }
        )));
    }

    #[test]
    fn all_tracked_examples_parse_and_compile() {
        let examples = [
            include_str!("../examples/village_and_nether.json"),
            include_str!("../examples/advanced.json"),
            include_str!("../examples/sulfur_caves.json"),
            include_str!("../examples/random_search.json"),
            include_str!("../examples/village_blacksmith.json"),
            include_str!("../examples/stronghold_eyes.json"),
        ];
        for json in examples {
            let file: FileConfig = serde_json::from_str(json).expect("示例 JSON 应有效");
            CompiledFilter::compile(file.conditions).expect("示例条件应可编译");
        }
    }
}
