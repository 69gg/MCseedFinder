use std::cmp::min;
use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;

use crate::config::{Anchor, CompiledCondition, CompiledFilter, MINECRAFT_VERSION};
use crate::domain::{Dimension, Position, SpawnInfo, project_to_nether};
use crate::native::{self, NativeContext, NativeHit};

pub const DEFAULT_START: i128 = 0;
pub const DEFAULT_END: i128 = 1_000_000;
pub const DEFAULT_RESULTS: usize = 1;
pub const DEFAULT_BATCH_SIZE: usize = 4_096;
pub const FULL_SEED_SPACE: u128 = 1_u128 << 64;

const RANDOM_SEARCH_DOMAIN: &str = "mcseed-finder/random-search/v1";
const SPLITMIX64_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
const SPLITMIX64_MIX_1: u64 = 0xbf58_476d_1ce4_e5b9;
const SPLITMIX64_MIX_2: u64 = 0x94d0_49bb_1331_11eb;

#[derive(Debug, Clone, Copy)]
pub enum SearchMode {
    Sequential {
        start: i128,
        end: i128,
    },
    Random {
        random_seed: u64,
        max_attempts: u128,
    },
}

#[derive(Debug, Clone)]
pub struct SearchSettings {
    pub mode: SearchMode,
    pub results: usize,
    pub threads: usize,
    pub batch_size: usize,
    pub progress: bool,
}

impl SearchSettings {
    pub fn validate(&self) -> Result<()> {
        match self.mode {
            SearchMode::Sequential { start, end } => {
                const END_AFTER_I64_MAX: i128 = i64::MAX as i128 + 1;
                if start < i64::MIN as i128 {
                    bail!("--start 小于 Minecraft 可用的最小种子 {}", i64::MIN);
                }
                if end > END_AFTER_I64_MAX {
                    bail!("--end 最大只能是 {END_AFTER_I64_MAX}（i64::MAX 后一个值）");
                }
                if start >= end {
                    bail!("种子范围必须满足 start < end，当前为 {start}..{end}");
                }
            }
            SearchMode::Random { max_attempts, .. } => {
                if max_attempts == 0 {
                    bail!("--max-attempts 必须大于 0");
                }
                if max_attempts > FULL_SEED_SPACE {
                    bail!("--max-attempts 不能超过完整 64 位种子空间 {FULL_SEED_SPACE}");
                }
            }
        }
        if self.results == 0 {
            bail!("结果数量必须大于 0");
        }
        if self.threads == 0 {
            bail!("线程数必须大于 0");
        }
        if self.batch_size == 0 {
            bail!("批大小必须大于 0");
        }
        Ok(())
    }
}

impl SearchMode {
    fn candidate_count(self) -> u128 {
        match self {
            Self::Sequential { start, end } => {
                u128::try_from(end - start).expect("已验证的顺序种子范围长度应可表示为 u128")
            }
            Self::Random { max_attempts, .. } => max_attempts,
        }
    }

    fn seed_at(self, index: u128) -> Result<i64> {
        match self {
            Self::Sequential { start, .. } => {
                let offset = i128::try_from(index).context("顺序种子偏移超出 i128 范围")?;
                let seed = start
                    .checked_add(offset)
                    .ok_or_else(|| anyhow!("顺序种子计算溢出"))?;
                i64::try_from(seed).with_context(|| format!("种子 {seed} 超出 i64 范围"))
            }
            Self::Random { random_seed, .. } => {
                let index = u64::try_from(index).context("随机种子索引超出 64 位空间")?;
                let bits = splitmix64_permutation(index.wrapping_add(random_seed));
                Ok(i64::from_ne_bytes(bits.to_ne_bytes()))
            }
        }
    }
}

/// 生成当前搜索会话使用的随机化键。它只决定遍历顺序，不参与 Minecraft 世界生成。
pub fn generate_random_search_seed() -> u64 {
    RandomState::new().hash_one(RANDOM_SEARCH_DOMAIN)
}

/// SplitMix64 的输出变换由加法、奇数乘法和可逆 xor-shift 组成，因此是 u64 全排列。
fn splitmix64_permutation(value: u64) -> u64 {
    let mut mixed = value.wrapping_add(SPLITMIX64_GAMMA);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(SPLITMIX64_MIX_1);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(SPLITMIX64_MIX_2);
    mixed ^ (mixed >> 31)
}

#[derive(Debug, Clone, Serialize)]
pub struct HitReport {
    pub name: String,
    pub dimension: Dimension,
    pub position: Position,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_position: Option<Position>,
    pub distance: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConditionReport {
    pub condition: String,
    pub matched: bool,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_count: Option<u64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub count_is_lower_bound: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hits: Vec<HitReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<ConditionReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SeedReport {
    pub version: &'static str,
    pub seed: i64,
    pub matched: bool,
    pub spawn: SpawnInfo,
    pub filter: ConditionReport,
}

#[derive(Debug)]
pub struct SearchOutcome {
    pub reports: Vec<SeedReport>,
    pub checked: u128,
    pub elapsed: Duration,
    pub mode: SearchMode,
    pub requested_results: usize,
    pub exhausted: bool,
}

#[derive(Debug, Clone, Copy)]
struct WorkItem {
    start: u128,
    end: u128,
    result_limit: usize,
}

struct IndexedReport {
    index: u128,
    report: SeedReport,
}

struct WorkResult {
    reports: Vec<IndexedReport>,
    checked: u128,
}

enum WorkerReply {
    Ready(Result<()>),
    Complete(Result<WorkResult>),
}

struct Evaluation<'a> {
    native: &'a mut NativeContext,
    spawn: Option<SpawnInfo>,
}

struct ConditionOutcome {
    matched: bool,
    report: Option<ConditionReport>,
}

pub fn evaluate_seed(seed: i64, filter: &CompiledFilter) -> Result<SeedReport> {
    let mut native = NativeContext::new()?;
    native.set_seed(seed);
    let mut evaluation = Evaluation::new(&mut native);
    evaluation.report(seed, filter)
}

pub fn search(filter: Arc<CompiledFilter>, settings: &SearchSettings) -> Result<SearchOutcome> {
    settings.validate()?;
    let started = Instant::now();
    let candidate_count = settings.mode.candidate_count();
    let worker_count = min(
        settings.threads,
        usize::try_from(candidate_count).unwrap_or(settings.threads),
    );

    std::thread::scope(|scope| -> Result<SearchOutcome> {
        let mut task_senders = Vec::with_capacity(worker_count);
        let mut reply_receivers = Vec::with_capacity(worker_count);
        let mut handles = Vec::with_capacity(worker_count);

        for worker in 0..worker_count {
            let (task_sender, task_receiver) = mpsc::channel::<WorkItem>();
            let (reply_sender, reply_receiver) = mpsc::channel::<WorkerReply>();
            let filter = Arc::clone(&filter);
            let mode = settings.mode;
            let handle = std::thread::Builder::new()
                .name(format!("mcseed-worker-{}", worker + 1))
                .spawn_scoped(scope, move || {
                    let mut native = match NativeContext::new() {
                        Ok(native) => native,
                        Err(error) => {
                            let _ = reply_sender.send(WorkerReply::Ready(Err(error)));
                            return;
                        }
                    };
                    if reply_sender.send(WorkerReply::Ready(Ok(()))).is_err() {
                        return;
                    }
                    while let Ok(work) = task_receiver.recv() {
                        let result = search_index_range(work, mode, &filter, &mut native);
                        if reply_sender.send(WorkerReply::Complete(result)).is_err() {
                            break;
                        }
                    }
                })
                .with_context(|| format!("无法启动搜索工作线程 {}", worker + 1))?;
            handles.push(handle);
            task_senders.push(task_sender);
            reply_receivers.push(reply_receiver);
        }

        let coordinator_result = (|| -> Result<SearchOutcome> {
            for receiver in &reply_receivers {
                match receiver.recv().context("搜索工作线程在初始化时断开")? {
                    WorkerReply::Ready(result) => result?,
                    WorkerReply::Complete(_) => bail!("搜索工作线程返回了无效的初始化消息"),
                }
            }

            let mut cursor = 0_u128;
            let mut checked = 0_u128;
            let mut reports = Vec::new();

            while cursor < candidate_count && reports.len() < settings.results {
                let batch_end = min(
                    cursor.saturating_add(settings.batch_size as u128),
                    candidate_count,
                );
                let batch_length = batch_end - cursor;
                let active_workers = min(worker_count, batch_length as usize);
                let remaining = settings.results - reports.len();

                for (worker, sender) in task_senders.iter().take(active_workers).enumerate() {
                    let start = cursor + batch_length * worker as u128 / active_workers as u128;
                    let end = cursor + batch_length * (worker + 1) as u128 / active_workers as u128;
                    sender
                        .send(WorkItem {
                            start,
                            end,
                            result_limit: remaining,
                        })
                        .with_context(|| format!("无法向搜索工作线程 {} 派发任务", worker + 1))?;
                }

                let mut batch_reports = Vec::new();
                for (worker, receiver) in reply_receivers.iter().take(active_workers).enumerate() {
                    let result = match receiver
                        .recv()
                        .with_context(|| format!("搜索工作线程 {} 意外断开", worker + 1))?
                    {
                        WorkerReply::Complete(result) => result?,
                        WorkerReply::Ready(_) => bail!("搜索工作线程重复发送初始化消息"),
                    };
                    checked = checked
                        .checked_add(result.checked)
                        .ok_or_else(|| anyhow!("已检查种子数溢出"))?;
                    batch_reports.extend(result.reports);
                }

                batch_reports.sort_by_key(|report| report.index);
                reports.extend(
                    batch_reports
                        .into_iter()
                        .take(remaining)
                        .map(|indexed| indexed.report),
                );
                cursor = batch_end;

                if settings.progress {
                    let rate = checked as f64 / started.elapsed().as_secs_f64().max(0.001);
                    eprintln!(
                        "已检查 {checked} 个种子，找到 {} 个，{rate:.0} seeds/s",
                        reports.len()
                    );
                }
            }

            Ok(SearchOutcome {
                exhausted: cursor >= candidate_count && reports.len() < settings.results,
                reports,
                checked,
                elapsed: started.elapsed(),
                mode: settings.mode,
                requested_results: settings.results,
            })
        })();

        drop(task_senders);
        let mut worker_panicked = false;
        for handle in handles {
            worker_panicked |= handle.join().is_err();
        }
        if worker_panicked {
            bail!("搜索工作线程意外退出");
        }
        coordinator_result
    })
}

fn search_index_range(
    work: WorkItem,
    mode: SearchMode,
    filter: &CompiledFilter,
    native: &mut NativeContext,
) -> Result<WorkResult> {
    let mut reports = Vec::new();
    let mut checked = 0_u128;
    for index in work.start..work.end {
        let seed = mode.seed_at(index)?;
        native.set_seed(seed);
        let mut evaluation = Evaluation::new(native);
        let matched = evaluation.evaluate(&filter.root, false)?.matched;
        checked += 1;
        if matched {
            reports.push(IndexedReport {
                index,
                report: evaluation.report(seed, filter)?,
            });
            if reports.len() >= work.result_limit {
                break;
            }
        }
    }
    Ok(WorkResult { reports, checked })
}

impl<'a> Evaluation<'a> {
    fn new(native: &'a mut NativeContext) -> Self {
        Self {
            native,
            spawn: None,
        }
    }

    fn report(&mut self, seed: i64, filter: &CompiledFilter) -> Result<SeedReport> {
        let filter_report = self
            .evaluate(&filter.root, true)?
            .report
            .ok_or_else(|| anyhow!("内部错误：缺少条件报告"))?;
        let spawn = self.spawn()?.clone();
        Ok(SeedReport {
            version: MINECRAFT_VERSION,
            seed,
            matched: filter_report.matched,
            spawn,
            filter: filter_report,
        })
    }

    fn evaluate(
        &mut self,
        condition: &CompiledCondition,
        collect: bool,
    ) -> Result<ConditionOutcome> {
        match condition {
            CompiledCondition::SpawnBiome { targets } => {
                let spawn = self.spawn()?.clone();
                let matched = targets.iter().any(|target| target.id == spawn.biome_id);
                let report = collect.then(|| ConditionReport {
                    condition: "spawn_biome".to_owned(),
                    matched,
                    description: format!(
                        "出生生物群系 {}，目标为 [{}]",
                        spawn.biome,
                        join_names(targets.iter().map(|target| target.name.as_str()))
                    ),
                    observed_count: Some(u64::from(matched)),
                    count_is_lower_bound: false,
                    hits: Vec::new(),
                    children: Vec::new(),
                });
                Ok(ConditionOutcome { matched, report })
            }
            CompiledCondition::BiomeNear {
                dimension,
                targets,
                anchor,
                radius,
                y_min,
                y_max,
                min_count,
                max_count,
            } => {
                let (anchor_x, anchor_z) = self.resolve_anchor(*anchor)?;
                let limit = scan_limit(*min_count, *max_count)?;
                let ids: Vec<i32> = targets.iter().map(|target| target.id).collect();
                let scan = self.native.find_biomes(
                    *dimension, &ids, anchor_x, anchor_z, *radius, *y_min, *y_max, limit, collect,
                )?;
                let matched = count_matches(scan.found, *min_count, *max_count);
                let report = collect.then(|| ConditionReport {
                    condition: "biome_near".to_owned(),
                    matched,
                    description: format!(
                        "{} 中 [{}] 距 {} ({anchor_x}, {anchor_z}) 不超过 {} 格，Y={}..={}；{}",
                        dimension,
                        join_names(targets.iter().map(|target| target.name.as_str())),
                        anchor.label(),
                        radius,
                        y_min,
                        y_max,
                        count_description(scan.found, scan.limit_reached, *min_count, *max_count)
                    ),
                    observed_count: Some(scan.found),
                    count_is_lower_bound: scan.limit_reached,
                    hits: scan
                        .hits
                        .iter()
                        .map(|hit| biome_hit(*dimension, hit, anchor_x, anchor_z))
                        .collect(),
                    children: Vec::new(),
                });
                Ok(ConditionOutcome { matched, report })
            }
            CompiledCondition::StructureNear {
                dimension,
                targets,
                anchor,
                radius,
                min_count,
                max_count,
            } => {
                let (anchor_x, anchor_z) = self.resolve_anchor(*anchor)?;
                let limit = scan_limit(*min_count, *max_count)?;
                let mut found = 0_u64;
                let mut lower_bound = false;
                let mut hits = Vec::new();
                for target in targets {
                    if found >= limit {
                        lower_bound = true;
                        break;
                    }
                    let scan = self.native.find_structure(
                        target.id,
                        anchor_x,
                        anchor_z,
                        *radius,
                        limit - found,
                        collect,
                    )?;
                    found = found
                        .checked_add(scan.found)
                        .ok_or_else(|| anyhow!("结构计数溢出"))?;
                    lower_bound |= scan.limit_reached;
                    if collect {
                        hits.extend(scan.hits.iter().map(|hit| HitReport {
                            name: target.name.clone(),
                            dimension: *dimension,
                            position: hit.position,
                            parent_position: None,
                            distance: horizontal_distance(
                                hit.position.x,
                                hit.position.z,
                                anchor_x,
                                anchor_z,
                            ),
                        }));
                    }
                }
                let matched = count_matches(found, *min_count, *max_count);
                let report = collect.then(|| ConditionReport {
                    condition: "structure_near".to_owned(),
                    matched,
                    description: format!(
                        "{} 中 [{}] 距 {} ({anchor_x}, {anchor_z}) 不超过 {} 格；{}",
                        dimension,
                        join_names(targets.iter().map(|target| target.name.as_str())),
                        anchor.label(),
                        radius,
                        count_description(found, lower_bound, *min_count, *max_count)
                    ),
                    observed_count: Some(found),
                    count_is_lower_bound: lower_bound,
                    hits,
                    children: Vec::new(),
                });
                Ok(ConditionOutcome { matched, report })
            }
            CompiledCondition::StructurePieceNear {
                parent,
                selectors,
                anchor,
                radius,
                min_count,
                max_count,
            } => {
                let (anchor_x, anchor_z) = self.resolve_anchor(*anchor)?;
                let limit = scan_limit(*min_count, *max_count)?;
                let scan = self.native.find_structure_pieces(
                    parent.id, selectors, anchor_x, anchor_z, *radius, limit, collect,
                )?;
                let matched = count_matches(scan.found, *min_count, *max_count);
                let report = collect.then(|| ConditionReport {
                    condition: "structure_piece_near".to_owned(),
                    matched,
                    description: format!(
                        "{} 中 {} 的子结构 [{}]；父结构起点距 {} ({anchor_x}, {anchor_z}) 不超过 {} 格；{}",
                        parent.dimension,
                        parent.name,
                        join_names(selectors.iter().map(String::as_str)),
                        anchor.label(),
                        radius,
                        count_description(
                            scan.found,
                            scan.limit_reached,
                            *min_count,
                            *max_count
                        )
                    ),
                    observed_count: Some(scan.found),
                    count_is_lower_bound: scan.limit_reached,
                    hits: scan
                        .hits
                        .iter()
                        .map(|hit| HitReport {
                            name: hit.name.clone(),
                            dimension: parent.dimension,
                            position: hit.position,
                            parent_position: Some(hit.parent_position),
                            distance: horizontal_distance(
                                hit.parent_position.x,
                                hit.parent_position.z,
                                anchor_x,
                                anchor_z,
                            ),
                        })
                        .collect(),
                    children: Vec::new(),
                });
                Ok(ConditionOutcome { matched, report })
            }
            CompiledCondition::All(children) => {
                let mut matched = true;
                let mut reports = Vec::new();
                for child in children {
                    let outcome = self.evaluate(child, collect)?;
                    matched &= outcome.matched;
                    if let Some(report) = outcome.report {
                        reports.push(report);
                    }
                    if !collect && !matched {
                        break;
                    }
                }
                let report = collect.then(|| ConditionReport {
                    condition: "all".to_owned(),
                    matched,
                    description: format!("全部 {} 个条件均需满足", children.len()),
                    observed_count: None,
                    count_is_lower_bound: false,
                    hits: Vec::new(),
                    children: reports,
                });
                Ok(ConditionOutcome { matched, report })
            }
            CompiledCondition::Any(children) => {
                let mut matched = false;
                let mut reports = Vec::new();
                for child in children {
                    let outcome = self.evaluate(child, collect)?;
                    matched |= outcome.matched;
                    if let Some(report) = outcome.report {
                        reports.push(report);
                    }
                    if !collect && matched {
                        break;
                    }
                }
                let report = collect.then(|| ConditionReport {
                    condition: "any".to_owned(),
                    matched,
                    description: format!("{} 个条件中至少满足一个", children.len()),
                    observed_count: None,
                    count_is_lower_bound: false,
                    hits: Vec::new(),
                    children: reports,
                });
                Ok(ConditionOutcome { matched, report })
            }
            CompiledCondition::Not(child) => {
                let outcome = self.evaluate(child, collect)?;
                let matched = !outcome.matched;
                let report = collect.then(|| ConditionReport {
                    condition: "not".to_owned(),
                    matched,
                    description: "子条件必须不成立".to_owned(),
                    observed_count: None,
                    count_is_lower_bound: false,
                    hits: Vec::new(),
                    children: outcome.report.into_iter().collect(),
                });
                Ok(ConditionOutcome { matched, report })
            }
        }
    }

    fn spawn(&mut self) -> Result<&SpawnInfo> {
        if self.spawn.is_none() {
            self.spawn = Some(self.native.spawn()?);
        }
        self.spawn
            .as_ref()
            .ok_or_else(|| anyhow!("内部错误：出生点缓存为空"))
    }

    fn resolve_anchor(&mut self, anchor: Anchor) -> Result<(i32, i32)> {
        match anchor {
            Anchor::Origin => Ok((0, 0)),
            Anchor::Coordinates { x, z } => Ok((x, z)),
            Anchor::Spawn => {
                let spawn = self.spawn()?;
                Ok((spawn.position.x, spawn.position.z))
            }
            Anchor::NetherSpawn => {
                let spawn = self.spawn()?;
                Ok((
                    project_to_nether(spawn.position.x),
                    project_to_nether(spawn.position.z),
                ))
            }
        }
    }
}

fn scan_limit(minimum: u64, maximum: Option<u64>) -> Result<u64> {
    if let Some(maximum) = maximum {
        maximum
            .checked_add(1)
            .ok_or_else(|| anyhow!("max_count 太大，无法计算停止阈值"))
    } else {
        Ok(minimum.max(1))
    }
}

fn count_matches(found: u64, minimum: u64, maximum: Option<u64>) -> bool {
    found >= minimum && maximum.is_none_or(|maximum| found <= maximum)
}

fn count_description(found: u64, lower_bound: bool, minimum: u64, maximum: Option<u64>) -> String {
    let observed = if lower_bound {
        format!("至少找到 {found} 个")
    } else {
        format!("找到 {found} 个")
    };
    match maximum {
        Some(maximum) => format!("{observed}，要求 {minimum}..={maximum} 个"),
        None => format!("{observed}，要求至少 {minimum} 个"),
    }
}

fn biome_hit(dimension: Dimension, hit: &NativeHit, anchor_x: i32, anchor_z: i32) -> HitReport {
    HitReport {
        name: native::biome_name(hit.id).unwrap_or_else(|| format!("biome_{}", hit.id)),
        dimension,
        position: hit.position,
        parent_position: None,
        distance: horizontal_distance(hit.position.x, hit.position.z, anchor_x, anchor_z),
    }
}

fn horizontal_distance(x: i32, z: i32, anchor_x: i32, anchor_z: i32) -> f64 {
    let dx = i64::from(x) - i64::from(anchor_x);
    let dz = i64::from(z) - i64::from(anchor_z);
    ((dx * dx + dz * dz) as f64).sqrt()
}

fn join_names<'a>(names: impl Iterator<Item = &'a str>) -> String {
    names.collect::<Vec<_>>().join(", ")
}

pub fn default_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{FULL_SEED_SPACE, SearchMode, SearchSettings, count_matches, scan_limit};

    #[test]
    fn count_bounds_support_absence_filters() {
        assert!(count_matches(0, 0, Some(0)));
        assert!(!count_matches(1, 0, Some(0)));
        assert!(count_matches(2, 1, Some(3)));
        assert_eq!(scan_limit(0, Some(0)).expect("阈值"), 1);
    }

    #[test]
    fn search_range_accepts_one_past_i64_max_as_end() {
        let settings = SearchSettings {
            mode: SearchMode::Sequential {
                start: i64::MAX as i128,
                end: i64::MAX as i128 + 1,
            },
            results: 1,
            threads: 1,
            batch_size: 1,
            progress: false,
        };
        settings.validate().expect("范围应有效");
    }

    #[test]
    fn random_mode_accepts_the_complete_seed_space() {
        let settings = SearchSettings {
            mode: SearchMode::Random {
                random_seed: 42,
                max_attempts: FULL_SEED_SPACE,
            },
            results: 1,
            threads: 1,
            batch_size: 1,
            progress: false,
        };
        settings.validate().expect("完整 64 位随机空间应有效");
        settings
            .mode
            .seed_at(FULL_SEED_SPACE - 1)
            .expect("最后一个随机索引应可映射");
    }

    #[test]
    fn random_permutation_is_reproducible_and_has_no_sample_duplicates() {
        let mode = SearchMode::Random {
            random_seed: 20_260_725,
            max_attempts: FULL_SEED_SPACE,
        };
        let repeated = SearchMode::Random {
            random_seed: 20_260_725,
            max_attempts: FULL_SEED_SPACE,
        };
        let another = SearchMode::Random {
            random_seed: 20_260_726,
            max_attempts: FULL_SEED_SPACE,
        };
        let mut unique = HashSet::with_capacity(65_536);
        for index in 0..65_536 {
            let seed = mode.seed_at(index).expect("随机索引应可映射");
            assert_eq!(seed, repeated.seed_at(index).expect("相同键应可复现"));
            assert!(unique.insert(seed), "随机全排列的样本中不应出现重复");
        }
        assert_ne!(
            mode.seed_at(0).expect("索引应可映射"),
            another.seed_at(0).expect("索引应可映射")
        );
    }
}
