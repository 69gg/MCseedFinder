use std::io::{self, Write};

use anyhow::{Context, Result};

use crate::accelerator::AcceleratorInfo;
use crate::cli::OutputFormat;
use crate::engine::{ConditionReport, SearchMode, SearchOutcome, SearchSettings, SeedReport};

pub fn write_search_start(settings: &SearchSettings, accelerator: &AcceleratorInfo) {
    eprintln!("搜索后端：{}", accelerator.description());
    if let SearchMode::Random {
        random_seed,
        max_attempts,
    } = settings.mode
    {
        eprintln!(
            "随机搜索：random_seed={random_seed}，目标 {} 个，最多检查 {max_attempts} 个不重复种子",
            settings.results
        );
    }
}

pub fn write_seed(report: &SeedReport, format: OutputFormat) -> Result<()> {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    match format {
        OutputFormat::Text => write_seed_text(&mut output, report),
        OutputFormat::Jsonl => {
            serde_json::to_writer(&mut output, report).context("无法序列化 JSON 输出")?;
            writeln!(output).context("无法写入标准输出")
        }
    }
}

pub fn write_search_summary(outcome: &SearchOutcome) {
    let seconds = outcome.elapsed.as_secs_f64();
    let rate = outcome.checked as f64 / seconds.max(0.001);
    eprintln!(
        "完成：检查 {} 个种子，找到 {} 个，用时 {:.3}s（{:.0} seeds/s）",
        outcome.checked,
        outcome.reports.len(),
        seconds,
        rate
    );
    if outcome.gpu_candidates > 0 {
        let retained = outcome.gpu_survivors as f64 * 100.0 / outcome.gpu_candidates as f64;
        if outcome.gpu_coarse_candidates > 0 {
            let coarse_retained =
                outcome.gpu_coarse_survivors as f64 * 100.0 / outcome.gpu_coarse_candidates as f64;
            eprintln!(
                "GPU 出生点粗筛：保留 {}/{}（{coarse_retained:.3}%）",
                outcome.gpu_coarse_survivors, outcome.gpu_coarse_candidates
            );
        }
        eprintln!(
            "GPU 预筛选：{}，保留 {}/{}（{retained:.3}%）",
            outcome.accelerator.description(),
            outcome.gpu_survivors,
            outcome.gpu_candidates
        );
    }
    if outcome.exhausted {
        eprintln!(
            "搜索空间或最大尝试次数已耗尽：目标 {} 个，实际找到 {} 个",
            outcome.requested_results,
            outcome.reports.len()
        );
    }
    if let SearchMode::Random { random_seed, .. } = outcome.mode {
        eprintln!("可使用 --random --random-seed {random_seed} 复现相同遍历顺序");
    }
}

fn write_seed_text(output: &mut impl Write, report: &SeedReport) -> Result<()> {
    let marker = if report.matched {
        "匹配"
    } else {
        "不匹配"
    };
    let spawn = &report.spawn.position;
    let y = spawn
        .y
        .map(|value| value.to_string())
        .unwrap_or_else(|| "?".to_owned());
    writeln!(
        output,
        "seed={} [{}] spawn=({}, {}, {}) biome={}",
        report.seed, marker, spawn.x, y, spawn.z, report.spawn.biome
    )
    .context("无法写入标准输出")?;
    write_condition(output, &report.filter, 0)
}

fn write_condition(output: &mut impl Write, report: &ConditionReport, depth: usize) -> Result<()> {
    let indent = "  ".repeat(depth + 1);
    let marker = if report.matched { "✓" } else { "✗" };
    writeln!(output, "{indent}{marker} {}", report.description).context("无法写入标准输出")?;
    for hit in &report.hits {
        let y = hit
            .position
            .y
            .map(|value| format!(", {value}"))
            .unwrap_or_default();
        let parent = hit
            .parent_position
            .map(|position| format!("；父结构起点 ({}, {})", position.x, position.z))
            .unwrap_or_default();
        let eyes = hit
            .eye_count
            .zip(hit.eye_mask)
            .map(|(count, mask)| format!("；已有末影之眼 {count}/12（mask={mask:#05x}）"))
            .unwrap_or_default();
        writeln!(
            output,
            "{indent}  - {} {} ({x}{y}, {z}){parent}{eyes} 距离 {distance:.1}",
            hit.dimension,
            hit.name,
            x = hit.position.x,
            z = hit.position.z,
            distance = hit.distance,
        )
        .context("无法写入标准输出")?;
    }
    for child in &report.children {
        write_condition(output, child, depth + 1)?;
    }
    Ok(())
}
