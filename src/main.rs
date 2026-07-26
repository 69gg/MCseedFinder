mod cli;
mod config;
mod domain;
mod engine;
mod native;
mod output;

use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{Result, bail};
use clap::Parser;
use serde_json::json;

use crate::cli::{Cli, Command, FilterArgs, ListKind};
use crate::config::{CompiledFilter, FileConfig, conditions_from_flags, load_file};
use crate::engine::{
    DEFAULT_BATCH_SIZE, DEFAULT_END, DEFAULT_RESULTS, DEFAULT_START, FULL_SEED_SPACE, SearchMode,
    SearchSettings, default_thread_count, evaluate_seed, generate_random_search_seed, search,
};

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("错误：{error:#}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<bool> {
    let cli = Cli::parse();
    match cli.command {
        Command::Find(arguments) => {
            let (file, filter) = load_filter(&arguments.filters)?;
            let settings = search_settings(&arguments, &file)?;
            output::write_search_start(&settings);
            let outcome = search(Arc::new(filter), &settings)?;
            for report in &outcome.reports {
                output::write_seed(report, arguments.format)?;
            }
            output::write_search_summary(&outcome);
            Ok(!outcome.reports.is_empty())
        }
        Command::Check(arguments) => {
            let (_, filter) = load_filter(&arguments.filters)?;
            let report = evaluate_seed(arguments.seed, &filter)?;
            output::write_seed(&report, arguments.format)?;
            Ok(report.matched)
        }
        Command::List(arguments) => {
            match arguments.kind {
                ListKind::Biomes => {
                    reject_piece_list_filter(arguments.structure.as_deref())?;
                    list_biomes(arguments.json)?;
                }
                ListKind::Structures => {
                    reject_piece_list_filter(arguments.structure.as_deref())?;
                    list_structures(arguments.json)?;
                }
                ListKind::Pieces => list_pieces(arguments.json, arguments.structure.as_deref())?,
            }
            Ok(true)
        }
    }
}

fn search_settings(arguments: &crate::cli::FindArgs, file: &FileConfig) -> Result<SearchSettings> {
    let random = arguments.random || file.search.random.unwrap_or(false);
    let configured_start = arguments.start.or(file.search.start);
    let configured_end = arguments.end.or(file.search.end);
    let configured_random_seed = arguments.random_seed.or(file.search.random_seed);
    let configured_max_attempts = arguments.max_attempts.or(file.search.max_attempts);

    let mode = if random {
        if arguments.start.is_some() || arguments.end.is_some() {
            bail!("随机模式不能设置 start/end；请使用 --max-attempts 限制检查数量");
        }
        if !arguments.random && (file.search.start.is_some() || file.search.end.is_some()) {
            bail!("JSON 随机模式不能同时设置 start/end");
        }
        SearchMode::Random {
            random_seed: configured_random_seed.unwrap_or_else(generate_random_search_seed),
            max_attempts: configured_max_attempts.unwrap_or(FULL_SEED_SPACE),
        }
    } else {
        if configured_random_seed.is_some() || configured_max_attempts.is_some() {
            bail!("random_seed/max_attempts 只能用于随机模式；请同时设置 --random");
        }
        SearchMode::Sequential {
            start: configured_start.unwrap_or(DEFAULT_START),
            end: configured_end.unwrap_or(DEFAULT_END),
        }
    };

    let settings = SearchSettings {
        mode,
        results: arguments
            .results
            .or(file.search.results)
            .unwrap_or(DEFAULT_RESULTS),
        threads: arguments
            .threads
            .or(file.search.threads)
            .unwrap_or_else(default_thread_count),
        batch_size: arguments
            .batch_size
            .or(file.search.batch_size)
            .unwrap_or(DEFAULT_BATCH_SIZE),
        progress: arguments.progress,
    };
    settings.validate()?;
    Ok(settings)
}

fn load_filter(arguments: &FilterArgs) -> Result<(FileConfig, CompiledFilter)> {
    let mut file = load_file(arguments.config.as_deref())?;
    file.conditions.extend(conditions_from_flags(
        &arguments.spawn_biome,
        &arguments.biome_near,
        &arguments.structure_near,
        &arguments.piece_near,
        &arguments.stronghold_eyes,
    )?);
    let filter = CompiledFilter::compile(file.conditions.clone())?;
    Ok((file, filter))
}

fn list_biomes(as_json: bool) -> Result<()> {
    let mut entries = native::biomes()?;
    entries.sort_by(|left, right| {
        left.dimension
            .as_raw()
            .cmp(&right.dimension.as_raw())
            .then_with(|| left.name.cmp(&right.name))
    });
    if as_json {
        let values: Vec<_> = entries
            .iter()
            .map(|entry| {
                json!({
                    "name": entry.name,
                    "dimension": entry.dimension,
                    "id": entry.id,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&values)?);
    } else {
        for entry in entries {
            println!("{:<10} {}", entry.dimension, entry.name);
        }
    }
    Ok(())
}

fn list_structures(as_json: bool) -> Result<()> {
    let mut entries = native::structures()?;
    entries.sort_by(|left, right| {
        left.dimension
            .as_raw()
            .cmp(&right.dimension.as_raw())
            .then_with(|| left.name.cmp(&right.name))
    });
    if as_json {
        let values: Vec<_> = entries
            .iter()
            .map(|entry| {
                json!({
                    "name": entry.name,
                    "dimension": entry.dimension,
                    "accuracy": entry.accuracy.description(),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&values)?);
    } else {
        for entry in entries {
            println!(
                "{:<10} {:<24} {}",
                entry.dimension,
                entry.name,
                entry.accuracy.description()
            );
        }
    }
    Ok(())
}

fn reject_piece_list_filter(structure: Option<&str>) -> Result<()> {
    if structure.is_some() {
        bail!("--structure 只能与 `list pieces` 一起使用");
    }
    Ok(())
}

fn list_pieces(as_json: bool, structure: Option<&str>) -> Result<()> {
    let selected_structure = structure.map(native::structure_by_name).transpose()?;
    let structures = native::structures()?;
    let mut entries = native::pieces()?;
    if let Some(parent) = &selected_structure {
        entries.retain(|entry| entry.structure_id == parent.id);
    }
    entries.sort_by(|left, right| {
        let left_parent = structures
            .iter()
            .find(|entry| entry.id == left.structure_id)
            .map(|entry| entry.name.as_str())
            .unwrap_or("");
        let right_parent = structures
            .iter()
            .find(|entry| entry.id == right.structure_id)
            .map(|entry| entry.name.as_str())
            .unwrap_or("");
        left_parent
            .cmp(right_parent)
            .then_with(|| right.is_group.cmp(&left.is_group))
            .then_with(|| left.name.cmp(&right.name))
    });
    if as_json {
        let values = entries
            .iter()
            .map(|entry| {
                let parent = structures
                    .iter()
                    .find(|candidate| candidate.id == entry.structure_id)
                    .map(|candidate| candidate.name.as_str())
                    .unwrap_or("unknown");
                json!({
                    "structure": parent,
                    "name": entry.name,
                    "kind": if entry.is_group { "group" } else { "piece" },
                    "accuracy": entry.accuracy.description(),
                })
            })
            .collect::<Vec<_>>();
        println!("{}", serde_json::to_string_pretty(&values)?);
    } else {
        for entry in entries {
            let parent = structures
                .iter()
                .find(|candidate| candidate.id == entry.structure_id)
                .map(|candidate| candidate.name.as_str())
                .unwrap_or("unknown");
            println!(
                "{:<20} {:<6} {:<64} {}",
                parent,
                if entry.is_group { "group" } else { "piece" },
                entry.name,
                entry.accuracy.description()
            );
        }
    }
    Ok(())
}
