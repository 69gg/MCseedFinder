use std::collections::HashSet;
use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::Value;

fn run(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mcseed-finder"))
        .args(arguments)
        .output()
        .expect("CLI 应能启动")
}

fn json_lines(output: &Output) -> Vec<Value> {
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| serde_json::from_str(line).expect("每行应为 JSON"))
        .collect()
}

#[test]
fn lists_all_26_2_structure_families() {
    let output = run(&["list", "structures", "--json"]);
    assert!(output.status.success());
    let entries: Vec<Value> = serde_json::from_slice(&output.stdout).expect("结构列表 JSON");
    assert_eq!(entries.len(), 22);
    assert!(entries.iter().any(|entry| entry["name"] == "nether_fossil"));
    assert!(
        entries
            .iter()
            .any(|entry| entry["name"] == "trial_chambers")
    );
    assert!(entries.iter().any(|entry| entry["name"] == "stronghold"));
}

#[test]
fn checks_known_seed_with_tracked_example_config() {
    let config = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/village_and_nether.json");
    let output = Command::new(env!("CARGO_BIN_EXE_mcseed-finder"))
        .args(["check", "0", "--config"])
        .arg(config)
        .args(["--format", "jsonl"])
        .output()
        .expect("CLI 应能启动");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = &json_lines(&output)[0];
    assert_eq!(report["matched"], true);
    assert_eq!(report["spawn"]["position"]["x"], -32);
    assert_eq!(report["spawn"]["biome"], "forest");
}

#[test]
fn parallel_search_keeps_numeric_result_order() {
    let common = [
        "find",
        "--start",
        "0",
        "--end",
        "64",
        "--count",
        "3",
        "--batch-size",
        "16",
        "--spawn-biome",
        "forest",
        "--format",
        "jsonl",
    ];
    let mut single_arguments = common.to_vec();
    single_arguments.extend(["--threads", "1"]);
    let mut parallel_arguments = common.to_vec();
    parallel_arguments.extend(["--threads", "4"]);

    let single = run(&single_arguments);
    let parallel = run(&parallel_arguments);
    assert!(single.status.success());
    assert!(parallel.status.success());
    let single_seeds: Vec<i64> = json_lines(&single)
        .iter()
        .map(|value| value["seed"].as_i64().expect("seed"))
        .collect();
    let parallel_seeds: Vec<i64> = json_lines(&parallel)
        .iter()
        .map(|value| value["seed"].as_i64().expect("seed"))
        .collect();
    assert_eq!(single_seeds, vec![0, 1, 2]);
    assert_eq!(parallel_seeds, single_seeds);
}

#[test]
fn random_search_is_reproducible_across_thread_counts() {
    let common = [
        "find",
        "--random",
        "--random-seed",
        "42",
        "--max-attempts",
        "128",
        "--count",
        "3",
        "--batch-size",
        "16",
        "--spawn-biome",
        "forest",
        "--format",
        "jsonl",
    ];
    let mut single_arguments = common.to_vec();
    single_arguments.extend(["--threads", "1"]);
    let mut parallel_arguments = common.to_vec();
    parallel_arguments.extend(["--threads", "4"]);

    let single = run(&single_arguments);
    let parallel = run(&parallel_arguments);
    assert!(single.status.success());
    assert!(parallel.status.success());
    let seeds = |output: &Output| -> Vec<i64> {
        json_lines(output)
            .iter()
            .map(|value| value["seed"].as_i64().expect("seed"))
            .collect()
    };
    assert_eq!(seeds(&single).len(), 3);
    assert_eq!(seeds(&single), seeds(&parallel));
    assert!(String::from_utf8_lossy(&single.stderr).contains("random_seed=42"));
}

#[test]
fn random_search_honors_requested_count_without_duplicates() {
    let output = run(&[
        "find",
        "--random",
        "--random-seed",
        "20260725",
        "--max-attempts",
        "64",
        "--count",
        "7",
        "--threads",
        "4",
        "--batch-size",
        "16",
        "--format",
        "jsonl",
    ]);
    assert!(output.status.success());
    let seeds: Vec<i64> = json_lines(&output)
        .iter()
        .map(|value| value["seed"].as_i64().expect("seed"))
        .collect();
    assert_eq!(seeds.len(), 7);
    assert_eq!(seeds.iter().copied().collect::<HashSet<_>>().len(), 7);
}

#[test]
fn random_search_settings_can_come_from_json_config() {
    let config = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/random_search.json");
    let output = Command::new(env!("CARGO_BIN_EXE_mcseed-finder"))
        .args(["find", "--config"])
        .arg(config)
        .args(["--count", "1", "--threads", "2", "--format", "jsonl"])
        .output()
        .expect("CLI 应能启动");
    assert!(output.status.success());
    assert_eq!(json_lines(&output).len(), 1);
    assert!(String::from_utf8_lossy(&output.stderr).contains("random_seed=20260725"));
}

#[test]
fn explicit_random_mode_overrides_a_configs_sequential_range() {
    let config = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/village_and_nether.json");
    let output = Command::new(env!("CARGO_BIN_EXE_mcseed-finder"))
        .args(["find", "--random", "--config"])
        .arg(config)
        .args([
            "--random-seed",
            "20260725",
            "--max-attempts",
            "128",
            "--count",
            "1",
            "--threads",
            "2",
            "--format",
            "jsonl",
        ])
        .output()
        .expect("CLI 应能启动");
    assert!(output.status.success());
    assert_eq!(json_lines(&output).len(), 1);
}

#[test]
fn random_only_options_reject_sequential_or_mixed_modes() {
    let mixed = run(&["find", "--random", "--start", "0"]);
    assert_eq!(mixed.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&mixed.stderr).contains("不能设置 start/end"));

    let missing_mode = run(&["find", "--random-seed", "42"]);
    assert_eq!(missing_mode.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&missing_mode.stderr).contains("只能用于随机模式"));

    let zero_attempts = run(&["find", "--random", "--max-attempts", "0"]);
    assert_eq!(zero_attempts.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&zero_attempts.stderr).contains("必须大于 0"));
}

#[test]
fn build_and_runtime_inputs_do_not_reference_ignored_game_sources() {
    let tracked_inputs = [
        include_str!("../Cargo.toml"),
        include_str!("../build.rs"),
        include_str!("../src/main.rs"),
        include_str!("../src/cli.rs"),
        include_str!("../src/config.rs"),
        include_str!("../src/engine.rs"),
        include_str!("../src/native.rs"),
        include_str!("../src/output.rs"),
        include_str!("../native/bridge.c"),
    ];
    for input in tracked_inputs {
        assert!(!input.contains("sources/"));
        assert!(!input.contains("server.jar"));
        assert!(!input.contains("client.jar"));
    }
}

#[test]
fn exit_codes_distinguish_no_match_from_invalid_input() {
    let no_match = run(&["check", "0", "--spawn-biome", "desert", "--format", "jsonl"]);
    assert_eq!(no_match.status.code(), Some(1));
    assert_eq!(json_lines(&no_match)[0]["matched"], false);

    let invalid = run(&["check", "0", "--structure-near", "not_a_structure:64"]);
    assert_eq!(invalid.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("未知结构"));
}
