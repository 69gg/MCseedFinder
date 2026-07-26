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
fn lists_village_groups_and_exact_piece_names() {
    let output = run(&["list", "pieces", "--structure", "village", "--json"]);
    assert!(output.status.success());
    let entries: Vec<Value> = serde_json::from_slice(&output.stdout).expect("村庄子结构列表 JSON");
    assert!(entries.len() > 500);
    assert!(
        entries
            .iter()
            .any(|entry| { entry["name"] == "blacksmith" && entry["kind"] == "group" })
    );
    assert!(entries.iter().any(|entry| {
        entry["name"] == "village/plains/houses/plains_weaponsmith_1" && entry["kind"] == "piece"
    }));
}

#[test]
fn checks_known_village_blacksmith_and_reports_its_parent() {
    let output = run(&[
        "check",
        "0",
        "--piece-near",
        "village:blacksmith:1024",
        "--format",
        "jsonl",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = &json_lines(&output)[0];
    let condition = &report["filter"]["children"][0];
    assert_eq!(condition["condition"], "structure_piece_near");
    assert_eq!(
        condition["hits"][0]["name"],
        "village/plains/houses/plains_weaponsmith_1"
    );
    assert_eq!(condition["hits"][0]["position"]["x"], 267);
    assert_eq!(condition["hits"][0]["position"]["y"], 71);
    assert_eq!(condition["hits"][0]["position"]["z"], 960);
    assert_eq!(condition["hits"][0]["parent_position"]["x"], 272);
    assert_eq!(condition["hits"][0]["parent_position"]["z"], 944);
}

#[test]
fn reports_and_filters_the_eye_located_stronghold_portal() {
    let output = run(&["check", "0", "--stronghold-eyes", "2", "--format", "jsonl"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = &json_lines(&output)[0];
    let condition = &report["filter"]["children"][0];
    assert_eq!(condition["condition"], "stronghold_eyes");
    assert_eq!(condition["observed_count"], 2);
    assert_eq!(condition["hits"][0]["eye_count"], 2);
    assert_eq!(condition["hits"][0]["eye_mask"], 0x030);
    assert_eq!(condition["hits"][0]["position"]["x"], -196);
    assert_eq!(condition["hits"][0]["position"]["z"], -1728);
    assert_eq!(condition["hits"][0]["parent_position"]["x"], -204);
    assert_eq!(condition["hits"][0]["parent_position"]["z"], -1692);

    let no_match = run(&["check", "0", "--stronghold-eyes", "3"]);
    assert_eq!(no_match.status.code(), Some(1));

    let invalid = run(&["check", "0", "--stronghold-eyes", "13"]);
    assert_eq!(invalid.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("不能超过 12"));

    let reversed = run(&["check", "0", "--stronghold-eyes", "5..2"]);
    assert_eq!(reversed.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&reversed.stderr).contains("不能小于"));
}

#[test]
fn rejects_a_piece_selector_not_owned_by_the_parent_structure() {
    let output = run(&["check", "0", "--piece-near", "shipwreck:blacksmith:1024"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("不支持子结构选择器"));
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
fn village_piece_search_is_reproducible_across_threads() {
    let common = [
        "find",
        "--start",
        "0",
        "--end",
        "16",
        "--count",
        "3",
        "--batch-size",
        "4",
        "--piece-near",
        "village:blacksmith:1024",
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
}

#[test]
fn stronghold_eye_search_is_reproducible_across_threads() {
    let common = [
        "find",
        "--start",
        "0",
        "--end",
        "8",
        "--count",
        "3",
        "--batch-size",
        "2",
        "--stronghold-eyes",
        "0..12",
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
    assert_eq!(seeds(&single), vec![0, 1, 2]);
    assert_eq!(seeds(&parallel), seeds(&single));
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
fn cpu_accelerator_rejects_a_gpu_device_and_reports_selection() {
    let invalid = run(&[
        "find",
        "--start",
        "0",
        "--end",
        "1",
        "--accelerator",
        "cpu",
        "--gpu-device",
        "0",
    ]);
    assert_eq!(invalid.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("gpu-device"));

    let cpu = run(&[
        "find",
        "--start",
        "0",
        "--end",
        "1",
        "--accelerator",
        "cpu",
        "--format",
        "jsonl",
    ]);
    assert!(cpu.status.success());
    assert!(String::from_utf8_lossy(&cpu.stderr).contains("搜索后端：cpu"));

    let portable_auto = run(&[
        "find",
        "--start",
        "0",
        "--end",
        "1",
        "--accelerator",
        "auto",
        "--gpu-device",
        "0",
        "--structure-near",
        "village:1024",
        "--format",
        "jsonl",
    ]);
    assert!(
        portable_auto.status.success(),
        "{}",
        String::from_utf8_lossy(&portable_auto.stderr)
    );
}

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
#[test]
fn explicit_gpu_acceleration_requires_a_gpu_build() {
    let output = run(&[
        "find",
        "--start",
        "0",
        "--end",
        "1",
        "--accelerator",
        "cuda",
        "--structure-near",
        "village:128",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("--features cuda"));
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
#[test]
fn compiled_gpu_search_matches_cpu_on_a_small_fixed_range_when_available() {
    let backend = if cfg!(feature = "cuda") {
        "cuda"
    } else {
        "rocm"
    };
    let common = [
        "find",
        "--start",
        "0",
        "--end",
        "64",
        "--count",
        "5",
        "--threads",
        "4",
        "--batch-size",
        "32",
        "--structure-near",
        "village:1024",
        "--format",
        "jsonl",
    ];
    let mut cpu_arguments = common.to_vec();
    cpu_arguments.extend(["--accelerator", "cpu"]);
    let mut gpu_arguments = common.to_vec();
    gpu_arguments.extend(["--accelerator", backend]);
    let cpu = run(&cpu_arguments);
    let gpu = run(&gpu_arguments);
    if gpu.status.code() == Some(2) {
        let stderr = String::from_utf8_lossy(&gpu.stderr);
        if stderr.contains("没有检测到")
            || stderr.contains("运行时不可用")
            || stderr.contains("检测 GPU 设备失败")
        {
            return;
        }
    }
    assert!(cpu.status.success());
    assert!(
        gpu.status.success(),
        "{}",
        String::from_utf8_lossy(&gpu.stderr)
    );
    assert_eq!(json_lines(&gpu), json_lines(&cpu));
    let gpu_stderr = String::from_utf8_lossy(&gpu.stderr);
    assert!(gpu_stderr.contains(&format!("搜索后端：{backend}:")));
    assert!(gpu_stderr.contains("GPU 出生点粗筛："));
    assert!(gpu_stderr.contains("GPU 预筛选："));
}

#[test]
fn build_and_runtime_inputs_do_not_reference_ignored_game_sources() {
    let tracked_inputs = [
        include_str!("../Cargo.toml"),
        include_str!("../build.rs"),
        include_str!("../src/main.rs"),
        include_str!("../src/accelerator.rs"),
        include_str!("../src/cli.rs"),
        include_str!("../src/config.rs"),
        include_str!("../src/engine.rs"),
        include_str!("../src/native.rs"),
        include_str!("../src/output.rs"),
        include_str!("../native/bridge.c"),
        include_str!("../native/jigsaw.c"),
        include_str!("../native/jigsaw.h"),
        include_str!("../native/version.h"),
        include_str!("../native/gpu/abi.h"),
        include_str!("../native/gpu/placement.h"),
        include_str!("../native/gpu/reference.c"),
        include_str!("../native/gpu/backend.cu"),
        include_str!("../native/generated/village_26_2.inc"),
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
