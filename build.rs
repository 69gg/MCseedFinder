use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn collect_c_sources(directory: &Path, sources: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("无法读取 {}: {error}", directory.display()))
        .map(|entry| entry.expect("无法读取目录项").path())
        .collect();
    entries.sort();

    for path in entries {
        if path.is_dir() {
            let name = path.file_name().and_then(|value| value.to_str());
            if !matches!(name, Some(".github" | "docs" | "loot")) {
                collect_c_sources(&path, sources);
            }
        } else if path.extension().and_then(|value| value.to_str()) == Some("c")
            && path.file_name().and_then(|value| value.to_str()) != Some("tests.c")
        {
            sources.push(path);
        }
    }
}

fn main() {
    let vendor_root = Path::new("vendor/cubiomes");
    let mut sources = vec![PathBuf::from("native/bridge.c")];
    collect_c_sources(vendor_root, &mut sources);

    let mut build = cc::Build::new();
    build
        .files(&sources)
        .include(vendor_root)
        .std("c11")
        .warnings(false)
        .flag_if_supported("-ffunction-sections")
        .flag_if_supported("-fdata-sections")
        .compile("mcseed_worldgen");

    println!("cargo:rerun-if-changed=native/bridge.c");
    println!("cargo:rerun-if-changed=native/bridge.h");
    println!("cargo:rerun-if-changed=vendor/cubiomes");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if matches!(
        target_os.as_str(),
        "linux" | "freebsd" | "openbsd" | "netbsd"
    ) {
        println!("cargo:rustc-link-lib=m");
    }
}
