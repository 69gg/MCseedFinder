use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn minecraft_version() -> String {
    let header = fs::read_to_string("native/version.h").expect("无法读取 native/version.h");
    header
        .lines()
        .find_map(|line| {
            let value = line
                .trim()
                .strip_prefix("#define MCSEED_VERSION_NAME ")?
                .trim();
            let value = value.strip_prefix('"')?.strip_suffix('"')?;
            (!value.is_empty()).then(|| value.to_owned())
        })
        .expect("native/version.h 缺少 MCSEED_VERSION_NAME")
}

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

fn command_path(command: &str) -> Option<PathBuf> {
    let candidate = Path::new(command);
    if candidate.components().count() > 1 {
        return candidate.exists().then(|| candidate.to_owned());
    }
    env::var_os("PATH").and_then(|paths| {
        env::split_paths(&paths)
            .map(|directory| directory.join(command))
            .find(|path| path.is_file())
    })
}

fn rocm_root(hipcc: &Path) -> Option<PathBuf> {
    for variable in ["ROCM_PATH", "HIP_PATH"] {
        if let Some(value) = env::var_os(variable) {
            let path = PathBuf::from(value);
            if path.is_dir() {
                return Some(path);
            }
        }
    }

    if let Some(hipconfig) = command_path("hipconfig")
        && let Ok(output) = Command::new(hipconfig).arg("--path").output()
        && output.status.success()
    {
        let value = String::from_utf8_lossy(&output.stdout);
        let path = PathBuf::from(value.trim());
        if path.is_dir() {
            return Some(path);
        }
    }

    hipcc
        .canonicalize()
        .ok()
        .and_then(|path| path.parent()?.parent().map(Path::to_owned))
        .filter(|path| path.is_dir())
}

fn compile_gpu_backend(target_os: &str, target_arch: &str) {
    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let rocm = env::var_os("CARGO_FEATURE_ROCM").is_some();
    if cuda && rocm {
        panic!("cuda 与 rocm feature 不能同时启用；请为目标 GPU 选择一个后端");
    }
    if !cuda && !rocm {
        return;
    }
    if target_os != "linux" || target_arch != "x86_64" {
        panic!("GPU 加速目前只支持 Linux x86_64，当前目标为 {target_os}/{target_arch}");
    }

    if cuda {
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .cudart("shared")
            .ccbin(false)
            .file("native/gpu/backend.cu")
            .include("native/gpu")
            .define("MCSEED_GPU_CUDA", None)
            .std("c++17")
            .warnings(false)
            .flag_if_supported("-O3")
            .compile("mcseed_gpu_backend");
    }

    if rocm {
        let hipcc_name = env::var("HIPCC").unwrap_or_else(|_| "hipcc".to_owned());
        let hipcc = command_path(&hipcc_name).unwrap_or_else(|| {
            panic!("启用 rocm feature 需要 hipcc；请安装 ROCm HIP SDK 或设置 HIPCC")
        });
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .compiler(&hipcc)
            .file("native/gpu/backend.cu")
            .include("native/gpu")
            .define("MCSEED_GPU_HIP", None)
            .std("c++17")
            .warnings(false)
            .flag_if_supported("-O3")
            .compile("mcseed_gpu_backend");

        if let Some(root) = rocm_root(&hipcc) {
            for directory in [root.join("lib"), root.join("lib64")] {
                if directory.is_dir() {
                    println!("cargo:rustc-link-search=native={}", directory.display());
                }
            }
        }
        println!("cargo:rustc-link-lib=dylib=amdhip64");
    }
}

fn main() {
    let minecraft_version = minecraft_version();
    let vendor_root = Path::new("vendor/cubiomes");
    let mut sources = Vec::new();
    collect_c_sources(Path::new("native"), &mut sources);
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

    println!("cargo:rerun-if-changed=native");
    println!("cargo:rerun-if-changed=vendor/cubiomes");
    println!("cargo:rustc-env=MCSEED_MINECRAFT_VERSION={minecraft_version}");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    compile_gpu_backend(&target_os, &target_arch);

    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDAARCHS");
    println!("cargo:rerun-if-env-changed=AMDGPU_TARGETS");
    if matches!(
        target_os.as_str(),
        "linux" | "freebsd" | "openbsd" | "netbsd"
    ) {
        println!("cargo:rustc-link-lib=m");
    }
}
