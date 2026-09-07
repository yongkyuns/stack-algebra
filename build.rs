#[cfg(feature = "eigen-compare")]
use std::{env, path::Path, process::Command};

// Compile out the bridge, rather than checking an environment variable after
// its build dependencies have already been compiled.
#[cfg(not(feature = "eigen-compare"))]
fn main() {}

#[cfg(feature = "eigen-compare")]
fn main() {
    println!("cargo:rerun-if-changed=eigen/eigen_bridge.cpp");
    println!("cargo:rerun-if-env-changed=EIGEN3_INCLUDE_DIR");

    let include_dir = env::var("EIGEN3_INCLUDE_DIR")
        .ok()
        .or_else(include_dir_from_pkg_config)
        .or_else(|| {
            let homebrew_path = "/usr/local/include/eigen3";
            Path::new(homebrew_path).exists().then(|| homebrew_path.to_owned())
        })
        .expect(
            "Eigen headers were not found. Set EIGEN3_INCLUDE_DIR or install the eigen3 pkg-config package.",
        );

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("eigen/eigen_bridge.cpp")
        .include(include_dir)
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O3")
        .define("NDEBUG", None)
        .define("EIGEN_NO_DEBUG", None)
        .compile("stack_algebra_eigen");
}

#[cfg(feature = "eigen-compare")]
fn include_dir_from_pkg_config() -> Option<String> {
    let output = Command::new("pkg-config")
        .args(["--cflags-only-I", "eigen3"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let flags = String::from_utf8(output.stdout).ok()?;
    flags
        .split_whitespace()
        .find_map(|flag| flag.strip_prefix("-I").map(str::to_owned))
}
