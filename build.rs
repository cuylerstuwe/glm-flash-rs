fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_LIB_DIR");
    if let Ok(path) = std::env::var("CUDA_LIB_DIR") {
        println!("cargo:rustc-link-search=native={path}");
    } else if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/sbsa-linux/lib/stubs");
    }
    println!("cargo:rustc-link-lib=dylib=cuda");
}
