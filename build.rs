#![allow(warnings)]
use std::{env, process::Command, ffi::CString};

fn main() {
  { // dnnl
    let dst = cmake::Config::new("oneDNN")
      .define("DNNL_LIBRARY_TYPE", "STATIC")
      .define("DNNL_BUILD_EXAMPLES", "OFF")
      .define("DNNL_BUILD_TESTS", "OFF")
      .build();
    println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=static=dnnl");
    if cfg!(target_family = "unix") {
      let machine = Command::new("gcc")
        .arg("-dumpmachine")
        .output()
        .unwrap()
        .stdout;
      let machine = CString::new(machine)
        .unwrap()
        .into_string()
        .unwrap();
      let machine = machine.lines()
        .next()
        .unwrap();
      let version = Command::new("gcc")
        .arg("-dumpversion")
        .output()
        .unwrap()
        .stdout;
      let version = CString::new(version)
        .unwrap()
        .into_string()
        .unwrap();
      let version = version.lines()
        .next()
        .unwrap();
      println!("cargo:rustc-link-search=native=/usr/lib/gcc/{}/{}", machine, version);
      println!("cargo:rustc-link-lib=dylib=gomp");
    }
    cpp_build::Config::new()
      .include(dst.join("include").display().to_string())
      .build("src/lib.rs");
  }  
  #[cfg(feature="cuda")]
  {
    { // compile custom cuda source
      println!("cargo:rustc-rerun-if-changed=src/cuda/kernels.cu");
      let status = Command::new("nvcc")
        .arg("src/cuda/kernels.cu")
        .arg("--ptx")
        .arg("-odir")
        .arg(env::var("OUT_DIR").unwrap())
        .status()
        .unwrap();
      assert!(status.success());
    }
  }
}
