use shaderc::{CompileOptions, Compiler, ShaderKind, SourceLanguage, IncludeType, ResolvedInclude, IncludeCallbackResult};
use std::path::PathBuf;
use std::fs;
use std::iter::once;

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

static SCALAR_TYPES: &[(&'static str, &'static str)] = &[
    ("u8", "uint"),
    ("u16", "uint"),
    //("bf16", "uint"),
    ("u32", "uint"),
    ("i32", "int"),
    ("f32", "float"),
];

static NUM_TYPES: &[(&'static str, &'static str)] = &[
    //("bf16", "uint"),
    ("u32", "uint"),
    ("i32", "int"),
    ("f32", "float"),
];

static UNSIGNED_TYPES: &[(&'static str, &'static str)] = &[
    ("u8", "uint"),
    ("u16", "uint"),
    ("u32", "uint"),
];

static FLOAT_TYPES: &[(&'static str, &'static str)] = &[
    //("bf16", "uint"),
    ("f32", "float"),
];

fn glsl_options() -> CompileOptions<'static> {
    let mut options = CompileOptions::new().unwrap();
    options.set_auto_bind_uniforms(true);
    options.set_source_language(SourceLanguage::GLSL);
    options.set_include_callback(glsl_include_callback);
    options
}

fn glsl_include_callback(name: &str, _include_type: IncludeType, _src_name: &str, _src_depth: usize) -> IncludeCallbackResult {
    match name {
        "buffer_macros.comp" => Ok(ResolvedInclude {
                resolved_name: name.into(),
                content: include_str!("src/glsl/buffer_macros.comp").into()
        }),
        _ => Err(format!("Unrecognized include {:?}!", name))
    }
}

/*
#[allow(unused)]
#[derive(Clone, Copy)]
enum Behavior {
    Enable,
    Require,
    Warn,
    Disable
}

impl Behavior {
    fn to_str(&self) -> &'static str {
        use Behavior::*;
        match self {
            Enable => "enable",
            Require => "require",
            Warn => "warn",
            Disable => "disable",
        }
    }
}

fn glsl_extension(src: &mut String, extension: &str, behavior: Behavior) {
    let version_str = "#version 450";
    let index = src.find(version_str)
        .map(|i| i + version_str.len())
        .expect(src);
    src.insert_str(index, &format!("\n#extension {} : {}\n", extension, behavior.to_str()));
}

fn add_load_store_macros(options: &mut CompileOptions, rust_ty: &str) {
    if rust_ty == "bf16_as_f32" {
        options.add_macro_definition("LOAD(x)",  Some("uintBitsToFloat(uint(x) << 16)"));
        options.add_macro_definition("STORE(x)", Some("uint16_t(floatBitsToUint(x) >> 16)"));
    } else {
        options.add_macro_definition("LOAD(x)",  Some("x"));
        options.add_macro_definition("STORE(x)", Some("x"));
    }
}*/

fn compile_glsl(
    compiler: &mut Compiler,
    src: &str,
    name: &str,
    options: Option<&CompileOptions>,
) -> Result<()> {
    // This is so that the preprocessed source will get dumped on failure.
    let src_artifact = compiler.preprocess(src, name, "main", options)?;
    eprintln!("{}", src_artifact.as_text());
    let artifact = compiler.compile_into_spirv(src, ShaderKind::Compute, name, "main", options)?;
    let glsl_path = PathBuf::from("../")
        .join("src")
        .join("shaders")
        .join("glsl");
    validate_spirv(name, artifact.as_binary())?;
    fs::create_dir_all(&glsl_path)?;
    let fpath = glsl_path.join(&name).with_extension("spv");
    println!("cargo:rerun-if-changed={}", fpath.to_string_lossy());
    fs::write(&fpath, artifact.as_binary_u8())?;
    Ok(())
}

fn validate_spirv(name: &str, spirv: &[u32]) -> Result<()> {
    use spirv_cross::{spirv::{self, ExecutionModel}, glsl, msl, hlsl};
    let module = spirv::Module::from_words(spirv);
    let shader_dir = PathBuf::from(std::env::var("OUT_DIR")?).join("shaders").join(name);
    fs::create_dir_all(&shader_dir)?;
    let mut glsl_ast = spirv::Ast::<glsl::Target>::parse(&module)
        .map_err(|e| format!("GLSL parsing of {name} failed: {e}", name=name, e=e))?;
    let mut glsl_compiler_options = glsl::CompilerOptions::default();
    //glsl_compiler_options.version = glsl::Version::V1_1;
    for entry_point in glsl_ast.get_entry_points()? {
        glsl_compiler_options.entry_point.replace((entry_point.name.clone(), ExecutionModel::GlCompute));
        glsl_ast.set_compiler_options(&glsl_compiler_options)?;
        let glsl = glsl_ast.compile()
            .map_err(|e| format!("GLSL compilation of {name}::{entry} failed: {e}", name=name, entry=&entry_point.name, e=e))?;
        let entry_name = entry_point.name.replace("::", "__");
        fs::write(shader_dir.join(entry_name).with_extension("glsl"), &glsl)?;
    }
    let mut metal_ast = spirv::Ast::<msl::Target>::parse(&module)
        .map_err(|e| format!("Metal parsing of {name} failed: {e}", name=name, e=e))?;
    let mut metal_compiler_options = msl::CompilerOptions::default();
    metal_compiler_options.version = msl::Version::V1_2;
    for entry_point in metal_ast.get_entry_points()? {
        metal_compiler_options.entry_point.replace((entry_point.name.clone(), ExecutionModel::GlCompute));
        metal_ast.set_compiler_options(&metal_compiler_options)?;
        let metal = metal_ast.compile()
            .map_err(|e| format!("Metal compilation of {name}::{entry} failed: {e}", name=name, entry=&entry_point.name, e=e))?;
        let entry_name = entry_point.name.replace("::", "__");
        fs::write(shader_dir.join(entry_name).with_extension("metal"), &metal)?;
    }
    let mut hlsl_ast = spirv::Ast::<hlsl::Target>::parse(&module)
        .map_err(|e| format!("HLSL parsing of {name} failed: {e}", name=name, e=e))?;
    let mut hlsl_compiler_options = hlsl::CompilerOptions::default();
    hlsl_compiler_options.shader_model = hlsl::ShaderModel::V5_1;
    for entry_point in hlsl_ast.get_entry_points()? {
        hlsl_compiler_options.entry_point.replace((entry_point.name.clone(), ExecutionModel::GlCompute));
        hlsl_ast.set_compiler_options(&hlsl_compiler_options)?;
        let hlsl = hlsl_ast.compile()
            .map_err(|e| format!("HLSL compilation of {name}::{entry} failed: {e}", name=name, entry=&entry_point.name, e=e))?;
        let entry_name = entry_point.name.replace("::", "__");
        fs::write(shader_dir.join(entry_name).with_extension("hlsl"), &hlsl)?;
    }
    Ok(())
}
/*
fn validate_spirv(name: &str, spirv: &[u32]) -> Result<()> {
    use spirv_cross::{spirv, hlsl, msl};
    let module = spirv::Module::from_words(spirv);
    spirv::Ast::<msl::Target>::parse(&module)
        .map_err(|e| format!("Metal parsing of {name} failed: {e}", name=name, e=e))?
        .compile()
        .map_err(|e| format!("Metal compilation of {name} failed: {e}", name=name, e=e))?;
    spirv::Ast::<hlsl::Target>::parse(&module)
        .map_err(|e| format!("HLSL parsing of {name} failed: {e}", name=name, e=e))?
        .compile()
        .map_err(|e| format!("HLSL compilation of {name} failed: {e}", name=name, e=e))?;
    Ok(())
}*/


fn glsl_accuracy(compiler: &mut Compiler) -> Result<()> {
    let src = include_str!("src/glsl/accuracy.comp");
    for &(rust_ty, c_ty) in UNSIGNED_TYPES.iter() {
        let mut options = glsl_options();
        if rust_ty == "u8" {
            options.add_macro_definition("U8", None);
        } else {
            options.add_macro_definition("T", Some(c_ty));
        }
        compile_glsl(compiler, src, &format!("accuracy_{}", rust_ty), Some(&options))?;
    }
    Ok(())
}

fn glsl_bias_backward(compiler: &mut Compiler) -> Result<()> {
    let src = include_str!("src/glsl/bias_backward.comp");
    for &(rust_ty, c_ty) in FLOAT_TYPES.iter() {
        let mut options = glsl_options();
        if rust_ty == "bf16" {
            options.add_macro_definition("BF16", None);
        } else {
            options.add_macro_definition("T", Some(c_ty));
        }
        compile_glsl(compiler, src, &format!("bias_backward_{}", rust_ty), Some(&options))?;
    }
    Ok(())
}

fn glsl_fill(compiler: &mut Compiler) -> Result<()> {
    let options = glsl_options();
    {
        let src = include_str!("src/glsl/fill_u32.comp");
        compile_glsl(compiler, src, "fill_u32", Some(&options))?;
    }
    {
        let src = include_str!("src/glsl/fill_u64.comp");
        compile_glsl(compiler, src, "fill_u64", Some(&options))?;
    }
    Ok(())
}

fn glsl_gemm(compiler: &mut Compiler) -> Result<()> {
    let src = include_str!("src/glsl/gemm.comp");
    for &(rust_ty, c_ty) in NUM_TYPES.iter() {
        let mut options = glsl_options();
        if rust_ty == "bf16" {
            options.add_macro_definition("BF16", None);
        } else {
            options.add_macro_definition("T", Some(c_ty));
        }
        compile_glsl(compiler, src, &format!("gemm_{}", rust_ty), Some(&options))?;
    }
    for &(rust_ty, c_ty) in FLOAT_TYPES.iter() {
        {
            // Relu
            let mut options = glsl_options();
            if rust_ty == "bf16" {
                options.add_macro_definition("BF16", None);
            } else {
                options.add_macro_definition("T", Some(c_ty));
            }
            options.add_macro_definition("RELU", None);
            compile_glsl(
                compiler,
                src,
                &format!("gemm_relu_{}", rust_ty),
                Some(&options),
            )?;
        }
        {
            // Bias
            let mut options = glsl_options();
            if rust_ty == "bf16" {
                options.add_macro_definition("BF16", None);
            } else {
                options.add_macro_definition("T", Some(c_ty));
            }
            options.add_macro_definition("BIAS", None);
            compile_glsl(
                compiler,
                src,
                &format!("gemm_bias_{}", rust_ty),
                Some(&options),
            )?;
        }
        {
            // Bias + Relu
            let mut options = glsl_options();
            if rust_ty == "bf16" {
                options.add_macro_definition("BF16", None);
            } else {
                options.add_macro_definition("T", Some(c_ty));
            }
            options.add_macro_definition("BIAS", None);
            options.add_macro_definition("RELU", None);
            compile_glsl(
                compiler,
                src,
                &format!("gemm_bias_relu_{}", rust_ty),
                Some(&options),
            )?;
        }
    }
    Ok(())
}

fn glsl_binary(compiler: &mut Compiler) -> Result<()> {
    let src = include_str!("src/glsl/binary.comp");
    for op in ["add", "scaled_add"].iter() {
        for &(rust_ty, c_ty) in NUM_TYPES.iter() {
            let mut options = glsl_options();
            if rust_ty == "bf16" {
                options.add_macro_definition("BF16", None);
            } else {
                options.add_macro_definition("T", Some(c_ty));
            }
            if op.contains("add") {
                options.add_macro_definition("OP", Some("+"));
            }
            if op.contains("scaled") {
                options.add_macro_definition("ASSIGN", None);
            }
            compile_glsl(compiler, src, &format!("{}_{}", op, rust_ty), Some(&options))?;
        }
    }
    Ok(())
}

fn glsl_cast(compiler: &mut Compiler) -> Result<()> {
    let src = include_str!("src/glsl/cast.comp");
    // TODO: impl for Scalar
    for &(rust_ty_1, c_ty_1) in SCALAR_TYPES.iter().chain(once(&("u8", "uint"))) {
        for &(rust_ty_2, c_ty_2) in NUM_TYPES.iter() {
            let mut options = glsl_options();
            if rust_ty_1 == "u8" {
                options.add_macro_definition("U8", None);
            } else if rust_ty_1 == "u16" {
                options.add_macro_definition("U16", None);
            } else if rust_ty_1 == "bf16" {
                options.add_macro_definition("BF16", None);
            } else {
                options.add_macro_definition("T", Some(c_ty_1));
            }
            if rust_ty_2 == "bf16" {
                options.add_macro_definition("T2_BF16", None);
            } else {
                options.add_macro_definition("T2", Some(c_ty_2));
            }
            compile_glsl(compiler, src, &format!("scaled_cast_{}_{}", rust_ty_1, rust_ty_2), Some(&options))?;
        }
    }
    for &(rust_ty_2, c_ty_2) in NUM_TYPES.iter() {
        let mut options = glsl_options();
        if rust_ty_2 == "bf16" {
            options.add_macro_definition("T2_BF16", None);
        } else {
            options.add_macro_definition("T2", Some(c_ty_2));
        }
        options.add_macro_definition("INPLACE", None);
        compile_glsl(compiler, src, &format!("scale_mut_{}", rust_ty_2), Some(&options))?;
    }
    Ok(())
}

fn glsl_cross_entropy_loss(compiler: &mut Compiler) -> Result<()> {
    let src = include_str!("src/glsl/cross_entropy_loss.comp");
    for &(rust_ty, c_ty) in FLOAT_TYPES.iter() {
        for &c in [64, 256, 1024].iter() {
            let mut options = glsl_options();
            if rust_ty == "bf16" {
                options.add_macro_definition("BF16", None);
            } else {
                options.add_macro_definition("T", Some(c_ty));
            }
            options.add_macro_definition("C", Some(&c.to_string()));
            compile_glsl(compiler, src, &format!("cross_entropy_loss_{}_{}", rust_ty, c), Some(&options))?;
        }
    }
    Ok(())
}

fn glsl_cross_entropy_loss_backward(compiler: &mut Compiler) -> Result<()> {
    let src = include_str!("src/glsl/cross_entropy_loss_backward.comp");
    for &(rust_ty, c_ty) in FLOAT_TYPES.iter() {
        let mut options = glsl_options();
        if rust_ty == "bf16" {
            options.add_macro_definition("BF16", None);
        } else {
            options.add_macro_definition("T", Some(c_ty));
        }
        compile_glsl(compiler, src, &format!("cross_entropy_loss_backward_{}", rust_ty), Some(&options))?;
    }
    Ok(())
}

fn glsl_one_hot(compiler: &mut Compiler) -> Result<()> {
    let src = include_str!("src/glsl/one_hot.comp");
    for &(rust_ty_1, c_ty_1) in UNSIGNED_TYPES.iter() {
        for &(rust_ty_2, c_ty_2) in NUM_TYPES.iter() {
            let mut options = glsl_options();
            if rust_ty_1 == "u8" {
                options.add_macro_definition("U8", None);
            } else if rust_ty_1 == "u16" {
                options.add_macro_definition("U16", None);
            } else {
                options.add_macro_definition("T", Some(c_ty_1));
            }
            if rust_ty_2 == "bf16" {
                options.add_macro_definition("T2_BF16", None);
            } else {
                options.add_macro_definition("T2", Some(c_ty_2));
            }
            compile_glsl(compiler, src, &format!("one_hot_{}_{}", rust_ty_1, rust_ty_2), Some(&options))?;
        }
    }
    Ok(())
}

fn glsl_reduce(compiler: &mut Compiler) -> Result<()> {
    let src = include_str!("src/glsl/reduce_final.comp");
    for op in ["sum", "mean", "min", "max", "argmin", "argmax"].iter() {
        for &(rust_ty, c_ty) in NUM_TYPES.iter() {
            let mut options = glsl_options();
            if rust_ty == "bf16" {
                options.add_macro_definition("BF16", None);
            } else {
                options.add_macro_definition("T", Some(c_ty));
            }
            if rust_ty == "bf16" || rust_ty == "f32" {
                options.add_macro_definition("FLOAT", None);
            } else if rust_ty == "u32" {
                options.add_macro_definition("UINT", None);
            } else if rust_ty == "i32" {
                options.add_macro_definition("INT", None);
            }
            options.add_macro_definition(&op.to_uppercase(), None);
            compile_glsl(compiler, src, &format!("reduce_{}_final_{}", op, rust_ty), Some(&options))?;
        }
    }
    Ok(())
}

fn glsl_index_select(compiler: &mut Compiler) -> Result<()> {
    let src = include_str!("src/glsl/index_select.comp");
    for &(rust_ty, c_ty) in NUM_TYPES.iter() {
        let mut options = glsl_options();
        if rust_ty == "bf16" {
            options.add_macro_definition("BF16", None);
        } else {
            options.add_macro_definition("T", Some(c_ty));
        }
        compile_glsl(compiler, src, &format!("index_select_{}", rust_ty), Some(&options))?;
    }
    Ok(())
}

fn glsl_kmeans(compiler: &mut Compiler) -> Result<()> {
    {
        let src = include_str!("src/glsl/kmeans_distance.comp");
        for &(rust_ty, c_ty) in FLOAT_TYPES.iter() {
            let mut options = glsl_options();
            if rust_ty == "bf16" {
                options.add_macro_definition("BF16", None);
            } else {
                options.add_macro_definition("T", Some(c_ty));
            }
            compile_glsl(compiler, src, &format!("kmeans_distance_{}", rust_ty), Some(&options))?;
        }
    }
    {
        let src = include_str!("src/glsl/kmeans_accumulate_next_centroids.comp");
        for &(rust_ty, c_ty) in FLOAT_TYPES.iter() {
            let mut options = glsl_options();
            if rust_ty == "bf16" {
                options.add_macro_definition("BF16", None);
            } else {
                options.add_macro_definition("T", Some(c_ty));
            }
            compile_glsl(compiler, src, &format!("kmeans_accumulate_next_centroids_{}", rust_ty), Some(&options))?;
        }
    }
    {
        let src = include_str!("src/glsl/kmeans_update_centroids.comp");
        for &(rust_ty, c_ty) in FLOAT_TYPES.iter() {
            let mut options = glsl_options();
            if rust_ty == "bf16" {
                options.add_macro_definition("BF16", None);
            } else {
                options.add_macro_definition("T", Some(c_ty));
            }
            compile_glsl(compiler, src, &format!("kmeans_update_centroids_{}", rust_ty), Some(&options))?;
        }
    }
    Ok(())
}



fn main() -> Result<()> {
    let mut compiler = Compiler::new().unwrap();
    //glsl_accuracy(&mut compiler)?;
    glsl_bias_backward(&mut compiler)?;
    glsl_fill(&mut compiler)?;
    glsl_gemm(&mut compiler)?;
    glsl_binary(&mut compiler)?;
    glsl_cast(&mut compiler)?;
    glsl_cross_entropy_loss(&mut compiler)?;
    glsl_cross_entropy_loss_backward(&mut compiler)?;
    glsl_one_hot(&mut compiler)?;
    glsl_reduce(&mut compiler)?;
    glsl_index_select(&mut compiler)?;
    glsl_kmeans(&mut compiler)?;
    Ok(())
}
