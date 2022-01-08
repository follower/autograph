use anyhow::{anyhow, bail};
use fxhash::FxHashMap;
use hibitset::BitSet;
use naga::Module as NagaModule;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use rspirv::{
    binary::{Disassemble, Parser},
    dr::Loader,
};
use serde::{de::Error as _, Deserialize, Deserializer, Serialize};
use std::{
    borrow::Cow,
    fmt::{self, Debug},
};
type Result<T, E = anyhow::Error> = std::result::Result<T, E>;

//pub(super) const PUSH_CONSTANT_SIZE: usize = 256;
pub(super) const SPECIALIZATION_SIZE: usize = 32;

static MODULE_IDS: Lazy<Mutex<BitSet>> = Lazy::new(Mutex::default);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct ModuleId(pub(super) u32);

impl ModuleId {
    fn create() -> Result<Self> {
        let mut ids = MODULE_IDS.lock();
        for id in 0.. {
            if !ids.contains(id) {
                ids.add(id);
                return Ok(Self(id));
            }
        }
        Err(anyhow!("Too many modules!"))
    }
    fn deserialize_create<'de, D: Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Self::create().map_err(D::Error::custom)
    }
    fn destroy(self) {
        MODULE_IDS.lock().remove(self.0);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct EntryId(pub(super) u32);

/// A compute shader module.
///
/// A module has an [SPIRV](https://www.khronos.org/spir/) source and info about each entry (shader function).
///
/// # Limits
/// - Up to 4 Buffer arguments.
/// - Up to 64 bytes of push constants.
#[derive(Serialize, Deserialize)]
pub struct Module {
    pub(super) spirv: Cow<'static, [u8]>,
    pub(super) descriptor: ModuleDescriptor,
    #[serde(skip_serializing, deserialize_with = "ModuleId::deserialize_create")]
    pub(super) id: ModuleId,
    name: String,
}

impl Module {
    /// Parses the spirv into a Module.
    ///
    /// Note: If `spirv` is not aligned to 4 bytes, will clone the data (generally this will happen when using the include_bytes! macro).
    ///
    /// **Errors**
    /// - Will error if the `spirv` is invalid.
    pub fn from_spirv(spirv: impl Into<Cow<'static, [u8]>>) -> Result<Self> {
        let mut spirv = spirv.into();
        if bytemuck::try_cast_slice::<u8, u32>(spirv.as_ref()).is_err() {
            spirv.to_mut();
        }
        let descriptor = ModuleDescriptor::parse(&spirv)?;
        let id = ModuleId::create()?;
        Ok(Self {
            spirv,
            descriptor,
            id,
            name: String::new(),
        })
    }
    /// Names the module.
    ///
    /// The name will be used in error messages.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
    pub(super) fn name(&self) -> Option<&str> {
        if !self.name.is_empty() {
            Some(self.name.as_str())
        } else {
            None
        }
    }
    #[doc(hidden)]
    pub fn rspirv_module(&self) -> rspirv::dr::Module {
        let mut loader = Loader::new();
        Parser::new(&self.spirv, &mut loader).parse().unwrap();
        loader.module()
    }
    #[doc(hidden)]
    pub fn disassemble(&self) -> String {
        self.rspirv_module().disassemble()
    }
    #[doc(hidden)]
    pub fn descriptor_to_string(&self) -> String {
        format!("{:#?}", &self.descriptor)
    }

    #[cfg(test)]
    pub(crate) fn to_metal(&self) -> Result<String> {
        use naga::back::msl::{Options, PipelineOptions, Writer};
        let module = spirv_to_naga(&self.spirv)?;
        let info = validate_naga(&module)?;
        let mut writer = Writer::new(String::new());
        writer.write(
            &module,
            &info,
            &Options::default(),
            &PipelineOptions::default(),
        )?;
        let metal = writer.finish();
        Ok(metal)
    }

    #[cfg(test)]
    pub(crate) fn to_hlsl(&self) -> Result<FxHashMap<String, String>> {
        use anyhow::Context;
        use spirv_cross::{
            hlsl::{CompilerOptions, Target},
            spirv::{Ast, ExecutionModel, Module},
        };
        let name: Cow<str> = self
            .name()
            .map_or_else(|| format!("{:?}", self).into(), Into::into);
        let module = Module::from_words(bytemuck::cast_slice(&self.spirv));
        let mut ast = Ast::<Target>::parse(&module)
            .with_context(|| format!("Parsing of {name} failed!", name = name))?;
        let mut options = CompilerOptions::default();
        options.shader_model = spirv_cross::hlsl::ShaderModel::V5_1;
        let mut hlsl =
            FxHashMap::with_capacity_and_hasher(self.descriptor.entries.len(), Default::default());
        for entry in self.descriptor.entries.keys() {
            options
                .entry_point
                .replace((entry.clone(), ExecutionModel::GlCompute));
            ast.set_compiler_options(&options)?;
            let output = ast.compile().map(Into::into).with_context(|| {
                format!(
                    "Compilation of {name}: {entry} failed!",
                    name = name,
                    entry = entry
                )
            })?;
            hlsl.insert(entry.clone(), output);
        }
        Ok(hlsl)
    }
}

impl Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(name) = self.name() {
            f.debug_tuple("Module").field(&name).finish()
        } else {
            f.debug_tuple("Module").field(&self.id.0).finish()
        }
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        self.id.destroy()
    }
}

fn validate_naga(
    module: &NagaModule,
) -> Result<naga::valid::ModuleInfo, naga::WithSpan<naga::valid::ValidationError>> {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    Validator::new(ValidationFlags::all(), Capabilities::PUSH_CONSTANT).validate(module)
}

fn spirv_to_naga(spirv: &[u8]) -> Result<NagaModule> {
    Ok(naga::front::spv::parse_u8_slice(
        spirv,
        &Default::default(),
    )?)
}

#[derive(Default, Debug, Serialize, Deserialize)]
pub(super) struct ModuleDescriptor {
    pub(super) entries: FxHashMap<String, EntryDescriptor>,
}

impl ModuleDescriptor {
    fn parse(spirv: &[u8]) -> Result<Self> {
        Self::parse_naga(&spirv_to_naga(spirv)?)
    }
    fn parse_naga(naga_module: &NagaModule) -> Result<Self> {
        parse_naga(naga_module)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct EntryDescriptor {
    pub(super) id: EntryId,
    pub(super) local_size: [u32; 3],
    pub(super) buffers: Vec<bool>,
    pub(super) push_constant_size: u8,
    pub(super) spec_constants: Vec<SpecConstant>,
}

impl EntryDescriptor {
    pub(super) fn push_constant_size(&self) -> usize {
        self.push_constant_size as usize
    }
    pub(super) fn specialization_size(&self) -> usize {
        self.spec_constants.iter().map(|x| x.spec_type.size()).sum()
    }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub(super) struct SpecConstant {
    id: u32,
    spec_type: SpecType,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub(super) enum SpecType {
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
}

impl SpecType {
    pub(super) fn size(&self) -> usize {
        use SpecType::*;
        match self {
            U32 | I32 | F32 => 4,
            U64 | I64 | F64 => 8,
        }
    }
}
/*
#[derive(Clone, Debug)]
struct EntryPoint {
    name: String,
    local_size: [u32; 3],
}

#[derive(Clone, Copy, Debug)]
struct BufferBinding {
    descriptor_set: u32,
    binding: u32,
    mutable: bool,
}*/

fn parse_naga(module: &NagaModule) -> Result<ModuleDescriptor> {
    use naga::{valid::GlobalUse, ShaderStage, StorageClass, TypeInner};

    let module_info = validate_naga(module)?;
    let mut entries =
        FxHashMap::with_capacity_and_hasher(module.entry_points.len(), Default::default());

    for (i, entry_point) in module.entry_points.iter().enumerate() {
        let mut storage_buffers = Vec::new();
        let mut push_constant_size = 0;
        let entry = entry_point.name.as_str();
        let entry_info = module_info.get_entry_point(i);

        if entry_point.stage != ShaderStage::Compute {
            bail!(
                "{entry:?}: Only compute shaders supported! Found {stage:?}!",
                entry = entry_point.name.as_str(),
                stage = entry_point.stage
            );
        }

        for (v, var) in module.global_variables.iter() {
            let global_use = entry_info[v];
            if global_use.contains(GlobalUse::READ) || global_use.contains(GlobalUse::WRITE) {
                match &var.class {
                    StorageClass::Storage { .. } => {
                        if let Some(binding) = &var.binding {
                            let binding = binding.binding as usize;
                            while storage_buffers.len() <= binding {
                                storage_buffers.push(None);
                            }
                            if storage_buffers[binding].is_some() {
                                bail!("{entry:?}: bindings must be unique!", entry = entry);
                            }
                            let mutable = global_use.contains(GlobalUse::WRITE);
                            storage_buffers[binding].replace(mutable);
                        }
                    }
                    StorageClass::PushConstant => {
                        let ty = &module.types[var.ty];
                        if let TypeInner::Struct { span, .. } = ty.inner {
                            if push_constant_size != 0 {
                                bail!(
                                    "{entry:?}: cannot have more than 1 push constant block!",
                                    entry = entry
                                );
                            }
                            if span > 256 {
                                bail!("{entry:?}: push constant size must be less than 256 bytes! Found {span}!", entry=entry, span=span);
                            }
                            push_constant_size = span as u8;
                        }
                    }
                    _ => (),
                }
            }
        }

        let mut buffers = Vec::with_capacity(storage_buffers.len());
        for buffer in storage_buffers.iter() {
            if let Some(mutable) = buffer {
                buffers.push(*mutable);
            } else {
                bail!(
                    "{entry:?}: bindings must be in order from 0 .. N! {buffers:?}",
                    entry = entry,
                    buffers = storage_buffers
                );
            }
        }
        entries.insert(
            entry.to_string(),
            EntryDescriptor {
                id: EntryId(i as u32),
                local_size: entry_point.workgroup_size,
                buffers,
                push_constant_size,
                spec_constants: Vec::new(),
            },
        );
    }
    Ok(ModuleDescriptor { entries })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_module_from_spirv() -> Result<()> {
        Module::from_spirv(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/shaders/rust/core.spv"
            ))
            .as_ref(),
        )?;
        Ok(())
    }
}
