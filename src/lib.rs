#[macro_use]
mod macros {
    #[macro_export]
    macro_rules! include_spirv {
        ($file:expr) => {{
            #[repr(C)]
            pub struct AlignedAs<Align, Bytes: ?Sized> {
                pub _align: [Align; 0],
                pub bytes: Bytes,
            }

            #[allow(unused)]
            static ALIGNED: &AlignedAs<u32, [u8]> = &AlignedAs {
                _align: [],
                bytes: *include_bytes!($file),
            };

            &ALIGNED.bytes
        }};
    }

    macro_rules! include_shader {
        ($file:expr) => {{
            include_spirv!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/", $file))
        }};
    }
}

pub mod backend;
pub mod dataset;
pub mod tensor;
pub mod learn;
pub mod cluster;
mod util;

pub use anyhow::Result;
pub use ndarray;
