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
pub mod error;
pub mod neural_network;
pub mod tensor;
mod util;

pub use ndarray;

pub type Result<T, E = error::Error> = std::result::Result<T, E>;
