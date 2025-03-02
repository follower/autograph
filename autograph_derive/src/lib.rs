/*!
# Usage
You can derive Layer and Forward for structs and tuple structs, (enums not yet implemented).
```
use autograph::{
    result::Result,
    learn::neural_network::{
        layer::{Layer, Forward},
        autograd::{VariableD, ParameterD},
    },
};

// Derive Layer for custom layers.
#[derive(Layer)]
struct MyLayer {
    // Note that parameters must be dynamic dimensional to be compatible with
    // Layer::parameters_mut().
    #[autograph(parameter)]
    weight: ParameterD,
    #[autograph(optional_parameter)]
    bias: Option<ParameterD>,
    #[autograph(optional_layer)]
    activation: Option<MyActivation>,
}

impl Forward for MyLayer {
    fn forward(&self, input: VariableD) -> Result<VariableD> {
        todo!() // custom impl
    }
}

/// Layer can be derived for functional layers without parameters.
#[derive(Layer)]
struct MyActivation {}

impl Forward for MyActivation {
    fn forward(&self, input: VariableD) -> Result<VariableD> {
        todo!() // custom impl
    }
}

// Forward can be derived for a sequence of layers, where each output is passed to the next
// layer in order.
#[derive(Layer, Forward)]
struct MyNetwork {
    #[autograph(layer)]
    layer1: MyLayer,
    #[autograph(optional_layer)]
    layer2: Option<MyLayer>,
}
```

Autograd can be derived as well for structs.
```
#[derive(Autograd)]
struct DenseBackward {
    // Use vertex / optional_vertex for Variables and Parameters
    #[autograph(vertex)]
    input: Variable2,
    #[autograph(vertex)]
    weight: Parameter2,
    #[autograph(optional_vertex)]
    bias: Option<Parameter1>,
}

#[derive(Autograd)]
struct ReluBackward {
    // Use gradient / optional_gradient for VariableGradient and ParameterGradient
    #[autograph(gradient)]
    input_grad: VariableGradientD,
    output: FloatArcTensorD,
}
```
*/

use proc_macro::TokenStream;
use proc_macro2::{Literal, TokenStream as TokenStream2};
use quote::{quote, ToTokens};
use syn::{
    Attribute, Data, DataStruct, DeriveInput, Fields, FnArg, Index, ItemFn, Meta, NestedMeta,
};

fn autograph_path(attributes: &[Attribute]) -> TokenStream2 {
    for attribute in attributes {
        if let Ok(Meta::List(meta_list)) = attribute.parse_meta() {
            if &meta_list.path.to_token_stream().to_string() == "autograph" {
                if let Some(NestedMeta::Meta(Meta::Path(path))) = meta_list.nested.first() {
                    if path.to_token_stream().to_string() == "crate" {
                        return quote! {
                            crate
                        };
                    }
                }
            }
        }
    }
    quote! {
        ::autograph
    }
}

enum FieldKind {
    Parameter,
    OptionalParameter,
    Layer,
    OptionalLayer,
    Vertex,
    OptionalVertex,
    Gradient,
    OptionalGradient,
}

impl FieldKind {
    fn parse(attrs: &[Attribute]) -> Option<Self> {
        for attr in attrs {
            if let Ok(Meta::List(meta_list)) = attr.parse_meta() {
                if &meta_list.path.to_token_stream().to_string() == "autograph" {
                    if let Some(NestedMeta::Meta(Meta::Path(path))) = meta_list.nested.first() {
                        let s = path.to_token_stream().to_string();
                        match s.as_str() {
                            "parameter" => {
                                return Some(Self::Parameter);
                            }
                            "optional_parameter" => {
                                return Some(Self::OptionalParameter);
                            }
                            "layer" => {
                                return Some(Self::Layer);
                            }
                            "optional_layer" => {
                                return Some(Self::OptionalLayer);
                            }
                            "vertex" => {
                                return Some(Self::Vertex);
                            }
                            "optional_vertex" => {
                                return Some(Self::OptionalVertex);
                            }
                            "gradient" => {
                                return Some(Self::Gradient);
                            }
                            "optional_gradient" => {
                                return Some(Self::OptionalGradient);
                            }
                            _ => (),
                        }
                    }
                }
            }
        }
        None
    }
}

fn get_struct_fields(data_struct: &DataStruct) -> Vec<(TokenStream2, FieldKind)> {
    let fields = match &data_struct.fields {
        Fields::Named(fields) => &fields.named,
        Fields::Unnamed(fields) => &fields.unnamed,
        Fields::Unit => {
            return Vec::new();
        }
    };
    fields
        .iter()
        .enumerate()
        .filter_map(|(i, field)| {
            let ident = if let Some(ident) = &field.ident {
                ident.to_token_stream()
            } else {
                let index = Index::from(i);
                quote! { #index }
            };
            Some((ident, FieldKind::parse(&field.attrs)?))
        })
        .collect()
}

fn derive_layer_struct(input: &DeriveInput, data_struct: &DataStruct) -> TokenStream {
    let autograph_path = autograph_path(&input.attrs);
    let fields = get_struct_fields(data_struct);
    let mut parameters_len_impl = TokenStream2::new();
    let mut collect_parameters_impl = TokenStream2::new();
    let mut collect_parameters_mut_impl = TokenStream2::new();
    let mut layers_impl = TokenStream2::new();
    let mut layers_mut_impl = TokenStream2::new();
    if !fields.is_empty() {
        let mut parameters_len_inner = Vec::new();
        let mut collect_parameters_inner = TokenStream2::new();
        let mut collect_parameters_mut_inner = TokenStream2::new();
        let mut layers_inner = Vec::new();
        let mut layers_mut_inner = Vec::new();
        for (ident, kind) in fields.iter() {
            match kind {
                FieldKind::Parameter => {
                    parameters_len_inner.push(quote! {
                        1
                    });
                    collect_parameters_inner.extend(quote! {
                        parameters.push(self. #ident .clone());
                    });
                    collect_parameters_mut_inner.extend(quote! {
                        parameters.push(&mut self. #ident);
                    });
                }
                FieldKind::OptionalParameter => {
                    parameters_len_inner.push(quote! {
                        if self. #ident .is_some() { 1 } else { 0 }
                    });
                    collect_parameters_inner.extend(quote! {
                        parameters.extend(self. #ident.clone());
                    });
                    collect_parameters_mut_inner.extend(quote! {
                        parameters.extend(self. #ident .as_mut());
                    });
                }
                FieldKind::Layer => {
                    parameters_len_inner.push(quote! {
                        self. #ident .parameters_len()
                    });
                    collect_parameters_inner.extend(quote! {
                        self. #ident .collect_parameters(parameters);
                    });
                    collect_parameters_mut_inner.extend(quote! {
                        self. #ident .collect_parameters_mut(parameters);
                    });
                    layers_inner.push(quote! {
                        ::core::iter::once(&self. #ident as &dyn Layer)
                    });
                    layers_mut_inner.push(quote! {
                        ::core::iter::once(&mut self. #ident as &mut dyn Layer)
                    });
                }
                FieldKind::OptionalLayer => {
                    parameters_len_inner.push(quote! {
                        self. #ident .as_ref().map_or(0, Layer::parameters_len)
                    });
                    collect_parameters_inner.extend(quote! {
                        if let Some(layer) = self. #ident .as_ref() {
                            layer.collect_parameters(parameters);
                        }
                    });
                    collect_parameters_mut_inner.extend(quote! {
                        if let Some(layer) = self. #ident .as_mut() {
                            layer.collect_parameters_mut(parameters);
                        }
                    });
                    layers_inner.push(quote! {
                        self. #ident .as_ref().map(|layer| layer as &dyn Layer)
                    });
                    layers_mut_inner.push(quote! {
                        self. #ident .as_mut().map(|layer| layer as &mut dyn Layer)
                    });
                }
                _ => (),
            }
        }
        parameters_len_impl = quote! {
            fn parameters_len(&self) -> usize {
                #(#parameters_len_inner)+*
            }
        };
        collect_parameters_impl = quote! {
            fn collect_parameters(
                &self,
                parameters: &mut Vec<#autograph_path::learn::neural_network::autograd::ParameterD>,
            ) {
                #collect_parameters_inner
            }
        };
        collect_parameters_mut_impl = quote! {
            fn collect_parameters_mut<'a>(
                &'a mut self,
                parameters: &mut Vec<&'a mut #autograph_path::learn::neural_network::autograd::ParameterD>,
            ) {
                #collect_parameters_mut_inner
            }
        };
        layers_impl = quote! {
            fn layers(&self) -> Vec<&dyn Layer> {
                ::core::iter::empty::<&dyn Layer>()
                    #(.chain(#layers_inner))*
                    .collect()
            }
        };
        layers_mut_impl = quote! {
            fn layers_mut(&mut self) -> Vec<&mut dyn Layer> {
                ::core::iter::empty::<&mut dyn Layer>()
                    #(.chain(#layers_mut_inner))*
                    .collect()
            }
        };
    }
    let ident = &input.ident;
    TokenStream::from(quote! {
        #[automatically_derived]
        #[#autograph_path ::learn::neural_network::layer::async_trait]
        impl Layer for #ident {
            #parameters_len_impl
            #collect_parameters_impl
            #collect_parameters_mut_impl
            #layers_impl
            #layers_mut_impl
        }
    })
}

#[proc_macro_derive(Layer, attributes(autograph))]
pub fn derive_layer(input: TokenStream) -> TokenStream {
    let input: DeriveInput = syn::parse(input).unwrap();

    match &input.data {
        Data::Struct(data_struct) => derive_layer_struct(&input, data_struct),
        Data::Enum(_data_enum) => TokenStream::from(quote! {
            compile_error!("Not yet implemented!")
        }),
        Data::Union(_) => TokenStream::from(quote! {
            compile_error!("Unions unsupported!")
        }),
    }
}

fn derive_forward_struct(input: &DeriveInput, data_struct: &DataStruct) -> TokenStream {
    let fields = get_struct_fields(data_struct);
    let mut forward_inner = Vec::new();
    for (ident, kind) in fields.iter() {
        match kind {
            FieldKind::Layer => {
                forward_inner.push(quote! {
                    x = self. #ident .forward(x)?;
                });
            }
            FieldKind::OptionalLayer => {
                forward_inner.push(quote! {
                    if let Some(layer) = self. #ident .as_ref() {
                        x = layer.forward(x)?;
                    }
                });
            }
            FieldKind::Parameter | FieldKind::OptionalParameter => {
                forward_inner.push(quote! {
                    compile_error!("Cannot derive Forward with Parameters!");
                });
            }
            _ => (),
        }
    }
    let autograph_path = autograph_path(&input.attrs);
    let ident = &input.ident;
    TokenStream::from(quote! {
        #[automatically_derived]
        impl Forward for #ident {
            fn forward(&self, input: #autograph_path::learn::neural_network::autograd::VariableD) -> #autograph_path::result::Result<#autograph_path::learn::neural_network::autograd::VariableD> {
                let mut x = input;
                #(#forward_inner)*
                Ok(x)
            }
        }
    })
}

#[proc_macro_derive(Forward, attributes(autograph))]
pub fn derive_forward(input: TokenStream) -> TokenStream {
    let input: DeriveInput = syn::parse(input).unwrap();

    match &input.data {
        Data::Struct(data_struct) => derive_forward_struct(&input, data_struct),
        Data::Enum(_data_enum) => TokenStream::from(quote! {
            compile_error!("Not yet implemented!")
        }),
        Data::Union(_) => TokenStream::from(quote! {
            compile_error!("Unions unsupported!")
        }),
    }
}

fn derive_autograd_struct(input: &DeriveInput, data_struct: &DataStruct) -> TokenStream {
    let autograph_path = autograph_path(&input.attrs);
    let fields = get_struct_fields(data_struct);
    let mut grads_inner = Vec::<TokenStream2>::new();
    if !fields.is_empty() {
        for (ident, kind) in fields.iter() {
            match kind {
                FieldKind::Vertex | FieldKind::Parameter => {
                    grads_inner.push(quote! {
                        self.#ident .grad().map(|grad| grad.into_gradient().into_dyn())
                    });
                }
                FieldKind::OptionalVertex | FieldKind::OptionalParameter => {
                    grads_inner.push(quote! {
                        self.#ident .as_ref().map(|v| v.grad()).flatten().map(|grad| grad.into_gradient().into_dyn())
                    });
                }
                FieldKind::Gradient => {
                    grads_inner.push(quote! {
                        ::core::iter::once(self.#ident .clone().into_gradient().into_dyn())
                    });
                }
                FieldKind::OptionalGradient => {
                    grads_inner.push(quote! {
                        self.#ident .as_ref().map(|grad| grad.clone().into_gradient().into_dyn()).into_iter()
                    });
                }
                _ => (),
            }
        }
    }
    let ident = &input.ident;
    let name_impl = quote! {
        fn name(&self) -> ::std::borrow::Cow<'static, str> {
            stringify!(#ident).into()
        }
    };
    let grads_impl = quote! {
        fn grads(&self) -> Vec<#autograph_path::learn::neural_network::autograd::GradientD> {
            ::core::iter::empty::<#autograph_path::learn::neural_network::autograd::GradientD>()
                #(.chain(#grads_inner))*
                .collect()
        }
    };
    TokenStream::from(quote! {
        #[automatically_derived]
        impl Autograd for #ident {
            #name_impl
            #grads_impl
        }
    })
}

#[proc_macro_derive(Autograd, attributes(autograph))]
pub fn derive_autograd(input: TokenStream) -> TokenStream {
    let input: DeriveInput = syn::parse(input).unwrap();

    match &input.data {
        Data::Struct(data_struct) => derive_autograd_struct(&input, data_struct),
        Data::Enum(_data_enum) => TokenStream::from(quote! {
            compile_error!("Not yet implemented!")
        }),
        Data::Union(_) => TokenStream::from(quote! {
            compile_error!("Unions unsupported!")
        }),
    }
}

#[doc(hidden)]
// Generates descriptor_set = 0, binding = 0.. for each #[spirv(storage_buffer)]
#[proc_macro_attribute]
pub fn autobind(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut func: ItemFn = syn::parse(item).unwrap();
    let mut binding: usize = 0;

    for fn_arg in func.sig.inputs.iter_mut() {
        let pat_type = match fn_arg {
            FnArg::Receiver(_) => continue,
            FnArg::Typed(pat_type) => pat_type,
        };
        for mut attr in pat_type.attrs.iter_mut() {
            if let Ok(Meta::List(meta_list)) = attr.parse_meta() {
                if &meta_list.path.to_token_stream().to_string() == "spirv" {
                    let mut found = false;
                    for nested in meta_list.nested.iter() {
                        if let NestedMeta::Meta(Meta::Path(path)) = nested {
                            if path.to_token_stream().to_string() == "storage_buffer" {
                                found = true;
                                break;
                            }
                        }
                    }
                    if found {
                        let mut nested = meta_list.nested;
                        let descriptor_set_name_value = syn::parse(
                            quote! {
                                descriptor_set = 0
                            }
                            .into(),
                        )
                        .expect("Unable to parse descriptor_set!");
                        nested.push(NestedMeta::Meta(Meta::NameValue(descriptor_set_name_value)));
                        let binding_lit = Literal::usize_unsuffixed(binding);
                        let binding_name_value = syn::parse(
                            quote! {
                                binding = #binding_lit
                            }
                            .into(),
                        )
                        .expect("Unable to parse binding!");
                        nested.push(NestedMeta::Meta(Meta::NameValue(binding_name_value)));
                        attr.tokens = quote! {
                            (#nested)
                        };
                        binding += 1;
                        break;
                    }
                }
            }
        }
    }

    TokenStream::from(quote! {
        #func
    })
}
