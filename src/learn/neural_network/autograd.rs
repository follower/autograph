#[cfg(doc)]
use crate::tensor::float::FloatTensorBase;
use crate::{
    buffer::float::FloatBuffer,
    device::Device,
    glsl_shaders,
    linalg::{Dot, DotAcc, DotBias},
    result::Result,
    scalar::FloatType,
    tensor::float::{
        FloatArcTensor, FloatTensor, FloatTensor2, FloatTensorD, FloatTensorView, FloatTensorView0,
        FloatTensorView2, FloatTensorViewMut, FloatTensorViewMut1, FloatTensorViewMut2,
    },
};
use anyhow::{anyhow, bail, Context};
use ndarray::{Dimension, IntoDimension, Ix0, Ix1, Ix2, IxDyn, ShapeBuilder};
use parking_lot::{Mutex, MutexGuard};
use std::{
    convert::{TryFrom, TryInto},
    fmt::{self, Debug},
    iter::once,
    mem::take,
    sync::Arc,
};

mod sealed {
    pub trait NodeBase {}
}
use sealed::NodeBase;

trait Backward: Send + Sync + 'static {
    fn backward(&self, vertices: VertexGuardDVec, output_grad: FloatTensorD) -> Result<()>;
}

impl<F: Send + Sync + 'static> Backward for F
where
    F: Fn(VertexGuardDVec, FloatTensorD) -> Result<()>,
{
    fn backward(&self, vertices: VertexGuardDVec, output_grad: FloatTensorD) -> Result<()> {
        self(vertices, output_grad)
    }
}

/// Marker trait for [`VertexBase`] nodes.
pub trait Node: Into<VertexNode> + Clone + NodeBase {
    #[doc(hidden)]
    fn name(&self) -> Option<&str>;
    #[doc(hidden)]
    fn set_name(&mut self, name: String) -> Result<()>;
    #[doc(hidden)]
    fn try_into_variable(self) -> Result<VariableNode, Self> {
        Err(self)
    }
    #[doc(hidden)]
    fn try_from_variable(node: VariableNode) -> Result<Self, VariableNode> {
        Err(node)
    }
}

struct VariableNodeBase {
    vertices: Vec<VertexD>,
    backward: Option<Box<dyn Backward>>,
}

/// A [`Variable`] node.
#[derive(Clone, Default)]
pub struct VariableNode {
    base: Option<Arc<VariableNodeBase>>,
    name: Option<Arc<String>>,
    training: bool,
    graph: bool,
}

impl NodeBase for VariableNode {}

impl Node for VariableNode {
    fn name(&self) -> Option<&str> {
        self.name.as_deref().map(String::as_str)
    }
    fn set_name(&mut self, name: String) -> Result<()> {
        if let Some(_name) = self.name.as_mut() {
            let _name = Arc::get_mut(_name).ok_or_else(|| anyhow!("Variable not exclusive!"))?;
            *_name = name;
        } else {
            self.name.replace(name.into());
        }
        Ok(())
    }
    fn try_into_variable(self) -> Result<VariableNode, Self> {
        Ok(self)
    }
    fn try_from_variable(node: Self) -> Result<Self, Self> {
        Ok(node)
    }
}

/// A [`Parameter`] node.
#[derive(Clone, Default)]
pub struct ParameterNode {
    name: Option<Arc<String>>,
}

impl NodeBase for ParameterNode {}

impl Node for ParameterNode {
    fn name(&self) -> Option<&str> {
        self.name.as_deref().map(String::as_str)
    }
    fn set_name(&mut self, name: String) -> Result<()> {
        if let Some(_name) = self.name.as_mut() {
            let _name = Arc::get_mut(_name).ok_or_else(|| anyhow!("Parameter not exclusive!"))?;
            *_name = name;
        } else {
            self.name.replace(name.into());
        }
        Ok(())
    }
}

/// A [`Vertex`] node.
#[allow(missing_docs)]
#[derive(Clone)]
pub enum VertexNode {
    Variable(VariableNode),
    Parameter(ParameterNode),
}

impl VertexNode {
    fn as_variable_node(&self) -> Option<&VariableNode> {
        match self {
            Self::Variable(node) => Some(node),
            _ => None,
        }
    }
    fn as_parameter_node(&self) -> Option<&ParameterNode> {
        match self {
            Self::Parameter(node) => Some(node),
            _ => None,
        }
    }
}

impl From<VariableNode> for VertexNode {
    fn from(node: VariableNode) -> Self {
        Self::Variable(node)
    }
}

impl From<ParameterNode> for VertexNode {
    fn from(node: ParameterNode) -> Self {
        Self::Parameter(node)
    }
}

impl NodeBase for VertexNode {}

impl Node for VertexNode {
    fn name(&self) -> Option<&str> {
        match self {
            Self::Variable(node) => node.name(),
            Self::Parameter(node) => node.name(),
        }
    }
    fn set_name(&mut self, name: String) -> Result<()> {
        match self {
            Self::Variable(node) => node.set_name(name),
            Self::Parameter(node) => node.set_name(name),
        }
    }
    fn try_into_variable(self) -> Result<VariableNode, Self> {
        match self {
            Self::Variable(node) => Ok(node),
            _ => Err(self),
        }
    }
    fn try_from_variable(node: VariableNode) -> Result<Self, VariableNode> {
        Ok(node.into())
    }
}

/// A vertex for autograd ops.
///
/// [`Variable`] and [`Parameter`] are vertices, convertible into [`Vertex`]. In the forward pass, a graph is built connecting outputs to inputs, storing backward ops. In the backward pass, the graph is traversed from the loss to the inputs, computing the gradients of variables and parameters in order.
#[derive(Clone)]
pub struct VertexBase<D: Dimension, N: Node> {
    value: FloatArcTensor<D>,
    grad: Option<Arc<Mutex<Option<FloatBuffer>>>>,
    node: N,
}

/// A dynamically typed vertex [`Variable`] / [`Parameter`].
pub type Vertex<D> = VertexBase<D, VertexNode>;
/// A dynamic dimensional vertex
pub type VertexD = Vertex<IxDyn>;

/// A variable of a network.
///
/// Inputs to the network are wrapped in variables, which potentially construct the graph of backward ops as the forward pass is computed. Use the [`.backward()`](VertexBase::backward()) method to execute the backward pass, computing the gradients.
pub type Variable<D> = VertexBase<D, VariableNode>;
/// A variable with 1 element
pub type Variable0 = Variable<Ix0>;
/// A variable with 2 dimensions
pub type Variable2 = Variable<Ix2>;
/// A dynamic dimensional variable
pub type VariableD = Variable<IxDyn>;

/// A parameter of a network.
///
/// The parameters are [updated](super::layer::Layer::update()) after one or more forward + backward passes, training the network.
pub type Parameter<D> = VertexBase<D, ParameterNode>;
/// A parameter with 1 dimension
pub type Parameter1 = Parameter<Ix1>;
/// A parameter with 2 dimensions
pub type Parameter2 = Parameter<Ix2>;
/// A dynamic dimensional parameter
pub type ParameterD = Parameter<IxDyn>;

impl<D: Dimension, N: Node> VertexBase<D, N> {
    /// Returns a reference to the value of the vertex.
    pub fn value(&self) -> &FloatArcTensor<D> {
        &self.value
    }
    /// Returns the value of the vertex.
    pub fn into_value(self) -> FloatArcTensor<D> {
        self.value
    }
    /// Returns a mutable reference to the value of the vertex.
    pub fn value_mut(&mut self) -> &mut FloatArcTensor<D> {
        &mut self.value
    }
    /// Creates a vertex of type `float_type` on `device` with `shape` filled with 0's.
    ///
    /// See [`FloatTensor::zeros()`](FloatTensorBase::zeros()).
    pub fn zeros<Sh>(float_type: FloatType, device: Device, shape: Sh) -> Result<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
        N: Default,
    {
        Ok(Self {
            value: FloatArcTensor::zeros(float_type, device, shape)?,
            grad: None,
            node: N::default(),
        })
    }
    /// The device of the vertex.
    pub fn device(&self) -> Device {
        self.value.device()
    }
    /// The dimensions of the vertex in pattern form.
    pub fn dim(&self) -> D::Pattern {
        self.value.dim()
    }
    /// The dimensions of the vertex.
    pub fn raw_dim(&self) -> D {
        self.value.raw_dim()
    }
    /// The dimensions of the vertex as a slice.
    pub fn shape(&self) -> &[usize] {
        self.value.shape()
    }
    /// The strides of the vertex as a slice.
    pub fn strides(&self) -> &[isize] {
        self.value.strides()
    }
    /// The length of the vertex.
    pub fn len(&self) -> usize {
        self.value.len()
    }
    /// Whether the vertex is empty.
    pub fn is_empty(&self) -> bool {
        self.value.is_empty()
    }
    /// The dimensionality of the vertex.
    pub fn ndim(&self) -> usize {
        self.value.ndim()
    }
    /// The [`FloatType`] of the vertex.
    pub fn float_type(&self) -> FloatType {
        self.value.float_type()
    }
    /// Converts the vertex into dimension `D2`.
    ///
    /// See [`FloatTensor::into_dimensionality()`](FloatTensorBase::into_dimensionality()).
    pub fn into_dimensionality<D2>(self) -> Result<VertexBase<D2, N>>
    where
        D2: Dimension,
    {
        Ok(VertexBase {
            value: self.value.into_dimensionality()?,
            grad: self.grad,
            node: self.node,
        })
    }
    /// Returns the vertex with dim `shape`.
    ///
    /// See [`FloatTensor::into_shape()`](FloatTensorBase::into_dimensionality()).
    pub fn into_shape<E>(self, shape: E) -> Result<VertexBase<E::Dim, N>>
    where
        E: IntoDimension,
    {
        Ok(VertexBase {
            value: self.value.into_shape(shape)?,
            grad: self.grad,
            node: self.node,
        })
    }
    /// Reverses (transposes) the axes of the vertex.
    pub fn reversed_axes(mut self) -> Self {
        self.value = self.value.reversed_axes();
        self
    }
    /// Clones self with reversed (transposed) axes.
    pub fn t(&self) -> Self {
        self.clone().reversed_axes()
    }
    #[allow(unused)]
    pub(crate) fn flatten(self) -> Result<VertexBase<Ix2, N>> {
        Ok(VertexBase {
            value: self.value.flatten()?,
            grad: self.grad,
            node: self.node,
        })
    }
    /// Converts the dimensionality of the vertex to [`IxDyn`](type@ndarray::IxDyn).
    pub fn into_dyn(self) -> VertexBase<IxDyn, N> {
        VertexBase {
            value: self.value.into_dyn(),
            grad: self.grad,
            node: self.node,
        }
    }
    /// Converts into a [`Vertex`].
    pub fn into_vertex(self) -> Vertex<D> {
        Vertex {
            value: self.value,
            grad: self.grad,
            node: self.node.into(),
        }
    }
    /// Potentially adds a gradient to the vertex.
    ///
    /// # Note
    /// - If `requires_grad`, gradients **can** be computed for this vertex. The gradient is lazily allocated during the backward pass.
    ///   - For [`Variable`]'s, gradients will be computed for all inputs which require a gradient.
    ///   - For [`Parameter`]'s, gradients will be computed if any input variables are training. See [`.with_training()`](VertexBase::with_training()).
    /// - If `requires_grad` is false, drops the gradient. Gradients will not be computed for this vertex.
    /// - This method must be called on parameters (with `requires_grad` = true) before the first backward pass.
    /// - It *does not* need to be called for each pass.
    /// - Use [`Variable::with_training()`](VertexBase::with_training()) to control which model / parameters are trained (ie parameter gradients computed).
    /// - Input variables typically do not require a gradient.
    ///     - Input variables can have a gradient, this can be used to connect several backward passes together for distributed training.
    pub fn require_grad_mut(&mut self, requires_grad: bool) {
        if requires_grad {
            if self.grad.is_none() {
                self.grad.replace(Arc::default());
            }
        } else {
            self.grad.take();
        }
    }
    /// Potentially adds a gradient to the vertex.
    ///
    /// See [.require_grad_mut()](Self::require_grad_mut()).
    pub fn require_grad(mut self, require_grad: bool) -> Self {
        self.require_grad_mut(require_grad);
        self
    }
    /// Whether the vertex requires grad.
    pub fn requires_grad(&self) -> bool {
        self.grad.is_some()
    }
    /// Takes the gradient from the vertex.
    ///
    /// Returns None if:
    /// - The vertex does not require grad.
    /// - The vertex is not exclusively held.
    /// - The gradient was not computed.
    ///
    /// Typically this method should be called in [`Optimizer::update()`](super::optimizer::Optimizer::update()), after one or more backwards passes.
    pub fn take_grad(&mut self) -> Option<FloatTensor<D>> {
        if let Some(grad) = self.grad.as_mut().map(Arc::get_mut).flatten() {
            Some(
                FloatTensor::from(grad.get_mut().take()?)
                    .into_shape(self.value.dim())
                    .unwrap(),
            )
        } else {
            None
        }
    }
    fn lock(&self) -> VertexGuard<D> {
        VertexGuard {
            value: self.value.view(),
            grad: self.grad.as_deref().map(Mutex::lock),
        }
    }
    /// Sets the `name` of the vertex.
    ///
    /// This method is intended for use on vertex construction. The `name` does not have to be unique, and will be used in error messages as well as in rendering the graph.
    ///
    /// **Errors**
    ///
    /// Errors if the vertex is not exclusive.
    pub fn set_name(&mut self, name: impl Into<String>) -> Result<()> {
        self.node.set_name(name.into())
    }
    /// Adds a `name` to the vertex.
    ///
    /// See [``.set_name()`](Self::set_name()).
    pub fn with_name(mut self, name: impl Into<String>) -> Result<Self> {
        self.set_name(name)?;
        Ok(self)
    }
    /// Transfers the vertex into `device`.
    ///
    /// NOOP when the vertex is on `device`.
    ///
    /// # Autograd
    /// For a variable with a gradient, the output will have a backward op to compute the input gradient.
    ///
    /// **Panics**
    /// - Not yet implemented when the input has multiple dependent gradients.
    ///
    /// See [`FloatArcTensor::into_device_shared()`](FloatTensorBase::into_device_shared()).
    pub async fn into_device(self, device: Device) -> Result<Self> {
        if self.device() == device {
            Ok(self)
        } else {
            match self.try_into_variable() {
                Ok(variable) => {
                    let name = variable
                        .node
                        .name()
                        .map(|name| format!("{}_into_{:?}", name, &device));
                    let output_value = variable.value.clone().into_device_shared(device).await;
                    let mut output = variable
                        .into_builder()
                        .backward(|vertices, output_grad| {
                            let [mut x] = vertices.try_into_array().expect("Expected 1 vertex!");
                            if let Some(grad) = x.grad.as_deref_mut() {
                                let output_grad = smol::block_on(
                                    output_grad.to_slice()?.into_device(x.value.device()),
                                )?;
                                if let Some(_grad) = grad {
                                    todo!() // accumulate the gradient
                                } else {
                                    grad.replace(output_grad);
                                }
                            }
                            Ok(())
                        })
                        .build(|| output_value)?;
                    if let Some(name) = name {
                        output.set_name(name)?;
                    }
                    Ok(Self::try_from_variable(output).ok().unwrap())
                }
                Err(this) => {
                    let value = this.value.into_device_shared(device).await?;
                    let grad = this.grad.as_ref().map(|_| Arc::default());
                    Ok(Self {
                        value,
                        grad,
                        node: this.node,
                    })
                }
            }
        }
    }
    fn try_into_variable(self) -> Result<Variable<D>, Self> {
        match self.node.try_into_variable() {
            Ok(node) => Ok(Variable {
                value: self.value,
                grad: self.grad,
                node,
            }),
            Err(node) => Err(Self {
                value: self.value,
                grad: self.grad,
                node,
            }),
        }
    }
    fn try_from_variable(variable: Variable<D>) -> Result<Self, Variable<D>> {
        match N::try_from_variable(variable.node) {
            Ok(node) => Ok(Self {
                value: variable.value,
                grad: variable.grad,
                node,
            }),
            Err(node) => Err(Variable {
                value: variable.value,
                grad: variable.grad,
                node,
            }),
        }
    }
}

impl<D: Dimension> From<Variable<D>> for Vertex<D> {
    fn from(variable: Variable<D>) -> Self {
        Self {
            value: variable.value,
            grad: variable.grad,
            node: variable.node.into(),
        }
    }
}

impl<D: Dimension> From<Parameter<D>> for Vertex<D> {
    fn from(parameter: Parameter<D>) -> Self {
        Self {
            value: parameter.value,
            grad: parameter.grad,
            node: parameter.node.into(),
        }
    }
}

impl<D: Dimension, N: Node + Default> From<FloatArcTensor<D>> for VertexBase<D, N> {
    fn from(tensor: FloatArcTensor<D>) -> Self {
        Self {
            value: tensor,
            grad: None,
            node: N::default(),
        }
    }
}

impl<D: Dimension, N: Node + Default> From<FloatTensor<D>> for VertexBase<D, N> {
    fn from(tensor: FloatTensor<D>) -> Self {
        Self {
            value: tensor.into(),
            grad: None,
            node: N::default(),
        }
    }
}

impl<D: Dimension, N: Node> Debug for VertexBase<D, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ty = if self.node.clone().try_into_variable().is_ok() {
            "Variable"
        } else {
            "Parameter"
        };
        let mut builder = f.debug_struct(ty);
        builder
            .field("float_type", &self.float_type())
            .field("device", &self.device())
            .field("shape", &self.shape());
        let strides = self.strides();
        if strides != bytemuck::cast_slice(self.raw_dim().default_strides().slice()) {
            builder.field("strides", &strides);
        }
        builder.finish()
    }
}

impl<D: Dimension> Variable<D> {
    /// Whether to train the parameters.
    ///
    /// If true, autograd ops applied via [`VariableBuilder`] will have a gradient if they have at least one parameter, and those parameters will have a gradient provided to the backward op.
    ///
    /// This can be toggled on and off, for example to train a generator but not a discriminator (and vice versa) in a Generative Adversarial Network.
    pub fn with_training(mut self, training: bool) -> Self {
        self.node.training = training;
        self
    }
    /*
    /// Forces the graph to be built.
    ///
    /// By default, the internal graph is built only when required for the backward pass. Use this method when rendering the graph. See [`.graph()`]
    pub fn with_graph(mut self, graph: bool) -> Self {
        self.graph = graph;
        self
    }
    pub fn graph(&self) -> Graph {
        todo!()
    }*/
    /// Returns a [`VariableBuilder`] for autograd ops.
    pub fn builder(&self) -> VariableBuilder {
        self.clone().into_builder()
    }
    /// Converts the variable into a [`VariableBuilder`].
    pub fn into_builder(self) -> VariableBuilder {
        VariableBuilder::from_vertices(once(self.into_dyn().into()))
    }
    /// Runs the backward pass.
    ///
    /// Recursively calls all backward ops, which compute the gradients of variables and parameters.
    pub fn backward(&mut self) -> Result<()> {
        let grad = if let Some(mut grad) = self.grad.take() {
            if let Some(grad) = Arc::get_mut(&mut grad) {
                let grad = grad.get_mut();
                if grad.is_none() {
                    let float_type = self.value.float_type();
                    let device = self.value.device();
                    let len = self.value.len();
                    grad.replace(FloatBuffer::ones(float_type, device, len)?);
                }
            } else {
                bail!("Variable not exclusive!");
            }
            grad
        } else {
            bail!("Variable does not require grad!");
        };
        let var = Variable {
            value: self.value.clone().into_dyn(),
            grad: Some(grad),
            node: take(&mut self.node),
        };
        let mut stack = vec![var];
        while let Some(mut var) = stack.pop() {
            if let Some(output_grad) = var.take_grad() {
                let node = var.node;
                let name = &node.name;
                if let Some(Ok(base)) = node.base.map(Arc::try_unwrap) {
                    if let Some(backward) = base.backward {
                        let guards = base.vertices.iter().map(Vertex::lock).collect();
                        backward
                            .backward(VertexGuardDVec(guards), output_grad)
                            .with_context(|| {
                                format!(
                                    "Backward failed{}!",
                                    name.as_ref()
                                        .map_or(String::new(), |name| format!(" at {:?}", name)),
                                )
                            })?;
                        stack.extend(
                            base.vertices
                                .into_iter()
                                .filter_map(|v| v.try_into_variable().ok()),
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
/*
pub struct Graph {

}

impl Graph {
    fn render(&self) {
        todo!()
    }
}
*/

/// A guard for [`Vertex`], with a value and a locked gradient.
#[derive(Debug)]
pub struct VertexGuard<'a, D: Dimension> {
    value: FloatTensorView<'a, D>,
    grad: Option<MutexGuard<'a, Option<FloatBuffer>>>,
}

/// Dynamically dimensional [`VertexGuard`].
pub type VertexGuardD<'a> = VertexGuard<'a, IxDyn>;

impl<'a, D: Dimension> VertexGuard<'a, D> {
    /// The value of the vertex as a view.
    pub fn value(&self) -> FloatTensorView<'a, D> {
        self.value.clone()
    }
    /*
    /// Returns the gradient as a mutable view.
    ///
    /// If the gradient has not been computed, allocates the gradient.
    ///
    /// Returns None if the vertex does not require grad.
    ///
    /// Use this method in the closure provided to [`VariableBuilder::backward()`].
    ///
    /// # Safety
    /// The gradient is not initialized. See [`.grad_zeroed_mut()`](Self::grad_zeroed_mut()) for a safe alternative.
    ///
    /// **Errors**
    ///
    /// See [`FloatTensor::alloc()`](FloatTensorBase::alloc()).
    pub unsafe fn grad_alloc_mut(&mut self) -> Result<Option<FloatTensorViewMut<D>>> {
        if let Some(mut grad) = self.grad.as_mut() {
            if grad.is_none() {
                let float_type = self.value.float_type();
                let device = self.value.device();
                let len = self.value.len();
                grad.replace(FloatBuffer::alloc(float_type, device, len)?);
            }
            Ok(Some(FloatTensorViewMut::from(grad.as_mut().unwrap().as_slice_mut())
                .into_shape(self.value.dim())?))
        } else {
            Ok(None)
        }
    }*/
    /// Returns the gradient as a mutable view.
    ///
    /// If the gradient has not been computed, zeroes the gradient.
    ///
    /// Returns None if the vertex does not require grad.
    ///
    /// Use this method in the closure provided to [`VariableBuilder::backward()`].
    ///
    /// **Errors**
    ///
    /// See [`FloatTensor::zeros()`](FloatTensorBase::zeros()).
    pub fn grad_zeroed_mut(&mut self) -> Result<Option<FloatTensorViewMut<D>>> {
        if let Some(grad) = self.grad.as_mut() {
            if grad.is_none() {
                let float_type = self.value.float_type();
                let device = self.value.device();
                let len = self.value.len();
                grad.replace(FloatBuffer::zeros(float_type, device, len)?);
            }
            Ok(Some({
                let tensor = FloatTensorViewMut::from(grad.as_mut().unwrap().as_slice_mut())
                    .into_shape(self.value.dim())?;
                unsafe { tensor.with_raw_strides(self.value.raw_strides()) }
            }))
        } else {
            Ok(None)
        }
    }
    /// Converts the guard into dimension `D2`.
    ///
    /// See [`FloatTensor::into_dimensionality()`](FloatTensorBase::into_dimensionality()).
    pub fn into_dimensionality<D2>(self) -> Result<VertexGuard<'a, D2>>
    where
        D2: Dimension,
    {
        Ok(VertexGuard {
            value: self.value.into_dimensionality()?,
            grad: self.grad,
        })
    }
}

/// A vec of [`VertexGuard`]'s.
#[derive(Debug)]
pub struct VertexGuardDVec<'a>(Vec<VertexGuardD<'a>>);

impl<'a> VertexGuardDVec<'a> {
    /// Converts the vec into an array.
    ///
    /// **Errors**
    ///
    /// Returns self if `N` is not equal to the length of the vec.
    pub fn try_into_array<const N: usize>(self) -> Result<[VertexGuardD<'a>; N], Self> {
        self.try_into()
    }
}

impl<'a, const N: usize> TryFrom<VertexGuardDVec<'a>> for [VertexGuardD<'a>; N] {
    type Error = VertexGuardDVec<'a>;
    fn try_from(vec: VertexGuardDVec<'a>) -> Result<Self, Self::Error> {
        Self::try_from(vec.0).map_err(VertexGuardDVec)
    }
}

/// A builder for [`Variable`] autograd functions.
#[derive(Default)]
pub struct VariableBuilder {
    vertices: Vec<VertexD>,
    backward: Option<Box<dyn Backward>>,
    training: bool,
    graph: bool,
}

impl VariableBuilder {
    /// Creates a new builder from `vertices`.
    ///
    /// # Example
    /**
    ```no_run
    # use autograph::{result::Result, learn::neural_network::autograd::{VariableD, ParameterD, VariableBuilder}};
    fn my_func(x: VariableD, w: ParameterD) -> Result<VariableD> {
        VariableBuilder::from_vertices([x.clone().into(), w.clone().into()])
            .backward(|vertex_guards, output_grad| {
            let [mut x, mut w] = vertex_guards.try_into_array().expect("Expected 2 vertices!");
            # #[allow(unused_mut)]
            if let Some(mut x_grad) = x.grad_zeroed_mut()? {
                // accumulate the input grad
            }
            # #[allow(unused_mut)]
            if let Some(mut w_grad) = w.grad_zeroed_mut()? {
                // accumulate the weight grad
            }
            Ok(())
        }).build(|| todo!() /* compute output */)?
            .with_name("my_func")
    }
    ```
    */
    pub fn from_vertices<I>(vertices: I) -> Self
    where
        I: IntoIterator<Item = VertexD>,
    {
        Self::default().with_vertices(vertices)
    }
    /// Adds `vertices` to the builder.
    ///
    /// Vertices are inputs to the function (either parameters or variables).

    pub fn with_vertices<I>(mut self, vertices: I) -> Self
    where
        I: IntoIterator<Item = VertexD>,
    {
        self.vertices.extend(vertices);
        let nodes = self
            .vertices
            .iter()
            .filter_map(|v| v.node.as_variable_node());
        for node in nodes {
            self.training |= node.training;
            self.graph |= node.graph;
        }
        self
    }
    /// Adds a backward op.
    ///
    /// The closure `backward` takes the vertices as a [`VertexGuardDVec`] and the output gradient, and computes the gradient of each vertex (input / parameter).
    ///
    /// # Example
    /**
    ```no_run
    # use autograph::{result::Result, device::Device, float::FloatType, learn::neural_network::autograd::Variable};
    # fn main() -> Result<()> {
    # let device = Device::new()?;
    # let x = Variable::zeros(FloatType::F32, device, 1)?;
    # let my_function = |x| Result::<_, autograph::error::Error>::Ok(x);
    let y = x.builder().backward(|vertex_guards, output_grad| {
        # let my_function_grad = |x, dx, dy| Result::<_, autograph::error::Error>::Ok(());
        let [mut x] = vertex_guards.try_into_array().ok().expect("Expected 1 vertex!");
        let x_value = x.value();
        # #[allow(unused_mut)]
        if let Some(mut x_grad) = x.grad_zeroed_mut()? {
            my_function_grad(x_value, x_grad, output_grad)?;
        }
        Ok(())
    }).build(|| my_function(x.into_value()))?
        .with_name("my_function")?;
    # Ok(())
    # }
    ```
    */
    pub fn backward<F>(mut self, backward: F) -> Self
    where
        F: Fn(VertexGuardDVec, FloatTensorD) -> Result<()> + Send + Sync + 'static,
    {
        let variable_grad = self
            .vertices
            .iter()
            .filter(|v| v.node.as_variable_node().is_some())
            .any(|v| v.grad.is_some());
        let parameter_grad = self
            .vertices
            .iter()
            .filter(|v| v.node.as_parameter_node().is_some())
            .any(|v| v.grad.is_some());
        if (self.training && parameter_grad) || variable_grad {
            self.backward.replace(Box::new(backward));
        }
        self
    }
    /// Builds the variable.
    ///
    /// Returns a variable with value computed by `value_fn`. If not needed for building the graph,
    /// the vertices will be dropped prior to calling `value_fn`.
    ///
    /// **Errors**
    ///
    /// If `value_fn` fails, returns the error.
    pub fn build<D, F>(self, value_fn: F) -> Result<Variable<D>>
    where
        D: Dimension,
        F: FnOnce() -> Result<FloatArcTensor<D>>,
    {
        let grad = if self.backward.is_some() {
            Some(Arc::default())
        } else {
            None
        };
        let node_base = if self.graph || self.backward.is_some() {
            Some(Arc::new(VariableNodeBase {
                vertices: self.vertices,
                backward: self.backward,
            }))
        } else {
            std::mem::drop(self.vertices);
            None
        };
        let node = VariableNode {
            base: node_base,
            name: None,
            training: self.training,
            graph: self.graph,
        };
        Ok(Variable {
            value: value_fn()?,
            grad,
            node,
        })
    }
}

// linalg
impl<N1: Node, N2: Node> Dot<VertexBase<Ix2, N2>> for VertexBase<Ix2, N1> {
    type Output = Variable2;
    fn dot(&self, rhs: &VertexBase<Ix2, N2>) -> Result<Variable2> {
        let vertices = [
            self.clone().into_dyn().into_vertex(),
            rhs.clone().into_dyn().into_vertex(),
        ];
        VariableBuilder::from_vertices(vertices)
            .backward(|vertices, dy| {
                let dy = dy.into_dimensionality()?;
                let [x, w] = vertices.try_into_array().unwrap();
                let mut x = x.into_dimensionality()?;
                let mut w = w.into_dimensionality()?;
                if let Some(mut dx) = x.grad_zeroed_mut()? {
                    dy.dot_acc(&w.value(), &mut dx)?;
                }
                if let Some(mut dw) = w.grad_zeroed_mut()? {
                    dy.t().dot_acc(&x.value(), &mut dw)?;
                }
                Ok(())
            })
            .build(|| self.value().dot(rhs.value()).map(Into::into))?
            .with_name("dot")
    }
}

fn bias_backward(
    bias_grad: &mut FloatTensorViewMut1,
    output_grad: &FloatTensorView2,
) -> Result<()> {
    let (n, c) = output_grad.dim();
    if bias_grad.dim() != c {
        bail!(
            "bias_grad dim {:?} (expected {:?}) not valid for output_grad dim {:?}!",
            bias_grad.shape(),
            [c],
            output_grad.shape(),
        );
    }
    if bias_grad.float_type() != output_grad.float_type() {
        bail!(
            "bias_grad float_type {:?} != output_grad float_type {:?}!",
            bias_grad.float_type(),
            output_grad.float_type(),
        );
    }
    let output_grad_slice = output_grad.to_slice()?;
    let builder = glsl_shaders::module(&format!(
        "bias_backward_{}",
        bias_grad.float_type().as_str()
    ))?
    .compute_pass("main")?
    .float_slice_mut(bias_grad.as_raw_slice_mut())?
    .float_slice(output_grad_slice.as_slice())?
    .push([n as u32, c as u32])?;
    unsafe { builder.submit([c as u32, 1, 1]) }
}

impl<N1: Node, N2: Node, N3: Node> DotBias<VertexBase<Ix2, N2>, VertexBase<Ix1, N3>>
    for VertexBase<Ix2, N1>
{
    fn dot_bias(
        &self,
        rhs: &VertexBase<Ix2, N2>,
        bias: Option<&VertexBase<Ix1, N3>>,
    ) -> Result<Variable2> {
        if let Some(bias) = bias {
            let vertices = [
                self.clone().into_dyn().into_vertex(),
                rhs.clone().into_dyn().into_vertex(),
                bias.clone().into_dyn().into_vertex(),
            ];
            VariableBuilder::from_vertices(vertices)
                .backward(|vertices, dy| {
                    let dy = dy.into_dimensionality()?;
                    let [x, w, b] = vertices.try_into_array().unwrap();
                    let mut x = x.into_dimensionality()?;
                    let mut w = w.into_dimensionality()?;
                    let mut b = b.into_dimensionality()?;
                    if let Some(mut dx) = x.grad_zeroed_mut()? {
                        dy.dot_acc(&w.value(), &mut dx)?;
                    }
                    if let Some(mut dw) = w.grad_zeroed_mut()? {
                        x.value().t().dot_acc(&dy.view(), &mut dw)?;
                    }
                    if let Some(mut db) = b.grad_zeroed_mut()? {
                        bias_backward(&mut db.view_mut(), &dy.view())?;
                    }
                    Ok(())
                })
                .build(|| self.value().dot(rhs.value()).map(Into::into))?
                .with_name("dot_bias")
        } else {
            self.dot(rhs)
        }
    }
}
/*
impl Variable2 {
    pub(crate) fn dense(&self, rhs: &Parameter2, bias: Option<&Parameter1>) -> Result<Variable2> {
        if let Some(bias) = bias {
            let vertices = [
                self.clone().into_dyn().into_vertex(),
                rhs.clone().into_dyn().into_vertex(),
                bias.clone().into_dyn().into_vertex(),
            ];
            VariableBuilder::from_vertices(vertices)
                .backward(|vertices, dy| {
                    let dy = dy.into_dimensionality()?;
                    let [x, w, b] = vertices.try_into_array().unwrap();
                    let mut x = x.into_dimensionality()?;
                    let mut w = w.into_dimensionality()?;
                    let mut b = b.into_dimensionality()?;
                    if let Some(mut dx) = x.grad_zeroed_mut()? {
                        dy.dot_acc(&w.value(), &mut dx)?;
                    }
                    if let Some(mut dw) = w.grad_zeroed_mut()? {
                        dy.t().dot_acc(&x.value(), &mut dw)?;
                    }
                    if let Some(mut db) = b.grad_zeroed_mut()? {
                        bias_backward(&mut db.view_mut(), &dy.view())?;
                    }
                    Ok(())
                })
                .build(|| self.value().dot(&rhs.value().t()).map(Into::into))?
                .with_name("dot_bias")
        } else {
            todo!()
        }
    }
}
*/
#[allow(unused_variables)]
fn cross_entropy_loss(input: &FloatTensorView2, target: &FloatTensorView2) -> Result<FloatTensor2> {
    if input.shape() != target.shape() {
        bail!(
            "input shape {:?} != target shape {:?}!",
            input.shape(),
            target.shape(),
        );
    }
    if input.float_type() != target.float_type() {
        bail!(
            "input float_type {:?} != target float_type {:?}!",
            input.float_type(),
            target.float_type(),
        )
    }
    let device = input.device();
    let (n, nclasses) = input.dim();
    let float_type = input.float_type();
    let mut output = match float_type {
        FloatType::BF16 => FloatTensor::zeros(float_type, device, [n, nclasses])?,
        FloatType::F32 => unsafe { FloatTensor::alloc(float_type, device, [n, nclasses])? },
    };
    let nclasses_str = if nclasses <= 64 {
        "64"
    } else if nclasses <= 256 {
        "256"
    } else if nclasses <= 1024 {
        "1024"
    } else {
        bail!("nclasses > 1024 unimplemented!");
    };
    let input_slice = input.to_slice()?;
    let target_slice = target.to_slice()?;
    let builder = glsl_shaders::module(&format!(
        "cross_entropy_loss_{}_{}",
        float_type.as_str(),
        nclasses_str
    ))?
    .compute_pass("main")?
    .float_slice(input_slice.as_slice())?
    .float_slice(target_slice.as_slice())?
    .float_slice_mut(output.as_raw_slice_mut())?
    .push([n as u32, nclasses as u32])?;
    unsafe {
        builder.submit([n as u32, 1, 1])?;
    }
    Ok(output)
}

fn cross_entropy_loss_backward(
    input: &FloatTensorView2,
    input_grad: &mut FloatTensorViewMut2,
    target: &FloatTensorView2,
    output_grad: &FloatTensorView0,
) -> Result<()> {
    let n = input.dim().0 as u32;
    let float_type = input.float_type();
    let input_slice = input.to_slice()?;
    let target_slice = target.to_slice()?;
    let output_grad_slice = output_grad.to_slice()?;
    let builder = glsl_shaders::module(&format!(
        "cross_entropy_loss_backward_{}",
        float_type.as_str()
    ))?
    .compute_pass("main")?
    .float_slice(input_slice.as_slice())?
    .float_slice_mut(input_grad.as_raw_slice_mut())?
    .float_slice(target_slice.as_slice())?
    .float_slice(output_grad_slice.as_slice())?
    .push(n)?;
    unsafe { builder.submit([n, 1, 1]) }
}

impl<D: Dimension> Variable<D> {
    pub(super) fn cross_entropy_loss(&self, target: FloatArcTensor<D>) -> Result<Variable0> {
        let target = target.into_dimensionality()?;
        let output =
            cross_entropy_loss(&self.value().view().into_dimensionality()?, &target.view())?
                .sum()?
                .into_shared()?;
        let target = Variable::from(target).into_dyn().with_name("target")?;
        self.builder()
            .with_vertices(once(target.into_vertex()))
            .backward(|vertices, dy| {
                let dy = dy.into_dimensionality()?;
                let [x, t] = vertices.try_into_array().unwrap();
                let mut x = x.into_dimensionality()?;
                let t = t.into_dimensionality()?;
                let x_value = x.value();
                if let Some(mut dx) = x.grad_zeroed_mut()? {
                    cross_entropy_loss_backward(&x_value.view(), &mut dx, &t.value(), &dy.view())?;
                }
                Ok(())
            })
            .build(|| Ok(output))?
            .with_name("cross_entropy_loss")
    }
}

#[cfg(all(test, feature = "device_tests"))]
mod tests {
    use super::*;
    use crate::{scalar::Float, tensor::CowTensor, util::type_eq};
    use approx::assert_relative_eq;
    use half::bf16;
    use ndarray::{Array, Array2, ArrayView2, ArrayViewMut1, Axis};
    use num_traits::FromPrimitive;

    fn array_bias_backward<T: Copy + Into<f32> + FromPrimitive>(
        db: &mut ArrayViewMut1<T>,
        dy: &ArrayView2<T>,
    ) {
        for (db, dy) in db.iter_mut().zip(dy.axis_iter(Axis(1))) {
            let acc: f32 = dy.iter().copied().map(Into::into).sum();
            *db = T::from_f32((*db).into() + acc).unwrap();
        }
    }

    async fn test_bias_backward<T: Float + From<u8> + Into<f32> + FromPrimitive>() -> Result<()> {
        let batch_size = 3;
        let units = 7;
        let data = (0..batch_size * units)
            .into_iter()
            .map(|x| (x as u8).into())
            .collect();
        let dy_array = Array::from_shape_vec([batch_size, units], data)?;
        let mut db_true = Array::from_elem(units, T::one());
        array_bias_backward(&mut db_true.view_mut(), &dy_array.view());
        let device = Device::new()?;
        let _s = device.acquire().await;
        let dy = CowTensor::from(dy_array.view())
            .into_device(device.clone())
            .await?
            .into_float();
        let mut db = FloatTensor::ones(T::float_type(), device, units)?;
        bias_backward(&mut db.view_mut(), &dy.view())?;
        let db_array = db.cast_into::<T>()?.read().await?;
        assert_eq!(db_array.as_array(), db_true.view());
        Ok(())
    }

    #[tokio::test]
    async fn bias_backward_bf16() -> Result<()> {
        test_bias_backward::<bf16>().await
    }

    #[tokio::test]
    async fn bias_backward_f32() -> Result<()> {
        test_bias_backward::<f32>().await
    }

    fn array_cross_entropy_loss(x: &ArrayView2<f32>, t: &ArrayView2<f32>) -> Array2<f32> {
        let mut y = Array::zeros(x.raw_dim());
        for (mut y, (x, t)) in y.outer_iter_mut().zip(x.outer_iter().zip(t.outer_iter())) {
            let m = x
                .iter()
                .copied()
                .fold(x[0], |m, x| if x > m { x } else { m });
            let x = x.map(|x| x - m);
            let s: f32 = x.iter().map(|x| x.exp()).sum();
            for (y, (x, t)) in y.iter_mut().zip(x.iter().copied().zip(t.iter().copied())) {
                *y = (s.ln() - x) * t;
            }
        }
        y
    }

    async fn test_cross_entropy_loss<T: Float + From<u8> + FromPrimitive>() -> Result<()> {
        let n = 67;
        let c = 9;
        let x_data: Vec<T> = (0..n * c).into_iter().map(|x| (x as u8).into()).collect();
        let t_data: Vec<T> = x_data.iter().copied().rev().collect();
        let x_array = Array::from_shape_vec([n, c], x_data)?;
        let t_array = Array::from_shape_vec([n, c], t_data)?;
        let y_true = {
            let x_array = x_array.map(|x| x.to_f32().unwrap());
            let t_array = t_array.map(|t| t.to_f32().unwrap());
            array_cross_entropy_loss(&x_array.view(), &t_array.view())
        };
        let device = Device::new()?;
        let _s = device.acquire().await;
        let x = CowTensor::from(x_array.view())
            .into_device(device.clone())
            .await?
            .into_float();
        let t = CowTensor::from(t_array.view())
            .into_device(device.clone())
            .await?
            .into_float();
        let y = cross_entropy_loss(&x.view(), &t.view())?;
        let y_array = y.cast_into::<T>()?.read().await?;
        let y_array = y_array.as_array().map(|x| x.to_f32().unwrap());
        if type_eq::<T, bf16>() {
            assert_relative_eq!(y_array, y_true, epsilon = 0.01, max_relative = 0.01);
        } else {
            assert_relative_eq!(y_array, y_true, max_relative = 0.000_1);
        }
        Ok(())
    }

    #[tokio::test]
    async fn cross_entropy_loss_bf16() -> Result<()> {
        test_cross_entropy_loss::<bf16>().await
    }

    #[tokio::test]
    async fn cross_entropy_loss_f32() -> Result<()> {
        test_cross_entropy_loss::<f32>().await
    }
}
