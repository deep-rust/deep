use deep::*;
use deep_backend_tools::*;
use ndarray::{ArcArray, IxDyn};
use rand_core::RngCore;
use std::collections::HashMap;
use std::iter::{Extend, FromIterator};

pub type Tsor = ArcArray<f32, IxDyn>;

pub fn tsor0(n: f32) -> Tsor {
    ndarray::arr0(n).into_shared().into_dyn()
}

pub fn tsor1(n: &[f32]) -> Tsor {
    ndarray::arr1(n).into_shared().into_dyn()
}

pub fn tsor2<V>(n: &[V]) -> Tsor
where
    V: ndarray::FixedInitializer<Elem = f32> + Clone,
{
    ndarray::arr2(n).into_shared().into_dyn()
}

pub fn tsor3<V, U>(n: &[V]) -> Tsor
where
    V: ndarray::FixedInitializer<Elem = U> + Clone,
    U: ndarray::FixedInitializer<Elem = f32> + Clone,
{
    ndarray::arr3(n).into_shared().into_dyn()
}

pub trait Handler {
    /// This returns the op ty that this handler can execute.
    fn op(&self) -> OpTy;

    /// This generates the trainable state for this graph node.
    fn generate_state(&self, op: &Op, rng: &mut dyn RngCore) -> Vec<Tsor>;

    /// This performs forward propogation for the op.
    fn forward(&self, imop: ImOp<Native>, state: &[Tsor]) -> Vec<Tsor>;

    /// This performs backward propogation for the op.
    ///
    /// Returns an `ImOp` of the input gradients along with the trainable state deltas (if any).
    fn backward(
        &self,
        imop: ImOp<Native>,
        state: &[Tsor],
        output_delta: (usize, Tsor),
    ) -> (ImOp<Native>, Vec<Tsor>);
}

#[derive(Default)]
pub struct Native {
    handlers: HashMap<OpTy, Box<dyn Handler>>,
}

impl Native {
    pub fn new() -> Self {
        Self::default()
    }

    /// Use this to add one handler.
    pub fn handler<H>(mut self, h: H) -> Self
    where
        H: Handler + 'static,
    {
        self.handlers
            .insert(h.op(), Box::new(h) as Box<dyn Handler>);
        self
    }

    /// Use this to add several handlers at once, such as from a library.
    pub fn handlers<I>(mut self, iter: I) -> Self
    where
        I: IntoIterator<Item = Box<dyn Handler>>,
    {
        self.extend(iter);
        self
    }
}

impl Extend<Box<dyn Handler>> for Native {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = Box<dyn Handler>>,
    {
        self.handlers
            .extend(iter.into_iter().map(|handler| (handler.op(), handler)))
    }
}

impl FromIterator<Box<dyn Handler>> for Native {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Box<dyn Handler>>,
    {
        let mut backend = Self::new();
        backend.extend(iter);
        backend
    }
}

impl Backend for Native {
    /// The input is a feed dict from strings to tensors.
    type Inputs = HashMap<String, Tsor>;
    /// The internal type stores all the intermediary computations of the whole graph.
    type Internal = Tape<Self>;
    /// Tensor type is `ndarray`'s `ArcArray` over dynamic dimension.
    type Tensor = Tsor;
    /// The delta stores a map from nodes in the graph to their recieved gradient.
    type Delta = AccumulateTensors<Tsor>;
    /// State contains all state data (internal tensors that are being trained or static).
    type State = Vec<Vec<Tsor>>;
    /// Error is the error type for the native backend.
    type Error = Error;

    /// Generates the initial state for a graph.
    fn state<R>(&self, graph: &Graph, mut rng: R) -> Result<Self::State>
    where
        R: RngCore,
    {
        graph
            .ops
            .iter()
            .map(|op| {
                let ty = op.into();
                self.handlers
                    .get(&ty)
                    .ok_or_else(|| Error::OpHasNoHandler { ty })
                    .map(|handler| handler.generate_state(op, &mut rng))
            })
            .collect()
    }

    /// Gets the output of solving the requested tensor.
    fn forward(
        &self,
        graph: &Graph,
        state: &Self::State,
        inputs: &Self::Inputs,
        tensor: Input,
    ) -> Result<(Self::Tensor, Self::Internal)> {
        let mut tape = Tape::new();
        tape.solve(self, graph, &state[..], inputs, tensor)
            .map(|tensor| (tensor, tape))
    }

    /// Propogates a delta from the output back to the input via chain rule
    /// and produces a `Delta` that can be used to update the graph
    /// with an optimizer. The `Delta` contains all the dE/dx of all internal
    /// variables.
    fn backward(
        &self,
        graph: &Graph,
        state: &Self::State,
        internal: &Self::Internal,
        inputs: &Self::Inputs,
        tensor: Input,
        output_delta: Self::Tensor,
    ) -> Result<Self::Delta> {
        internal.backprop(
            self,
            graph,
            &state[..],
            inputs,
            tensor,
            output_delta,
            AccumulateTensors::new(),
        )
    }

    /// Applies a delta to the graph.
    fn train(&self, state: &mut Self::State, delta: &Self::Delta) -> Result<()> {
        unimplemented!()
    }
}

impl Immediate for Native {
    fn solve(&self, imop: ImOp<Self>, state: &[Tsor]) -> Option<Vec<Tsor>> {
        let ty = (&imop).into();
        self.handlers
            .get(&ty)
            .map(|handler| handler.forward(imop, state))
    }
}

impl Propogate for Native {
    fn propogate(
        &self,
        imop: ImOp<Self>,
        state: &[Tsor],
        output_delta: (usize, Tsor),
    ) -> Option<(ImOp<Self>, Vec<Tsor>)> {
        let ty = (&imop).into();
        self.handlers
            .get(&ty)
            .map(|handler| handler.backward(imop, state, output_delta))
    }
}
