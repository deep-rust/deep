use deep::*;
use deep_backend_tools::*;
use ndarray::ArrayD;
use rand::RngCore;
use std::collections::HashMap;
use std::rc::Rc;

type Tensor = Rc<ArrayD<f32>>;

pub trait Handler {
    /// This returns the op ty that this handler can execute.
    fn op(&self) -> OpTy;

    /// This generates the trainable state for this graph node.
    fn generate_state(&self, op: &Op, rng: &mut dyn RngCore) -> Vec<Tensor>;

    /// This performs forward propogation for the op.
    fn forward(&self, imop: ImOp<Native>, state: &[Tensor]) -> Vec<Tensor>;

    /// This performs backward propogation for the op.
    ///
    /// Returns an `ImOp` of the input gradients along with the trainable state deltas (if any).
    fn backward(
        &self,
        imop: ImOp<Native>,
        state: &[Tensor],
        output_deltas: &[Tensor],
    ) -> (ImOp<Native>, Vec<Tensor>);
}

pub struct Native {
    handlers: HashMap<OpTy, Box<dyn Handler>>,
}

impl Backend for Native {
    /// The input is a feed dict from strings to tensors.
    type Inputs = HashMap<String, Tensor>;
    /// The internal type stores all the intermediary computations of the whole graph.
    type Internal = HashMap<Internal, Tensor>;
    /// Tensor type is `ndarray`'s `ArrayD`.
    type Tensor = Tensor;
    /// The delta stores a map from nodes in the graph to their recieved gradient.
    type Delta = HashMap<usize, Vec<Tensor>>;
    /// State contains all state data (internal tensors that are being trained or static).
    type State = Vec<Vec<Tensor>>;
    /// Error is the error type for the native backend.
    type Error = Error;

    /// Generates the initial state for a graph.
    fn state<R>(&self, graph: &Graph, rng: &mut R) -> Result<Self::State>
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
                    .map(|handler| handler.generate_state(op, rng))
            })
            .collect()
    }

    /// Gets the output of solving the requested tensor.
    fn forward(
        &self,
        graph: &Graph,
        state: &Self::State,
        inputs: Self::Inputs,
        tensor: Input,
    ) -> Result<(Self::Tensor, Self::Internal)> {
        unimplemented!()
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
        inputs: Self::Inputs,
        tensor: Input,
        output_delta: &Self::Tensor,
    ) -> Result<Self::Delta> {
        unimplemented!()
    }

    /// Applies a delta to the graph.
    fn train(&self, state: &mut Self::State, delta: &Self::Delta) -> Result<()> {
        unimplemented!()
    }
}

impl Immediate for Native {
    fn solve(&self, imop: ImOp<Self>, state: &[Tensor]) -> Option<Vec<Tensor>> {
        let ty = (&imop).into();
        self.handlers
            .get(&ty)
            .map(|handler| handler.forward(imop, state))
    }
}
