use deep::*;
use deep_backend_tools::*;
use ndarray::ArrayD;
use std::collections::HashMap;
use std::rc::Rc;

type Tensor = Rc<ArrayD<f32>>;

struct Native;

impl Backend for Native {
    /// The input is a feed dict from strings to tensors.
    type Inputs = HashMap<String, Tensor>;
    /// The internal type stores all the intermediary computations of the whole graph.
    type Internal = HashMap<Internal, Tensor>;
    /// Tensor type is `ndarray`'s `ArrayD`.
    type Tensor = Tensor;
    /// The delta stores a map from nodes in the graph to their recieved gradient.
    type Delta = HashMap<usize, Tensor>;
    type Error = Error;

    /// Gets the output of solving the requested tensor.
    fn forward(
        &self,
        graph: &Graph,
        inputs: Self::Inputs,
        tensor: Input,
    ) -> Result<(Self::Tensor, Self::Internal), Error> {
        unimplemented!()
    }

    /// Propogates a delta from the output back to the input via chain rule
    /// and produces a `Delta` that can be used to update the graph
    /// with an optimizer. The `Delta` contains all the dE/dx of all internal
    /// variables.
    fn backward(
        &self,
        graph: &Graph,
        internal: &Self::Internal,
        inputs: Self::Inputs,
        tensor: Input,
        output_delta: &Self::Tensor,
    ) -> Result<Self::Delta, Error> {
        unimplemented!()
    }

    /// Applies a delta to the graph.
    fn train(&self, graph: &mut Graph, delta: &Self::Delta) -> Result<(), Error> {
        unimplemented!()
    }
}
