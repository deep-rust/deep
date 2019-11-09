use deep::*;
use ndarray::ArrayD;
use std::collections::HashMap;

struct Native;

impl Backend for Native {
    /// The input is a feed dict from strings to tensors.
    type Inputs = HashMap<String, ArrayD<f32>>;
    /// The internal type stores all the intermediary computations of the whole graph.
    type Internal = HashMap<Internal, ArrayD<f32>>;
    /// The output is a tensor.
    type Output = ArrayD<f32>;
    /// The delta stores a map from nodes in the graph to their recieved gradient.
    type Delta = HashMap<usize, ArrayD<f32>>;

    /// Gets the output of solving the requested tensor.
    fn forward(
        &self,
        graph: &Graph,
        inputs: Self::Inputs,
        tensor: Input,
    ) -> (Self::Output, Self::Internal) {
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
        output_delta: &Self::Output,
    ) -> Self::Delta {
        unimplemented!()
    }

    /// Applies a delta to the graph.
    fn train(&self, graph: &mut Graph, delta: &Self::Delta) {
        unimplemented!()
    }
}
