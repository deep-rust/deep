enum Tensor {
    Input(String),
    Internal {
        /// The node to pull the input tensor from.
        node: usize,
        /// The specific output to pull from.
        output: usize,
    },
}

struct Node {
    /// The name of the operation.
    op: String,
    /// The inputs to the operation.
    inputs: Vec<Tensor>,
}

struct Graph {
    /// A series of nodes refering to each other's outputs for their input.
    nodes: Vec<Node>,
}

trait Backend {
    type Inputs;
    type Internal;
    type Output;
    type Delta;

    /// Gets all the outputs of solving the requested tensors.
    fn forward(
        &self,
        graph: &Graph,
        inputs: Self::Inputs,
        tensor: Tensor,
    ) -> (Self::Output, Self::Internal);

    /// Propogates a delta from the output back to the input via chain rule
    /// and produces a `Delta` that can be used to update the graph
    /// with an optimizer. The `Delta` contains all the dE/dx of all internal
    /// variables.
    fn backward(
        &self,
        graph: &Graph,
        internal: &Self::Internal,
        inputs: Self::Inputs,
        tensor: Tensor,
        output_delta: &Self::Output,
    ) -> Self::Delta;

    /// Applies a delta to the graph.
    fn train(&self, graph: &mut Graph, delta: &Self::Delta);
}
