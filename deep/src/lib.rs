#[macro_use]
extern crate strum_macros;

mod tensor;

pub use tensor::Tensor;

use rand_core::RngCore;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Internal {
    /// The node to pull the input tensor from.
    pub node: usize,
    /// The specific output to pull from.
    pub output: usize,
}

impl Internal {
    fn shift_inputs(&mut self, shift: usize) {
        self.node += shift;
    }
}

#[derive(Clone, Debug, EnumDiscriminants)]
#[strum_discriminants(name(OpTy), derive(Hash))]
pub enum Op {
    Add(Input, Input),
    Sub(Input, Input),
    Square(Input),
}

impl Op {
    fn shift_inputs(&mut self, shift: usize) {
        match self {
            Self::Add(a, b) => {
                a.shift_inputs(shift);
                b.shift_inputs(shift);
            }
            Self::Sub(a, b) => {
                a.shift_inputs(shift);
                b.shift_inputs(shift);
            }
            Self::Square(a) => {
                a.shift_inputs(shift);
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum Input {
    // An input from the feed dict.
    Feed(String),
    // An input from another node in the graph.
    Internal(Internal),
}

impl Input {
    fn shift_inputs(&mut self, shift: usize) {
        if let Self::Internal(n) = self {
            n.shift_inputs(shift);
        }
    }
}

impl From<&str> for Input {
    fn from(s: &str) -> Input {
        Input::Feed(s.to_owned())
    }
}

#[derive(Clone, Default, Debug)]
pub struct Graph {
    /// A series of ops refering to each other's outputs for their input.
    pub ops: Vec<Op>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn merge(&mut self, other: Graph) {
        let current = self.ops.len();
        self.ops.extend(other.ops);
        for op in &mut self.ops[current..] {
            op.shift_inputs(current);
        }
    }

    pub fn merge_input(&mut self, other: Graph, mut input: Input) -> Input {
        let current = self.ops.len();
        self.merge(other);
        input.shift_inputs(current);
        input
    }
}

pub trait Backend {
    type Inputs;
    type Internal;
    type Tensor;
    type Delta;
    type State;
    type Error;

    /// Generates the initial state for a graph.
    fn state<R>(&self, graph: &Graph, rng: R) -> Result<Self::State, Self::Error>
    where
        R: RngCore;

    /// Gets the output of solving the requested tensor.
    fn forward(
        &self,
        graph: &Graph,
        state: &Self::State,
        inputs: &Self::Inputs,
        tensor: Input,
    ) -> Result<(Self::Tensor, Self::Internal), Self::Error>;

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
        output_delta: &Self::Tensor,
    ) -> Result<Self::Delta, Self::Error>;

    /// Applies a delta to the graph's state.
    fn train(&self, state: &mut Self::State, delta: &Self::Delta) -> Result<(), Self::Error>;
}
