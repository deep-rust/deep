use crate::{Backend, Graph, Input, Internal, Op};
use rand_core::RngCore;
use std::cell::RefCell;
use std::ops::{Add, Sub};
use std::rc::Rc;

pub struct Tensor {
    graph: Rc<RefCell<Graph>>,
    input: Input,
}

impl Tensor {
    /// Creates the state for the tensor.
    pub fn gen_state<B>(&self, backend: &B, rng: impl RngCore) -> Result<B::State, B::Error>
    where
        B: Backend,
    {
        backend.state(&self.graph.borrow(), rng)
    }

    /// Evaluates the tensor
    pub fn eval<B>(
        &self,
        backend: &B,
        state: &B::State,
        inputs: &B::Inputs,
    ) -> Result<B::Tensor, B::Error>
    where
        B: Backend,
    {
        backend
            .forward(&self.graph.borrow(), state, inputs, self.input.clone())
            .map(|(output, _)| output)
    }
}

impl From<&str> for Tensor {
    fn from(s: &str) -> Tensor {
        Tensor {
            graph: Default::default(),
            input: s.into(),
        }
    }
}

fn merge2_1(a: Tensor, b: Tensor, make_op: impl Fn(Input, Input) -> Op) -> Tensor {
    let graph = a.graph;
    let a = a.input;
    let b = graph
        .borrow_mut()
        .merge_input(b.graph.borrow().clone(), b.input);
    graph.borrow_mut().ops.push(make_op(a, b));
    let node = graph.borrow().ops.len() - 1;
    Tensor {
        graph,
        input: Input::Internal(Internal { node, output: 0 }),
    }
}

impl Add for Tensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        merge2_1(self, rhs, Op::Add)
    }
}

impl Sub for Tensor {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        merge2_1(self, rhs, Op::Sub)
    }
}
