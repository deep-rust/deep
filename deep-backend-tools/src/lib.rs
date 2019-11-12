use deep::*;
use failure::Fail;
use std::collections::HashMap;

#[derive(Debug, Fail)]
pub enum Error {
    #[fail(display = "input not provided for \"{}\"", name)]
    InputNotProvided { name: String },
    #[fail(display = "no handler for \"{:?}\"", ty)]
    OpHasNoHandler { ty: OpTy },
}

pub type Result<T> = std::result::Result<T, Error>;

pub trait Feed: Backend {
    fn feed(&self, inputs: &Self::Inputs, name: &str) -> Option<Self::Tensor>;
}

impl<B, T> Feed for B
where
    B: Backend<Inputs = HashMap<String, T>, Tensor = T>,
    T: Clone,
{
    fn feed(&self, inputs: &HashMap<String, T>, name: &str) -> Option<T> {
        inputs.get(name).cloned()
    }
}

pub trait Immediate: Backend {
    fn solve(&self, imop: ImOp<Self>, state: &[Self::Tensor]) -> Option<Vec<Self::Tensor>>;
}

pub trait Propogate: Backend {
    fn propogate(
        &self,
        imop: ImOp<Self>,
        state: &[Self::Tensor],
        output_deltas: &[Self::Tensor],
    ) -> Option<(ImOp<Self>, Vec<Self::Tensor>)>;
}

pub struct Tape<B: Backend> {
    solved: HashMap<Internal, Vec<B::Tensor>>,
}

impl<B, T> Default for Tape<B>
where
    B: Feed + Immediate + Backend<Tensor = T>,
    T: Clone,
{
    fn default() -> Self {
        Self {
            solved: Default::default(),
        }
    }
}

impl<B, T> Tape<B>
where
    B: Feed + Immediate + Backend<Tensor = T>,
    T: Clone,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn solve(
        &mut self,
        backend: &B,
        graph: &Graph,
        state: &[Vec<B::Tensor>],
        inputs: &B::Inputs,
        input: Input,
    ) -> Result<B::Tensor> {
        match input {
            Input::Feed(name) => backend
                .feed(inputs, &name)
                .ok_or_else(|| Error::InputNotProvided { name }),
            Input::Internal(internal) => {
                use std::collections::hash_map::Entry;
                let op = match self.solved.entry(internal) {
                    Entry::Occupied(o) => return Ok(o.get()[internal.output].clone()),
                    Entry::Vacant(_) => graph.ops[internal.node].clone(),
                };
                let ty = (&op).into();
                ImOp::solve(op, self, backend, graph, state, inputs).and_then(|imop| {
                    backend
                        .solve(imop, &state[internal.node][..])
                        .map(|solutions| {
                            let output = solutions[internal.output].clone();
                            self.solved.insert(internal, solutions);
                            output
                        })
                        .ok_or_else(|| Error::OpHasNoHandler { ty })
                })
            }
        }
    }

    /// Propogates the output from `output_delta` to all of the pieces that contributed to
    /// the output specified by `input`.
    ///
    /// This process will produce the `Backend::Delta` that can be used to train the state.
    pub fn backprop(
        &mut self,
        backend: &B,
        graph: &Graph,
        state: &[Vec<B::Tensor>],
        inputs: &B::Inputs,
        input: Input,
        output_delta: B::Tensor,
    ) -> Result<B::Delta> {
        unimplemented!()
    }
}

pub enum ImOp<B: Backend + ?Sized> {
    Add(B::Tensor, B::Tensor),
    Sub(B::Tensor, B::Tensor),
    Square(B::Tensor),
}

impl<B, T> ImOp<B>
where
    B: Feed + Immediate + Backend<Tensor = T>,
    T: Clone,
{
    fn solve<'a>(
        op: Op,
        tape: &'a mut Tape<B>,
        backend: &B,
        graph: &Graph,
        state: &[Vec<B::Tensor>],
        inputs: &B::Inputs,
    ) -> Result<Self> {
        let mut tensor = |input| tape.solve(backend, graph, state, inputs, input);
        let mut double = |a, b, f: fn(B::Tensor, B::Tensor) -> Self| {
            tensor(a).and_then(|a| tensor(b).map(|b| f(a, b)))
        };
        match op {
            Op::Add(a, b) => double(a, b, ImOp::Add),
            Op::Sub(a, b) => double(a, b, ImOp::Sub),
            Op::Square(a) => tensor(a).map(ImOp::Square),
        }
    }
}

impl<B> From<&ImOp<B>> for OpTy
where
    B: Backend,
{
    fn from(imop: &ImOp<B>) -> Self {
        match imop {
            ImOp::Add(..) => OpTy::Add,
            ImOp::Sub(..) => OpTy::Sub,
            ImOp::Square(..) => OpTy::Square,
        }
    }
}
