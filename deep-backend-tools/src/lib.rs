mod accumulate_tensors;

pub use accumulate_tensors::AccumulateTensors;

use deep::*;
use failure::Fail;
use std::collections::{hash_map::Entry, HashMap};

#[derive(Debug, Fail)]
pub enum Error {
    #[fail(display = "input not provided for \"{}\"", name)]
    InputNotProvided { name: String },
    #[fail(
        display = "internal node \"{}\" (\"{:?}\") was not found in the feed dict (not computed)",
        node, ty
    )]
    InternalNotComputed { node: usize, ty: Option<OpTy> },
    #[fail(display = "no handler for \"{:?}\"", ty)]
    OpHasNoHandler { ty: OpTy },
}

pub type Result<T> = std::result::Result<T, Error>;
type SResult<T, E> = std::result::Result<T, E>;

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
    /// This is given an operation with inputs specified in `imop` (as per the original `solve`),
    /// internal state (trainable or otherwise) specified in `state`, and one delta in the output
    /// that must be propogated specified by `output_delta`. It must only perform backprop for the
    /// output specified. If the output specified doesn't exist (addition gets `1`, when there is only `0`),
    /// then it should panic at runtime as that is a developer error. It should not return `None`.
    /// Returning `None` should only be done if the operation was not registered with the backend.
    /// All other issues are programmatic issues and should panic.
    fn propogate(
        &self,
        imop: ImOp<Self>,
        state: &[Self::Tensor],
        output_delta: (usize, Self::Tensor),
    ) -> Option<(ImOp<Self>, Vec<Self::Tensor>)>;
}

pub struct Tape<B: Backend> {
    solved: HashMap<Internal, Vec<B::Tensor>>,
}

impl<B, T> Default for Tape<B>
where
    B: Backend<Tensor = T>,
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
    B: Backend<Tensor = T>,
    T: Clone,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn input(
        &self,
        backend: &B,
        inputs: &B::Inputs,
        graph: &Graph,
        input: Input,
    ) -> Result<B::Tensor>
    where
        B: Feed,
    {
        match input {
            Input::Feed(name) => backend
                .feed(inputs, &name)
                .ok_or_else(|| Error::InputNotProvided { name }),
            Input::Internal(internal) => self
                .solved
                .get(&internal)
                .and_then(|v| v.get(internal.output))
                .cloned()
                .ok_or_else(|| Error::InternalNotComputed {
                    node: internal.node,
                    ty: graph.ops.get(internal.node).map(|op| op.into()),
                }),
        }
    }

    pub fn solve(
        &mut self,
        backend: &B,
        graph: &Graph,
        state: &[Vec<B::Tensor>],
        inputs: &B::Inputs,
        input: Input,
    ) -> Result<B::Tensor>
    where
        B: Immediate + Feed,
    {
        match input {
            Input::Feed(name) => backend
                .feed(inputs, &name)
                .ok_or_else(|| Error::InputNotProvided { name }),
            Input::Internal(internal) => {
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
    ///
    /// This delta is accumulated in the `deltas` parameter utilising its `Extend` impl.
    pub fn backprop<E>(
        &self,
        backend: &B,
        graph: &Graph,
        state: &[Vec<B::Tensor>],
        inputs: &B::Inputs,
        input: Input,
        output_delta: B::Tensor,
        deltas: E,
    ) -> Result<E>
    where
        B: Propogate + Feed,
        E: Extend<(usize, Vec<B::Tensor>)>,
    {
        match input {
            Input::Feed(_) => Ok(deltas),
            Input::Internal(internal) => {
                let op = graph
                    .ops
                    .get(internal.node)
                    .expect("node requested in backprop but does not exist");
                ImOp::backprop(
                    op.clone(),
                    internal,
                    self,
                    backend,
                    graph,
                    state,
                    inputs,
                    output_delta,
                    deltas,
                )
            }
        }
    }
}

#[derive(Clone)]
pub enum ImOp<B: Backend + ?Sized> {
    Add(B::Tensor, B::Tensor),
    Sub(B::Tensor, B::Tensor),
    Square(B::Tensor),
    TrainConst,
}

impl<B> ImOp<B>
where
    B: Backend,
{
    pub fn add(self) -> SResult<(B::Tensor, B::Tensor), Self> {
        if let ImOp::Add(a, b) = self {
            Ok((a, b))
        } else {
            Err(self)
        }
    }

    pub fn sub(self) -> SResult<(B::Tensor, B::Tensor), Self> {
        if let ImOp::Sub(a, b) = self {
            Ok((a, b))
        } else {
            Err(self)
        }
    }

    pub fn square(self) -> SResult<B::Tensor, Self> {
        if let ImOp::Square(a) = self {
            Ok(a)
        } else {
            Err(self)
        }
    }
}

impl<B, T> ImOp<B>
where
    B: Backend<Tensor = T>,
    T: Clone,
{
    fn solve<'a>(
        op: Op,
        tape: &'a mut Tape<B>,
        backend: &B,
        graph: &Graph,
        state: &[Vec<B::Tensor>],
        inputs: &B::Inputs,
    ) -> Result<Self>
    where
        B: Feed + Immediate,
    {
        let mut tensor = |input| tape.solve(backend, graph, state, inputs, input);
        let mut double = |a, b, f: fn(B::Tensor, B::Tensor) -> Self| {
            tensor(a).and_then(|a| tensor(b).map(|b| f(a, b)))
        };
        match op {
            Op::Add(a, b) => double(a, b, ImOp::Add),
            Op::Sub(a, b) => double(a, b, ImOp::Sub),
            Op::Square(a) => tensor(a).map(ImOp::Square),
            Op::TrainConst(..) => Ok(ImOp::TrainConst),
        }
    }

    /// This takes the output delta of a particular output from the op and propogates it backwards to the inputs.
    fn backprop<'a, E>(
        op: Op,
        internal: Internal,
        tape: &Tape<B>,
        backend: &B,
        graph: &Graph,
        state: &[Vec<B::Tensor>],
        inputs: &B::Inputs,
        output_delta: B::Tensor,
        deltas: E,
    ) -> Result<E>
    where
        B: Propogate + Feed,
        E: Extend<(usize, Vec<B::Tensor>)>,
    {
        // Get the op type.
        let ty = (&op).into();

        // Get one tensor that is either an input or has been precomputed.
        // Anything else is an error.
        let tensor = |input, tape: &Tape<B>| tape.input(backend, inputs, graph, input);

        // This calls backend.propogate to invoke the actual implementation of the backprop for this op.
        let gradients = |imop| {
            backend
                .propogate(
                    imop,
                    state
                        .get(internal.node)
                        .expect("operation doesn't have any state")
                        .as_slice(),
                    (internal.output, output_delta),
                )
                .ok_or_else(|| Error::OpHasNoHandler { ty })
        };

        // This is to appease the borrow checker because I was getting moved closure errors.
        let gradients1 = gradients.clone();
        let gradients2 = gradients.clone();

        // This recursively backprops to send the gradient to a new graph node.
        let backprop = |input, output_delta, tape: &Tape<B>, deltas| {
            tape.backprop(backend, graph, state, inputs, input, output_delta, deltas)
        };

        // This performs the backprop for an op with two parameters.
        // It will update the delta for this op and recursively backprop to its inputs.
        // This requires the two inputs, a function to turn the inputs into an ImOp, and a function to decompose the
        // ImOp into a tuple tensors to pass the gradient backwards.
        let binary = |ia: Input,
                      ib: Input,
                      fimop: fn(B::Tensor, B::Tensor) -> Self,
                      fundo: fn(ImOp<B>) -> SResult<(B::Tensor, B::Tensor), Self>,
                      mut deltas: E| {
            tensor(ia.clone(), tape)
                .and_then(|a| tensor(ib.clone(), tape).map(|b| fimop(a, b)))
                .and_then(gradients2)
                .map(|(input_gradients, train_gradients)| {
                    deltas.extend(std::iter::once((internal.node, train_gradients)));
                    input_gradients
                })
                .map(|imop| {
                    fundo(imop).unwrap_or_else(|imop| {
                        let imop_ty: OpTy = (&imop).into();
                        panic!("op \"{:?}\" gave back ImOp type \"{:?}\"", ty, imop_ty);
                    })
                })
                .and_then(|(ta, tb)| {
                    let deltas = backprop(ia, ta, tape, deltas)?;
                    backprop(ib, tb, tape, deltas)
                })
        };

        // This performs the backprop for an op with one parameter.
        // It will update the delta for this op and recursively backprop to its inputs.
        // This requires the input, a function to turn the input into an ImOp, and a function to decompose the
        // ImOp into its tensor to pass the gradient backwards.
        let unary = |ia: Input,
                     fimop: fn(B::Tensor) -> Self,
                     fundo: fn(ImOp<B>) -> SResult<B::Tensor, Self>,
                     mut deltas: E| {
            tensor(ia.clone(), tape)
                .map(fimop)
                .and_then(gradients1)
                .map(|(input_gradients, train_gradients)| {
                    deltas.extend(std::iter::once((internal.node, train_gradients)));
                    input_gradients
                })
                .map(|imop| {
                    fundo(imop).unwrap_or_else(|imop| {
                        let imop_ty: OpTy = (&imop).into();
                        panic!("op \"{:?}\" gave back ImOp type \"{:?}\"", ty, imop_ty);
                    })
                })
                .and_then(|ta| backprop(ia, ta, tape, deltas))
        };

        // This updates the delta for this op only. It has no runtime inputs, so it does not recurse.
        let nullary = |imop: Self, mut deltas: E| {
            gradients(imop).map(|(_, train_gradients)| {
                deltas.extend(std::iter::once((internal.node, train_gradients)));
                deltas
            })
        };

        match op {
            Op::Add(a, b) => binary(a, b, ImOp::Add, ImOp::add, deltas),
            Op::Sub(a, b) => binary(a, b, ImOp::Sub, ImOp::sub, deltas),
            Op::Square(a) => unary(a, ImOp::Square, ImOp::square, deltas),
            Op::TrainConst(..) => nullary(ImOp::TrainConst, deltas),
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
            ImOp::TrainConst => OpTy::TrainConst,
        }
    }
}
