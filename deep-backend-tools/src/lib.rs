use deep::*;
use failure::Fail;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Fail)]
pub enum Error {
    #[fail(display = "input not provided: {}", name)]
    InputNotProvided { name: String },
}

type Result<T> = std::result::Result<T, Error>;

pub trait Feed: Backend {
    fn feed(&self, inputs: &Self::Inputs, name: &str) -> Option<Rc<Self::Tensor>>;
}

impl<B, T> Feed for B
where
    B: Backend<Inputs = HashMap<String, Rc<T>>, Tensor = T>,
{
    fn feed(&self, inputs: &HashMap<String, Rc<T>>, name: &str) -> Option<Rc<T>> {
        inputs.get(name).cloned()
    }
}

pub trait Immediate: Backend {
    fn solve(&self, imop: ImOp<Self>) -> Vec<Rc<Self::Tensor>>;
}

pub struct Tape<B: Backend> {
    solved: HashMap<Internal, Vec<Rc<B::Tensor>>>,
}

impl<B> Tape<B>
where
    B: Feed + Immediate,
{
    pub fn solve(
        &mut self,
        backend: &B,
        graph: &Graph,
        inputs: &B::Inputs,
        input: Input,
    ) -> Result<Rc<B::Tensor>> {
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
                ImOp::solve(op, self, backend, graph, inputs).map(|imop| {
                    let solutions = backend.solve(imop);
                    let output = solutions[internal.output].clone();
                    self.solved.insert(internal, solutions);
                    output
                })
            }
        }
    }
}

pub enum ImOp<B: Backend + ?Sized> {
    Add(Rc<B::Tensor>, Rc<B::Tensor>),
    Sub(Rc<B::Tensor>, Rc<B::Tensor>),
    Square(Rc<B::Tensor>),
}

impl<B> ImOp<B>
where
    B: Feed + Immediate,
{
    fn solve<'a>(
        op: Op,
        tape: &'a mut Tape<B>,
        backend: &B,
        graph: &Graph,
        inputs: &B::Inputs,
    ) -> Result<Self> {
        let mut tensor = |input| tape.solve(backend, graph, inputs, input);
        let mut double = |a, b, f: fn(Rc<B::Tensor>, Rc<B::Tensor>) -> Self| {
            tensor(a).and_then(|a| tensor(b).map(|b| f(a, b)))
        };
        match op {
            Op::Add(a, b) => double(a, b, ImOp::Add),
            Op::Sub(a, b) => double(a, b, ImOp::Sub),
            Op::Square(a) => tensor(a).map(ImOp::Square),
        }
    }
}
