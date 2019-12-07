use deep::*;
use deep_backend_tools::*;
use deep_native::*;
use maplit::hashmap;
use ndarray::arr1;
use rand::{thread_rng, RngCore};

struct Add;

impl Handler for Add {
    fn op(&self) -> OpTy {
        OpTy::Add
    }

    fn generate_state(&self, _op: &Op, _rng: &mut dyn RngCore) -> Vec<Tsor> {
        // There are no internal variables to an add operation.
        vec![]
    }

    fn forward(&self, imop: ImOp<Native>, _state: &[Tsor]) -> Vec<Tsor> {
        if let ImOp::Add(a, b) = imop {
            vec![a + b]
        } else {
            panic!("got {:?} when OpTy::Add was expected", OpTy::from(&imop));
        }
    }

    fn backward(
        &self,
        imop: ImOp<Native>,
        _state: &[Tsor],
        (_, output_delta): (usize, Tsor),
    ) -> (ImOp<Native>, Vec<Tsor>) {
        let ty: OpTy = (&imop).into();
        assert_eq!(ty, OpTy::Add);
        (
            ImOp::Add(output_delta.clone(), output_delta.clone()),
            vec![],
        )
    }
}

struct Zeros;

impl Handler for Zeros {
    fn op(&self) -> OpTy {
        OpTy::Zeros
    }

    fn generate_state(&self, op: &Op, _rng: &mut dyn RngCore) -> Vec<Tsor> {
        if let Op::Zeros(shape) = op {
            vec![Tsor::zeros(&shape[..])]
        } else {
            panic!("got {:?} when Op::Zeros was expected", OpTy::from(op));
        }
    }

    fn forward(&self, imop: ImOp<Native>, state: &[Tsor]) -> Vec<Tsor> {
        if let ImOp::Zeros = imop {
            vec![state[0].clone()]
        } else {
            panic!("got {:?} when OpTy::Zeros was expected", OpTy::from(&imop));
        }
    }

    fn backward(
        &self,
        imop: ImOp<Native>,
        _state: &[Tsor],
        (_, output_delta): (usize, Tsor),
    ) -> (ImOp<Native>, Vec<Tsor>) {
        let ty: OpTy = (&imop).into();
        assert_eq!(ty, OpTy::Zeros);
        (ImOp::Zeros, vec![output_delta])
    }
}

#[test]
fn forward_add() {
    let backend = Native::new().handler(Add);
    // Inputs
    let feed = hashmap! {
        "a".to_owned() => tsor1(&[2.0]),
        "b".to_owned() => tsor1(&[3.0]),
    };

    // Add two input tensors to make an output tensor.
    let c = Tensor::from("a") + Tensor::from("b");

    // Generate the state for training the graph for the tensor.
    let state = c
        .gen_state(&backend, thread_rng())
        .expect("unable to generate state");

    // Evaluate the tensor given the inputs.
    let output = c.eval(&backend, &state, &feed).expect("unable to eval");

    // Validate the output.
    let expected = arr1(&[5.0]).into_shared().into_dyn();
    assert_eq!(output, expected);
}
