use deep::*;
use deep_backend_tools::*;
use deep_native::*;
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
        _imop: ImOp<Native>,
        _state: &[Tsor],
        output_deltas: &[Tsor],
    ) -> (ImOp<Native>, Vec<Tsor>) {
        (
            // The gradient goes to both inputs the same.
            ImOp::Add(output_deltas[0].clone(), output_deltas[0].clone()),
            // There are no internal variables to provide gradient.
            vec![],
        )
    }
}

#[test]
fn forward_add() {
    let backend = Native::new().handler(Add);
    // Inputs
    let feed = vec![
        ("a".to_owned(), arr1(&[2.0]).into_shared().into_dyn()),
        ("b".to_owned(), arr1(&[3.0]).into_shared().into_dyn()),
    ]
    .into_iter()
    .collect();

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
