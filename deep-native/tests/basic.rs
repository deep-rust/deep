use deep::*;
use deep_backend_tools::*;
use deep_native::*;
use ndarray::arr1;
use rand::{thread_rng, RngCore};
use std::collections::HashMap;

struct Add;

impl Handler for Add {
    fn op(&self) -> OpTy {
        OpTy::Add
    }

    fn generate_state(&self, op: &Op, rng: &mut dyn RngCore) -> Vec<Tsor> {
        // There are no internal variables to an add operation.
        vec![]
    }

    fn forward(&self, imop: ImOp<Native>, state: &[Tsor]) -> Vec<Tsor> {
        if let ImOp::Add(a, b) = imop {
            vec![a + b]
        } else {
            panic!("got {:?} when OpTy::Add was expected", OpTy::from(&imop));
        }
    }

    fn backward(
        &self,
        imop: ImOp<Native>,
        state: &[Tsor],
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

    let mut graph = Graph::new();
    graph.ops.push(Op::Add(
        Input::Feed("a".to_owned()),
        Input::Feed("b".to_owned()),
    ));
    let state = backend
        .state(&graph, &mut thread_rng())
        .expect("unable to generate state");
    let feed = vec![
        ("a".to_owned(), arr1(&[2.0]).into_shared().into_dyn()),
        ("b".to_owned(), arr1(&[3.0]).into_shared().into_dyn()),
    ]
    .into_iter()
    .collect();
    let target = Input::Internal(Internal { node: 0, output: 0 });
    let (output, _) = backend
        .forward(&graph, &state, feed, target)
        .expect("unable to do forward prop");
    let expected = arr1(&[5.0]).into_shared().into_dyn();
    assert_eq!(output, expected);
}
