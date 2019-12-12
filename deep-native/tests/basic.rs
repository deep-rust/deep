use deep::*;
use deep_backend_tools::*;
use deep_native::*;
use maplit::hashmap;
use ndarray::arr1;
use rand::{thread_rng, Rng, RngCore};

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

struct Sub;

impl Handler for Sub {
    fn op(&self) -> OpTy {
        OpTy::Sub
    }

    fn generate_state(&self, _op: &Op, _rng: &mut dyn RngCore) -> Vec<Tsor> {
        // There are no internal variables to a sub operation.
        vec![]
    }

    fn forward(&self, imop: ImOp<Native>, _state: &[Tsor]) -> Vec<Tsor> {
        if let ImOp::Sub(a, b) = imop {
            vec![a - b]
        } else {
            panic!("got {:?} when OpTy::Sub was expected", OpTy::from(&imop));
        }
    }

    fn backward(
        &self,
        imop: ImOp<Native>,
        _state: &[Tsor],
        (_, output_delta): (usize, Tsor),
    ) -> (ImOp<Native>, Vec<Tsor>) {
        let ty: OpTy = (&imop).into();
        assert_eq!(ty, OpTy::Sub);
        (ImOp::Sub(output_delta.clone(), -output_delta), vec![])
    }
}

struct Square;

impl Handler for Square {
    fn op(&self) -> OpTy {
        OpTy::Square
    }

    fn generate_state(&self, _op: &Op, _rng: &mut dyn RngCore) -> Vec<Tsor> {
        // There are no internal variables to a square operation.
        vec![]
    }

    fn forward(&self, imop: ImOp<Native>, _state: &[Tsor]) -> Vec<Tsor> {
        if let ImOp::Square(a) = imop {
            vec![a.mapv(|n| n.powi(2)).to_shared()]
        } else {
            panic!("got {:?} when OpTy::Square was expected", OpTy::from(&imop));
        }
    }

    fn backward(
        &self,
        imop: ImOp<Native>,
        _state: &[Tsor],
        (_, output_delta): (usize, Tsor),
    ) -> (ImOp<Native>, Vec<Tsor>) {
        let ty: OpTy = (&imop).into();
        assert_eq!(ty, OpTy::Square);
        if let ImOp::Square(a) = imop {
            (ImOp::Square(2.0 * a * output_delta), vec![])
        } else {
            panic!("got {:?} when OpTy::Square was expected", OpTy::from(&imop));
        }
    }
}

struct TrainConst;

impl Handler for TrainConst {
    fn op(&self) -> OpTy {
        OpTy::TrainConst
    }

    fn generate_state(&self, op: &Op, _rng: &mut dyn RngCore) -> Vec<Tsor> {
        if let Op::TrainConst(shape, value) = op {
            vec![Tsor::zeros(&shape[..]) + *value as f32]
        } else {
            panic!("got {:?} when Op::TrainConst was expected", OpTy::from(op));
        }
    }

    fn forward(&self, imop: ImOp<Native>, state: &[Tsor]) -> Vec<Tsor> {
        if let ImOp::TrainConst = imop {
            vec![state[0].clone()]
        } else {
            panic!(
                "got {:?} when OpTy::TrainConst was expected",
                OpTy::from(&imop)
            );
        }
    }

    fn backward(
        &self,
        imop: ImOp<Native>,
        _state: &[Tsor],
        (_, output_delta): (usize, Tsor),
    ) -> (ImOp<Native>, Vec<Tsor>) {
        let ty: OpTy = (&imop).into();
        assert_eq!(ty, OpTy::TrainConst);
        (ImOp::TrainConst, vec![output_delta])
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

#[test]
fn train_add() {
    let backend = Native::new()
        .handler(Add)
        .handler(Sub)
        .handler(Square)
        .handler(TrainConst);

    // Add two input tensors to make an output tensor.
    let y = Tensor::from("x") + Tensor::train_const(vec![], 0.0);

    // The loss function.
    let loss = (y - Tensor::from("y")).squared();

    // Generate the state for training the graph for the tensor.
    let mut state = loss
        .gen_state(&backend, thread_rng())
        .expect("unable to generate state");

    // The actual difference.
    let m = 5.0;

    // The learning rate.
    let learning_rate = 0.01;

    let mut loss_value = std::f32::NAN;

    for _ in 0..1000 {
        // Random x value
        let x = thread_rng().gen();
        // Compute y
        let y = x + m;
        // Inputs
        let feed = hashmap! {
            "x".to_owned() => tsor0(x),
            "y".to_owned() => tsor0(y)
        };

        // Train the network and get back the loss value.
        loss_value = loss
            .gradient_descent(
                &backend,
                &mut state,
                &feed,
                learning_rate,
                |t| *t.iter().next().unwrap(),
                tsor0,
            )
            .expect("unable to train");
        eprintln!("loss at {}", loss_value);
    }

    // Loss starts around 25.
    assert!(loss_value < 0.1);
}
