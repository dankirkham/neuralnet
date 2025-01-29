use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Standard;
use rayon::prelude::*;
use tracing::{info_span, instrument};

use crate::Example;

#[derive(Debug)]
pub struct Network {
    /// Number of layers in the network.
    num_layers: usize,
    /// Number of neurons in each respective layer.
    layer_sizes: Vec<usize>,
    /// Biases for each neuron.
    biases: Vec<Array2<f32>>,
    /// Weights for each connection to each neuron.
    weights: Vec<Array2<f32>>,
}

impl Default for Network {
    fn default() -> Self {
        let layer_sizes = vec![5, 7, 3];
        Network::new(layer_sizes)
    }
}

struct Nabla {
    b: Array2<f32>,
    w: Array2<f32>,
}

pub struct BatchWork {
    batch_length: usize,
    nablas: Vec<Nabla>,
}

impl Network {
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        let mut biases = Vec::with_capacity(layer_sizes.len());
        let mut weights = Vec::with_capacity(layer_sizes.len());

        for i in 1..layer_sizes.len() {
            biases.push(Array::random((layer_sizes[i], 1), Standard) * 2. - 1.);
            weights.push(Array::random((layer_sizes[i], layer_sizes[i - 1]), Standard) * 2. - 1.);
        }

        Self {
            num_layers: layer_sizes.len(),
            layer_sizes,
            biases,
            weights,
        }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        assert_eq!(input.len(), self.layer_sizes[0]);

        let mut input: Array2<f32> = input.clone();
        for i in 1..self.num_layers {
            let output = &self.weights[i - 1].dot(&input);
            input = sigmoid(&(output + &self.biases[i - 1]));
        }

        input
    }

    #[instrument(skip_all)]
    pub fn alloc_batch_work(&self, batch_length: usize) -> BatchWork {
        let nablas: Vec<_> = self.biases.iter().zip(self.weights.iter()).map(|(b, w)| {
            let b = Array::zeros((b.shape()[0], b.shape()[1]));
            let w = Array::zeros((w.shape()[0], w.shape()[1]));
            Nabla { b, w }
        }).collect();

        BatchWork {
            batch_length,
            nablas,
        }
    }

    #[instrument(skip_all)]
    pub fn update(&mut self, work: &BatchWork, eta: f32) {
        for (b, nb) in self.biases.iter_mut().zip(work.nablas.iter()) {
            *b = &*b - (eta / (work.batch_length as f32)) * &nb.b;
        }

        for (w, nb) in self.weights.iter_mut().zip(work.nablas.iter()) {
            *w = &*w - (eta / (work.batch_length as f32)) * &nb.w;
        }
    }

    #[instrument(skip_all)]
    pub fn process_mini_batch(&self, work: &mut BatchWork, batch: &[Example]) {
        assert_eq!(work.batch_length, batch.len());
        let alloc_span = info_span!("alloc").entered();
        work.nablas.iter_mut().for_each(|nabla| {
            nabla.b.iter_mut().for_each(|v| *v = 0.);
            nabla.w.iter_mut().for_each(|v| *v = 0.);
        });
        drop(alloc_span);

        let layer_nablas: Vec<_> = batch.iter().map(|example| self.backprop(&example.x, &example.y)).collect();

        let sum_span = info_span!("sum").entered();
        for nabla in layer_nablas.into_iter() {
            work.nablas.iter_mut().zip(nabla.into_iter()).for_each(|(nabla, delta_nabla)| {
                nabla.b = &nabla.b + &delta_nabla.b;
                nabla.w = &nabla.w + &delta_nabla.w;
            });
        }
        drop(sum_span);
    }

    #[instrument(skip_all)]
    fn backprop(&self, x: &Array2<f32>, y: &Array2<f32>) -> Vec<Nabla> {
        // let mut nabla: Vec<_> = self.biases.iter().zip(self.weights.iter()).map(|(b, w)| {
        //     let b = Array::zeros((b.shape()[0], b.shape()[1]));
        //     let w = Array::zeros((w.shape()[0], w.shape()[1]));
        //     Nabla { b, w }
        // }).collect();

        let alloc_span = info_span!("alloc").entered();
        let mut nablas = Vec::with_capacity(self.num_layers);
        let mut activations = Vec::with_capacity(self.num_layers + 1);
        let mut zs = Vec::with_capacity(self.num_layers);
        drop(alloc_span);

        // Feed forward
        let ff_span = info_span!("feed_forward").entered();
        let mut activation = x.clone();
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let mat_op = info_span!("mat_op").entered();
            let z = w.dot(&activation) + b;
            drop(mat_op);
            activations.push(activation);
            activation = sigmoid(&z);
            zs.push(z);
        }
        activations.push(activation);
        drop(ff_span);

        // Output Error
        let oe_span = info_span!("output_error").entered();
        let mut delta: Array2<f32> = Network::cost_derivative(&activations.last().unwrap(), &y) *
            sigmoid_prime(zs.last().unwrap());
        // let Nabla { b, w } = nabla.last_mut().unwrap();
        let mat_op = info_span!("mat_op").entered();
        let w = delta.dot(&activations[activations.len() - 2].t());
        drop(mat_op);
        let b = delta.clone();
        nablas.push(Nabla { b, w });
        drop(oe_span);

        // Backpropagate
        for i in 2..self.num_layers {
            let _bp_span = info_span!("backprop_layer", layer = i).entered();
            let z = &zs[zs.len() - i];
            let sp = sigmoid_prime(z);
            let mat_op = info_span!("mat_op").entered();
            delta = self.weights[self.weights.len() - i + 1].t().dot(&delta) * sp;
            drop(mat_op);
            // let idx = nabla.len() - i;
            // let Nabla { b, w } = nabla.get_mut(idx).unwrap();
            let mat_op = info_span!("mat_op").entered();
            let w = delta.dot(&activations[activations.len() - i - 1].t());
            drop(mat_op);
            let b = delta.clone();
            nablas.push(Nabla { b, w });
        }

        let rev_span = info_span!("reverse").entered();
        nablas.reverse();
        drop(rev_span);
        nablas
    }

    fn cost_derivative(activation: &Array2<f32>, y: &Array2<f32>) -> Array2<f32> {
        activation - y
    }
}

fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
    1. / (1. + (-x).exp())
}

fn sigmoid_prime(x: &Array2<f32>) -> Array2<f32> {
    sigmoid(x) * (1.0 - sigmoid(x))
}

