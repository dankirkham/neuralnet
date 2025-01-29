use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Standard;
use rayon::prelude::*;

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

    pub fn update(&mut self, batch_length: usize, nabla_b: Vec<Array2<f32>>, nabla_w: Vec<Array2<f32>>, eta: f32) {
        for (b, nb) in self.biases.iter_mut().zip(nabla_b.into_iter()) {
            *b = &*b - (eta / (batch_length as f32)) * nb;
        }

        for (w, nw) in self.weights.iter_mut().zip(nabla_w.into_iter()) {
            *w = &*w - (eta / (batch_length as f32)) * nw;
        }
    }

    pub fn process_mini_batch(&self, batch: &[Example]) -> (usize, Vec<Array2<f32>>, Vec<Array2<f32>>) {
        let mut nabla_b: Vec<Array2<f32>> = self.biases
            .iter()
            .map(|b| Array::zeros((b.shape()[0], b.shape()[1])))
            .collect();

        let mut nabla_w: Vec<Array2<f32>> = self.weights
            .iter()
            .map(|w| Array::zeros((w.shape()[0], w.shape()[1])))
            .collect();

        let deltas: Vec<_> = batch.par_iter().map(|example| self.backprop(&example.x, &example.y)).collect();

        for delta in deltas.into_iter() {
            let (delta_b, delta_w) = delta;
            nabla_b.iter_mut().zip(delta_b.into_iter()).for_each(|(nb, dnb)| *nb = &*nb + &dnb);
            nabla_w.iter_mut().zip(delta_w.into_iter()).for_each(|(nw, dnw)| *nw = &*nw + &dnw);
        }

        (batch.len(), nabla_b, nabla_w)
    }

    fn backprop(&self, x: &Array2<f32>, y: &Array2<f32>) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {
        let mut nabla_b: Vec<Array2<f32>> = self.biases
            .iter()
            .map(|b| Array::zeros((b.shape()[0], b.shape()[1])))
            .collect();

        let mut nabla_w: Vec<Array2<f32>> = self.weights
            .iter()
            .map(|w| Array::zeros((w.shape()[0], w.shape()[1])))
            .collect();

        let mut activations = Vec::with_capacity(self.num_layers + 1);
        let mut zs = Vec::with_capacity(self.num_layers);

        // Feed forward
        let mut activation = x.clone();
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let z = w.dot(&activation) + b;
            zs.push(z.clone());
            activations.push(activation);
            activation = sigmoid(&z);
        }
        activations.push(activation);

        // Output Error
        let mut delta: Array2<f32> = Network::cost_derivative(&activations.last().unwrap(), &y) *
            sigmoid_prime(zs.last().unwrap());
        *nabla_w.last_mut().unwrap() = delta.dot(&activations[activations.len() - 2].t());
        *nabla_b.last_mut().unwrap() = delta.clone();

        // Backpropagate
        for i in 2..self.num_layers {
            let z = &zs[zs.len() - i];
            let sp = sigmoid_prime(z);
            delta = self.weights[self.weights.len() - i + 1].t().dot(&delta) * sp;
            nabla_w[self.weights.len() - i] = delta.dot(&activations[activations.len() - i - 1].t());
            nabla_b[self.biases.len() - i] = delta.clone();
        }

        (nabla_b, nabla_w)
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

