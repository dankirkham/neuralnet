mod mnist;
mod network;

use std::{fs::File, io::BufWriter};

use kdam::tqdm;
use ndarray::{Array2, Axis};
use ndarray_stats::QuantileExt;
use rand::seq::SliceRandom;
use rand::thread_rng;
use tracing_flame::FlameLayer;
use tracing_subscriber::{registry::Registry, prelude::*, fmt};

use mnist::load_data;
use network::Network;

fn setup_global_subscriber() -> impl Drop {
    let fmt_layer = fmt::Layer::default();

    let (flame_layer, _guard) = FlameLayer::with_file("./tracing.folded").unwrap();

    let subscriber = Registry::default()
        .with(fmt_layer)
        .with(flame_layer);

    tracing::subscriber::set_global_default(subscriber).expect("Could not set global default");
    _guard
}

fn main() {
    let _guard = setup_global_subscriber();

    let mut rng = thread_rng();

    let train_data = load_data("./data/train").unwrap();
    let mut train_data: Vec<Example> = train_data.into_iter().map(|v| v.into()).collect();
    dbg!(train_data.len());

    let test_data = load_data("./data/t10k").unwrap();
    let test_data: Vec<Example> = test_data.into_iter().map(|v| v.into()).collect();
    dbg!(test_data.len());

    let mut network = Network::new(vec![784, 30, 10]);
    let batch_size = 16;
    let epochs = 10;
    let eta = 0.15;

    let mut work = network.alloc_batch_work(batch_size);

    for _ in tqdm!(0..epochs) {
        train_data.shuffle(&mut rng);
        for batch in train_data.chunks(batch_size) {
            network.process_mini_batch(&mut work, batch);
            network.update(&work, eta);
        }
    }

    // let mut pass = 0;
    // let mut misses: Vec<&Example> = Vec::new();
    // for case in test_data.iter() {
    //     let result = network.forward(&case.x);
    //     let (class, _) = result.argmax().unwrap();
    //     if class == case.class {
    //         pass += 1;
    //     } else {
    //         misses.push(case);
    //     }
    // }
    // dbg!(pass);
    // let accuracy = (pass as f64) / (test_data.len() as f64);
    // dbg!(accuracy);

    // for case in misses.iter().take(5) {
    //     for row in case.x.axis_chunks_iter(Axis(0), 28) {
    //         for col in row.iter() {
    //             if *col > 0.8 {
    //                 print!("█");
    //             } else if *col > 0.6 {
    //                 print!("▓");
    //             } else if *col > 0.4 {
    //                 print!("▒");
    //             } else if *col > 0.2 {
    //                 print!("░");
    //             } else {
    //                 print!(" ");
    //             }
    //         }
    //         println!("");
    //     }

    //     let result = network.forward(&case.x);
    //     let (class, _) = result.argmax().unwrap();
    //     println!("Guess: {}, Actual: {}", class, case.class);
    // }
}

pub struct Example {
    x: Array2<f32>,
    y: Array2<f32>,
    class: usize,
}
