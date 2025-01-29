use std::fs::File;
use std::io::Read;

use byteorder::{BigEndian, ReadBytesExt};
use ndarray::Array2;

use crate::Example;

#[derive(Debug)]
struct MnistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl MnistData {
    fn new(f: &File) -> Result<MnistData, std::io::Error> {
        // let mut gz = flate2::read::GzDecoder::new(f);
        // let mut contents: Vec<u8> = Vec::new();
        // gz.read_to_end(&mut contents)?;
        // let mut r = Cursor::new(&contents);
        let mut r = f;

        let magic_number = r.read_i32::<BigEndian>()?;

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }

        r.read_to_end(&mut data)?;

        Ok(MnistData { sizes, data })
    }
}

#[derive(Debug)]
pub struct MnistImage {
    pub image: Array2<f64>,
    pub classification: u8,
}

impl From<MnistImage> for Example {
    fn from(value: MnistImage) -> Self {
        let x: Vec<_> = value.image.into_iter().map(|v| v as f32).collect();
        let x = Array2::from_shape_vec((784, 1), x).unwrap();
        let mut y = Array2::zeros((10, 1));
        y[[value.classification as usize, 0]] = 1.;

        Self {
            x,
            y,
            class: value.classification as usize,
        }
    }
}

pub fn load_data(dataset_name: &str) -> Result<Vec<MnistImage>, std::io::Error> {
    let filename = format!("{}-labels.idx1-ubyte", dataset_name);
    let label_data = &MnistData::new(&(File::open(filename))?)?;
    let filename = format!("{}-images.idx3-ubyte", dataset_name);
    let images_data = &MnistData::new(&(File::open(filename))?)?;
    let mut images: Vec<Array2<f64>> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        let image_data: Vec<f64> = image_data.into_iter().map(|x| x as f64 / 255.).collect();
        images.push(Array2::from_shape_vec((image_shape, 1), image_data).unwrap());
    }

    let classifications: Vec<u8> = label_data.data.clone();

    let mut ret: Vec<MnistImage> = Vec::new();

    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        ret.push(MnistImage {
            image,
            classification,
        })
    }

    Ok(ret)
}
