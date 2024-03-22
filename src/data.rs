use std::fs;
use image;
use burn::{
    data::{self, dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
};
use nannou::draw::{background::new, primitive::path};
#[derive(Debug,Clone)]
pub struct NumbersItem {
    pub number: [[f32; 28*28]; 3],
    pub label: i32,
}
#[derive(Debug,Clone)]
pub struct NumbersBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> NumbersBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}
#[derive(Debug,Clone)]
pub struct NumbersBatch<B: Backend> {
    pub numbers: Tensor<B, 3>,
    pub labels: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<NumbersItem,NumbersBatch<B>> for NumbersBatcher<B> {
    fn batch(&self, items: Vec<NumbersItem>) -> NumbersBatch<B> {
        let numbers = items.iter()
        .map(|item| Data::<f32,2>::from(item.number).convert())
        .map(|data| {
            Tensor::<B,2>::from_data(data, &self.device)
        })
        .map(|tensor| tensor.reshape([1, 28, 28]))
        .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
        .collect();
        let labels: Vec<Tensor<B,1,Int>> = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data([(item.label as i64).elem()], &self.device ))
            .collect();
        let numbers = Tensor::cat(numbers, 0);
        let labels = Tensor::cat(labels, 0);

        NumbersBatch {
            numbers, labels
        }
    }
}

pub struct NumbersDataset {
    dataset: Vec<NumbersItem>,
}

impl Dataset<NumbersItem> for NumbersDataset {
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Option<NumbersItem> {
        self.dataset.get(index).cloned()
    }
}

impl NumbersDataset {
    pub fn new(path: &str) -> Self {
        let dataset = NumbersDataset::parse(path);
        Self {
            dataset
        }
    }
    pub fn test() -> Self {
        NumbersDataset::new("test")
    }
    pub fn train() -> Self {
        NumbersDataset::new("train")
    }
    fn parse(path: &str) -> Vec<NumbersItem>{
        let mut numbers:Vec<NumbersItem> = Vec::new();
        let dir_path = format!("E:/coding/rust/project/burn_nannou_MNIST/MNIST/{}", path);
        let dir = fs::read_dir(dir_path).unwrap();
        for sub_dir in dir {
            let sub_dir = sub_dir.unwrap();
            let label = sub_dir.file_name().into_string().unwrap().parse::<i32>().unwrap();
            for image in fs::read_dir(sub_dir.path()).unwrap() {
                let mut item = NumbersItem {
                    number: [[0.0; 28*28]; 3],
                    label,
                };
                let image = image.unwrap();
                let image = image::open(image.path()).unwrap().to_rgb8();
                for (n,pix) in image.pixels().enumerate() {
                    if pix.0[0] > 0 {
                        item.number[0][n] = 1.0;
                    } else if pix.0[1] > 0 {
                        item.number[1][n] = 1.0;
                    } else if pix.0[2] > 0 {
                        item.number[2][n] = 1.0;
                    }
    
                }
                numbers.push(item);
            }
        }
        numbers
    }
}