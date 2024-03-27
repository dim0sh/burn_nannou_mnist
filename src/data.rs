use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset}, tensor::{backend::Backend, Data, ElementConversion, Int, Tensor}
};

use image;
use nannou::rand::random_range;
use rayon::prelude::*;
use std::fs::{self, DirEntry};

#[derive(Debug, Clone)]
pub struct NumbersItem {
    pub number: [[f32; 28 * 28]; 3],
    pub label: i32,
}
#[derive(Debug, Clone)]
pub struct NumbersBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> NumbersBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}
#[derive(Debug, Clone)]
pub struct NumbersBatch<B: Backend> {
    pub numbers: Tensor<B, 3>,
    pub labels: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<NumbersItem, NumbersBatch<B>> for NumbersBatcher<B> {
    fn batch(&self, items: Vec<NumbersItem>) -> NumbersBatch<B> {
        let numbers = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.number).convert())
            .map(|data| Tensor::<B, 2>::from_data(data))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();
        let labels: Vec<Tensor<B, 1, Int>> = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data([(item.label as i64).elem()]))
            .collect();
        let numbers = Tensor::cat(numbers, 0);
        let labels = Tensor::cat(labels, 0);

        NumbersBatch { numbers, labels }
    }
}
#[derive(Debug, Clone)]
pub struct NumbersDataset {
    pub dataset: Vec<NumbersItem>,
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
        Self { dataset }
    }
    pub fn test() -> Self {
        NumbersDataset::new("val")
    }
    pub fn train() -> Self {
        NumbersDataset::new("train")
    }
    fn parse(path: &str) -> Vec<NumbersItem> {
        let mut numbers: Vec<NumbersItem> = Vec::new();
        let dir_path = format!("./MNIST/{}", path);
        let dir = fs::read_dir(dir_path).unwrap();
        for sub_dir in dir {
            let sub_dir = sub_dir.unwrap();
            let label = sub_dir
                .file_name()
                .into_string()
                .unwrap()
                .parse::<i32>()
                .unwrap();
            let sub_numbers = fs::read_dir(sub_dir.path())
                .unwrap()
                .par_bridge()
                .map(|image| NumbersDataset::parse_image(image.unwrap(), label))
                .collect::<Vec<NumbersItem>>();
            numbers.extend(sub_numbers);
        }
        numbers
    }
    fn parse_image(image: DirEntry, label: i32) -> NumbersItem {
        let mut item = NumbersItem {
            number: [[0.0; 28 * 28]; 3],
            label,
        };
        let image = image::open(image.path()).unwrap().to_rgb8();
        for (n, pix) in image.pixels().enumerate() {
            item.number[0][n] = pix.0[0] as f32;
            item.number[1][n] = pix.0[1] as f32;
            item.number[2][n] = pix.0[2] as f32;
        }
        item
    }
    pub fn blur_dataset(&mut self) -> Self{
        let new_data = self.dataset.clone();
            
        new_data
            .iter()
            .take(10000)
            .par_bridge()
            .map(|item| NumbersDataset::blur_filter(item.clone()))
            .collect::<Vec<NumbersItem>>();

        self.dataset.extend(new_data);
        self.clone()
    }

    pub fn noise_dataset(&mut self) -> Self{
        let new_data = self.dataset.clone();
        new_data
            .iter()
            .take(10000)
            .par_bridge()
            .map(|item| NumbersDataset::noise_filter(item.clone()))
            .collect::<Vec<NumbersItem>>();

        self.dataset.extend(new_data);
        self.clone()
    }

    pub fn blur_filter(image: NumbersItem) -> NumbersItem {
        let mut item = NumbersItem {
            number: [[0.0; 28 * 28]; 3],
            label: image.label,
        };
        
        for i in 0..3 {
            for j in 0..28 {
                for k in 0..28 {
                    let mut sum = 0.0;
                    for x in -1..2 {
                        for y in -1..2 {
                            let x = j as i32 + x;
                            let y = k as i32 + y;
                            if x >= 0 && x < 28 && y >= 0 && y < 28 {
                                sum += image.number[i][y as usize * 28 + x as usize]
                            }
                        }
                    }
                    item.number[i][j * 28 + k] = sum / 9.0;
                }
            }
        }
        item
    }

    fn noise_filter(image: NumbersItem) -> NumbersItem {
        let mut item = NumbersItem {
            number: [[0.0; 28 * 28]; 3],
            label: image.label,
        };
        for i in 0..3 {
            for j in 0..28 {
                for k in 0..28 {
                    item.number[i][j * 28 + k] = (image.number[i][j * 28 + k] + random_range(0.0, 255.0) * 0.1)%255.0;
                }
            }
        }
        item
    }
}
