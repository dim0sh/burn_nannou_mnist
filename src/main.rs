use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use burn::optim::AdamConfig;
use burn_nannou_mnist::data::NumbersDataset;
use burn_nannou_mnist::model::ModelConfig;

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    burn_nannou_mnist::training::train::<MyAutodiffBackend>(
        "/tmp",
        burn_nannou_mnist::training::TrainingConfig::new(
            ModelConfig::new(10, 512),
            AdamConfig::new(),
        ),
        device.clone(),
    );
    let mut positive = [0; 10];
    let mut all = [0; 10];
    NumbersDataset::new("test").dataset.iter().for_each(|item| {
        let (predicted, expected) =
            burn_nannou_mnist::inference::infer::<MyBackend>("/tmp", device.clone(), item.clone());
            all[expected as usize] += 1;
            if predicted == expected {
                positive[expected as usize] += 1;
            }
    });
    let accuracy = positive.iter().zip(all.iter()).map(|(p, a)| *p as f32 / *a as f32).collect::<Vec<f32>>();
    let total_accuracy = positive.iter().sum::<i32>() as f32 / all.iter().sum::<i32>() as f32;
    println!("Accuracy per number: {:?}, Total Accuracy: {:?}", accuracy, total_accuracy);
}
