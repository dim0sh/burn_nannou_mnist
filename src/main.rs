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
    let mut positive = 0;
    let mut all = 0;
    NumbersDataset::new("test").dataset.iter().for_each(|item| {
        let (predicted, expected) =
            burn_nannou_mnist::inference::infer::<MyBackend>("/tmp", device.clone(), item.clone());
        all += 1;
        if predicted == expected {
            positive += 1;
        }
    });
    println!("Accuracy: {}", positive as f32 / all as f32);
}
