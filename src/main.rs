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
        match expected {
            0 => {
                all[0] += 1;
                if predicted == 0 {
                    positive[0] += 1;
                }
            }
            1 => {
                all[1] += 1;
                if predicted == 1 {
                    positive[1] += 1;
                }
            }
            2 => {
                all[2] += 1;
                if predicted == 2 {
                    positive[2] += 1;
                }
            }
            3 => {
                all[3] += 1;
                if predicted == 3 {
                    positive[3] += 1;
                }
            }
            4 => {
                all[4] += 1;
                if predicted == 4 {
                    positive[4] += 1;
                }
            }
            5 => {
                all[5] += 1;
                if predicted == 5 {
                    positive[5] += 1;
                }
            }
            6 => {
                all[6] += 1;
                if predicted == 6 {
                    positive[6] += 1;
                }
            }
            7 => {
                all[7] += 1;
                if predicted == 7 {
                    positive[7] += 1;
                }
            }
            8 => {
                all[8] += 1;
                if predicted == 8 {
                    positive[8] += 1;
                }
            }
            9 => {
                all[9] += 1;
                if predicted == 9 {
                    positive[9] += 1;
                }
            }
            _ => {}
        }
    });
    let accuracy = positive.iter().zip(all.iter()).map(|(p, a)| *p as f32 / *a as f32).collect::<Vec<f32>>();
    println!("{:?}", accuracy);
}
