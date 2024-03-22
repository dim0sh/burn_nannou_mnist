use burn::optim::AdamConfig;
use burn::backend::{Autodiff, Wgpu, wgpu::AutoGraphicsApi};
use burn_nannou_MNIST::model::ModelConfig;

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    burn_nannou_MNIST::training::train::<MyAutodiffBackend>(
        "/tmp",
        burn_nannou_MNIST::training::TrainingConfig::new(ModelConfig::new(10, 128), AdamConfig::new()),
        device.clone(),
    );

    // burn_nannou_MNIST::inference::infer::<MyBackend>(
    //     "/tmp",
    //     device,
    //     burn_nannou_MNIST::data::NumbersItem {
    //         number: [[1.0, 2.0]],
    //         label: 1,
    //     },
    // );
}