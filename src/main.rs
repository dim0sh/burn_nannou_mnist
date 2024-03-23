use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use burn::optim::AdamConfig;
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

    // burn_nannou_mnist::inference::infer::<MyBackend>(
    //     "/tmp",
    //     device,
    //     burn_nannou_mnist::data::NumbersItem {
    //         number: [[1.0, 2.0]],
    //         label: 1,
    //     },
    // );
}
