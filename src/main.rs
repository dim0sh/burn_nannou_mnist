use burn::backend::wgpu::AutoGraphicsApi;
use burn::backend::{WgpuAutodiffBackend, WgpuBackend};
use burn::optim::AdamConfig;
use burn_nannou_mnist::data::{NumbersDataset, NumbersItem};
use burn_nannou_mnist::model::ModelConfig;
use burn_nannou_mnist::app;

fn main() {
    let artifact_dir = "./tmp";
    type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = WgpuAutodiffBackend<AutoGraphicsApi, f32, i32>;
    let device = burn::backend::wgpu::WgpuDevice::default();
    // enable training by setting train to true
    let train = false;
    let batch_infer_dataset = false;
    let ui_bool = true;

    if train {
        burn_nannou_mnist::training::train::<MyAutodiffBackend>(
            artifact_dir,
            burn_nannou_mnist::training::TrainingConfig::new(
                ModelConfig::new(10, 512),
                AdamConfig::new(),
            ),
            device.clone(),
        );
    }

    if batch_infer_dataset {
        let items = NumbersDataset::new("test").dataset.into_iter().collect::<Vec<NumbersItem>>();
        burn_nannou_mnist::inference::infer_batch::<MyBackend>(artifact_dir, device.clone(), items);
    }
    if ui_bool {
        nannou::app(app::model)
            .view(app::view)
            .update(app::update)
            .run();

    }
    
}

