use crate::{data::{NumbersBatcher, NumbersItem}, training::TrainingConfig};
use burn::{
    data::dataloader::batcher::Batcher,
    config::Config,
    tensor::backend::Backend,
    record::{CompactRecorder, Recorder},
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: NumbersItem) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init_with::<B>(record);

    let label = item.label;
    let batcher = NumbersBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.numbers);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} Expected {}", predicted, label);
}