use crate::{
    data::{NumbersBatcher, NumbersItem},
    training::TrainingConfig,
};
use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    record::{CompactRecorder, Recorder},
    tensor::backend::Backend,


};

pub fn infer<B: Backend<IntElem = i32>>(
    artifact_dir: &str,
    device: B::Device,
    item: NumbersItem,
) -> (i32, i32) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into())
        .expect("Trained model should exist");

    let model = config.model.init_with::<B>(record);

    let label = item.label;
    let batcher = NumbersBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.numbers);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    (predicted, label)
}

pub fn infer_batch<B: Backend<IntElem = i32>>(
    artifact_dir: &str,
    device: B::Device,
    items: Vec<NumbersItem>,
) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into())
        .expect("Trained model should exist");

    let model = config.model.init_with::<B>(record);

    let batcher = NumbersBatcher::new(device);
    let batch = batcher.batch(items);
    let output = model.forward(batch.numbers);
    let mut tp_sum = 0;
    let mut all_sum = 0;
    output
        .argmax(1)
        .flatten::<1>(0, 1)
        .iter_dim(0)
        .zip(batch.labels.iter_dim(0))
        .for_each(|(predicted, label)| {
            tp_sum += if predicted.into_scalar() == label.into_scalar() { 1 } else { 0 };
            all_sum += 1;
        });
    println!("Accuracy: {}", tp_sum as f32 / all_sum as f32);
}
