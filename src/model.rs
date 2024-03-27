use burn::{
    config::Config,
    module::Module, 
    nn::{
        conv::{
            Conv2d, 
            Conv2dConfig
        },
        pool::{
            AvgPool2d,
            AvgPool2dConfig
        },
        Dropout, 
        DropoutConfig, 
        Linear, 
        LinearConfig, 
        ReLU}, 
        tensor::{
            backend::Backend, 
            Tensor
        }
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    pool: AvgPool2d,
    linear1: Linear<B>,
    linear2: Linear<B>,
    dropout: Dropout,
    activation: ReLU,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, _device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1,4], [3,3],).init(),
            pool: AvgPool2dConfig::new([8, 8]).init(),
            linear1: LinearConfig::new(4 * 19 * 19, self.hidden_size).init(),
            linear2: LinearConfig::new(self.hidden_size, 128).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: ReLU::new(),
        }
    }
    pub fn init_with<B: Backend>(&self, record: ModelRecord<B>) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1,4], [3,3],).init_with(record.conv1),
            pool: AvgPool2dConfig::new([8, 8]).init(),
            linear1: LinearConfig::new(4 * 19 * 19, self.hidden_size).init_with(record.linear1),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes)
                .init_with(record.linear2),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: ReLU::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = input.dims();

        let x = input.reshape([batch_size, 1, height, width]);

        let x = self.conv1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x);
        let x = x.reshape([batch_size, 4 * 19 * 19]);

        let x = self.linear1.forward(x);
        // let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x)
    }
}
