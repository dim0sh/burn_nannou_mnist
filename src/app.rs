use image::RgbImage;
use burn::backend::{WgpuBackend,wgpu::AutoGraphicsApi};
use nannou::prelude::*;
use crate::{
    inference,
    data::NumbersItem
};

struct InputImage {
    ui_vector: Vec<u8>,
}

impl InputImage {
    fn new() -> Self {
        let ui_vector = vec![0; 28*28];
        InputImage {
            ui_vector,
        }
    }
}

const WINDOW_SIZE: (u32,u32) = (800, 800);
const INPUT_SIZE: (u32,u32) = (280, 280);
pub struct GraphModel {
    image: InputImage,
}

pub fn model(app: &App) -> GraphModel {
    app.new_window().size(WINDOW_SIZE.0, WINDOW_SIZE.1).view(view).build().unwrap();
    GraphModel {
        image: InputImage::new(),
    }
}

pub fn view(app: &App, _model: &GraphModel, frame: Frame) {
    frame.clear(BLACK);
    let draw = app.draw();
    draw.background().color(DARKGREY);
    for i in 0..28 {
        for j in 0..28 {
            let x = i as f32 * 10.0 - INPUT_SIZE.0 as f32 / 2.0;
            let y = j as f32 * 10.0 - INPUT_SIZE.1 as f32 / 2.0;
            let color = _model.image.ui_vector[j*28 + i];
            draw.rect().x_y(x, y).w_h(10.0, 10.0).color(Rgb::new(color, color, color));
        }
    }
    draw.to_frame(app, &frame).unwrap();
}

pub fn update(app: &App, model: &mut GraphModel, _update: Update) {
    if app.mouse.buttons.left().is_down() {
        let (x, y) = (app.mouse.x, app.mouse.y);
        if x > -(INPUT_SIZE.0 as f32) / 2.0 && x < INPUT_SIZE.0 as f32 / 2.0 && y > (-(INPUT_SIZE.1 as f32) / 2.0) && y < (INPUT_SIZE.1 as f32 / 2.0) {
            let x = (x + INPUT_SIZE.0 as f32 / 2.0) as u32;
            let y = (y + INPUT_SIZE.1 as f32 / 2.0) as u32;
            model.image.ui_vector[(y/10 * 28 + x/10) as usize] = 255;
        }
    }
    if app.keys.down.contains(&Key::C) {
        model.image.ui_vector = vec![0; 28*28];
    }
    if app.keys.down.contains(&Key::E) {
        let mut image = RgbImage::new(28, 28);
        for (n, mut pix) in image.enumerate_pixels_mut().enumerate() {
            let color = model.image.ui_vector[n as usize];
            pix.2 = &mut image::Rgb([color, color, color]);
        }
        type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
        let device = burn::backend::wgpu::WgpuDevice::default();
        let artifact_dir = "./tmp";
        let item = parse_image(image, 1);
        let (predicted, _) = inference::infer::<MyBackend>(artifact_dir, device, item);
        println!("Predicted: {}", predicted);
    }
}

fn parse_image(image: RgbImage, label: i32) -> NumbersItem {
    let mut item = NumbersItem {
        number: [[0.0; 28 * 28]; 3],
        label,
    };
    for (n, pix) in image.pixels().enumerate() {
        item.number[0][n] = pix.0[0] as f32;
        item.number[1][n] = pix.0[1] as f32;
        item.number[2][n] = pix.0[2] as f32;
    }
    item
}