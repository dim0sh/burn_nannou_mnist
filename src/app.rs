// use image::RgbImage;
use burn::backend::{wgpu::AutoGraphicsApi, WgpuBackend};
use nannou::{prelude::*, image::RgbImage};
use crate::{
    inference,
    data::NumbersItem
};

struct InputImage {
    ui_image: RgbImage,
    image_pos_size: (Vec2, f32),
}

impl InputImage {
    fn new() -> Self {
        let ui_image = RgbImage::new(28, 28);
        let image_pos_size = (Vec2::new(0.0, 0.0), 280.0);
        InputImage {
            ui_image,
            image_pos_size,
        }
    }
}

const WINDOW_SIZE: (u32,u32) = (800, 800);
pub struct GraphModel {
    image: InputImage,
}

pub fn model(app: &App) -> GraphModel {
    app.new_window().size(WINDOW_SIZE.0, WINDOW_SIZE.1).view(view).build().unwrap();
    GraphModel {
        image: InputImage::new(),
    }
}

pub fn view(app: &App, model: &GraphModel, frame: Frame) {
    frame.clear(BLACK);
    let draw = app.draw();
    draw.background().color(DARKGREY);
    let img = nannou::image::DynamicImage::ImageRgb8(model.image.ui_image.clone());
    draw.texture(&nannou::wgpu::Texture::from_image(app, &img))
        .w_h(model.image.image_pos_size.1, model.image.image_pos_size.1)
        .xy(model.image.image_pos_size.0);
    draw.to_frame(app, &frame).unwrap();
}

pub fn update(app: &App, model: &mut GraphModel, _update: Update) {
    let input_size = model.image.image_pos_size.1;
    if app.mouse.buttons.left().is_down() {
        let (x, y) = (app.mouse.x, app.mouse.y);
        if x > -(input_size/2.0) && x < (input_size/2.0) && y > -(input_size/2.0) && y < (input_size/2.0) {
            println!("x:{},y:{}",x, y);
            let x = ((x + input_size/2.0)/10.0)as u32;
            let y = (((y + input_size/2.0)/10.0)-27.0).abs() as u32;

            let surrounding = 1;
            for i in -surrounding..surrounding {
                for j in -surrounding..surrounding {
                    let x = x as i32 + i;
                    let y = y as i32 + j;
                    if x >= 0 && x < 28 && y >= 0 && y < 28 {
                        model.image.ui_image.put_pixel(x as u32, y as u32, nannou::image::Rgb([222, 222, 222]));
                    }
                }
            }

            model.image.ui_image.put_pixel(x, y, nannou::image::Rgb([255, 255, 255]));
        }
    }
    if app.keys.down.contains(&Key::C) {
        model.image.ui_image = RgbImage::new(28, 28);
    }
    if app.keys.down.contains(&Key::E) {
        let mut item = NumbersItem {
            number: [[0.0; 28 * 28]; 3],
            label: 0,
        };
        model.image.ui_image.save("./tmp.png").unwrap();
        
        for (n, color) in model.image.ui_image.pixels().enumerate() {
            item.number[0][n] = color.0[0] as f32;
            item.number[1][n] = color.0[1] as f32;
            item.number[2][n] = color.0[2] as f32;
        }
        type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
        let device = burn::backend::wgpu::WgpuDevice::default();
        let artifact_dir = "./tmp";
        let (predicted, _) = inference::infer::<MyBackend>(artifact_dir, device, item);
        println!("Predicted: {}", predicted);
    }
}