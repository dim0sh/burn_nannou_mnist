use nannou::prelude::*;
pub struct GraphModel {
    
}

pub fn model(app: &App) -> GraphModel {
    app.new_window().size(800, 800).view(view).build().unwrap();
    GraphModel {}
}

pub fn view(app: &App, _model: &GraphModel, frame: Frame) {
    frame.clear(BLACK);
    let draw = app.draw();
    draw.background().color(BLACK);
    draw.to_frame(app, &frame).unwrap();
}

pub fn update(_app: &App, _model: &mut GraphModel, _update: Update) {
}