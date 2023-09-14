"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
from PIL import Image

import torch
from flask import Flask, request

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5"

@app.route("/")
def index():
    return "<h1>Hello!</h1>"
@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if request.method != "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        results = model(img, size=640) # reduce size=320 for faster inference
        return results.pandas().xyxy[0].to_json(orient="records")

if __name__ == "__main__":
    from waitress import serve
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/best.pt', force_reload=True)
    serve(app, host="0.0.0.0", port=8080)
