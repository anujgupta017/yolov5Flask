"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
from PIL import Image

import torch
from flask import Flask, request

from flask import json
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5"

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    cause = e.original_exception.args[0]
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
        "mainreason" : cause
    })
    response.content_type = "application/json"
    return response

@app.route("/")
def index():
    return "<h1>Hello!</h1>"
@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if request.method != "POST":
        return "<h1>no post method detected</h1>"

    if request.files.get("image"):
            try:
                image_file = request.files["image"]
                image_bytes = image_file.read()
                img = Image.open(io.BytesIO(image_bytes))
                results = model(img, size=640) # reduce size=320 for faster inference
                return results.pandas().xyxy[0].to_json(orient="records")
            except Exception as ex:
                raise ex
    else:
         return "<h1>no image</h1>"

if __name__ == "__main__":
    #for production

    from waitress import serve
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/best.pt', force_reload=True)
    serve(app, host="0.0.0.0", port=8080)

    #debug

    #model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/best.pt', force_reload=True)
    #app.run() 


 
