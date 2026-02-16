from flask import Flask, jsonify, render_template, request
from PIL import Image
import os
from model import segment_image

app = Flask(__name__)
UPLOAD_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET","POST"])
def index():
    if request.method=="POST":
        file = request.files["image"]
        img = Image.open(file).convert("RGB")
        input_path = os.path.join(UPLOAD_FOLDER,"input.png")
        img.save(input_path)
        result = segment_image(img)
        out_path = os.path.join(UPLOAD_FOLDER,"output.png")
        Image.fromarray(result).save(out_path)
        return render_template("index.html", input_image="outputs/input.png",output_image="outputs/output.png")
    return render_template("index.html", output_image=None)
@app.route("/predict", methods =["POST"])
def predict():
    file = request.files["image"]
    output_path = "static/output.png"
    segmented = segment_image(file)
    segmented.save(output_path)

    return jsonify({"image_url": "/static/output.png"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
