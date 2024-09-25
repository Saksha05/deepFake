from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['UPLOAD_FOLDER'] = 'uploads'

model = load_model('model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def detect_deepfake(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    probability = prediction[0][0]
    if probability < 0.5:
        result = "Fake"
    else:
        result = "Real"
    return result

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")

@app.route("/demo", methods=["GET", "POST"])
def demo():
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = detect_deepfake(file_path)
            if result == "Fake":
                flash(f"Prediction result: {result}", 'danger')
            else:
                flash(f"Prediction result: {result}", 'success')
            return redirect(request.url)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
