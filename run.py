from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import io
import os
import torch
import cv2
import base64
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///feedback.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Feedback(db.Model):
    feedback_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(50), nullable=False)
    message = db.Column(db.String(1000), nullable=False)

    def __repr__(self):
        return f"{self.name} - {self.email} - {self.message}"

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---- Load TorchScript segmentation model (CPU only) ----
torch.set_num_threads(1)
seg_model = torch.jit.load("segmentation_model_torchscript.pt", map_location="cpu")
seg_model.eval()

def preprocess_tf(image_pil):
    image = image_pil.resize((150, 150))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def preprocess_torch(image_pil):
    image = image_pil.resize((256, 256))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC → CHW
    image = np.expand_dims(image, axis=0)
    return torch.from_numpy(image)

def generate_overlay(original_pil, mask_tensor):
    mask = mask_tensor.squeeze().detach().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255

    original = np.array(original_pil.resize((256, 256)))

    red_mask = np.zeros_like(original)
    red_mask[:, :, 0] = mask

    overlay = cv2.addWeighted(original, 0.8, red_mask, 0.4, 0)

    _, buffer = cv2.imencode(".png", overlay)
    return base64.b64encode(buffer).decode("utf-8")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('predict.html', message="No file part")

        file = request.files['image']

        if file.filename == '':
            return render_template('predict.html', message="No selected file")

        if not allowed_file(file.filename):
            return render_template('predict.html', message="Invalid file type")

        try:
            image_pil = Image.open(io.BytesIO(file.read())).convert('RGB')

            # ---- TFLite Classification ----
            tf_input = preprocess_tf(image_pil)
            interpreter.set_tensor(input_details[0]['index'], tf_input)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]

            class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
            predicted_index = int(np.argmax(prediction))
            tumor_type = class_names[predicted_index]
            confidence = round(float(prediction[predicted_index]) * 100, 2)

            # Default values
            tumor_detected = True
            display_img = None

            if tumor_type == "notumor":
                # No tumor case
                tumor_detected = False
                tumor_type = "No tumor detected"
                confidence = 100

                # Convert original image to base64
                buffer = io.BytesIO()
                image_pil.save(buffer, format="PNG")
                display_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
            else:
                # Tumor detected: run segmentation
                torch_input = preprocess_torch(image_pil)
                with torch.no_grad():
                    mask = seg_model(torch_input)
                display_img = generate_overlay(image_pil, mask)

            return render_template(
                "result.html",
                tumor_detected=tumor_detected,
                tumor_type=tumor_type,
                confidence=confidence,
                display_img=display_img
            )

        except Exception as e:
            return render_template('predict.html', message=str(e))

    return render_template('predict.html')


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        feedback_entry = Feedback(name=name, email=email, message=message)
        db.session.add(feedback_entry)
        db.session.commit()

        return render_template('index.html', message="Thank you for your feedback!")
    return render_template('index.html')

@app.route('/download/<filename>')
def download_sample(filename):
    sample_folder = os.path.join(app.root_path, 'static', 'samples')
    return send_from_directory(directory=sample_folder, path=filename, as_attachment=True)

@app.errorhandler(413)
def too_large(e):
    return render_template('predict.html', message="File too large. Max size is 2MB.")

if __name__ == '__main__':
    # Ensure database tables exist inside app context
    with app.app_context():
        db.create_all()
    
    app.run(debug=False)
