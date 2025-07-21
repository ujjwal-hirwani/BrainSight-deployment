from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import io
import tensorflow as tf
from flask import send_from_directory
import os

# Initialize Flask app
app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///feedback.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Feedback(db.Model):
    feedback_id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(50), nullable=False)
    message = db.Column(db.String(1000), nullable=False)

    def __repr__(self):
        return f"{self.name} - {self.email} - {self.message}"
    
# Load your Keras model (replace with your path)
model = tf.keras.models.load_model('cnn_model.h5')

# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route: Home page with upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route: Handle prediction
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if 'image' not in request.files:
        return render_template('predict.html')

    file = request.files['image']

    if file.filename == '':
        return render_template('predict.html', message = "No selected file")

    if file and allowed_file(file.filename):
        try:
            # Read image in-memory
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            image = image.resize((150, 150))  # resize to match your model input
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)  # add batch dimension

            # Predict
            class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

            prediction = model.predict(image)
            predicted_index = np.argmax(prediction, axis=1)[0]
            predicted_label = class_names[predicted_index]

            return redirect(url_for('result', predicted_label = predicted_label))

        except Exception as e:
            return render_template('predict.html', message = f"Error processing image: {str(e)}")
    else:
        return render_template('predict.html', message = "Invalid file type. Only jpg, jpeg, png are allowed.")

@app.route('/result')
def result():
    predicted_label = request.args.get('predicted_label', default="Unknown")
    return render_template('result.html', label=predicted_label)

# Error handler for oversized files
@app.errorhandler(413)
def too_large(e):
    return render_template('predict.html', message = "File too large. Max size is 2MB.")

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        feedback = Feedback(name = name, email = email, message = message)
        db.session.add(feedback)
        db.session.commit()
    return render_template('index.html', message = "")

@app.route('/download/<filename>')
def download_sample(filename):
    sample_folder = os.path.join(app.root_path, 'static', 'samples')
    return send_from_directory(directory=sample_folder, path=filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
