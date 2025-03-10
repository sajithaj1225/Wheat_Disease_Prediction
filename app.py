from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import os
from PIL import Image

app = Flask(__name__)

# Load the updated model
model = tf.keras.models.load_model("wheat_disease_fixed.h5", compile=False, safe_mode=False)

# Define class labels
class_labels = ['Brown_rust', 'Healthy', 'Yellow_rust']

# Set up upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('prediction.html', error='No file uploaded')
        file = request.files['file']
        if file.filename == '':
            return render_template('prediction.html', error='No file selected')
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Load and preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Predict the class
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        
        return render_template('prediction.html', prediction=predicted_class, img_path=filepath)
    
    return render_template('prediction.html')

@app.route('/resource')
def resource():
    return render_template('resource.html')

if __name__ == '__main__':
    print(f"Using TensorFlow version: {tf.__version__}")  # Check TensorFlow version
    app.run(debug=True)
