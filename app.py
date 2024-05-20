from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import os

app = Flask(__name__)

# Load the trained model
model = load_model('C:\\Users\\nour2\\OneDrive\\Desktop\\BrainTumor\\nouveau_modele_entrene.keras')

# Define route for home page
@app.route('/')
def home():
    return render_template('menu.html')

# Define route for the upload page
@app.route('/upload')
def upload_page():
    return render_template('index.html')

# Define route for the text page
@app.route('/text')
def text_page():
    return render_template('text.html')

# Define route for handling image upload and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['image']

        # Save the uploaded image temporarily
        img_path = 'temp_img.jpg'
        file.save(img_path)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make predictions using the loaded model
        predictions = model.predict(img_array)

        # Get the predicted class
        if predictions[0][0] > 0.5:
            prediction = 'Healthy'
        else:
            prediction = 'Tumor'

        # Delete the temporary image file
        os.remove(img_path)

        # Render the result template with the prediction
        return render_template('result.html', prediction=prediction)

if __name__ == '_main_':
    app.run(debug=True,port=5001)