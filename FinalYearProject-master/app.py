import os
from flask import Flask, render_template, request
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model
import numpy as np

app = Flask(__name__)
MODEL_PATH = 'models/resnet50_pneu_model.keras'

# Load the model
model = load_model(MODEL_PATH)

# Ensure the static directory exists
STATIC_DIR = './static'
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Save the uploaded image
    imagefile = request.files["imagefile"]
    image_path = os.path.join(STATIC_DIR, imagefile.filename)
    imagefile.save(image_path)

    # Resize the image to (224, 224) and ensure it's in RGB format
    img = load_img(image_path, target_size=(224, 224))  # ResNet50 expects (224, 224, 3)
    x = img_to_array(img)  # Convert image to array
    x = x / 255.0  # Normalize pixel values to [0, 1]
    x = np.expand_dims(x, axis=0)  # Add batch dimension

    # Predict using the model
    classes = model.predict(x)
    result = classes[0][0]

    # Interpret the prediction result
    classification = "Positive" if result >= 0.5 else "Negative"
    confidence = result * 100 if result >= 0.5 else (1 - result) * 100
    output = f"{classification} ({confidence:.2f}%)"

    # Return the prediction result to the web page
    return render_template('index.html', prediction=output, imagePath=image_path)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
