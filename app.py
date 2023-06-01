from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('my_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the image file from the POST request
        image_file = request.files['file']
        
        # Open the image file using PIL
        img = Image.open(image_file)
        # img = Image.open('battery.jpeg')
        
        # Preprocess the image so it matches the input your model expects
        img = img.resize((150, 150))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make the prediction
        prediction = model.predict(img)

        class_names = ['Cardboard', 'Glass', 'Metal', 'Battery', 'Plastic', 'Trash']
        predicted_class_indices = np.argmax(prediction, axis=1)

        predicted_class_names = [class_names[idx] for idx in predicted_class_indices]
        
        # Return the prediction
        return str(predicted_class_names)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
