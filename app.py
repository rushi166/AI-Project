from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

import numpy as np
import os

app = Flask(__name__)
model = load_model('model/greenclassify_cnn_model.h5')

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        preds = model.predict(x)
        class_idx = np.argmax(preds)
        classes = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
                   'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
                   'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']
        prediction = classes[class_idx]

        return render_template('predict.html', prediction=prediction, image_path='/' + filepath)
    
    return render_template('predict.html')
    
@app.route('/contact')
def contact():
    return render_template('contact_us.html')  # <-- Contact Us Page    

@app.route('/logout')
def logout():
    return render_template('logout.html')

if __name__ == '__main__':
    app.run(debug=True)