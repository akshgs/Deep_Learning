from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model 
from PIL import Image, ImageOps  
app=Flask(__name__)
@app.route('/')
def chess():
    return render_template('rps.html')
@app.route('/predict',methods=['post'])
def predict():
    name=(request.values['file'])
    path="static/test/"+name
    print(path)
    np.set_printoptions(suppress=True)
    model = load_model("sps.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(path)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) 
    prediction = model.predict(data)
    index = np.argmax(predict)
    class_name = class_names[index]
    print("Class:", class_name[2:], end="")
    return render_template('rps.html',pre=class_name[2:])
if __name__ == '__main__':

    app.run(port=8000)