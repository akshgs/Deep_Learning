
import cv2
from glob import glob
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Conv3D,MaxPool3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
sps=pathlib.Path(r"D:\DSML\CNN DATA\sps")
rock=list(sps.glob('rocks/*.png'))
papper=list(sps.glob('papper/*.png'))
scissors=list(sps.glob('scissor/*.png'))
len(rock),len(papper),len(scissors)
sps_dict={"Rock":rock,
          "Papper":papper,
          "scissors":scissors}
sps_class={"Rock":0,
            "Papper":1,
            "scissors":2}
x=[]
y=[]
for i in sps_dict:
  sps_name=i
  sps_list=sps_dict[sps_name]
  for path in sps_list:
    img=cv2.imread(str(path))
    img=cv2.resize(img,(224,224))
    img=img/255
    x.append(img)
    cls=sps_class[sps_name]
    y.append(cls)
len(x)
x=np.array(x)
y=np.array(y)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.75,random_state=1)
len(xtrain),len(ytrain),len(xtest),len(ytest)
xtrain.shape
xtrain.shape,xtest.shape
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

print("[INFO] summary for base model...")
print(base_model.summary())
from tensorflow.keras.layers import MaxPooling2D
from keras.layers.core import Dropout
from tensorflow.keras.models import Model
headModel = base_model.output
headModel = MaxPooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(32, activation="relu")(headModel)
headModel = Dropout(0.2)(headModel)
headModel = Dense(3, activation="softmax")(headModel)
model = Model(inputs=base_model.input, outputs=headModel)
for layer in base_model.layers:
	layer.trainable = False
from tensorflow.keras.optimizers import Adam
print("[INFO] compiling model...")
opt = Adam(lr=1e-4)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
print("[INFO] training head...")
model_hist=model.fit(xtrain,ytrain,epochs=25,validation_data=(xtest,ytest),batch_size=180)
model.save("sps.h5")
from tensorflow.keras.preprocessing import image
def testing_image(image_directory):
    test_image = image.load_img(image_directory, target_size = (224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = test_image/255
    result = model.predict(x= test_image)
    print(result)
    if np.argmax(result)  == 0:
      prediction = 'Rock'
    elif np.argmax(result)  == 1:
      prediction = 'Papper'
    else:
      prediction = 'scissors'
    return prediction
print(testing_image(r"D:\DSML\CNN DATA\sps\scissor\2TAGoXw7yaK0bXBu.png"))
