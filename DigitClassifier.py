import pickle
from os import listdir
from os import path
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from PIL import Image
from os import listdir
from os import path


image='CustomTest/test7.png'

def ConvSingleChannel(img):
    imagePath=Image.open(img).convert('L')
    imgArray=np.asarray(imagePath)
    return imgArray


def ReshapeData(imgAr,shape):
    imgAr=imgAr.reshape(shape)
    imgAr=imgAr/255
    return imgAr

imgA=ConvSingleChannel(image)    
imgArr=ReshapeData(imgA,(28*28),)

# imgArr.shape
Arr=np.array([imgArr])

model=keras.models.load_model("DigitClassifierModel")

print("\n\n")

pred=model.predict(Arr)
predictions=np.argmax(pred)
print(predictions)