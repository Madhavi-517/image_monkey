# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 12:19:46 2021

@author: Madhavi
"""
from flask import Flask,render_template,request
from keras.models import load_model
from keras_preprocessing import image
import numpy as np


app=Flask(__name__,template_folder='tmp')

classes={0:'malted _hawler',1:'patas_monkey',2:'bald_uakari',3:'japanese_macaque',
         4:'pigmy_marmoset',5:'white_headed_capuchin',6:'silvery_marmoset',
         7:'common_squirrel_monkey',8:'black_headed_night_monkey',9:'nilgiri_langur'}


model=load_model('E:/image_classification/image-xception.hdf5')
    
def preprocessingImg(image_path):
    test_image=image.load_img(image_path,target_size=(299,299))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image, axis=0)
    test_image=test_image/255.
     
    prediction=model.predict(test_image)
    prediction=np.argmax(prediction)
    return classes[prediction]



@app.route('/',methods=['GET'])
def home():
    return render_template('home.html')



@app.route('/predict',methods=['POST'])
def predict():
    image=request.files['image']
    img_path='static/'+image.filename
    image.save(img_path)
    
    p=preprocessingImg(img_path)
   
    return  render_template('home.html',prediction=p,path=img_path)
 

if __name__=='__main__':
    app.run()
