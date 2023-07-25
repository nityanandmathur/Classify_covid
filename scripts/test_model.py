import os

import numpy as np
import tensorflow as tf


def test():
    model = tf.keras.models.load_model('./models/covid.h5')
    os.system('clear')
    img_path = '/workspaces/classify_covid/data/raw/Lung_Opacity/images/Lung_Opacity-1.png'
    
    #img = tf.keras.utils.load_img(img_path, target_size=(224,224)) -- newer versions
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
    #img_arr = tf.keras.utils.img_to_array(img) -- newer versions
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    img_arr = tf.expand_dims(img_arr, 0)

    predictions = model.predict(img_arr)
    print(predictions)

    class_names = ['Covid', 'Lung opacity', 'Normal']
    score = tf.nn.softmax(predictions[0])
    print(class_names[np.argmax(score)])

if __name__ == '__main__':
    test()