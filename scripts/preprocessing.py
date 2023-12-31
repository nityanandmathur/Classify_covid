import os
import random

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_df(c, p, n): 
  random.seed(42)
  filenames = os.listdir(c) + random.sample(os.listdir(n), 5000) + os.listdir(p)
  categories = []
  for filename in filenames:
    category = filename.split('-')[0]
    if category == '/workspaces/classify_covid/data/raw/COVID/images/COVID':
      categories.append(str(2))
    elif category == '/workspaces/classify_covid/data/raw/Viral Pneumonia/images/Viral Pneumonia':
      categories.append(str(1))
    else:
      categories.append(str(0))

  """converting to dataframe"""
  for i in range(len(filenames)):
    if 'COVID' in filenames[i]:
      filenames[i] = os.path.join(c, filenames[i])
    elif 'Viral Pneumonia' in filenames[i]:
      filenames[i] = os.path.join(p, filenames[i])
    else:
      filenames[i] = os.path.join(n, filenames[i])

  df = pd.DataFrame({
    'filename': filenames,
    'category': categories
  })
  df.to_csv(path_or_buf='/workspaces/classify_covid/data/raw/df_csv.csv', sep=',', index=False)
  # return df

def train_generator(train_data):
    train_data_gen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
    )
    train_generator = train_data_gen.flow_from_dataframe(
        train_data,
        x_col='filename',
        y_col='category',
        target_size=(224,224),
        class_mode='categorical',
        batch_size=15
    )
    return train_generator

def val_test_generator(data):
    val_test_data_gen = ImageDataGenerator(rescale=1./255)

    val_test_generator = val_test_data_gen.flow_from_dataframe(
        data,
        x_col='filename',
        y_col='category',
        target_size=(224,224),
        class_mode='categorical',
        batch_size=15
    )
    return val_test_generator
