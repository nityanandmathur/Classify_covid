import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *


def get_Model(freeze):  #model_cfg as input
  # cfg = model_cfg

  baseModel = DenseNet121(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

  if freeze:
    for layer in baseModel.layers:
      layer.trainable = False

  headModel = baseModel.output
  headModel = AveragePooling2D()(headModel)
  headModel = Flatten()(headModel)
  headModel = Dense(128, activation="relu")(headModel)
  headModel = Dropout(0.2)(headModel)
  headModel = Dense(3, activation='softmax')(headModel)

  model = Model(inputs=baseModel.input, outputs=headModel)
  return model