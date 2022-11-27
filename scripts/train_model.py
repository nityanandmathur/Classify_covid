import argparse
import os
import random
from typing import Text

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb.keras
from hydra import compose, initialize
from model import get_Model
from omegaconf import OmegaConf
from PIL import Image
from preprocessing import create_df, train_generator, val_test_generator
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

import wandb


def train(config_name: Text) -> None:
    # set the random seeds

    os.environ['TF_CUDNN_DETERMINISTIC'] = '2'
    random.seed(hash("setting random seeds") % 2**32 - 1)   #to control randomness
    np.random.seed(hash("improves reproductibility") % 2**32 - 1) #to make it reproducable as possible
    tf.random.set_seed(hash("by removing stochasticity") % 2**32 - 1)


    initialize(version_base=None, config_path="../configs")
    cfg=compose(config_name=config_name)
    print(OmegaConf.to_yaml(cfg))

    wandb.init(project="classify_covid", config=cfg)

    #data
    c = cfg.data.covid
    n = cfg.data.normal
    p = cfg.data.viral

    random.seed(42)
    filenames = os.listdir(c) + random.sample(os.listdir(n), 5000) + os.listdir(p)

    df= create_df(filenames, c, p, n)

    #train_test_split
    train_data, test_valid_data = train_test_split(df, test_size=0.2, random_state = 42, shuffle=True, stratify=df['category'])
    train_data = train_data.reset_index(drop=True)

    test_valid_data = test_valid_data.reset_index(drop=True)

    test_data, valid_data = train_test_split(test_valid_data, test_size=0.5, random_state = 42,
                                             shuffle=True, stratify=test_valid_data['category'])
    test_data = test_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)

    #data_augmentation
    train_gen = train_generator(train_data)
    valid_gen = val_test_generator(valid_data)
    test_gen = val_test_generator(test_data)

    model = get_Model(freeze=True)
    # loss=cfg.params.loss
    #lr=0.01

    lr = cfg.params.learning_rate
    epoch_num=cfg.params.training_epoch
    model.compile(optimizer=Adam(learning_rate=lr), metrics=["accuracy"], loss='categorical_crossentropy')

    history = model.fit(train_gen,
                        validation_data=valid_gen,
                        verbose=1,
                        epochs=epoch_num,
                        callbacks=[wandb.keras.WandbCallback()])
    
    model.save(cfg.model.save)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', dest='config', required=True)
    args = argparser.parse_args()

    train(config_name=args.config)