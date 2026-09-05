import os
import json
from easydict import EasyDict as edict
with open('src/config.json') as f:
    config = edict(json.load(f))
seed = config.random_seed
os.environ['PYTHONHASHSEED'] = f'{seed}'
import random
random.seed(seed)
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.random.set_seed(seed)
from dotenv import load_dotenv
import json
from easydict import EasyDict as edict
from src.models.train_model import SpatialAttention 


load_dotenv()


# CONFIG

DATA_DIR = config.data_dir  # Your folder with genre subfolders
CROPPED_DIR = config.cropped_dir
IMG_HEIGHT, IMG_WIDTH = config.img_height, config.img_width
MODEL_SAVE_PATH = config.model_save_path
PREDICTIONS_RESULTS_SAVE_PATH = config.predictions_results
MODEL_TYPE = config.model_type



def loadModel(modelNumber=None):
    """"Function to load latest model or specific model"""
    attentionModel = None
    modelPath = os.path.join(MODEL_SAVE_PATH, f'{modelNumber}.keras')
    if os.path.exists(modelPath):
        model = tf.keras.models.load_model(modelPath)
        print(f"Model loaded from {modelPath}")
    else: ##Take the latest model
        modelFiles = os.listdir(MODEL_SAVE_PATH)
        if modelFiles:
            modelNumber = max(modelFiles, key=lambda x: str(x.split('.')[0]))
            modelPath = os.path.join(MODEL_SAVE_PATH, f"{str(modelNumber.split('.')[0])}.keras")
            model = tf.keras.models.load_model(modelPath)
            print(f"Latest model loaded from {modelPath}")
        else:   
            print("No model files found in the directory.")
    if MODEL_TYPE == "cnn_attention":

        modelAttentionPath = os.path.join(MODEL_SAVE_PATH, f"{str(modelNumber.split('.')[0])}.attention.keras")
        attentionModel = tf.keras.models.load_model(modelAttentionPath)
        print(f"Attention model loaded from {modelAttentionPath}")

    return model, modelNumber, attentionModel