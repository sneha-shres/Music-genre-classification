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
from PIL import Image
import tensorflow as tf
tf.random.set_seed(seed)
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import json
from easydict import EasyDict as edict



load_dotenv()


# CONFIG

DATA_DIR = config.data_dir  # Your folder with genre subfolders
CROPPED_DIR = config.cropped_dir
IMG_HEIGHT, IMG_WIDTH = config.img_height, config.img_width
MODEL_SAVE_PATH = config.model_save_path
PREDICTIONS_RESULTS_SAVE_PATH = config.predictions_results
modelType = config.model_type
EVAL_TYPE = config.eval_type


""""Class to prepare the data for model trainning and evaluation"""
class DataLoader:
    def __init__(self,testSplit=0.2,valSplit=0.2, evaluationType=EVAL_TYPE):
        self.evaluationType = evaluationType
        self.testSplit=testSplit
        self.valSplit=valSplit

        self.dataDir = {"base": DATA_DIR, "edge": CROPPED_DIR}

        currentDataDir = self.dataDir[self.evaluationType] 
        self.load_images(currentDataDir)
        self.encodeY()

        if self.evaluationType=="base":
            self.splitData()
        else:
            self.X_test = self.X
            self.y_test = self.y


    def load_images(self,data_dir):
        X, y = [], []
        class_names = sorted(os.listdir(data_dir))
        for idx, genre in enumerate(class_names):
            genre_dir = os.path.join(data_dir, genre)
            if not os.path.isdir(genre_dir):
                continue
            for fname in os.listdir(genre_dir):
                if fname.endswith('.png'):
                    img_path = os.path.join(genre_dir, fname)
                    img = Image.open(img_path).convert('L').resize((IMG_WIDTH, IMG_HEIGHT))
                    arr = np.array(img) / 255.0  # Normalize
                    X.append(arr[..., np.newaxis])  # Add channel dim
                    y.append(idx)
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y)
        self.class_names=class_names

    
   


    def splitData(self):
        # First split: train+val and test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.testSplit, 
            random_state=42, 
            stratify=self.y
        )

        # Second split: train and val (e.g. 80% train, 20% val of remaining data)
        val_size = self.valSplit  # adjust as needed
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size, 
            random_state=42, 
            stratify=y_temp
        )



    def encodeY(self):
        y_cat = to_categorical(self.y, num_classes=len(self.class_names))
        self.y = y_cat
