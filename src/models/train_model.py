import os
import json
import pandas as pd
from easydict import EasyDict as edict
with open('src/config.json') as f:
    config = edict(json.load(f))
seed = config.random_seed
os.environ['PYTHONHASHSEED'] = f'{seed}'
import json
from collections import defaultdict
from easydict import EasyDict as edict
import random
random.seed(seed)
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.random.set_seed(seed)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, 
                                     Dense, Dropout, BatchNormalization, Activation,
                                     Multiply, Permute, Reshape)
from tensorflow.keras import Model, Input
from dotenv import load_dotenv
from enum import Enum
import logging
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping
import datetime
from src.visualization.visualize import Visualization
from src.data.data_loader import DataLoader
from tensorflow.keras.utils import register_keras_serializable




load_dotenv()

INPUT_SHAPE = tuple(config.input_shape)
NUM_CLASSES = config.num_classes
DROPOUT_RATE = config.dropout_rate 
MODEL_TYPE = config.model_type
BATCH_SIZE = config.batch_size
EPOCHS = config.epochs
PREDICTIONS_LOG_DIR = config.predictions_log_dir
MODEL_SAVE_PATH = config.model_save_path
PREDICTIONS_RESULTS_SAVE_PATH = config.predictions_results
DATA_DIR = config.data_dir  # Your folder with genre subfolders
CROPPED_DIR = config.cropped_dir


modelNumber = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


##Look at it later 
def get_tensorboard_callback(log_dir_root="models/logs/fit"):
    os.makedirs(os.path.dirname(log_dir_root), exist_ok=True) 
    log_dir = os.path.join(log_dir_root,modelNumber )
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback


class ModelType(Enum):
    BASIC_CNN = "basic_cnn"
    CNN_ATTENTION = "cnn_attention"

class MusicGenreCNN:
    def __init__(self,
                 model_type: ModelType = ModelType(MODEL_TYPE),
                 input_shape: tuple = INPUT_SHAPE,
                 num_classes: int = NUM_CLASSES,
                 dropout_rate: float = DROPOUT_RATE,
                 **model_kwargs):
        
        self.model_type = model_type
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model_kwargs = model_kwargs
        self.model = self.build_model()

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger.info(f"Initialized {self.model_type.value} model")

    def build_base_cnn(self):
        """Base CNN architecture"""
        model = Sequential()
        
        model.add(Conv2D(8, (3, 3),
                        input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(16, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(32, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        
        return model

    def build_attention_cnn(self):
        """CNN with spatial attention mechanism"""
        base_model = self.build_base_cnn()
        
        inputs = Input(shape=self.input_shape)
        x = base_model(inputs)

        # Add spatial attention
        attentionLayer = SpatialAttention()
        attentionOutput = attentionLayer(x) 

        # Now get the attention map (output of attentionLayer.conv)
        # Recompute it with the same input

        # avgPool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(inputs)
        avgPool = GlobalAverage()(inputs)
        maxPool = GlobalMax()(inputs)
        # maxPool = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(inputs)
        # concat = tf.keras.layers.Concatenate(axis=-1)([avgPool, maxPool])
        concat = tf.keras.layers.Concatenate(axis=-1)([avgPool, maxPool])
        
        attentionMap = attentionLayer.conv(concat)  # this is now a Tensor
        self.attentionModel = Model(inputs,attentionMap)
        
        # Classifier head
        x = Flatten()(attentionOutput)
        x = Dense(256, activation='relu', name='embedding')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
  
        return Model(inputs, outputs)

    def getAttentionModel(self):
        return self.attentionModel

    def build_model(self) -> tf.keras.Model:
        if self.model_type == ModelType.BASIC_CNN:
            model = self.build_base_cnn()
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(self.dropout_rate))
            model.add(Dense(self.num_classes, activation='softmax'))
            return model
            
        elif self.model_type == ModelType.CNN_ATTENTION:
            return self.build_attention_cnn()

    def compile(self, 
               optimizer: str = 'adam',
               loss: str = 'categorical_crossentropy',
               metrics: list = ['accuracy']):
        self.model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)
        self.logger.info("Model compiled successfully")

    def summary(self):
        return self.model.summary()

    def save_model_weights(self, filepath: str):
        self.model.save_weights(filepath)
        self.logger.info(f"Model weights saved to {filepath}")

@register_keras_serializable()
class GlobalAverage(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-1, keepdims=True)

@register_keras_serializable()
class GlobalMax(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=-1, keepdims=True)



@register_keras_serializable()
class SpatialAttention(tf.keras.layers.Layer):
    """Spatial attention mechanism for CNN feature maps"""
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv2D(1, kernel_size, padding='same', activation='sigmoid')

    def call(self, inputs):
        # Aggregate features across channel axis
        avg_pool = GlobalAverage()(inputs)
        max_pool = GlobalMax()(inputs)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return Multiply()([inputs, attention])

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.conv.kernel_size[0]})
        return config

if __name__ == "__main__":
    model_builder = MusicGenreCNN()
    model_builder.compile()
    model_builder.summary()

    dataObj = DataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = dataObj.X_train, dataObj.y_train, dataObj.X_val, dataObj.y_val, dataObj.X_test, dataObj.y_test

    tensorboard_callback = get_tensorboard_callback()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)   
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


    history = model_builder.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks = [tensorboard_callback, reduce_lr]
    )
    
    savePath = os.path.join(MODEL_SAVE_PATH, f'{modelNumber}.keras')
    
    if model_builder.model_type == ModelType.CNN_ATTENTION:
        saveAttentionPath = os.path.join(MODEL_SAVE_PATH, f'{modelNumber}.attention.keras')
        os.makedirs(os.path.dirname(saveAttentionPath), exist_ok=True)
        model_builder.getAttentionModel().save(saveAttentionPath)
        print(f"Attention model saved to {saveAttentionPath}")

    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    model_builder.model.save(savePath)
    print(f"Model saved to {savePath}")

    

    Visualization.plot_loss_curve(history, modelNumber)
    




