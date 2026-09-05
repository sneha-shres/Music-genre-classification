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
import pandas as pd
import tensorflow as tf
tf.random.set_seed(seed)
from dotenv import load_dotenv
import json
from easydict import EasyDict as edict
from src.visualization.visualize import Visualization
from sklearn.metrics import precision_score, f1_score, confusion_matrix, recall_score
from src.data.data_loader import DataLoader
from src.models.train_model import SpatialAttention 
from src.utils.model_utils import loadModel

load_dotenv()


# CONFIG

DATA_DIR = config.data_dir  # Your folder with genre subfolders
CROPPED_DIR = config.cropped_dir
IMG_HEIGHT, IMG_WIDTH = config.img_height, config.img_width
MODEL_SAVE_PATH = config.model_save_path
PREDICTIONS_RESULTS_SAVE_PATH = config.predictions_results
MODEL_TYPE = config.model_type
EVAL_TYPE = config.eval_type






if __name__ == "__main__":
    #Load latest model or specific model
    model, modelNumber, _ = loadModel()
    #DataLoader
    dataObj = DataLoader(evaluationType="edge")
    X_test, y_test = dataObj.X_test, dataObj.y_test
    classNames = dataObj.class_names
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    
    true_class_indices = np.argmax(y_test, axis=1)
    predicted_class_indices = np.argmax(predictions, axis=1)

    print(f"Test accuracy: {acc:.2%}")

    # Save predictions to CSV
    results = []
    csvSavePath = f'{PREDICTIONS_RESULTS_SAVE_PATH}/{EVAL_TYPE}/{modelNumber}.csv'
    os.makedirs(os.path.dirname(csvSavePath), exist_ok=True)

    
    for i in range(len(X_test)):
        true_genre = dataObj.class_names[true_class_indices[i]]
        predicted_genre = dataObj.class_names[predicted_class_indices[i]]

        results.append({
        'Sample': i,
        'True Genre': true_genre,
        'Predicted Genre': predicted_genre
    })
    
    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(csvSavePath, index=False)

    #Creating Visualization object
    viz = Visualization()

    #confusion matrix
    trueClassIndices = results_df['True Genre']
    predictedClassIndices = results_df['Predicted Genre']
    cm = confusion_matrix(trueClassIndices, predictedClassIndices)
    classNames = sorted(np.unique(trueClassIndices))
    viz.plot_confusion_matrix(cm, classNames, modelNumber)

    #precision, recall, f1
    macroPrecision = precision_score(trueClassIndices, predictedClassIndices, average='macro')
    macroRecall = recall_score(trueClassIndices, predictedClassIndices, average='macro')
    macroF1 = f1_score(trueClassIndices, predictedClassIndices, average='macro')
    print(f"Macro Precision: {macroPrecision:.4f}, Macro Recall: {macroRecall:.4f}, Macro F1: {macroF1:.4f}")

    # Plot embedding space
    embeddingModel = tf.keras.Model(inputs=model.input,
                                 outputs=model.get_layer('embedding').output)
    embeddings = embeddingModel.predict(X_test)

    viz.plotEmbbeddingSpace(embeddings, y_test, modelNumber,classNames)
    

    
