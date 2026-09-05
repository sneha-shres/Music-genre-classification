import logging
import os
from dotenv import load_dotenv
import random
import re
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from easydict import EasyDict as edict
import json
from sklearn.manifold import TSNE
with open('src/config.json') as f:
    config = edict(json.load(f))

model_type = config.model_type
evalType = config.eval_type

load_dotenv()

class Visualization:
    def __init__(self, vizType=None):
        self.vizType = vizType
    

        # logging setup
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



        self.plotFunction={"compare":self.compareWithOriginal}

        if vizType:
            plotFunction = self.plotFunction[vizType]
            plotFunction()
    

    def compareWithOriginal(self):
        logging.info("Plotting figure to compare spectrograms")
        RAW_SPECT_DIR =  os.path.join(os.getenv("RAW_DATA_DIR"),"images_original")
        PROCESSED_SPECT_DIR = os.path.join(os.getenv("PROCESSED_DATA_DIR"))
        SAMPLES = 5


        genres = sorted(os.listdir(RAW_SPECT_DIR))
        for genre in genres:
            logging.info(f"Plotting for {genre}...")
            rawGenre = os.path.join(RAW_SPECT_DIR, genre)
            processedGenre = os.path.join(PROCESSED_SPECT_DIR,genre)

            allFiles = os.listdir(rawGenre)
            
            # get random files
            randomRawFiles =   random.sample(allFiles, min(SAMPLES, len(allFiles))) # list, how many 


            fig, axs = plt.subplots(SAMPLES, 2, figsize=(10, SAMPLES * 2.5 +10))
            # fig.suptitle(f"Genre: {genre}", fontsize=16, y=1.02)
            for i, rawFile in enumerate(randomRawFiles):
                print(rawFile)
                numb = int(re.search(r'\d+', rawFile).group())
                processedFile= os.path.join(processedGenre,f"{genre}{numb}.png")
                
                img1 = Image.open(os.path.join(rawGenre, rawFile)).convert('L') # grayscale
                img2 = Image.open(processedFile).convert('L') 
     
                axs[i, 0].imshow(img1)
                axs[i, 0].set_title(f"{genre}, photo {numb}")
                axs[i, 0].axis('off')

                axs[i, 1].imshow(img2, cmap='magma', aspect='auto')
                axs[i, 1].set_title(f"{genre}, photo {numb}")
                axs[i, 1].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()
    

    def plot_confusion_matrix(self,cm, class_names, modelNumber, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

        os.makedirs(f"src/visualization/confusion_matrices/{model_type}/{evalType}", exist_ok=True)
        plt.savefig(f"src/visualization/confusion_matrices/{model_type}/{evalType}/{str(modelNumber.split('.')[0])}.png")
    

    def getAttentionLayer(self, avgAttentionDict):
        # Get global min and max for consistent color scale
        all_values = [val for mat in avgAttentionDict.values() for row in mat for val in row]
        vmin = min(all_values)
        vmax = max(all_values)

        genres = avgAttentionDict.keys()
        
        for genre in genres:
            plt.figure(figsize=(6, 5))
            heatmap = plt.imshow(avgAttentionDict[genre], cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)

            # Add colorbar as legend
            cbar = plt.colorbar(heatmap)
            cbar.set_label('Attention Intensity', rotation=270, labelpad=15)

            # Add title and clean axes
            plt.title(f'Average Attention Map for {genre} Genre')
            plt.axis('off')

            # Save the figure
            plt.savefig(f"{genre}Attention.png", dpi=300, bbox_inches='tight')
        

    def plot_loss_curve(history, modelNumber):
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f'{modelNumber}_losscurve.png')
    
    def plotEmbbeddingSpace(self,embeddings, labels, modelNumber,classNames):
        tsne = TSNE(n_components=2, random_state=0)
        embeddings_2d = tsne.fit_transform(embeddings)
        plt.figure(figsize=(8, 6))

        # Pick a colormap with enough distinct colors
        labels = np.argmax(labels, axis=1) # not one hot encoded
        genres = np.unique(labels)
        colorMap = plt.cm.get_cmap('tab10', len(genres))  # 'tab10' supports up to 10 distinct colors

        # Create a mapping: genre -> color
        genreColorMap = {genre: colorMap(i) for i, genre in enumerate(genres)}
        pointColors = [genreColorMap[label] for label in labels]

        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=labels, cmap='tab10', alpha=0.7)

        # Create the colorbar
        cbar = plt.colorbar(scatter, ticks=np.arange(len(genres)))
        cbar.ax.set_yticklabels(sorted(classNames))
        cbar.set_label('Genre')

        #scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=pointColors, cmap='viridis', alpha=0.5)
        plt.title('t-SNE Visualization of Embedding Space')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        #plt.colorbar(scatter, label='Genre')           
    
        os.makedirs(f"src/visualization/embbeddings/{model_type}/{evalType}", exist_ok=True)
        plt.savefig(f"src/visualization/embbeddings/{model_type}/{evalType}/{str(modelNumber.split('.')[0])}.png")
        
                


if __name__=="__main__":
    modelNum = 'trial'
    file = pd.read_csv(f'models/logs/predictionsResults/{modelNum}')
    trueClassIndices = file['True Genre'].tolist()
    predictedClassIndices = file['Predicted Genre'].tolist()
    cm = confusion_matrix(trueClassIndices, predictedClassIndices)
    classNames = sorted(np.unique(trueClassIndices))

    # obj = Visualization("compare")
    Visualization.plot_confusion_matrix(cm, classNames)