Tagging Music Sequences
==============================


## Table of Contents

1. [Project Overview](#project-overview)
2. [Data](#data)
3. [Model Architectures](#model-architectures)
4. [Results](#results)
5. [Discussion](#discussion)
6. [Getting Started](#getting-started)
7. [Authors](#authors)





### Project Overview
This project uses the GTZAN benchmark dataset to train and compare two deep‑learning models for music‑genre classification:

- **Baseline CNN** 
- **CNN with Spatial Attention** 

We augment the data to test robustness (shorter clips) and evaluate generalization.


### Data
We utilize the GTZAN dataset, which contains:

- **1 000 tracks** (approximately 30s each, 22 050 Hz sampling)  
- **10 genres** (100 tracks each):  
  blues, classical, country, disco, hip‑hop, jazz, metal, pop, reggae, rock  

**Preprocessing:**  
- Convert `.wav` → mel‑spectrogram (128 Mel bands, 2048‑sample window)  
- Normalize

**Data Augmentation:**  
- Pitch shift
- Add Gaussian noise
- Created 20 new samples per genre, yielding 200 additional spectrograms for training

*Edge Cases*
  - Crop audio (15s)
  - Created 10 new cropped audio per genre, yielding 100 new spectrograms for testing edge cases


### Model Architectures
We implemented two deep learning architectures for music genre classification using mel spectrograms as input:

**1. Baseline: Simple CNN**

Our initial model was a Convolutional Neural Network (CNN) trained on mel spectrograms extracted from the raw audio files with the following architecture:

- Conv2D(8, 3×3) → ReLU → MaxPool(2×2)
- Conv2D(16, 3×3) → ReLU → MaxPool(2×2)
- Conv2D(32, 3×3) → ReLU → MaxPool(2×2)
- Flatten()
- Dense(256) → ReLU → Dropout(0.25)
- Dense(10) → Softmax

**2. CNN with Convolutional Spatial Attention**

To improve upon the baseline, we extended the CNN architecture by adding an attention mechanism to the aforementioned structure:

- SpatialAttention():
- GlobalAvgPool + GlobalMaxPool along channel axis
- Concatenated → Conv2D(1, 7x7) → sigmoid mask
- Element-wise multiplication with input feature map
- Flatten()
- Dense(256) → ReLU → Dropout(0.25)
- Dense(10) → Softmax

### Results

*Base scores*

|                     | Accuracy  | F1-score | Precision |  Recall  |
|---------------------|-----------|----------|-----------|----------|
|    Baseline CNN     |    0.58   |   0.57   |   0.57    |   0.58   |
|  CNN with attention |    0.71   |   0.70   |   0.71    |   0.71   |


*Edge Cases*
|                     | Accuracy  | F1-score | Precision |  Recall  |
|---------------------|-----------|----------|-----------|----------|
|    Baseline CNN     |    0.15   |   0.06   |   0.04    |   0.15   |
|  CNN with attention |    0.11   |   0.03   |   0.11    |   0.11   |




### Discussion
Integrating an attention mechanism into our baseline CNN model led to a 12% improvement in overall test accuracy, indicating that attention helps the model focus on the most relevant parts of the spectrogram during classification. Despite this improvement, the model continued to struggle with accurately classifying certain genres, notably confusing disco with rock. This misclassification likely stems from the visual similarity in the spectrograms of these genres, as revealed by our analysis of attention maps. 

Both the baseline and attention-based models showed poor performance on edge cases, suggesting limited generalization to out-of-distribution or ambiguous examples in the test set. This limitation was further supported by the confusion matrices, which revealed a strong prediction bias. The baseline CNN predominantly predicted jazz and blues regardless of input. In addition, the attention-model also showed a bias towards blues almost exclusively. This suggests that although attention improved performance, the underlying model still lacks robustness and possibly suffers from insufficient feature diversity across genres.   

### Getting Started

1. Clone the github repo
```
git clone https://github.itu.dk/agku/Music-genre-classification.git
cd Music-Tagging-Sequences
```

2. Install requirements
```
pip install -r requirements.txt
```

3. Download the data
```
python src/data/make_dataset.py
```

4. Preproces the data
```
python src/features/build_features.py
```

5. Build and train model
```
python src/models/train_model.py
```

6. Predict 
```
python src/models/predict_model.py
```

7. Evaluate model

```
python src/models/evaluate_model.py
```

8. Test edge cases

```
python src/models/evaluate_edge_cases.py
```



### Authors
- **Agnieszka Kujawska** – [agku@itu.dk](mailto:agku@itu.dk)
- **Paula Menshikoff** – [pmen@itu.dk](mailto:pmen@itu.dk)
- **Sneha Shrestha** – [snsh@itu.dk](mailto:snsh@itu.dk)

--------
