This project aims to predict rainfall using a hybrid deep learning model that combines meteorological (meteo) data and cloud imagery data. The goal is to classify rainfall into four categories:

1.No Rain
2.Little Rain
3.Medium Rain
4.Tons of Rain

We will use an RNN (or Transformer) to process the sequential meteo data and a CNN to extract features from cloud images. The model will predict rainfall 1, 2, and 3 days in advance, focusing on daytime hours (since nighttime lacks visual satellite data).
Data Preparation
1. Cloud Imagery Data Processing
Source: Extract small cloud images from the CSF file.
Challenge: No useful data at night (visual satellite channel), so skip nighttime for imagery.
Preprocessing: Normalize images, resize if needed.
2. Meteorological Data Processing
Source: Time-series meteorological readings.
Challenge: Deciding whether to skip nighttime (optional).
Preprocessing: Standardize features, handle missing values.
3. Creating Time-Sequenced Data
Sequence Length: Use 3-day sequences as input.
Time-Series Dataset API:
Use tf.keras.preprocessing.timeseries_dataset_from_array to efficiently create sequences without redundant data.
Target Label:
Aggregate daily rainfall and classify into four categories.
4. Handling Imbalanced Data
Problem: Most data points are "No Rain" (Class 0), leading to misleading accuracy metrics.
Solution:
Weighted Loss Function to emphasize rain events.
Oversampling/Undersampling to balance classes.
F1-score instead of just accuracy.
5. Test Dataset Strategy
Key Rule: The test set should only contain future data, not random historical points.
Model Architecture
1. CNN for Cloud Images
Extracts spatial patterns from cloud formations.
Architecture: Conv2D layers → BatchNorm → MaxPooling → Flatten → Dense.
2. RNN/Transformer for Meteo Data
Captures temporal dependencies in meteorological trends.
Options:
LSTM/GRU: Suitable for sequential data.
Transformer: More powerful but needs large data.
3. Fusion of CNN and RNN/Transformer Outputs
Combine extracted features from both networks.
Fusion Techniques:
Concatenation: Merge CNN and RNN outputs before the final classification layer.
Attention Mechanism: Give dynamic importance to CNN or RNN features depending on relevance.
Gating Mechanism: A learned mechanism to weigh CNN vs. RNN impact.
4. Output Layer
Softmax activation to classify rainfall amount.
Model Training & Evaluation
Loss Function: Categorical Cross-Entropy (weighted for class imbalance).
Metrics:
Class-wise Accuracy: Focus on rain classes (not just "No Rain").
Confusion Matrix: Understand misclassifications.
Precision-Recall Curve: More insightful than accuracy.
