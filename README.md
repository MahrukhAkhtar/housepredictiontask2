# DevelopersHub-AIML-Internship

## Multimodal House Price Prediction (Tabular + Image Fusion)

### Project Overview
This project is part of the AI/ML Engineering Internship at DevelopersHub Corporation. It implements a state-of-the-art **Multimodal Deep Learning Architecture** using PyTorch to predict house sale prices. The model dynamically extracts visual features from house images and merges them with standardized tabular specifications (numeric & categorical data) to output a single continuous price prediction.

### DataSet
**Multimodal House Dataset**
* **Tabular Data:** Contains house specifications (numerical features processed via `SimpleImputer` and `StandardScaler`, and categorical features processed via `OneHotEncoder`).
* **Target Variable (`SalePrice`):** The continuous actual price of each house.
* **Image Data:** Color photos of corresponding houses (`image_id` mapped to matching physical images). 
* **Data Integration:** Automatic alignment verifies that only records containing both valid CSV specs and physical matching image files are processed.

### Tools & Libraries Used
* **Python** (Core Development)
* **PyTorch** (Neural Network Architecture & Custom Datasets)
* **Torchvision** (MobileNetV2 Pre-trained Weights & Image Augmentations)
* **scikit-learn** (Pipeline Preprocessing & Regressor Evaluation)
* **Matplotlib & Pillow** (Inference, Custom Visuals & Plots)
* **joblib** (Preprocessing Pipeline Export)

### Deep Learning Architecture
The model fuses two distinct feature extraction networks:

1. **Visual Branch (CNN):** 
   * Pre-trained **MobileNetV2** is utilized as a backbone feature extractor (frozen to preserve pre-trained ImageNet weights).
   * Feature map outputs are globally pooled and flattened to pass through a fully-connected layer (`Linear -> ReLU -> Dropout`).
2. **Tabular Branch (FCNN):** 
   * Tabular specs are transformed via scikit-learn's `ColumnTransformer` pipeline.
   * Features pass through a Multi-Layer Perceptron (MLP) mapping layer.
3. **Late Fusion Head:** 
   * Sub-features from both branches are concatenated (`128 + 64 = 192` dimensional vector) and fed to a final continuous Regression Head to predict the house's monetary value (`SalePrice`).

### Visualizations & Saved Assets
The training pipeline outputs valuable performance graphs and serialized pipeline states:

* **Saved Pipeline:** Tabular transformers are saved as `saved_model/tabular_pipeline.pkl`.
* **Saved Model:** The PyTorch model weights with the lowest validation loss are saved as `saved_model/multimodal_house_price_model.pth`.
* **Loss & MAE Curves:** Metric plots showing training vs. validation Mean Squared Error (MSE) and Mean Absolute Error (MAE) across epochs.
* **Predicted vs Actual Scatter Plot:** Displays regression alignment to check the quality of predictions.
* **Residual Error Plot:** Monitors error dispersion and model biases across price ranges.

### Key Features
* **Early Stopping Support:** Built-in validation patience tracker stops the process if the loss does not improve over a set number of epochs.
* **Reproducibility:** A customized `set_seed()` system freezes environment seeds across python, numpy, and PyTorch (including CuDNN deterministic settings).
