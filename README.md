# Breast Cancer Prediction using Random Forest Classifier

This project implements a Breast Cancer Prediction model using a Random Forest Classifier to predict whether a tumor is malignant or benign based on various features. The model is trained and evaluated on the Breast Cancer dataset from the UCI Machine Learning Repository.

## Project Description

This project aims to build a machine learning model that can predict breast cancer outcomes (malignant or benign) using data collected from breast cancer biopsies. The dataset consists of features like radius, texture, smoothness, and others, derived from digitized images of a breast mass.

The Random Forest Classifier is used due to its robustness and high performance in classification tasks. This model helps in making predictions on unseen data, providing a powerful tool for early detection of breast cancer.

## Features
- **Model Type**: Random Forest Classifier
- **Dataset**: Breast Cancer dataset (available from sklearn.datasets)
- **Evaluation Metrics**:
  - **Accuracy**: Measures the overall correctness of the model.
  - **Precision**: The proportion of positive predictions that are actually correct.
  - **Recall**: The proportion of actual positives correctly identified.
  - **F1-Score**: Harmonic mean of precision and recall, providing a balance between the two.

## Requirements

- Python 3.x
- Libraries:
  - sklearn
  - numpy
  - matplotlib
  - seaborn
  - pandas

You can install the necessary libraries using the following command:

```
pip install -r requirements.txt
```

**requirements.txt** file content:

```
scikit-learn==1.1.1
numpy==1.21.2
matplotlib==3.4.3
seaborn==0.11.2
pandas==1.3.3
```

## Setup and Usage

### 1. Install the required libraries
Run the following command to install all dependencies:

```
pip install -r requirements.txt
```

### 2. Training Process
- The script loads the Breast Cancer dataset.
- The dataset is split into training and test sets (80% for training, 20% for testing).
- A Random Forest Classifier is trained using the training set.
- Predictions are made on the test set and performance metrics (accuracy, precision, recall, F1-score) are displayed.

### 3. Visualization
- **Confusion Matrix**: Visualized using seaborn heatmaps to understand true vs. predicted classifications.
- **Feature Importance**: Displays which features were most important for the model's decision-making.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

### Contact

For any questions or issues, feel free to open an issue or contact me directly at **kathirvel15082k@gmail.com**.

---
