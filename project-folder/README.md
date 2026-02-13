# ❤️ Heart Disease Classification - ML Assignment 2

## a. Problem Statement

The goal of this project is to predict the presence of heart disease in a patient based on various medical attributes. This is a **binary classification** problem where the target variable indicates whether a patient has heart disease (1) or not (0). Six different machine learning classification models are implemented, evaluated, and compared to determine which model performs best on this dataset.

## b. Dataset Description

- **Dataset:** UCI Heart Disease Dataset
- **Source:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Total Instances:** 1025 (302 after removing duplicates)
- **Number of Features:** 13
- **Target Variable:** `target` (0 = No Heart Disease, 1 = Heart Disease)
- **Target Distribution:** 164 (Disease) | 138 (No Disease) — after deduplication

### Feature Description

| Feature    | Description                                          |
|------------|------------------------------------------------------|
| age        | Age of the patient (in years)                        |
| sex        | Sex (1 = Male, 0 = Female)                           |
| cp         | Chest pain type (0-3)                                |
| trestbps   | Resting blood pressure (mm Hg)                       |
| chol       | Serum cholesterol (mg/dl)                            |
| fbs        | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)|
| restecg    | Resting ECG results (0-2)                            |
| thalach    | Maximum heart rate achieved                          |
| exang      | Exercise induced angina (1 = Yes, 0 = No)            |
| oldpeak    | ST depression induced by exercise relative to rest   |
| slope      | Slope of the peak exercise ST segment (0-2)          |
| ca         | Number of major vessels colored by fluoroscopy (0-4) |
| thal       | Thalassemia (0-3)                                    |
| target     | 0 = No Disease, 1 = Heart Disease                    |

## c. Models Used

### Comparison Table

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|--------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression      | 0.8033   | 0.8712 | 0.8000    | 0.8485 | 0.8235 | 0.6031 |
| Decision Tree            | 0.8033   | 0.8019 | 0.8182    | 0.8182 | 0.8182 | 0.6039 |
| KNN                      | 0.7869   | 0.8377 | 0.7778    | 0.8485 | 0.8116 | 0.5702 |
| Naive Bayes              | 0.7869   | 0.8842 | 0.8333    | 0.7576 | 0.7937 | 0.5771 |
| Random Forest (Ensemble) | 0.7541   | 0.8588 | 0.7647    | 0.7879 | 0.7761 | 0.5038 |
| XGBoost (Ensemble)       | 0.7213   | 0.8323 | 0.7353    | 0.7576 | 0.7463 | 0.4376 |

### Model Observations

| ML Model Name            | Observation about model performance |
|--------------------------|-------------------------------------|
| Logistic Regression      | Logistic Regression achieved the highest F1 Score (0.8235) and the second-highest AUC (0.8712), making it one of the best overall performers. It provides a strong balance between precision and recall. Being a simple linear model, it generalizes well on this relatively small dataset without overfitting. Its high recall (0.8485) means it correctly identifies most patients with heart disease, which is critical in a medical diagnosis scenario. |
| Decision Tree            | Decision Tree matched Logistic Regression in accuracy (0.8033) and achieved the highest MCC (0.6039), indicating a well-balanced prediction across both classes. It provides equal precision and recall (0.8182), showing no bias toward either class. However, its AUC (0.8019) is the lowest among all models, suggesting its probability estimates are less reliable for ranking predictions. Decision Trees are prone to overfitting, which may limit generalization on unseen data. |
| KNN                      | KNN achieved a high recall (0.8485) equal to Logistic Regression, meaning it is effective at detecting heart disease cases. However, its precision (0.7778) is lower, resulting in more false positives compared to other models. The AUC of 0.8377 indicates decent discriminative ability. KNN's performance is sensitive to the choice of k and feature scaling, and it may struggle with larger or noisier datasets. |
| Naive Bayes              | Naive Bayes achieved the highest AUC (0.8842) among all models, indicating excellent ability to distinguish between the two classes based on predicted probabilities. It has the highest precision (0.8333), meaning when it predicts heart disease, it is most likely correct. However, its recall (0.7576) is the lowest, meaning it misses more actual disease cases. The strong AUC but lower recall suggests the default decision threshold may not be optimal for this model. |
| Random Forest (Ensemble) | Random Forest, despite being an ensemble method, performed below expectations with an accuracy of 0.7541. This could be due to the small dataset size (302 samples after deduplication), which limits the benefit of ensemble averaging. Its AUC (0.8588) is relatively strong, suggesting good probability calibration. The model shows balanced but moderate precision (0.7647) and recall (0.7879). With hyperparameter tuning and more training data, Random Forest could potentially outperform simpler models. |
| XGBoost (Ensemble)       | XGBoost had the lowest performance across most metrics — accuracy (0.7213), F1 Score (0.7463), and MCC (0.4376). This is likely due to the very small training set (241 samples), as XGBoost is designed for larger datasets where its gradient boosting approach can learn complex patterns. On small datasets, it tends to overfit the training data and generalize poorly. Both precision (0.7353) and recall (0.7576) are the lowest among all models. With larger datasets and proper hyperparameter tuning, XGBoost typically outperforms other models. |

## Project Structure

```
project-folder/
├── app.py                  # Streamlit web application
├── heart.csv               # Dataset
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── model/
    └── train_models.py     # Model training script
```

## How to Run Locally

1. Clone the repository:
   git clone https://github.com/hitman449/pratyush_machine_learning_assignment_2
   cd pratyush_machine_learning_assignment_2/project-folder


2. Install dependencies:
   pip install -r requirements.txt

3. Run the training script:
   cd model
   python train_models.py

4. Run the Streamlit app:
   streamlit run app.py

## Live Streamlit App

https://pratyushmachinelearningassignment2-cknmxtshveupytznyjgscc.streamlit.app/

## GitHub Repository

https://github.com/hitman449/pratyush_machine_learning_assignment_2/blob/main/project-folder

## Tech Stack

- Python 3.10
- Streamlit
- Scikit-learn
- XGBoost
- Pandas, NumPy
- Matplotlib, Seaborn