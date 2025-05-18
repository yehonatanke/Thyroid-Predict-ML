<div align="center">
  <img src="https://img.shields.io/badge/language-Python-%233776AB.svg?logo=python">
  <img src="https://img.shields.io/badge/uses-Machine%20Learning-%232A2F3D.svg">
</div>

# <p align="center"> Thyroid Activity Classification and Prediction <br> Using Machine Learning </p>

[Results Overview](#results-overview)

## Overview
This repository is part of a research initiative designed to leverage machine learning techniques for predicting thyroid gland activity. By analyzing clinical data, the project aims to classify thyroid conditions into hyperthyroidism, hypothyroidism, or normal functioning (euthyroid), thereby assisting in the early diagnosis and treatment of thyroid disorders.

## Introduction
This project aims to apply advanced data mining techniques to predict thyroid gland activity, focusing on classifying conditions as hyperthyroidism, hypothyroidism, or normal (euthyroid) states. Using a comprehensive dataset from clinical sources, the project leverages machine learning to enhance diagnostic accuracies, contributing to better patient outcomes in endocrine health.

### Dataset
The analysis utilized a dataset comprising 9,172 instances, each encoded with 30 clinically relevant attributes including demographic data, hormonal levels, and patient-reported symptoms. Prior to modeling, the dataset underwent extensive preprocessing to ensure quality and consistency, addressing challenges such as missing values and outlier effects.

### Methodological Framework

#### Data Preprocessing
- **Cleaning**: Removal of outliers and imputation of missing values to preserve data integrity.
- **Transformation**: Normalization and discretization techniques were applied to standardize the range of continuous variables and categorize them for effective analysis.

## Research Objective
The primary goal of this project is to develop predictive models that can accurately determine the activity of the thyroid gland based on clinical parameters. Utilizing decision tree algorithms, such as CART and C4.5, the project seeks to establish robust models that not only predict but also offer insights into the factors influencing thyroid conditions.

### Process Overview
The data preparation script is crucial for ensuring the quality and usability of the data for predictive modeling. Key steps include:

- **Data Cleaning**: Identifying and addressing missing values, outliers, and inconsistencies in the dataset.
- **Feature Engineering**: Transforming data through normalization and discretization to enhance model performance and interpretability.

#### Classify Thyroid Conditions
This describes the distribution of different thyroid conditions in the dataset.

<div style="display: flex; justify-content: center;">
    <img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/classify_thyroid_conditions.png" alt="classify thyroid conditions" width="300px" height="200px" style="margin-right: 10px;">
</div>

#### Discretization 
This illustrate the process of discretizing continuous variables, such as Free Thyroxine Index (FTI), Total T4 (TT4), and Thyroid Stimulating Hormone (TSH). 

<div style="display: flex; justify-content: center;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/discretization_visualization/FTI.png" alt="Missing Data" width="300px" height="200px" style="margin-right: 10px;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/discretization_visualization/TT4.png" alt="tt4" width="300px" height="200px" style="margin-right: 10px;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/discretization_visualization/T4U.png" alt="t4u" width="300px" height="200px" style="margin-right: 10px;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/discretization_visualization/FTI.png" alt="fti" width="300px" height="200px" style="margin-right: 10px;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/discretization_visualization/TSH.png" alt="Missing Data" width="300px" height="200px" style="margin-right: 10px;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/discretization_visualization/T3.png" alt="classify thyroid conditions" width="300px" height="200px" style="margin-right: 10px;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/discretization_visualization/age.png" alt="Missing Data" width="300px" height="200px" style="margin-right: 10px;">
</div>

#### Data Analysis
The main role of the script is to analyze, understand and locate problems in the data at the initial stage. 

<div style="display: flex; justify-content: center;">

<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/missing%20values%20visualization.png" alt="Missing Data" width="500px"  style="margin-right: 10px;">

<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/represent_data.png" alt="data rep" width="500px"  style="margin-right: 10px;">

</div>

#### Data Visualization
This module is responsible for data visualization to interpret the underlying patterns of the data and model performance. We want to understand the distribution of the data and produce charts that illustrate the accuracy and efficiency of the data preparation.

<div style="display: flex; justify-content: center;">

<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/Age%20Distribution%20by%20Gender%20(Percentage).png" alt="Age Distribution by Gender 1" width="400px" style="margin-right: 10px;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/Age%20Distribution%20by%20Gender%20(Frequency).png" alt="Age Distribution by Gender 2" width="400px" style="margin-right: 10px;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/discretization_visualization/facet_age.png" alt="Age Distribution 3" width="700px" style="margin-right: 10px;">
</div>

#### Workflow Explanation

The program follows a structured workflow to process, analyze, and visualize data. Here's an overview of each step:

1. **Loading Data**: The dataset is loaded from a specified file path.

2. **Setting Workflow**: Users can define a workflow dictionary where each key represents a specific task, such as plotting, preprocessing, analysis, or visualization. By setting the value of each key to `True` or `False`, users can activate or deactivate individual tasks as needed.

3. **Data Processing**:
    - *Graphic Representation*: If enabled, the program plots data distribution and non-numeric distribution.
    - *Missing Data Plotting*: If enabled, missing data is plotted for visualization.
    - *Modify Sex Column*: Standardizes 'sex' column values if enabled.
    - *Remove S and R*: Removes specific diagnoses ('S' and 'R') if enabled.
    - *Age Handling*: Analyzes and potentially modifies age data if enabled.

4. **Statistical Analysis**:
    - *Statistics Handling*: Calculates descriptive statistics if enabled.

5. **Preprocessing**:
    - *Preprocess*: Removes specified columns and types if enabled.

6. **Data Analysis**:
    - *Analysis*: Conducts data analysis, generating insights and statistics if enabled.

7. **Visualization**:
    - *Visualize*: Visualizes preprocessed data and analysis results if enabled.

8. **Feature Engineering**:
    - *Discretization*: Discretizes features, converting continuous variables to categorical if enabled.
    - *Classify Conditions*: Classifies thyroid conditions based on discretized features if enabled.

9. **File Editing**:
    - *Edit File*: Edits data files by removing specified columns and renaming columns if enabled.

Users can customize the workflow according to their specific requirements by toggling the tasks on or off.

## <p align="center"> Results Overview </p>

### Project Overview
This project focused on developing and evaluating machine learning models to classify thyroid gland activity, particularly identifying states of overactivity (hyperthyroidism) and underactivity (hypothyroidism) compared to normal gland function. Using a dataset processed and cleaned from the initial 5,984 instances with 22 attributes related to thyroid function, we applied two decision tree algorithms: CART (Classification and Regression Trees) and C4.5.

### Methodology
The methods chosen were CART and C4.5 due to their robustness in handling both categorical and continuous data, as well as their ability to generate comprehensible models. CART utilizes the Gini index to optimize splits, aiming for the highest homogeneity within nodes, while C4.5 uses information gain based on entropy, which considers both the purity of the node and the intrinsic information of a split, and includes a pruning step to reduce overfitting.

### Note
Both models were subjected to 10-fold cross-validation to ensure that the evaluation was robust and the model generalizable.

### Analyses Results

#### CART Results
- **Tree Complexity**: The CART tree has 37 leaf nodes and a total size of 73.
- **Performance Metrics**:
  - **Accuracy**: 89.8563% of instances correctly classified.
  - **Kappa Statistic**: 0.5455, indicating moderate agreement.
  - **ROC Area**: Overall ROC Area values are decent, indicating good classification ability across the classes.
  - **Class-Specific Performance**:
    - **Healthy**: High true positive rate (TPR) and precision, suggesting effective identification.
    - **Hyperthyroid**: Low recall but reasonable precision, indicating struggles with identifying all hyperthyroid cases but reliability when it does.
    - **Hypothyroid**: Good recall, showing effectiveness in identifying most hypothyroid cases, though with some errors.
    
#### C4.5 Results
- **Tree Complexity**: The C4.5 tree has 67 leaves and a total size of 106.
- **Performance Metrics**:
  - **Accuracy**: 90.2741% of instances correctly classified.
  - **Kappa Statistic**: 0.5801, which is slightly better than CART, suggesting a moderate to good agreement.
  - **ROC Area**: Slightly better than CART, especially in distinguishing between classes, which could indicate a more nuanced decision-making process.
  - **Class-Specific Performance**:
    - **Healthy**: Very similar performance to CART, strong in identifying healthy cases.
    - **Hyperthyroid**: Better recall than CART, suggesting improvements in identifying hyperthyroid cases.
    - **Hypothyroid**: Comparable recall to CART, maintaining good identification rates.

### Estimation of the Degree of Accuracy of Each Method

The accuracy of each method as computed during the analysis:
- **CART**: 89.8563%
- **C4.5**: 90.2741%

The slight advantage in accuracy by C4.5 can be attributed to its sophisticated handling of attribute selection and pruning which tends to avoid overfitting better than CART.

### Comparative Analysis and Conclusions

#### Comparative Analysis
- **Accuracy**: C4.5 slightly outperforms CART in overall accuracy and kappa statistic, indicating a better balance between sensitivity and specificity.
- **Complexity**: C4.5 has a larger tree, which might suggest a more complex model than CART. This complexity could be a factor in its slightly better performance but might also indicate a higher risk of overfitting despite the pruning process.
- **ROC and Precision-Recall (PRC) Areas**: C4.5 shows better ROC and PRC areas, suggesting it is more capable of distinguishing between the classes than CART.

### Conclusions
Both algorithms performed admirably with overall accuracies close to 90%, indicating strong predictive capabilities. C4.5 slightly outperformed CART in terms of accuracy and the kappa statistic, suggesting better consistency and reliability, likely due to its pruning mechanism which effectively reduces the complexity of the model and helps mitigate overfitting.

### Comparison with Existing Research
The classification of thyroid disorders using machine learning is well-documented in the literature, with various studies highlighting the potential of algorithms like decision trees due to their interpretability and effectiveness. Studies often emphasize the importance of feature selection and data quality, which were critical aspects of our project as well.

Our results align with these findings, demonstrating that decision trees can effectively distinguish between different states of thyroid activity, with performance metrics that are competitive with current standards. Furthermore, the relative success of C4.5 in this project corroborates research suggesting that methods which account for both the quality of splits and the complexity of the model (through mechanisms like pruning) tend to perform better, especially in datasets with a mix of attribute types and a substantial number of instances.

### Conclusion and Recommendations
The project confirms the applicability of decision tree models for the classification of thyroid gland activity. Given the slightly superior performance of C4.5, it is recommended for similar tasks in clinical settings where interpretability and accuracy are crucial. However, future work could explore the following to further enhance model performance and application:

- **Feature Engineering**: More sophisticated feature engineering might uncover relationships that are not immediately apparent but could significantly enhance model accuracy.
- **Ensemble Methods**: Combining multiple models in an ensemble method, such as Random Forests or boosting techniques, could provide improvements in predictive performance and robustness.
- **Advanced Algorithms**: Investigating other advanced machine learning algorithms that can handle complex interactions and non-linear relationships may also prove beneficial.
