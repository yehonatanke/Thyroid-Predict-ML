<div align="center">
  <img src="https://img.shields.io/badge/language-Python-%233776AB.svg?logo=python">
  <img src="https://img.shields.io/badge/uses-Machine%20Learning-%232A2F3D.svg">
  <img src="https://custom-icon-badges.demolab.com/github/license/denvercoder1/custom-icon-badges?logo=law">
</div>


# <p align="center"> Thyroid Activity Classification and Prediction <br> Using Machine Learning </p>

## Table of Contents

1. [Overview](#overview)
2. [Introduction](#introduction)
    - [Dataset](#dataset)
    - [Methodological Framework](#methodological-framework)
        - [Data Preprocessing](#data-preprocessing)
3. [Research Objective](#research-objective)
4. [Repository Structure](#repository-structure)
    - [Process Overview](#process-overview)
        - [Classify Thyroid Conditions](#classify-thyroid-conditions)
        - [Discretization Visualizations](#discretization-visualizations)
5. [Data Analysis (data_analysis.py)](#data-analysis-dataanalysispy)
6. [Data Visualization (data_visualization.py)](#data-visualization-datavisualizationpy)
7. [Main Script (main.py)](#main-script-mainpy)
    - [Libraries](#libraries)
8. [How to Run](#how-to-run)
9. [Contributions](#contributions)
10. [License](#license)
11. [Conclusion](#conclusion)

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

## Repository Structure
```
/data_visualization  # Folder for the data visualization outputs
/dataset             # Folder for datasets used and generated by the scripts
/Thyroid_Disease_Dataset_Analysis # Main folder
    ├── data_preparation.py       # Script for data cleaning and preprocessing
    ├── data_analysis.py          # Script for data analysis 
    ├── data_visualization.py     # Script for visualizing the data and results
    ├── models.py                 # Trained model files
    ├── main.py                   # Main script that orchestrates the data processing and analysis
/README.md          
```

### Process Overview
The data preparation script is crucial for ensuring the quality and usability of the data for predictive modeling. Key steps include:

- **Data Cleaning**: Identifying and addressing missing values, outliers, and inconsistencies in the dataset.
- **Feature Engineering**: Transforming data through normalization and discretization to enhance model performance and interpretability.

#### Classify Thyroid Conditions
This visualization depicts the distribution of different thyroid conditions in the dataset. Understanding these distributions is crucial for building accurate predictive models and gaining insights into thyroid disorders.

<div style="display: flex; justify-content: center;">
    <img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/classify_thyroid_conditions.png" alt="classify thyroid conditions" width="300px" height="200px" style="margin-right: 10px;">
</div>


#### Discretization Visualizations
These visualizations illustrate the process of discretizing continuous variables, such as Free Thyroxine Index (FTI), Total T4 (TT4), and Thyroid Stimulating Hormone (TSH). Discretization helps in categorizing continuous variables into meaningful intervals, facilitating better analysis and interpretation of thyroid function.

<div style="display: flex; justify-content: center;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/discretization_visualization/FTI.png" alt="Missing Data" width="300px" height="200px" style="margin-right: 10px;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/discretization_visualization/TT4.png" alt="tt4" width="300px" height="200px" style="margin-right: 10px;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/discretization_visualization/T4U.png" alt="t4u" width="300px" height="200px" style="margin-right: 10px;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/discretization_visualization/FTI.png" alt="fti" width="300px" height="200px" style="margin-right: 10px;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/discretization_visualization/TSH.png" alt="Missing Data" width="300px" height="200px" style="margin-right: 10px;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/discretization_visualization/T3.png" alt="classify thyroid conditions" width="300px" height="200px" style="margin-right: 10px;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/discretization_visualization/age.png" alt="Missing Data" width="300px" height="200px" style="margin-right: 10px;">
</div>

## Data Analysis (data_analysis.py)
The main role of the script is to analyze, understand and locate problems in the data at the initial stage. Among other things, the actions of the script are:
- **Display various statistics**
- **Locate missing data**
- **Handle Missing Data:** Edit, if necessary, the missing information (delete, replace with average/median values, etc. according to the user's request)

<div style="display: flex; justify-content: center;">

<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/missing%20values%20visualization.png" alt="Missing Data" width="500px"  style="margin-right: 10px;">

<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/represent_data.png" alt="data rep" width="500px"  style="margin-right: 10px;">

</div>

## Data Visualization (data_visualization.py)
Data visualization is crucial for interpreting the data’s underlying patterns and the model's performance. This script generates:
- Plots that display the distribution of data points.
- Charts that illustrate the accuracy and effectiveness of the data preparation.

<div style="display: flex; justify-content: center;">

<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/Age%20Distribution%20by%20Gender%20(Percentage).png" alt="Age Distribution by Gender 1" width="400px" style="margin-right: 10px;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/Age%20Distribution%20by%20Gender%20(Frequency).png" alt="Age Distribution by Gender 2" width="400px" style="margin-right: 10px;">
<img src="https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/data%20visualization/discretization_visualization/facet_age.png" alt="Age Distribution 3" width="700px" style="margin-right: 10px;">
</div>


## Main Script (main.py)
The main.py script integrates all the stages from data preparation to visualization, ensuring a seamless workflow from raw data to actionable insights.

### Libraries
This project requires the following Python libraries:
- `pandas` and `numpy` for data manipulation.
- `matplotlib` and `seaborn` for data visualization.
- `scikit-learn` for machine learning models and evaluations.
- `tensorflow.keras` for building neural network models.

## How to Run
To run this project, ensure you have Python installed and execute the following command:
```bash
python main.py
```

## Contributions
Contributions to this project are welcome. You can contribute by improving the scripts, adding new models, or enhancing the visualization techniques.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/LICENSE) file for details.

## Conclusion

This research contributes to the field of medical informatics by demonstrating the application of machine learning in diagnosing and understanding thyroid disorders. Through detailed data analysis and the application of decision trees, the project provides a methodological framework for further research and development in the area of thyroid health diagnostics.
