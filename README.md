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
8. [Libraries](#libraries)
9. [How to Run](#how-to-run)
10. [Contributions](#contributions)
11. [License](#license)
12. [Conclusion](#conclusion)

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
/Thyroid Predict ML               # Main folder
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

## Libraries
This project requires the following Python libraries:
- `pandas` and `numpy` for data manipulation.
- `matplotlib` and `seaborn` for data visualization.
- `scikit-learn` for machine learning models and evaluations.
- `tensorflow.keras` for building neural network models.

## How to Run

1. **Clone the Repository**: Begin by cloning or downloading the repository containing the program files to your local machine.

2. **Install Dependencies**: Ensure you have all the necessary dependencies installed. You may need to install packages such as pandas, matplotlib, and numpy if you haven't already. You can typically install these dependencies using pip:

    ```bash
    pip install pandas matplotlib numpy scikit-learn tensorflow
    ```

3. **Set Up Data**: Place your dataset in a location accessible to the program. Update the `dataset_path` variable in the `main()` function to point to the location of your dataset.

4. **Define Workflow**: Customize the workflow dictionary in the `main()` function to specify which tasks you want to execute. Set the value of each key to `True` or `False` based on your requirements.

5. **Execute the Program**: Run the `main()` function to start the program. Depending on the tasks enabled in the workflow, the program will process, analyze, and visualize the data accordingly.

6. **Review Results**: Once the program completes execution, review the results generated by each enabled task. This may include plots, statistics, preprocessed data, analysis insights, and any edited files.

7. **Adjust Workflow**: If necessary, modify the workflow settings or edit the code to suit your specific needs. You can rerun the program with different configurations as needed.

8. **Repeat as Needed**: You can run the program multiple times with different datasets or configurations to analyze various datasets or perform different tasks.

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

Users can customize the workflow according to their specific requirements by toggling the tasks on or off. This modular approach allows for flexibility in data processing and analysis.

## Contributions
Contributions to this project are welcome. You can contribute by improving the scripts, adding new models, or enhancing the visualization techniques.

## License
This project is licensed under the [MIT License](https://github.com/yehonatanke/Thyroid-Predict-ML/blob/main/LICENSE).

## Conclusion

This research contributes to the field of medical informatics by demonstrating the application of machine learning in diagnosing and understanding thyroid disorders. Through detailed data analysis and the application of decision trees, the project provides a methodological framework for further research and development in the area of thyroid health diagnostics.
