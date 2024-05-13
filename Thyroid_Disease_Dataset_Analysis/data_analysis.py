import pandas as pd
import numpy as np
from data_visualization import visualize_age_by_gender_frequency, visualize_age_by_gender_percentage


def calculate_statistics(data):
    """
    Calculate statistics for numerical columns in a DataFrame.

    :param data: pandas.DataFrame
        Input DataFrame containing numerical data.
    :return: dict
        A dictionary containing statistics for each numerical column.
        Keys are column names, and values are dictionaries with statistics.
    :raises TypeError:
        If the input data is not a pandas DataFrame.
    :raises ValueError:
        If no numerical columns are found in the DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")

    # Check if DataFrame contains numerical columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        raise ValueError("No numerical columns found in the DataFrame")

    statistics = {}

    # Iterate over each column
    for column in numeric_columns:
        # Calculate statistics for numerical columns
        column_stats = {}
        column_stats["Average"] = data[column].mean()
        column_stats["Minimum value"] = data[column].min()
        column_stats["Maximum value"] = data[column].max()
        column_stats["Standard deviation"] = data[column].std()
        statistics[column] = column_stats

    return statistics


def analyze_data(data):
    """
    Analyze data by calculating statistics for numeric attributes and counts for categorical attributes.

    :param data: pandas.DataFrame
        Input DataFrame containing data to analyze.
    :return: dict
        A dictionary containing analysis results.
        Keys are attribute names followed by '_stats' for numeric attributes,
        and '_counts' for categorical attributes.
        Values are dictionaries containing either statistics (for numeric attributes)
        or value counts (for categorical attributes).
    """
    if data is None:
        print("Error: No data to analyze.")
        return None

    analysis_results = {}

    # Calculate average, min, max, and std for numeric attributes
    numeric_attributes = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    for attribute in numeric_attributes:
        if attribute in data.columns:
            avg = data[attribute].mean()
            min_val = data[attribute].min()
            max_val = data[attribute].max()
            std = data[attribute].std()
            print(f"{attribute.capitalize()}:")
            print(f"  Average: {avg:.2f}")
            print(f"  Minimum: {min_val:.2f}")
            print(f"  Maximum: {max_val:.2f}")
            print(f"  Standard Deviation: {std:.2f}")
            print()
            analysis_results[attribute + '_stats'] = {'average': avg, 'min': min_val, 'max': max_val, 'std': std}
        else:
            print(f"{attribute.capitalize()} column not found. Skipping...")
            print()

    # Calculate counts for categorical attributes
    categorical_attributes = ['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
                              'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
                              'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
                              'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured']
    for attribute in categorical_attributes:
        if attribute in data.columns:
            print(f"{attribute.capitalize()}:")
            print(data[attribute].value_counts())
            print()
            analysis_results[attribute + '_counts'] = data[attribute].value_counts()
        else:
            print(f"{attribute.capitalize()} column not found. Skipping...")
            print()

    # Count unique diagnoses
    if 'diagnosis' in data.columns:
        print("Diagnosis:")
        print(data['diagnosis'].value_counts())
        print()
        analysis_results['diagnosis_counts'] = data['diagnosis'].value_counts()
    else:
        print("Diagnosis column not found. Skipping...")
        print()

    return analysis_results


def age_analysis(data, dataset_path=None, change_above=None, change_below=None, to_value='median', visualize=False):
    """
    Analyzes and modifies age data in a DataFrame.

    :param data: pandas.DataFrame
        Input DataFrame containing age data.
    :param dataset_path: str or None, optional
        File path to save the modified DataFrame as CSV.
        If None, the original DataFrame is not saved. Default is None.
    :param change_above: int or None, optional
        Change ages above this value. Default is None.
    :param change_below: int or None, optional
        Change ages below this value. Default is None.
    :param to_value: {'median', 'mean'}, optional
        Value to change ages to. Default is 'median'.
    :param visualize: bool, optional
        Whether to visualize the data distribution with a histogram. Default is False.
    :raises TypeError:
        If the input data is not a pandas DataFrame.
    :raises ValueError:
        If the input DataFrame is empty, or if the 'age' column is not found.
    """
    # Check the input type
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    # Check if the DataFrame is empty
    if data.empty:
        raise ValueError("Input DataFrame is empty.")

    # Check if 'age' column exists
    if 'age' not in data.columns:
        raise ValueError("Column 'age' not found in the DataFrame.")

    # Convert age column to numeric
    data['age'] = pd.to_numeric(data['age'], errors='coerce')

    # Print analysis - before
    print("Age Values Before the Changes:")
    print("Median age:", data['age'].median())
    print("Average age:", data['age'].mean())
    print("Maximum age:", data['age'].max())
    print("Minimum age:", data['age'].min())

    # Apply changes based on user input
    if change_above is not None:
        data.loc[data['age'] > change_above, 'age'] = data['age'].median() if to_value == 'median' else data[
            'age'].mean()
    if change_below is not None:
        data.loc[data['age'] < change_below, 'age'] = data['age'].median() if to_value == 'median' else data[
            'age'].mean()

    # Save modified DataFrame to file if save_path is provided
    if dataset_path is not None:
        data.to_csv(dataset_path, index=False)

    # Print analysis - after
    print("\nAge Values After the Changes:")
    print("Median age:", data['age'].median())
    print("Average age:", data['age'].mean())
    print("Maximum age:", data['age'].max())
    print("Minimum age:", data['age'].min())

    # Plot stacked histogram
    if visualize:
        visualize_age_by_gender_frequency(data)
        visualize_age_by_gender_percentage(data)
