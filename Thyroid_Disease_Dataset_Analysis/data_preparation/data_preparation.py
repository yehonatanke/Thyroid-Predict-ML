import numpy as np
import pandas as pd
from data_visualization import plot_condition_distribution, plot_feature_distribution, plot_facet_grid


def discretize_thyroid_data(data, discretize_flags):
    """
     Discretize selected attributes in thyroid data based on clinical knowledge,
     while ignoring missing values represented as '?' or NaN.

     :param data: pandas.DataFrame
         DataFrame containing the thyroid data.
     :param discretize_flags: dict
         A dictionary with attribute names as keys and boolean values indicating
         whether to discretize each attribute.

     :raises ValueError:
         If 'data' is not a pandas DataFrame or 'discretize_flags' is not a dictionary.
     :raises KeyError:
         If a key in 'discretize_flags' does not correspond to a column in 'data'.
     :raises TypeError:
         If the values in the columns are not appropriate for discretization (non-numeric where numeric is expected).

     :return: pandas.DataFrame
         A DataFrame with the discretized columns added, ignoring rows with missing data in the discretized columns.
     """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The 'data' parameter must be a pandas DataFrame.")

    if not isinstance(discretize_flags, dict):
        raise ValueError("The 'discretize_flags' parameter must be a dictionary.")

    # Convert '?' to NaN and ensure numeric columns can be coerced to float
    for column, discretize in discretize_flags.items():
        if discretize:
            data[column] = data[column].replace('?', np.nan)
            if data[column].dtype == object:
                try:
                    data[column] = pd.to_numeric(data[column])
                except ValueError:
                    raise TypeError(
                        f"The column '{column}' contains non-numeric data that cannot be converted to numeric type.")

    # Discretize 'age' if specified
    if discretize_flags.get('age', False):
        age_bins = list(range(0, 81, 10)) + [float('inf')]  # Bins from 0 to 80 and then 80+
        age_labels = [f'{i}-{i + 9}' for i in range(0, 80, 10)] + ['80+']
        data['age_binned'] = pd.cut(data['age'].dropna(), bins=age_bins, labels=age_labels, right=False)

    # Discretize 'TSH' if specified
    if discretize_flags.get('TSH', False):
        TSH_bins = [-float('inf'), 0.4, 4.0, float('inf')]
        TSH_labels = ['Low', 'Normal', 'High']
        data['TSH_binned'] = pd.cut(data['TSH'].dropna(), bins=TSH_bins, labels=TSH_labels)

    # Discretization for T3, TT4, T4U, FTI using quantiles
    for hormone in ['T3', 'TT4', 'T4U', 'FTI']:
        if discretize_flags.get(hormone, False):
            if data[hormone].dropna().empty:
                continue  # Skip if the column is empty after dropping NaNs
            data[f'{hormone}_binned'] = pd.qcut(data[hormone].dropna(), 3, labels=['Low', 'Medium', 'High'],
                                                duplicates='drop')

    return data


def discretize_attributes_v1(data, discretize_dict):
    """
    Discretizes selected attributes in the given DataFrame based on predefined intervals.

    :param data: pandas.DataFrame or None
        DataFrame containing the data to be discretized.
    :param discretize_dict: dict
        A dictionary containing boolean values for each attribute indicating whether to perform discretization.

    :return: pandas.DataFrame or None
        DataFrame with discretized attributes.

    Explanation of chosen intervals:
    - Age: Divided into age groups (0-20, 21-40, 41-60, 61+ years) to capture different life stages.
    - TSH (Thyroid Stimulating Hormone): Categorized into low, normal, and high levels (0-5, 5-10, >10 mIU/L)
      reflecting typical reference ranges for hypothyroidism and hyperthyroidism.
    - T3, TT4, T4U, FTI, TBG: Grouped into intervals reflecting clinically relevant thresholds
      (e.g., quartiles or percentiles of the distribution) to capture variations in thyroid hormone levels.

    Note:
        Adjustments to intervals may be necessary based on further analysis or domain expertise.
    """
    print("\nStarting Discretization Process.")
    # Check if input data is a pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input 'data' must be a pandas DataFrame.")

    # Check if discretize_dict is a dictionary
    if not isinstance(discretize_dict, dict):
        raise ValueError("Input 'discretize_dict' must be a dictionary.")

    discretized_data = data.copy()

    # Define intervals for discretization
    intervals = {
        'age': [0, 20, 40, 60, float('inf')],  # Age groups
        'TSH': [0, 5, 10, float('inf')],  # TSH levels (in mIU/L)
        'T3': [0, 2.5, 5, float('inf')],  # T3 levels (in nmol/L)
        'TT4': [0, 50, 100, 150, float('inf')],  # TT4 levels (in µg/dL)
        'T4U': [0, 0.8, 1.2, 1.6, float('inf')],  # T4U levels
        'FTI': [0, 50, 100, 150, float('inf')],  # FTI levels
        'TBG': [0, 20, 40, 60, float('inf')]  # TBG levels (in µg/dL)
    }

    for attribute, discretize in discretize_dict.items():
        # Check if attribute is present in the DataFrame
        if attribute not in discretized_data.columns:
            print(f"Warning: Attribute '{attribute}' not found in the DataFrame. Skipping discretization.")
            continue

        if discretize:
            if attribute in intervals:
                # Discretize the attribute using predefined intervals
                discretized_data[attribute] = pd.cut(discretized_data[attribute], bins=intervals[attribute],
                                                     labels=False, right=False)
                # Increment labels by 1 to start from 1 instead of 0
                discretized_data[attribute] += 1
            else:
                print(f"Warning: No intervals defined for '{attribute}'. Skipping discretization.")

    print("\nEnding Discretization Process.")
    return discretized_data


def standardize_sex_column(data, dataset_path=None):
    """
    Standardizes the 'sex' column in the DataFrame by replacing missing values ('?') with appropriate genders.

    :param data: pandas.DataFrame
        The DataFrame containing the 'sex' column.
    :param dataset_path: str or None, optional
        The path to save the modified DataFrame. If None, the DataFrame is not saved.

    :raises TypeError:
        If the input data is not a pandas DataFrame.
    :raises ValueError:
        If the DataFrame is empty or if the 'sex' column is not found.
    """
    # Check the input type
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    # Check if the DataFrame is empty
    if data.empty:
        raise ValueError("Input DataFrame is empty.")

    # Check if 'age' column exists
    if 'sex' not in data.columns:
        raise ValueError("Column 'sex' not found in the DataFrame.")

    # Count missing values initially (including '?')
    missing_count_before = (data['sex'].isnull() | (data['sex'] == '?')).sum()
    missing_percentage_before = (missing_count_before / len(data)) * 100

    # Change '?' to female where pregnant is marked
    data.loc[((data['sex'] == '?') | data['sex'].isnull()) & (data['pregnant'] == 't'), 'sex'] = 'F'

    # Count how many values were changed
    changed_count = missing_count_before - (data['sex'].isnull() | (data['sex'] == '?')).sum()
    changed_percentage = (changed_count / len(data)) * 100

    # Count missing values after changes
    missing_count_after = (data['sex'].isnull() | (data['sex'] == '?')).sum()
    missing_percentage_after = (missing_count_after / len(data)) * 100

    # Print the counts and percentages
    print("Total values:", len(data))
    print("Missing data before:", missing_count_before, f"({missing_percentage_before:.2f}%)")
    print("Data changed:", changed_count, f"({changed_percentage:.2f}%)")
    print("Missing data after:", missing_count_after, f"({missing_percentage_after:.2f}%)")

    # Save modified DataFrame to file if save_path is provided
    if dataset_path is not None:
        data.to_csv(dataset_path, index=False)

    # Change the rest of the missing 'sex' data
    process_missing_gender(data, dataset_path)


def process_missing_gender(data, dataset_path=None):
    """
   Processes missing gender data in the DataFrame by randomly assigning genders based on the existing male-to-female ratio.

   :param data: pandas.DataFrame
       The DataFrame containing thyroid data.
   :param dataset_path: str or None, optional
       The path to save the modified DataFrame. If None, the DataFrame is not saved.

   :raises TypeError:
       If the input data is not a pandas DataFrame.
   :raises ValueError:
       If the DataFrame is empty or if the 'sex' column is not found.

   :return: None
       Modifies the DataFrame in place by updating missing gender data and prints a summary of changes.
   """
    # Check the input type
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    # Check if the DataFrame is empty
    if data.empty:
        raise ValueError("Input DataFrame is empty.")

    # Check if 'age' column exists
    if 'sex' not in data.columns:
        raise ValueError("Column 'sex' not found in the DataFrame.")

    # Find missing gender data
    missing_gender = data[data['sex'].isnull() | (data['sex'] == '?')]

    # Count male and female data in the rest of the sample
    male_count_before = (data['sex'] == 'M').sum()
    female_count_before = (data['sex'] == 'F').sum()
    total_count_before = male_count_before + female_count_before

    # Calculate the ratio of male to female
    male_ratio = male_count_before / total_count_before
    female_ratio = 1 - male_ratio

    # Generate random genders based on the calculated ratio
    random_genders = np.random.choice(['M', 'F'], size=len(missing_gender), p=[male_ratio, female_ratio])

    # Update missing gender data with random genders
    missing_gender['sex'] = random_genders

    # Update the original data with the modified missing gender data
    data.update(missing_gender)

    # Count male and female data after changes
    male_count_after = (data['sex'] == 'M').sum()
    female_count_after = (data['sex'] == 'F').sum()
    total_count_after = male_count_after + female_count_after

    # Print before and after information
    print("\nBefore:")
    print("Male count:", male_count_before)
    print("Female count:", female_count_before)
    print("Total count:", total_count_before)
    print()
    print("After:")
    print("Male count:", male_count_after)
    print("Female count:", female_count_after)
    print("Total count:", total_count_after)

    # Save modified DataFrame to file if save_path is provided
    if dataset_path is not None:
        data.to_csv(dataset_path, index=False)

    # Print summary
    print("Number of missing gender data:", len(missing_gender))
    print("Randomly assigned genders based on the ratio of male to female.")


def remove_specific_diagnoses(data, codes_to_remove=['S', 'R']):
    """
    Removes rows from the DataFrame based on specific diagnosis codes, defaulting to 'S' and 'R'.

    :param data: pandas.DataFrame
        The DataFrame containing thyroid diagnosis data.
    :param codes_to_remove: list, optional
        List of diagnosis codes to remove from the DataFrame, defaults to ['S', 'R'].

    :return: pandas.DataFrame
        A DataFrame with the specified diagnosis codes removed.
    """

    # Check the input type
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    # Check if the data is empty
    if data.empty:
        raise ValueError("Input data is empty.")

    df_copy = data.copy()
    # Filter the data to exclude rows with the specified diagnosis codes
    df_copy = df_copy[~df_copy['diagnosis'].isin(codes_to_remove)]
    return df_copy


def classify_thyroid_conditions(df, plot_distribution=False):
    """
    Classifies thyroid conditions based on diagnosis codes in a pandas DataFrame,
    updates the 'diagnosis' column accordingly, and removes entries with unrecognized diagnosis codes.

    :param df: pandas.DataFrame
        The DataFrame containing thyroid diagnosis data.
    :param plot_distribution: bool, optional
        Whether to plot the distribution of classified conditions.

    :return: pandas.DataFrame
        A modified copy of the DataFrame with updated and valid diagnosis classifications.
    """
    condition_map = {
        '-': 'Healthy',
        'A': 'Hyperthyroid', 'B': 'Hyperthyroid', 'C': 'Hyperthyroid', 'D': 'Hyperthyroid',
        'E': 'Hypothyroid', 'F': 'Hypothyroid', 'G': 'Hypothyroid', 'H': 'Hypothyroid',
        'I': 'Healthy', 'J': 'Healthy', 'K': 'Healthy', 'L': 'Healthy', 'M': 'Hypothyroid',
        'N': 'Hyperthyroid', 'O': 'Hyperthyroid', 'P': 'Hyperthyroid', 'Q': 'Hyperthyroid',
        'R': 'Healthy', 'S': 'Healthy', 'T': 'Hyperthyroid'
    }

    df_copy = df.copy()

    if 'diagnosis' in df_copy.columns:
        invalid_entries = []
        for index, row in df_copy.iterrows():
            diagnosis_code = row['diagnosis'][0]
            if diagnosis_code in condition_map:
                df_copy.at[index, 'diagnosis'] = condition_map[diagnosis_code]
            else:
                invalid_entries.append(index)
                print(f"Deleting entry {index} with unrecognized diagnosis code: {row['diagnosis']}")

        df_copy = df_copy.drop(invalid_entries)

        print("Classification and cleanup successfully performed.")
        counts = df_copy['diagnosis'].value_counts()
        print("Counts of each condition:")
        print(counts)

        if plot_distribution:
            plot_condition_distribution(df_copy['diagnosis'].value_counts(), len(df_copy), 1.0)

        return df_copy
    else:
        print("The 'diagnosis' feature does not exist in the DataFrame.")
        return df


def discretize_features(df, show_plots=True):
    """
    Discretizes features in the DataFrame based on predefined intervals and labels.

    :param df: pandas.DataFrame
        The DataFrame containing thyroid data.
    :param show_plots: bool, optional
        Whether to display plots for each feature. Default is True.

    :return: pandas.DataFrame
        A copy of the DataFrame with discretized features added.

    Explanation of Bin Ranges:

    TSH (Thyroid-Stimulating Hormone):
        Low: Below 0.4 mIU/L, often associated with hyperthyroidism.
        Normal: 0.4 to 4.0 mIU/L, generally considered normal.
        High: Above 4.0 mIU/L, potentially indicative of hypothyroidism.
    T3 (Triiodothyronine):
        Low: Below 0.8 ng/mL, can indicate hypothyroidism.
        Normal: 0.8 to 2.0 ng/mL, within the normal range.
        High: Above 2.0 ng/mL to 4.0 ng/mL, suggestive of hyperthyroidism.
    TT4 (Total Thyroxine):
        Low: Below 50 mcg/dL, suggesting hypothyroidism.
        Normal: 50 to 120 mcg/dL, considered normal.
        High: Above 120 mcg/dL, may indicate hyperthyroidism.
    T4U (Thyroxine Uptake):
        Low: Below 0.7, indicating reduced uptake.
        Normal: 0.7 to 1.3, typically normal.
        High: Above 1.3, potentially high uptake.
    FTI (Free Thyroxine Index):
        Low: Below 70, often indicative of hypothyroidism.
        Normal: 70 to 130, considered normal.
        High: Above 130, may suggest hyperthyroidism.

    """

    feature_bins = {
        'age': [0, 10, 20, 30, 40, 50, 60, 70, 80, float('inf')],
        'TSH': [-float('inf'), 0.4, 4.0, float('inf')],  # Commonly used clinical cutoffs for TSH
        'T3': [0, 0.8, 2.0, 4.0],  # Adjusted upper range based on typical clinical standards
        'TT4': [0, 50, 120, 200],  # Adjusted for a broader clinical range
        'T4U': [0, 0.7, 1.3, 2.0],  # Adjusted to reflect a tighter control around the normal range
        'FTI': [0, 70, 130, 300]  # Adjusted to encapsulate a broader range of possible conditions
    }

    feature_labels = {
        'age': ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
        'TSH': ['Low', 'Normal', 'High'],
        'T3': ['Low', 'Normal', 'High'],
        'TT4': ['Low', 'Normal', 'High'],
        'T4U': ['Low', 'Normal', 'High'],
        'FTI': ['Low', 'Normal', 'High']
    }

    df_copy = df.replace('?', np.nan).dropna(subset=list(feature_bins.keys()))
    for feature, bins in feature_bins.items():
        if feature in df_copy.columns:
            labels = feature_labels.get(feature)
            if labels:
                # Create a Categorical type with order
                categorical_type = pd.CategoricalDtype(categories=labels, ordered=True)
                # Discretize the feature and ensure it uses the Categorical type
                df_copy[feature + '_Discrete'] = pd.cut(df_copy[feature].astype(float), bins=bins, labels=labels)
                df_copy[feature + '_Discrete'] = df_copy[feature + '_Discrete'].astype(categorical_type)

                if show_plots:
                    plot_feature_distribution(df_copy, feature, save_to_dir=False)
                    plot_facet_grid(df, feature, bins, labels, save_to_dir=False)
            else:
                print(f"Labels not provided for {feature}.")
        else:
            print(f"Feature {feature} not found in DataFrame.")
    return df_copy
