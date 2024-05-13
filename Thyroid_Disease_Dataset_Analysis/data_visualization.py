import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_condition_distribution(condition_series, total_count, proportion=1.0):
    """
    Plots a bar graph of the distribution of thyroid conditions with a scientific and professional appearance,
    including annotations on each bar for the number of subjects.

    Parameters:
        condition_series (pandas.Series): A series containing classified thyroid conditions.
        total_count (int): Total number of subjects in the data.
        proportion (float): Multiplier for setting the y-axis limit. Default is 1.0.

    Returns:
        None: Displays the plot directly.
    """
    plt.figure(figsize=(8, 5))
    sns.set(style="whitegrid")

    ax = sns.barplot(x=condition_series.index, y=condition_series.values, palette="coolwarm",
                     hue=condition_series.index, dodge=False)
    plt.title('Distribution of Thyroid Conditions Classification', fontsize=15)
    plt.xlabel('Condition')
    plt.ylabel('Number\nof Subjects')
    plt.xticks()

    # Set y-axis limit based on the proportion of total subjects
    ax.set_ylim(0, total_count * proportion)

    # Annotate each bar with its count
    for p in ax.patches:
        ax.annotate('{:d}'.format(int(p.get_height())),  # Formatting as integer
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points')

    # Add grid lines and a line showing the total count
    plt.axhline(y=total_count, color='gray', linestyle='--')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


def plot_non_numeric_distribution(data_path, save_to_dir=False):
    # Load data into pandas DataFrame
    df = pd.read_csv(data_path)

    # Specify columns of interest
    non_numeric_data = ['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
                        'sick', 'pregnant', 'thyroid_surgery', 'query_hypothyroid', 'query_hyperthyroid',
                        'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
                        ]

    # Set seaborn style
    sns.set_style("darkgrid")

    # Plot distribution for each specified column
    for col in non_numeric_data:
        distribution = df[col].value_counts()
        plt.figure(figsize=(8, 5))
        sns.barplot(x=distribution.index, y=distribution.values, hue=distribution.index, palette="Blues",
                    color='blue', dodge=False)
        plt.title(f'{col}', fontsize=15)
        plt.xlabel(col)
        plt.ylabel('Frequency')

        # Add numbers above each bar
        for i, count in enumerate(distribution):
            plt.text(i, count, str(count), ha='center', va='bottom')

        plt.tight_layout()
        if save_to_dir:
            # Save the plot
            plt.savefig(f"{dir_to_data}/distribution_{col}.png", bbox_inches="tight")
            plt.close()  # Close the plot to avoid displaying it
        else:
            plt.show()


# def plot_non_numeric_distribution(data_path):
#     # Load data into pandas DataFrame
#     df = pd.read_csv(data_path)
#
#     # Plot sick distribution
#     sick_distribution = df['sick'].value_counts()
#     ax = sick_distribution.plot(kind='bar', figsize=(8, 6))
#     plt.title('Distribution of Sick')
#     plt.xlabel('Sick')
#     plt.ylabel('Frequency')
#
#     # Add numbers above each bar
#     for i, count in enumerate(sick_distribution):
#         plt.text(i, count, str(count), ha='center', va='bottom')
#
#     plt.show()

def plot_non_numeric_distribution_v2(data_path):
    # Read data into pandas DataFrame
    df = pd.read_csv(data_path)

    # Filter out non-nominal and missing attributes
    nominal_cols = df.select_dtypes(include=['object']).columns
    df_nominal = df[nominal_cols].replace('?', pd.NA).dropna(axis=1, how='all')

    # Plot distribution for each nominal attribute
    for col in df_nominal.columns:
        plt.figure(figsize=(8, 6))
        counts = df_nominal[col].value_counts()
        counts.plot(kind='bar', color='skyblue')
        for i, count in enumerate(counts):
            plt.text(i, count + 0.1, str(count), ha='center')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()


# def plot_non_numeric_distribution(file_path):
#     # Load the dataset
#     data = pd.read_csv(file_path)
#
#     # Preprocess the data: replace '?' with NaN for clearer handling of missing values
#     data = data.replace('?', pd.NA)
#
#     # Identify non-numeric columns
#     non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
#
#     # Check if there are non-numeric columns to plot
#     if not non_numeric_columns:
#         print("No non-numeric columns to plot.")
#         return
#
#     # Set up the visualization
#     sns.set(style='whitegrid')
#
#     # Plotting distributions for each non-numeric column in separate figures
#     for col in non_numeric_columns:
#         plt.figure(figsize=(10, 6))
#         ax = sns.countplot(y=col, data=data, order=data[col].value_counts().index)
#         plt.title(f'Distribution of {col}', fontsize=15)
#         plt.xlabel('Count')
#         plt.ylabel(col)
#
#         # Annotate each bar with the count
#         for p in ax.patches:
#             ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2),
#                         xytext=(5, 0), textcoords='offset points', ha='left', va='center')
#
#         plt.show()

def plot_data_distribution(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Preprocess the data: replace '?' with NaN for cleaner handling of missing values
    data = data.replace('?', pd.NA)

    # Attempt to convert all columns to numeric, coercing errors (this will leave non-convertible columns unchanged)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Identify numeric columns after conversion
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    # Check if there are numeric columns to plot
    if not numeric_columns:
        print("No numeric columns to plot.")
        return

    # Set up the visualization
    sns.set(style='whitegrid')

    # Plotting distributions for each numeric column in a separate figure
    for col in numeric_columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data[col].dropna(), kde=True, color='blue')  # Using dropna() to avoid issues with NaNs
        plt.title(f'Distribution of {col}', fontsize=15)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()


def plot_data_distribution_v5(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Preprocess the data: replace '?' with NaN for a cleaner handling of missing values
    data = data.replace('?', pd.NA)

    numeric_columns = ['TSH', 'T3', 'TT4', 'T4U', 'FTI']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Set up the visualization
    sns.set(style='whitegrid')

    # Plotting distributions for numeric columns
    fig, axes = plt.subplots(len(numeric_columns), 1, figsize=(8, 5 * len(numeric_columns)))

    for i, col in enumerate(numeric_columns):
        sns.histplot(data[col], kde=True, ax=axes[i], color='blue')
        axes[i].set_title(f'Distribution of {col}', fontsize=15)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def plot_discretization(data, discretized_data, attributes):
    """
    Plot and compare the distributions of attributes before and after discretization,
    with custom handling for specific attributes.

    Parameters:
    - data (pandas.DataFrame): Original DataFrame.
    - discretized_data (pandas.DataFrame): DataFrame with discretized attributes.
    - attributes (list of str): List of attribute names to be visualized, treated differently based on their nature.
    """
    for attribute in attributes:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

        ax1, ax2 = axes

        # Handle non-numeric entries
        clean_data = pd.to_numeric(data[attribute].replace('?', np.nan), errors='coerce')

        if attribute == 'age':
            # Direct histogram for 'age'
            ax1.hist(clean_data.dropna(), bins=30, color='blue', alpha=0.7)
            ax1.set_xlabel('Age')
            ax1.set_title('Original Age')
        else:
            # Handle other attributes with potential transformations or different binning
            if clean_data.dropna().empty:
                print(f"No data available for {attribute}")
                continue

            # Check skewness to decide on log transformation
            if clean_data.skew() > 2 and clean_data.min() > 0:
                transformed_data = np.log(clean_data.dropna())
                ax1.hist(transformed_data, bins=30, color='blue', alpha=0.7)
                ax1.set_xlabel(f'Log({attribute})')
            else:
                ax1.hist(clean_data.dropna(), bins=30, color='blue', alpha=0.7)
                ax1.set_xlabel(attribute)

        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Original {attribute}')
        ax1.tick_params(axis='x', rotation=45, labelsize=10)

        # Plot discretized data
        if f'{attribute}_binned' in discretized_data.columns:
            discretized_counts = discretized_data[f'{attribute}_binned'].value_counts().sort_index()
            discretized_counts.plot(kind='bar', ax=ax2, color='green', alpha=0.7)
            ax2.set_title(f'Discretized {attribute}')
            ax2.set_xlabel(f'{attribute} (binned)')
            ax2.set_ylabel('Frequency')
            ax2.tick_params(axis='x', rotation=45, labelsize=10)

        plt.tight_layout()
        plt.show()


def plot_discretization_v1(data, discretized_data, attributes):
    """
    Plot and compare the distributions of attributes before and after discretization.

    Parameters:
    - data (pandas.DataFrame): Original DataFrame.
    - discretized_data (pandas.DataFrame): DataFrame with discretized attributes.
    - attributes (list of str): List of attribute names to be visualized.

    This function creates a two-panel plot for each attribute: the left panel shows the original
    distribution, and the right panel shows the discretized distribution.
    """
    num_attributes = len(attributes)
    fig, axes = plt.subplots(nrows=num_attributes, ncols=2, figsize=(12, 6 * num_attributes))

    for i, attribute in enumerate(attributes):
        ax1 = axes[i, 0] if num_attributes > 1 else axes[0]
        ax2 = axes[i, 1] if num_attributes > 1 else axes[1]

        # Plot original data
        if pd.api.types.is_numeric_dtype(data[attribute]):
            ax1.hist(data[attribute].dropna(), bins='auto', color='blue', alpha=0.7)
            ax1.set_yscale('log')  # Use log scale if data is skewed
        else:
            data[attribute].value_counts().plot(kind='bar', ax=ax1, color='blue', alpha=0.7)
        ax1.set_title(f'Original {attribute}')
        ax1.set_xlabel(attribute)
        ax1.set_ylabel('Frequency')
        ax1.tick_params(axis='x', rotation=45)  # Rotate labels for better readability

        # Plot discretized data
        if f'{attribute}_binned' in discretized_data.columns:
            discretized_counts = discretized_data[f'{attribute}_binned'].value_counts().sort_index()
            discretized_counts.plot(kind='bar', ax=ax2, color='green', alpha=0.7)
            ax2.set_title(f'Discretized {attribute}')
            ax2.set_xlabel(f'{attribute} (binned)')
            ax2.set_ylabel('Frequency')
            ax2.tick_params(axis='x', rotation=45)  # Rotate labels for better readability

    plt.tight_layout()
    plt.show()


def visualize_discretization_v1(original_data, discretized_data, discretize_dict):
    """
    Visualizes the original and discretized distributions of specified attributes.

    Parameters:
    - original_data: pandas DataFrame containing the original data.
    - discretized_data: pandas DataFrame containing the discretized data.
    - discretize_dict: dictionary indicating which attributes to visualize.

    Returns:
    - None (displays the plots for each attribute).
    """
    for attribute, should_discretize in discretize_dict.items():
        if should_discretize:
            # Check if input data is a pandas DataFrame
            if not isinstance(original_data, pd.DataFrame) or not isinstance(discretized_data, pd.DataFrame):
                raise ValueError("Input 'original_data' and 'discretized_data' must be pandas DataFrames.")

            # Check if the specified attribute is present in the DataFrame
            if attribute not in original_data.columns or attribute not in discretized_data.columns:
                raise ValueError(f"Attribute '{attribute}' not found in the DataFrame.")

            # Plot histograms of the original and discretized data side by side
            plt.figure(figsize=(12, 6))

            # Plot histogram of the original data
            plt.subplot(1, 2, 1)
            plt.hist(original_data[attribute].dropna(), bins='auto', color='skyblue', edgecolor='black', alpha=0.7)
            plt.title(f'Original {attribute} Distribution')
            plt.xlabel(attribute)
            plt.ylabel('Frequency')

            # Plot histogram of the discretized data
            plt.subplot(1, 2, 2)
            plt.hist(discretized_data[attribute].dropna(), bins='auto', color='lightgreen', edgecolor='black',
                     alpha=0.7)
            plt.title(f'Discretized {attribute} Distribution')
            plt.xlabel(attribute)
            plt.ylabel('Frequency')

            plt.tight_layout()
            plt.show()


def plot_population_pyramid(data_file):
    """
    Generates a population pyramid plot based on age and gender distribution from a CSV file.

    :param data_file: str
        The path to the CSV file containing the data.
    :return: None
        Displays the population pyramid plot.
    """
    # Read CSV file into a DataFrame
    data = pd.read_csv(data_file)

    # Define age bins
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 200]
    labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']

    # Create 'age_group' column
    data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)

    # Group data by age_group and sex
    grouped = data.groupby(['age_group', 'sex']).size().unstack()

    # Plot population pyramid
    fig, ax = plt.subplots()

    # Female bars
    ax.barh(grouped.index, grouped['F'], color='salmon', label='Female', height=0.5)

    # Male bars (reversed direction)
    ax.barh(grouped.index, -grouped['M'], color='skyblue', label='Male', height=0.5)

    # Set labels and title
    ax.set_xlabel('Number of Individuals')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title('Population Pyramid by Age and Gender')

    # Center the x-axis at zero
    ax.set_xlim(-max(grouped.max()), max(grouped.max()))

    # Remove y-axis line
    ax.spines['left'].set_visible(False)

    # Remove y-axis ticks
    ax.tick_params(left=False)

    # Remove x-axis
    ax.xaxis.set_visible(False)

    # Remove box around plot
    ax.set_frame_on(False)

    # Add legend
    ax.legend()

    plt.show()


def visualize_data(data, analysis_results):
    """
    Visualize the analysis results.

    :param data: pandas.DataFrame
        The original DataFrame containing the data.
    :param analysis_results: dict
        A dictionary containing analysis results.
        Should include 'diagnosis_counts' key with value as a pandas.Series containing diagnosis counts.
    """
    # Plot histogram of ages
    plt.figure(figsize=(8, 6))
    plt.hist(data['age'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Ages')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Plot bar chart of diagnosis counts
    plt.figure(figsize=(10, 6))
    analysis_results['diagnosis_counts'].plot(kind='bar', color='orange')
    plt.title('Diagnosis Counts')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def visualize_age_by_gender_frequency2(data):
    """
    Visualizes the age distribution by gender using separate bars for each gender with custom age ranges.

    :param data: pandas.DataFrame
        Input DataFrame containing age and gender data.
    :raises ValueError:
        If the input DataFrame is empty or if the 'age' or 'sex' column is not found.
    """
    # Check if DataFrame is empty
    if data.empty:
        raise ValueError("Input DataFrame is empty.")

    # Check if 'age' and 'sex' columns exist
    if 'age' not in data.columns or 'sex' not in data.columns:
        raise ValueError("Columns 'age' and 'sex' must be present in the DataFrame.")

    # Convert age column to numeric
    data['age'] = pd.to_numeric(data['age'], errors='coerce')

    # Define custom age ranges
    age_ranges = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, float('inf')]
    labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
              '60-64', '65-69', '70-74', '75-79', '80+']

    # Categorize age data into custom ranges
    data['Age Range'] = pd.cut(data['age'], bins=age_ranges, labels=labels, right=False)

    # Filter data by gender
    male_data = data[data['sex'] == 'M']
    female_data = data[data['sex'] == 'F']

    # Set plot style
    plt.style.use('ggplot')

    # Plot separate bars for each gender
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for males
    ax.bar(labels, male_data['Age Range'].value_counts().sort_index(), color='blue', label='Male')

    # Plot bars for females
    ax.bar(labels, female_data['Age Range'].value_counts().sort_index(), color='pink', label='Female',
           bottom=male_data['Age Range'].value_counts().sort_index())

    ax.set_xlabel('Age Range', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Age Distribution by Gender', fontsize=16)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()


def sex_frequency_comparison_visualization(data):
    """
    Visualizes the sex distribution with bars stacked on top of each other.

    :param data: pandas.DataFrame
        Input DataFrame containing sex data.
    :raises ValueError:
        If the input DataFrame is empty or if the 'sex' column is not found.
    """
    # Check if DataFrame is empty
    if data.empty:
        raise ValueError("Input DataFrame is empty.")

    # Check if 'sex' column exists
    if 'sex' not in data.columns:
        raise ValueError("Column 'sex' must be present in the DataFrame.")

    # Count the number of men and women
    gender_counts = data['sex'].value_counts()
    men_count = gender_counts.get('M', 0)
    women_count = gender_counts.get('F', 0)

    # Set plot style
    plt.style.use('ggplot')

    # Plot stacked bars
    labels = ['Male', 'Female']
    men_values = [men_count, 0]
    women_values = [0, women_count]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(labels, men_values, color='grey', label='Male')
    ax.bar(labels, women_values, bottom=men_values, color='c', label='Female')

    ax.set_xlabel('Sex', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Sex Distribution', fontsize=16)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()


def gender_compare_visualization(data):
    gender_counts = data['sex'].value_counts()
    men_count = gender_counts.get('M', 0)
    women_count = gender_counts.get('F', 0)
    labels = ['Men', 'Women']
    men_values = [men_count, 0]
    women_values = [0, women_count]

    plt.bar(labels, men_values, color='blue', label='Men')
    plt.bar(labels, women_values, bottom=men_values, color='pink', label='Women')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.title('Gender Distribution')
    plt.legend()
    plt.show()


def visualize_age_by_gender_frequency55(data):
    """
    Visualizes the age distribution by gender using a stacked bar chart with custom age ranges.

    :param data: pandas.DataFrame
        Input DataFrame containing age and gender data.
    :raises ValueError:
        If the input DataFrame is empty or if the 'age' or 'sex' column is not found.
    """
    # Check if DataFrame is empty
    if data.empty:
        raise ValueError("Input DataFrame is empty.")

    # Check if 'age' and 'sex' columns exist
    if 'age' not in data.columns or 'sex' not in data.columns:
        raise ValueError("Columns 'age' and 'sex' must be present in the DataFrame.")

    # Convert age column to numeric
    data['age'] = pd.to_numeric(data['age'], errors='coerce')

    # Define custom age ranges
    age_ranges = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, float('inf')]
    labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
              '60-64', '65-69', '70-74', '75-79', '80+']

    # Categorize age data into custom ranges
    data['Age Range'] = pd.cut(data['age'], bins=age_ranges, labels=labels, right=False)

    # Filter data by gender
    male_data = data[data['sex'] == 'M']
    female_data = data[data['sex'] == 'F']

    # Set plot style
    plt.style.use('ggplot')

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot stacked bars
    bar_width = 0.35
    index = range(len(labels))
    ax.bar(index, male_data.groupby('Age Range').size(), bar_width, color='blue', label='Male')
    ax.bar(index, female_data.groupby('Age Range').size(), bar_width, color='pink', label='Female',
           bottom=male_data.groupby('Age Range').size())

    ax.set_xlabel('Age Range', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Age Distribution by Gender', fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(fontsize=12)

    # Get the maximum y-value
    max_y_value = max(ax.get_yticks())

    # Set y-axis limit slightly above the maximum y-value
    ax.set_ylim(0, max_y_value * 1.1)

    plt.tight_layout()
    plt.show()


def visualize_age_by_gender_frequency_np(data):
    """
    Visualizes the age distribution by gender using a stacked histogram with custom age ranges.

    :param data: pandas.DataFrame
        Input DataFrame containing age and gender data.
    :raises ValueError:
        If the input DataFrame is empty or if the 'age' or 'sex' column is not found.
    """
    # Check if DataFrame is empty
    if data.empty:
        raise ValueError("Input DataFrame is empty.")

    # Check if 'age' and 'sex' columns exist
    if 'age' not in data.columns or 'sex' not in data.columns:
        raise ValueError("Columns 'age' and 'sex' must be present in the DataFrame.")

    # Convert age column to numeric
    data['age'] = pd.to_numeric(data['age'], errors='coerce')

    # Define custom age ranges
    age_ranges = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, float('inf')]
    labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
              '60-64', '65-69', '70-74', '75-79', '80+']

    # Filter data by gender
    male_data = data[data['sex'] == 'M']
    female_data = data[data['sex'] == 'F']

    # Set plot style
    plt.style.use('ggplot')

    # Calculate the width of each bar
    bar_width = 0.4

    # Plot stacked histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot stacked bars
    ax.hist([male_data['age'], female_data['age']], bins=age_ranges, color=['blue', 'pink'],
            edgecolor='black', label=['Male', 'Female'], stacked=True, width=bar_width)

    ax.set_xlabel('Age', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Age Distribution by Gender', fontsize=16)

    # Adjust x-axis positions for the bars
    x = np.arange(len(labels))
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    ax.legend(fontsize=12)

    # Get the maximum y-value
    max_y_value = max(ax.get_yticks())

    # Set y-axis limit slightly above the maximum y-value
    ax.set_ylim(0, max_y_value * 1.1)

    plt.tight_layout()
    plt.show()


# def visualize_age_by_gender_frequency_np_add_column(data):
#     """
#     Visualizes the age distribution by gender using a stacked histogram with custom age ranges.
#
#     :param data: pandas.DataFrame
#         Input DataFrame containing age and gender data.
#     :raises ValueError:
#         If the input DataFrame is empty or if the 'age' or 'sex' column is not found.
#     """
#     # Check if DataFrame is empty
#     if data.empty:
#         raise ValueError("Input DataFrame is empty.")
#
#     # Check if 'age' and 'sex' columns exist
#     if 'age' not in data.columns or 'sex' not in data.columns:
#         raise ValueError("Columns 'age' and 'sex' must be present in the DataFrame.")
#
#     # Convert age column to numeric
#     data['age'] = pd.to_numeric(data['age'], errors='coerce')
#
#     # Define custom age ranges
#     age_ranges = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, float('inf')]
#     labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
#               '60-64', '65-69', '70-74', '75-79', '80+']
#
#     # Categorize age data into custom ranges
#     data['Age Range'] = pd.cut(data['age'], bins=age_ranges, labels=labels, right=False)
#
#     # Filter data by gender
#     male_data = data[data['sex'] == 'M']
#     female_data = data[data['sex'] == 'F']
#
#     # Set plot style
#     plt.style.use('ggplot')
#
#     # Calculate the width of each bar
#     bar_width = 0.4
#
#     # Plot stacked histogram
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     # Plot stacked bars
#     ax.hist([male_data['Age Range'], female_data['Age Range']], bins=len(labels), range=(0, len(labels)),
#             color=['blue', 'pink'], edgecolor='black', label=['Male', 'Female'], stacked=True, width=bar_width)
#
#     ax.set_xlabel('Age Range', fontsize=14)
#     ax.set_ylabel('Frequency', fontsize=14)
#     ax.set_title('Age Distribution by Gender', fontsize=16)
#
#     # Adjust x-axis positions for the bars
#     x = np.arange(len(labels))
#     ax.set_xticks(x + bar_width / 2)
#     ax.set_xticklabels(labels, rotation=45, ha='right')
#
#     ax.legend(fontsize=12)
#
#     # Get the maximum y-value
#     max_y_value = max(ax.get_yticks())
#
#     # Set y-axis limit slightly above the maximum y-value
#     ax.set_ylim(0, max_y_value * 1.1)
#
#     plt.tight_layout()
#     plt.show()


def visualize_age_by_gender_frequency_v2(data):
    """
    Visualizes the age distribution by gender using a stacked histogram with custom age ranges.

    :param data: pandas.DataFrame
        Input DataFrame containing age and gender data.
    :raises ValueError:
        If the input DataFrame is empty or if the 'age' or 'sex' column is not found.
    """
    # Check if DataFrame is empty
    if data.empty:
        raise ValueError("Input DataFrame is empty.")

    # Check if 'age' and 'sex' columns exist
    if 'age' not in data.columns or 'sex' not in data.columns:
        raise ValueError("Columns 'age' and 'sex' must be present in the DataFrame.")

    # Convert age column to numeric
    data['age'] = pd.to_numeric(data['age'], errors='coerce')

    # Define custom age ranges
    age_ranges = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, float('inf')]
    labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
              '60-64', '65-69', '70-74', '75-79', '80+']

    # Categorize age data into custom ranges
    data['Age Range'] = pd.cut(data['age'], bins=age_ranges, labels=labels, right=False)

    # Filter data by gender
    male_data = data[data['sex'] == 'M']
    female_data = data[data['sex'] == 'F']

    # Set plot style
    plt.style.use('ggplot')

    # Plot stacked histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot stacked bars
    ax.hist([male_data['Age Range'], female_data['Age Range']], bins=len(labels), range=(0, len(labels)),
            color=['blue', 'pink'], edgecolor='black', label=['Male', 'Female'], stacked=True)

    ax.set_xlabel('Age Range', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Age Distribution by Gender', fontsize=16)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(fontsize=12)

    # Get the maximum y-value
    max_y_value = max(ax.get_yticks())

    # Set y-axis limit slightly above the maximum y-value
    ax.set_ylim(0, max_y_value * 1.1)

    plt.tight_layout()
    plt.show()


def visualize_age_by_gender_frequency(data):
    """
    Visualizes the age distribution by gender using a stacked histogram with custom age ranges.

    :param data: pandas.DataFrame
        Input DataFrame containing age and gender data.
    :raises ValueError:
        If the input DataFrame is empty or if the 'age' or 'sex' column is not found.
    """
    # Check if DataFrame is empty
    if data.empty:
        raise ValueError("Input DataFrame is empty.")

    # Check if 'age' and 'sex' columns exist
    if 'age' not in data.columns or 'sex' not in data.columns:
        raise ValueError("Columns 'age' and 'sex' must be present in the DataFrame.")

    # Convert age column to numeric
    data['age'] = pd.to_numeric(data['age'], errors='coerce')

    # Define custom age ranges
    age_ranges = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, float('inf')]
    labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
              '60-64', '65-69', '70-74', '75-79', '80+']

    # Categorize age data into custom ranges
    data['Age Range'] = pd.cut(data['age'], bins=age_ranges, labels=labels, right=False)

    # Filter data by gender
    male_data = data[data['sex'] == 'M']
    female_data = data[data['sex'] == 'F']

    # Calculate maximum y-axis value
    max_y = max(male_data['Age Range'].value_counts().max(), female_data['Age Range'].value_counts().max())

    # Set plot style
    plt.style.use('ggplot')

    # Set up plot
    plt.figure(figsize=(10, 6))
    plt.hist([male_data['Age Range'], female_data['Age Range']], bins=len(labels), range=(0, len(labels)),
             color=['#1f77b4', '#ff7f0e'], edgecolor='black', label=['Male', 'Female'], stacked=True)

    # Customize plot
    plt.xlabel('Age Range', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Age Distribution by Gender (Frequency)', fontsize=14)
    plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)

    # Calculate the maximum percentage value
    # max_percentage = total_percentage.max()

    plt.ylim(0, max_y)

    # Customize y-axis ticks with jumps of 5
    plt.yticks(range(0, int(max_y) * 3, 200))
    # Add horizontal grid lines
    for i in range(100, int(max_y) * 3, 100):
        plt.axhline(y=i, color='gray', linestyle='--', linewidth=0.1)

    # Show plot
    plt.tight_layout()
    plt.show()


def visualize_age_by_gender_percentage(data):
    """
    Visualizes the age distribution by gender using a stacked histogram with custom age ranges, normalized by percentage.

    :param data: pandas.DataFrame
        Input DataFrame containing age and gender data.
    :raises ValueError:
        If the input DataFrame is empty, or if the 'age' or 'sex' column is not found.
    """
    # Check if the DataFrame is empty
    if data.empty:
        raise ValueError("Input DataFrame is empty.")

    # Check if 'age' and 'sex' columns exist
    if 'age' not in data.columns or 'sex' not in data.columns:
        raise ValueError("Columns 'age' and 'sex' must be present in the DataFrame.")

    # Convert age column to numeric
    data['age'] = pd.to_numeric(data['age'], errors='coerce')

    # Define custom age ranges
    age_ranges = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, float('inf')]
    labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
              '60-64', '65-69', '70-74', '75-79', '80+']

    # Categorize age data into custom ranges
    data['Age Range'] = pd.cut(data['age'], bins=age_ranges, labels=labels, right=False)

    # Group data by gender and age range
    grouped_data = data.groupby(['sex', 'Age Range'], observed=True).size().unstack(fill_value=0)

    # Calculate percentage
    total_counts = grouped_data.sum(axis=1)
    grouped_data_percentage = grouped_data.divide(total_counts, axis=0) * 100

    # Set plot style
    plt.style.use('ggplot')

    # Plot stacked histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate total percentage for each age range
    total_percentage = grouped_data_percentage.loc['M'] + grouped_data_percentage.loc['F']

    # Plot stacked bars
    ax.bar(labels, grouped_data_percentage.loc['M'], color='grey', label='Male')
    ax.bar(labels, grouped_data_percentage.loc['F'], color='c', label='Female',
           bottom=grouped_data_percentage.loc['M'])

    ax.set_xlabel('Age Range', fontsize=14)
    ax.set_ylabel('Percentage', fontsize=14)
    ax.set_title('Age Distribution by Gender (Percentage)', fontsize=16)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(fontsize=12)

    # Calculate the maximum percentage value
    max_percentage = total_percentage.max()

    # Set y-axis limit slightly above the maximum percentage value
    ax.set_ylim(0, min(max_percentage * 3, 100))

    # Customize y-axis ticks with jumps of 5
    ax.set_yticks(range(0, int(max_percentage) * 3, 5))

    # Add horizontal grid lines
    for i in range(5, 100, 10):
        ax.axhline(y=i, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


def plot_missing_data(data_path):
    """
    Visualizes the percentage of missing data for each column using a bar plot.

    :param data_path: str
        File path to the CSV file containing the data.
    :raises ValueError:
        If the input DataFrame is empty.
    """
    # Load data
    data = pd.read_csv(data_path)

    # Calculate missing data percentage for each column
    missing_data = data.isnull().mean() * 100

    # Check if the DataFrame is empty
    if data.empty:
        raise ValueError("Input DataFrame is empty.")

    # Print the percentage of missing data for each nominal variable
    nominal_vars = data.select_dtypes(include='object').columns
    print("Percentage of Missing Data for Nominal Variables:")

    for var, percentage in missing_data.items():
        print(f"{var}: {percentage:.2f}%")

    # Set plot style
    plt.style.use('ggplot')

    # Plot the missing data
    plt.figure(figsize=(10, 6))
    bars = plt.bar(missing_data.index, missing_data.values, color='c')

    # Add the percentage of missing data above each bar
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f'{height:.2f}%', ha='center', va='bottom',
                     fontsize=8)

    # Add the number of missing data for each column
    for i, val in enumerate(missing_data):
        if val > 0:
            plt.text(i, val + 5, f'{data.isnull().sum().iloc[i]}', ha='center', va='bottom', fontsize=8, color='black')

    # Customize labels and title
    plt.title('Missing Data (Percentage)', fontsize=16)
    plt.ylabel('Missing Data (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=10)
    plt.ylim(0, 110)

    # Add horizontal grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_combined_distribution(df, feature, bins, labels):
    # Convert feature to float and discretize
    df['Value_Float'] = pd.to_numeric(df[feature], errors='coerce')
    df['Discrete'] = pd.cut(df['Value_Float'], bins=bins, labels=labels, include_lowest=True)

    # Drop NaN values for plotting
    df_dropna = df.dropna(subset=['Value_Float', 'Discrete'])

    # Creating a violin plot for the continuous data
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Discrete', y='Value_Float', data=df_dropna, scale='width', inner=None, palette='coolwarm')

    # Overlay discrete data points
    sns.stripplot(x='Discrete', y='Value_Float', data=df_dropna, color='k', alpha=0.5, jitter=True)

    plt.title(f'Combined Distribution and Discretization of {feature}')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.xticks()
    plt.show()


def plot_feature_distribution_v2(df, feature):
    # Set plot style
    plt.style.use('ggplot')

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    df[feature].astype(float).plot(kind='hist', bins=20, title=f'{feature} Distribution', color='c')
    plt.subplot(1, 2, 2)
    # Plot using the ordered Categorical type
    df[feature + '_Discrete'].value_counts().sort_index().plot(kind='bar', title=f'{feature} Discretized', color='c')
    plt.show()


def plot_feature_distribution(df, feature, save_to_dir=False):
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")  # Clean and modern plot background

    # Create a figure with specified size and two subplots
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram for the continuous feature distribution
    sns.histplot(df[feature].dropna().astype(float), bins=20, ax=ax[0], color='skyblue', edgecolor='black')
    ax[0].set_title(f'{feature} Distribution', fontsize=14)
    ax[0].set_xlabel(f'{feature} Value', fontsize=12)
    ax[0].set_ylabel('Frequency', fontsize=12)

    # Bar plot for the discretized feature distribution
    order = df[feature + '_Discrete'].cat.categories  # Ensuring the categories are in defined order
    sns.countplot(x=df[feature + '_Discrete'], ax=ax[1], order=order, palette='coolwarm', edgecolor='black')
    ax[1].set_title(f'{feature} Discretized', fontsize=14)
    ax[1].set_xlabel(f'{feature} Category', fontsize=12)
    ax[1].set_ylabel('Count', fontsize=12)
    ax[1].tick_params(axis='x')

    # Improve layout
    plt.tight_layout()

    if save_to_dir:
        # Save the plot
        plt.savefig(f"/path/to/dir/discretization_visualization/{feature}.png", bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_facet_grid(df, feature, bins, labels, save_to_dir=False):
    # Convert and categorize feature
    df['Value_Float'] = pd.to_numeric(df[feature], errors='coerce')
    df['Category'] = pd.cut(df['Value_Float'], bins=bins, labels=labels, include_lowest=True)

    # Create a facet grid of histograms
    g = sns.FacetGrid(df, col='Category', col_wrap=4, height=3)
    g.map(plt.hist, 'Value_Float', bins=15, color='skyblue', edgecolor='black')

    # Add titles and labels
    g.fig.suptitle(f'Histograms of {feature} by Category', y=1.05)
    g.set_axis_labels(f'{feature} Value', 'Count')
    g.set_titles("{col_name}")

    if save_to_dir:
        # Save the plot
        plt.savefig(f"/path/to/dir/discretization_visualization/facet_{feature}.png", bbox_inches="tight")
        plt.close()
    else:
        plt.show()
