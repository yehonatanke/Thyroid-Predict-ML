from data_preparation.data_preparation import discretize_attributes_v1, discretize_thyroid_data, standardize_sex_column, \
    classify_thyroid_conditions, remove_specific_diagnoses, discretize_features
from data_processing import load_data, preprocess_data, open_data, preprocess_csv
from data_analysis import analyze_data, calculate_statistics, age_analysis
from data_visualization import visualize_data, plot_missing_data, plot_discretization, plot_data_distribution, \
    plot_non_numeric_distribution


def main():
    """
    Orchestrates data processing, analysis, and visualization tasks following a defined workflow.

    The workflow includes:
    - Loading data from a specified file
    - Conducting statistical analysis
    - Preprocessing the data
    - Performing analysis
    - Visualizing results
    - Facilitating dataset modifications


    Parameters:
        None

    :return
        None

    Functions Activated:
    - graphic_representation:   Visualizes data distribution and non-numeric distribution if enabled.
    - missing_data_plotting:    Plots missing data if enabled.
    - modify_sex_column:        Standardizes 'sex' column values if enabled.
    - remove_s_and_r:           Removes specific diagnoses ('S' and 'R') if enabled.
    - age_handling:             Analyzes and potentially modifies age data if enabled.
    - statistics_handling:      Calculates descriptive statistics if enabled.
    - preprocess:               Preprocesses data by removing specified columns and types if enabled.
    - analysis:                 Conducts data analysis, generating insights and statistics if enabled.
    - visualize:                Visualizes preprocessed data and analysis results if enabled.
    - discretization:           Discretizes features, converting continuous variables to categorical if enabled.
    - classify_conditions:      Classifies thyroid conditions based on discretized features if enabled.
    - save_classified_df:       Saves the classified DataFrame to a file if enabled.
    - edit_file:                Edits data files by removing specified columns and renaming columns if enabled.
    """
    # The dataset path
    dataset_path = "/path/to/data"

    # Choose what functions to activate
    workflow = {
        "graphic_representation": False,
        "age_handling": False,
        "statistics_handling": False,
        "visualize": False,
        "preprocess": False,
        "missing_data_plotting": False,
        "modify_sex_column": False,
        "analysis": False,
        "discretization": False,
        "classify_conditions": False,
        "save_classified_df": False,
        "remove_s_and_r": False,
        "edit_file": True
    }
    # Load data
    original_data = open_data(dataset_path)
    data = original_data.copy()  # Make a copy to work with, preserving the original data

    if workflow["graphic_representation"]:
        plot_data_distribution(dataset_path)
        plot_non_numeric_distribution(dataset_path, save_to_dir=False)

    if workflow["missing_data_plotting"]:
        plot_missing_data(dataset_path)

    if workflow["modify_sex_column"]:
        standardize_sex_column(data)

    # Remove discordant assay results (R) and elevated TBG (S) diagnoses
    if workflow["remove_s_and_r"]:
        modified_data = remove_specific_diagnoses(data=data, codes_to_remove=['S', 'R'])

    if workflow["age_handling"]:
        # Change the age data
        age_analysis(data, dataset_path=dataset_path, change_above=120, change_below=0,
                     to_value='median', visualize=False)

    if workflow["statistics_handling"]:
        # Calculate statistics
        stats = calculate_statistics(data)
        for column, column_stats in stats.items():
            print(f"Column: {column}")
            for stat_name, stat_value in column_stats.items():
                print(f"{stat_name}: {stat_value}")
            print()

    if workflow["preprocess"]:
        # Preprocess data
        preprocessed_data = preprocess_data(data=data, remove_measured_type=True, remove_TBG=True, data_path=None,
                                            remove_referral_source=True)

        if workflow["analysis"]:
            # Analyze data
            analysis_results = analyze_data(preprocessed_data)

        if workflow["visualize"]:
            # Visualize data
            visualize_data(data, analysis_results)

    if workflow["analysis"]:
        # Analyze data
        analysis_results = analyze_data(data)

    if workflow["discretization"]:
        # Assuming 'df' is your pandas DataFrame
        new_df = discretize_features(preprocessed_data)
        print(new_df.head())
        # # Dictionary indicating which attributes to discretize
        # discretize_flags = {'age': True, 'TSH': True, 'T3': True, 'TT4': True,
        #                     'T4U': True, 'FTI': True}
        #
        # original_data = data.copy()
        # # Apply discretization
        # discretized_data = discretize_thyroid_data(data, discretize_flags)
        #
        # # Attributes you want to visualize, ensure to include the attributes you've discretized
        # attributes = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
        #
        # # Visualize the discretization
        # plot_discretization(original_data, discretized_data, attributes)

    if workflow["classify_conditions"]:
        classified_df = classify_thyroid_conditions(new_df, plot_distribution=True)

        if workflow["save_classified_df"]:
            # Specify the desired file path
            new_classified_path = '/path/to/data.csv'

            # Save the classified DataFrame to the specified path
            classified_df.to_csv(new_classified_path, index=False)

            print("[File Process Complete]::New file in 'new_classified_path' directory")

    if workflow["edit_file"]:
        input_file = "/path/to/data.csv"
        output_file = "/path/to/data.csv"
        columns_to_remove = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']

        column_name_mapping = {'age': 'Age',
                               'sex': 'Sex',
                               'on_thyroxine': 'On_Thyroxine',
                               'query_on_thyroxine': 'Query_On_Thyroxine',
                               'on_antithyroid_medication': 'On_Antithyroid_Medication',
                               'sick': 'Sick',
                               'pregnant': 'Pregnant',
                               'thyroid_surgery': 'Thyroid_Surgery',
                               'i131_treatment': 'I131_Treatment',
                               'query_hypothyroid': 'Query_Hypothyroid',
                               'query_hyperthyroid': 'Query_Hyperthyroid',
                               'lithium': 'Lithium', 'goitre': 'Goitre',
                               'tumor': 'Tumor',
                               'hypopituitary': 'Hypopituitary',
                               'psych': 'Psych',
                               'TSH_Discrete': 'TSH',
                               'T3_Discrete': 'T3',
                               'TT4_Discrete': 'TT4',
                               'T4U_Discrete': 'T4U',
                               'FTI_Discrete': 'FTI',
                               'age_Discrete': 'Age',
                               'diagnosis': 'Diagnosis'
                               }

        preprocess_csv(input_file, output_file, columns_to_remove, column_name_mapping)


if __name__ == "__main__":
    main()
