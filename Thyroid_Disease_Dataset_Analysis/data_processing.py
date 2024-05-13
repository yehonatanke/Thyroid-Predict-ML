import pandas as pd


def open_data(file_path):
    """
    Opens and reads a CSV file.

    :param filename:
        The path to the CSV file.
    :type filename:
        str

    :return:
        A DataFrame containing the loaded data if successful.
        Returns None if an error occurs during loading.
    :rtype:
        pandas.DataFrame or None

    :raises FileNotFoundError:
        If the specified file path does not exist.
    :raises Exception:
        If an error occurs during the file reading process.
    """
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        return None
    except Exception as e:
        print("An error occurred while loading the data:", str(e))
        return None


def load_data(filename):
    """
    Load data from a text file.

    :param filename: str
        The path to the text file containing the data.
    :return: pandas.DataFrame or None
        Returns a DataFrame containing the loaded data if successful.
        Returns None if an error occurs during loading.
    """
    try:
        # Specify column names and data types
        col_names = ['age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
                     'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
                     'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
                     'TSH_measured', 'TSH', 'T3_measured', 'T3', 'TT4_measured', 'TT4',
                     'T4U_measured', 'T4U', 'FTI_measured', 'FTI', 'TBG_measured', 'TBG',
                     'referral_source', 'diagnosis']
        col_types = {'age': 'float64', 'TSH': 'float64', 'T3': 'float64', 'TT4': 'float64',
                     'T4U': 'float64', 'FTI': 'float64', 'TBG': 'float64'}

        # Load data into DataFrame, skip the header row
        data = pd.read_csv(filename, names=col_names, header=None, skiprows=1, na_values='?', dtype=col_types)

        if data.empty:
            print("Error: Data file is empty.")
            return None

        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None


def preprocess_data(data, data_path, remove_missing_rows=False, remove_measured_type=False,
                    remove_TBG=False, remove_referral_source=False):
    """
    Preprocess the loaded data.

    :param data: pandas.DataFrame or None
        The DataFrame containing the loaded data.
    :param data_path: str or None
        The file path to save the preprocessed data. If None, data will not be saved.
    :param remove_missing_rows: bool, optional (default=False)
        If True, remove rows with missing values.
    :param remove_measured_type: bool, optional (default=False)
        If True, remove columns with names containing '_measured'.
    :param remove_TBG: bool, optional (default=False)
        If True, remove the 'TBG' column.
    :param remove_referral_source: bool, optional (default=False)
        If True, remove the 'referral_source' column.
    :return: pandas.DataFrame or None
        Returns the preprocessed DataFrame if successful.
        Returns None if an error occurs during preprocessing or if input data is None.
    """
    if data is None:
        print("Error: No data to preprocess.")
        return None

    if remove_missing_rows:
        # Drop rows with missing values
        data = data.dropna()

        if data_path is not None:
            # Save filtered data to the original file
            data.to_csv(data_path, index=False)

            # Print summary
            print("Rows with missing values have been removed.")
            print("Filtered data saved to:", data_path)

    if remove_measured_type:
        # Find columns with names containing '_measured'
        measured_columns = [col for col in data.columns if '_measured' in col]

        if measured_columns:
            # Remove columns with names containing '_measured'
            data = data.drop(columns=measured_columns)

            # Save filtered data to the original file
            if data_path is not None:
                # Save filtered data to the original file
                data.to_csv(data_path, index=False)
                print("Columns containing '_measured' data have been removed.")
                print("Filtered data saved to:", data_path)
        else:
            print("No columns containing '_measured' data found.")

        # Print summary
        print("Number of rows removed:", len(measured_columns))
        print("Rows containing 'measured' data have been removed.")
        print("Filtered data saved to:", data_path)

    if remove_TBG:
        if 'TBG' in data.columns:
            # Drop the 'TBG' column
            data = data.drop(columns=['TBG'])

            if data_path is not None:
                # Save filtered data to the original file
                data.to_csv(data_path, index=False)
                print("The 'TBG' column has been removed.")
                print("Filtered data saved to:", data_path)
        else:
            print("The 'TBG' column does not exist in the data.")

    if remove_referral_source:
        if 'referral_source' in data.columns:
            # Drop the 'referral_source' column
            data = data.drop(columns=['referral_source'])

            if data_path is not None:
                # Save filtered data to the original file
                data.to_csv(data_path, index=False)
                print("The 'referral_source' column has been removed.")
                print("Filtered data saved to:", data_path)
        else:
            print("The 'referral_source' column does not exist in the data.")

    return data


def preprocess_csv(input_file, output_file, columns_to_remove, column_name_mapping):
    """
    Preprocess a CSV file.

    :param input_file: str
        The path to the input CSV file.
    :param output_file: str
        The path to save the preprocessed CSV file.
    :param columns_to_remove: list of str
        Columns to remove from the DataFrame.
    :param column_name_mapping: dict
        Mapping of old column names to new column names.
    :return: None
    """
    try:
        # Read CSV into a DataFrame
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        return
    except pd.errors.ParserError:
        print(f"Error: Unable to parse '{input_file}' as CSV.")
        return

    # Check if all columns to remove are present in the DataFrame
    missing_columns = [col for col in columns_to_remove if col not in df.columns]
    if missing_columns:
        print(f"Error: Columns {missing_columns} not found in the input file.")
        return

    # Drop columns to remove
    df = df.drop(columns_to_remove, axis=1)

    # Rename columns
    df.rename(columns=column_name_mapping, inplace=True)

    # Move 'diagnosis' column to the end
    diagnosis_col = df.pop('Diagnosis')
    df['Diagnosis'] = diagnosis_col

    try:
        # Write processed DataFrame to a new CSV file
        df.to_csv(output_file, index=False)
        print(f"Preprocessed CSV saved to '{output_file}'.")
    except PermissionError:
        print(f"Error: Permission denied to write to '{output_file}'.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return
