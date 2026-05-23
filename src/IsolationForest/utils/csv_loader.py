import pandas as pd

def load(file_path: str):
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded tabular data.
    """
    df = pd.read_csv(file_path)

    return df