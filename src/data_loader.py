import pandas as pd
import os


def load_data(file_path: str) -> pd.DataFrame:
    """
    load the data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")
    
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise RuntimeError(f"Error al cargar el archivo {file_path}: {e}")
    