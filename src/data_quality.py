
import pandas as pd


def summary_general(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate general quality statistics for the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to analyze.
    Returns:
        pd.DataFrame: DataFrame containing general quality statistics.
    """
    summary_general =  pd.DataFrame({
        "rows": [len(df)],
        "columns": [df.shape[1]],
        "duplicated_rows": [df.duplicated().sum()],
        "percentage_duplicates": [df.duplicated().mean() * 100]
    })
    return summary_general


def summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate quality statistics for each column.

    Args:
        df (pd.DataFrame): DataFrame to analyze.
    Returns:
        pd.DataFrame: DataFrame containing quality statistics for each column.
    """
    summary_columns = pd.DataFrame({
        "types": df.dtypes,
        "null": df.isnull().sum(),
        "percentage_null": df.isnull().mean() * 100,
        "unique_values": df.nunique()
    })
    return summary_columns


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate descriptive statistics for numerical variables.

    Args:
        df (pd.DataFrame): DataFrame to analyze.
    Returns:
        pd.DataFrame: DataFrame containing descriptive statistics for numerical variables.
    """

    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        return pd.DataFrame()

    summary_statistics = numeric.describe(
        percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
    ).T

    return summary_statistics


def summary_quality(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Generate a complete data quality summary.

    Args:
        df (pd.DataFrame): DataFrame to analyze.
    Returns:
        dict[str, pd.DataFrame]: Dictionary containing general, column-wise, and statistical summaries.
    
    """
    dict_summary = {
        "general": summary_general(df),
        "columns": summary_columns(df),
        "statistics": summary_statistics(df)
    }

    return dict_summary