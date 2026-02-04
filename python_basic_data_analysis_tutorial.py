import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import os

# set working directory
os.chdir(
    "/Users/jrose01/Documents/AI for Data Analysis Course/2026 course development materials/Exercises/numpy-pandas-code"
)


# OR
# os.chdir('C:\\Users\\jrose01\\Documents\\AI for Data Analysis Course\\2026 course development materials\\Exercises\\numpy-pandas-code')

# ----------------------------------
# PANDAS BASICS
# ----------------------------------

# read in the data
df = pd.read_csv("penguins.csv")


def dataframe_overview(df: pd.DataFrame) -> dict:
    """
    Return basic information about a DataFrame.

    """

    n_rows = df.shape[0]
    n_columns = df.shape[1]

    # column names
    columns = list(df.columns)

    # Create a dictionary with the missing rate for each column
    # df.isna() returns a dataframe with boolean values (T/F) for each cell)
    # numeric values: True = 1, False = 0
    missing_rate = df.isna().mean().to_dict()

    # Return everything in a dictionary so it is easy to reuse later
    return {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "columns": columns,
        "missing_rate": missing_rate,
    }


# run the function
dataframe_overview(df)

# ----------------------------------
# NUMPY BASICS
# ----------------------------------


def numeric_summary(values: np.ndarray) -> dict:
    """
    Compute basic statistics for a numeric array.

    This function assumes:
    - values is already numeric
    - values contains no missing data
    """
    # np.mean computes the average
    mean = np.mean(values)

    # np.std computes the standard deviation
    std = np.std(values)

    # np.min and np.max give the range
    min_val = np.min(values)
    max_val = np.max(values)

    # Convert numpy values to Python floats for cleaner output
    return {
        "mean": float(mean),
        "std": float(std),
        "min": float(min_val),
        "max": float(max_val),
    }


# run the function
numeric_summary(df["flipper_length_mm"].dropna().values)

# ----------------------------------
# PANDAS & NUMPY COMBINED
# ----------------------------------


def summarize_column(df: pd.DataFrame, column: str) -> dict:
    """
    Compute summary statistics for one numeric column in a DataFrame.

    pandas:
    - selects the column
    - handles missing values

    numpy:
    - computes the statistics
    """

    # Always check that the column exists
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    # Select the column and remove missing values
    series = df[column].dropna()

    # Convert the pandas Series into a numpy array
    values = series.to_numpy()

    # Use numpy to compute statistics
    mean = np.mean(values)
    std = np.std(values)

    return {
        "column": column,
        "mean": float(mean),
        "std": float(std),
        "n": len(values),  # number of non-missing observations
    }


# run the function
summarize_column(df, "body_mass_g")

# ----------------------------------
# MATPLOTLIB BASICS
# ----------------------------------


def plot_histogram(
    df: pd.DataFrame, column: str
) -> None:  # expect a dataframe and a string column name, return nothing
    """
    Plot a histogram for a numeric column.

    This function does NOT return any output data.
    Its purpose is to create a visual output.
    """

    # Check that the column exists
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    # check that the column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric.")

    # Create a new figure so plots don't overlap
    # This is important when creating multiple plots in a loop or script
    plt.figure()

    # Plot the histogram using non-missing values
    plt.hist(df[column].dropna(), bins=20)

    # Label the plot so it is interpretable
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {column}")

    # Display the plot
    plt.show()


plot_histogram(df, "bill_length_mm")

# ----------------------------------
# STATSMODELS BASICS
# ----------------------------------


def run_linear_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    standardize: bool = False,
):
    """
    Fit a linear regression model.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    outcome : str
        The dependent variable.
    predictors : list[str]
        Predictor variables (numeric and/or categorical).
    standardize : bool, default False
        If True, numeric predictors are standardized (z-scored).

    """
    # check that outcome and predictors exist
    if outcome not in df.columns:
        raise ValueError("Outcome column not found.")

    for p in predictors:
        if p not in df.columns:
            raise ValueError(f"Predictor '{p}' not found.")

    # Check that outcome is numeric
    if not pd.api.types.is_numeric_dtype(df[outcome]):
        raise ValueError("Outcome column is not numeric.")

    # --- Optional standardization (default is False) ---
    if standardize:
        for p in predictors:
            # Only standardize numeric predictors
            if pd.api.types.is_numeric_dtype(df[p]):
                mean = df[p].mean()
                std = df[p].std()

                # Avoid division by zero
                if std == 0:
                    raise ValueError(f"Predictor '{p}' has zero variance.")

                df[p] = (df[p] - mean) / std

    # --- Build formula ---
    formula = outcome + " ~ " + " + ".join(predictors)

    # --- Fit model ---
    results = smf.ols(formula, data=df).fit()

    return results


# run the function without standardization
results = run_linear_regression(
    df, outcome="body_mass_g", predictors=["bill_length_mm", "flipper_length_mm", "sex"]
)
results.summary()

# run the function with standardization
results = run_linear_regression(
    df,
    outcome="body_mass_g",
    predictors=["bill_length_mm", "flipper_length_mm", "sex"],
    standardize=True,  # set to True to standardize numeric predictors
)
results.summary()
