#!/usr/bin/env python
import pandas as pd
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt

def configure_pandas_display():
    """
    Configures pandas display settings for better readability of output.
    ------------------------------------------
    INPUT:
        None

    OUTPUT:
        None
    """
    # Set padnas display options for readability
    # Show all columns
    pd.set_option("display.max_columns", None)
    #Increase width for visibility
    pd.set_option("display.width", 120)
    # Limit width of each column to prevent overflow
    pd.set_option("display.max_colwidth", 100)

def load_data(file_path):
    """
    Loads data from CSV file into pandas DataFrame
    --------------------------------------------------
    INPUT:
        file_path: (str) Path to CSV file

    OUTPUT:
        data: (pd.DataFrame) Loaded data
    """
    try:
        return pd.read_csv(file_path)

    except FileNotFoundError:
        print(f"Error: Could not find file at {file_path}")
        return None

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# Correlation analysis
def add_correlations(df, summary_df):
    """
    Adds correlation analysis for numeric columns to the summary DataFrame.

    The function provides a way to quickly identify which variables in dataset 
    are most strongly related to each other, which can be valuable for:

        - Feature selection in machine learning
        - Understanding relationships in your data
        - Identifying potential collinearity issues
        - Data exploration and visualization planning

    ! => Pearson's correlation coefficient is the COVARIANCE of two variables
    divided by the product of their standard deviations.  
    PRODUCT MOMENT -> The mean of the product of the mean-adjusted r.v.s.
    ------------------------------------------------------------------
    INPUT:
        df: (pd.DataFrame) 
        summary_df: (pd.DataFrame) Summary dataframe

    OUTPUT:
        summary_df: (pd.DataFrame) Summary dataframe with added correlation
        information
    """
    # Grabs only the numeric columns of the ORIGINAL dataset
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # As long as there's more than one column, since we need atleast two...
    if len(numeric_cols) > 1:
        # Computes pairwise correlation of columns, excluding NA/null values
        # from ORIGINAL dataset, resulting in a square matrix, where each cell
        # shows the correlation between two variables, -1 < x < 1
        corr_matrix = df[numeric_cols].corr()

        # Iteratively obtain top  correlation information
        # Extracts all correlations for that column from correlation matrix
        for col in summary_df["col_name"]:
            # If col_name in summary_df is also in original df, compare
            if col in numeric_cols:
                # removes correlation with itself, always 1.0
                correlations = corr_matrix[col].drop(col)
                # Selects three highest absolute correlation values
                top_corr = correlations.nlargest(3)
                
                # Stores the results
                # Uses pandas' loc accessor to find the row where col_name
                # matches current column
                # Converts correlations to dictionary, then string
                summary_df.loc[
                    summary_df["col_name"] == col,
                    "top_correlations"
                ] = str(dict(top_corr.round(3)))

    return summary_df

def suggest_dtype(col_info):
    """
    Suggests appropriate data types based on column characteristics
    -------------------------------------------------------------------
    INPUT:
        col_info: (dict) Dictionary containing column information

    OUTPUT:
        str: Suggetstion for data type optimization
    """
    if col_info["col_dtype"] == "object":
        if col_info["num_distinct_values"] == 1:
            return "Consider using a constant"

        if col_info["num_distinct_values"] <= 10:
            return "Consider using categorical type"

    return "Current type appears appropriate"

def column_summary(df, n_counts):
    """
    ? --> LEARN THIS SHIT!

    Provides a summary of the DataFrame columns including statistical measures.
    Data quality indicators, and distribution analysis.
    ------------------------------------------------------------
    INPUT:
        df: (pd.DataFrame) Input dataframe to analyze
        n_counts: (int) Number of top distinct values to show for each column

    OUTPUT:
        summary_df: (pd.DataFrame) Summary statistics and analysis for each
        column
    """
    results = []

    # Process each column
    for column in df.columns:
        print(f"Processing {column} column with {df[column].dtype} dtype")

        # Basic column information
        col_info = {

            "col_name": column,
            "col_dtype": df[column].dtype,
            "num_distinct_values": df[column].nunique()
        }

        # NUll value analysis
        total_rows = len(df)
        nulls = df[column].isnull().sum()
        col_info.update({
            "nulls_count": nulls,
            "nulls_percentage": round((nulls / total_rows) * 100, 2),
            "non_nulls_count": total_rows - nulls
        })

        # Get distinct values
        value_counts = df[column].value_counts()

        # Store top N distinct values and their counts
        top_values = value_counts.head(n_counts)
        col_info["distinct_values"] = dict(top_values)

        # For numeric columns, add statistical measures
        if np.issubdtype(df[column].dtype, np.number):
            # ?
            non_null_values = df[column].dropna()
            non_zero_values = non_null_values[non_null_values != 0]

            # Basic satatistics
            col_info.update({
                "min_value": non_null_values.min(),
                "max_value": non_null_values.max(),
                "median": non_null_values.median(),
                "mean": non_null_values.mean(),
                "std_dev": non_null_values.std(),
                "non_zero_mean": non_zero_values.mean(),

                # Distbution measures
                "skewness": non_null_values.skew(),
                "kurtosis": non_null_values.kurtosis(),

                # Zero analysis
                "zero_count": (non_null_values == 0).sum(),
                "zero_percentage": round((non_null_values == 0).mean() * 100, 2)
            })

            # Outlier detection using IQR method
            Q1 = non_null_values.quantile(0.25)
            Q3 = non_null_values.quantile(0.75)
            IQR = Q3 - Q1
            outlier_low = Q1 - 1.5 * IQR
            outlier_high = Q3 + 1.5 * IQR
            outliers = non_null_values[(non_null_values < outlier_low) |
                                     (non_null_values > outlier_high)]

            col_info.update({
                "outlier_count": len(outliers),
                "outlier_percentage": round((len(outliers) / len(non_null_values)) * 100, 2),
                "Q1": Q1,
                "Q3": Q3,
                "IQR": IQR
            })

        # For categorical columns, add category analysis
        elif df[column].dtype == 'object':
            # Calculate existing string metrics
            col_info.update({
                "avg_string_length": df[column].str.len().mean(),
                "max_string_length": df[column].str.len().max(),
                "empty_string_count": (df[column] == '').sum(),
                "whitespace_only_count": df[column].str.isspace().sum()
            })

            # Value precentage distributions
            total_count = len(df[column].dropna())
            col_info["value_percentages"] = {
                k: round((v/total_count) * 100, 2)
                for k, v in dict(top_values).items()
            }

        results.append(col_info)

    # Create DataFrame from results
    summary_df = pd.DataFrame(results)

    # Organize columns in a logical order
    column_order = [
        "col_name", "col_dtype", "num_distinct_values", "dtype_suggestion",
        "nulls_count", "nulls_percentage", "non_nulls_count",
        "min_value", "max_value", "mean", "median", "std_dev",
        "non_zero_mean", "skewness", "kurtosis",
        "zero_count", "zero_percentage",
        "outlier_count", "outlier_percentage",
        "Q1", "Q3", "IQR",
        "avg_string_length", "max_string_length",
        "empty_string_count", "whitespace_only_count",
        "value_percentages", "distinct_values", "top_correlations"
    ]

    # Reorder columns, keeping only those that exist
    existing_columns = [col for col in column_order if col in summary_df.columns]
    summary_df = summary_df[existing_columns]

    return summary_df

def truncate_dict_values(d, max_items=10):
    """
    Truncates columns that are too long to be readable.
    ----------------------------------------
    INPUT:
        d: (dict) Values to truncate
        max_items: (int) Limit to the number of items to truncate

    OUTPUT:
        truncated_output: (str)
    """
    if len(d) > max_items:
        # Only includes the max_items number of columns
        truncated = dict(list(d.items())[:max_items])
        # Converts the appropriate length and adds elipsis at end
        truncated_output = str(truncated)[:-1] + ", ...}"

        return truncated_output
    # otherwise just output the dictionary as a string
    return str(d)

def analyze_data(df, n_counts=10):
    """
    Perform comprehensive analysis on the provided DataFrame
    ------------------------------------------------------------
    INPUT:
        df: (pd.DataFrame) DataFrame to analyze
        n_counts: (int) Number of top distinct values to show

    OUTPUT:
        summary_df: (pd.DataFrame) Summary analysis
    """
    # Generate initial summary
    summary_df = column_summary(df, n_counts)

    # Add correlation analysis
    summary_df = add_correlations(df, summary_df)

    # Add dtype suggestions
    summary_df["dtype_suggestions"] = summary_df.apply(suggest_dtype, axis=1)

    return summary_df

#def main():
#    # Configure pandas display settings
#    configure_pandas_display()
#
#    # Define file path
#    file_path = "data/Utility_Energy_Registry_Monthly_County_Energy_Use__Beginning_2021_20241208.csv"
#
#    # Load data
#    df = load_data(file_path)
#    if df is None:
#        return None
#
#    # Perform analysis
#    try:
#        summary_df = analyze_data(df)
#
#        # Display results
#        print("\nData Analysis Summary:")
#        print("-" * 50)
#        display(summary_df)
#
#    except Exception as e:
#        print(f"Error during analysis: {str(e)}")
#
#
#if __name__ == "__main__":
#    main()
#
# ========================================================================================================
# testing and debugging
configure_pandas_display()
# Get dataframe
df = pd.read_csv(
    "data/Utility_Energy_Registry_Monthly_County_Energy_Use__Beginning_2021_20241208.csv"
)
summary_df = analyze_data(df)

## After computing summary_df
print("\nData Analysis Summary:")
print("-" * 50)
display(summary_df)
breakpoint()
