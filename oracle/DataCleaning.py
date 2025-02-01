#!/usr/bin/env python
import pandas as pd
import numpy as np


class DataCleaning(object):
    """
    Data Wrangling Class for handling missing data and categorical encodings.
    """
    def __init__(self, dataframe):
        """
        Establish dataframe and obtain columns and rows
        """
        self.dataframe = dataframe
        self.rows = dataframe.shape[0]
        self.columns = dataframe.shape[1]

    def column_summary(self, N):
        """
        Basic check on column datatype, null counts, distinct values, etc.
        Loops thru each column, using a dataframe to store the:
            column name
            column datatype
            number of nulls
            number of non-nulls
            number of distinct values
            min/max values
            median value
            average value (if number)
            number of non-zero values (if number)
            top N distinct values
        -----------------------------------------------------------
        INPUT:
            N: (int) Top N distinct values for dataframe

        OUTPUT:
            summary_df: (pd.DataFrame) DataFrame as summary of original
            dataframe
        """
        # Initialize summary dataframe
        summary_rows = []

        for col in self.dataframe.columns:
            column_name = col
            column_dtype = self.dataframe[col].dtype
            null_num = self.dataframe[col].isnull().sum()
            non_null_num = self.dataframe[col].notnull().sum()

            # Initialize default values
            min_val, max_val, median_val, avg_val, non_zero_num, top_N_unique = None, None, None, None, None, None

            # Calculate only for numerical columns
            if pd.api.types.is_numeric_dtype(self.dataframe[col]):
                min_val = self.dataframe[col].min()
                max_val = self.dataframe[col].max()
                median_val = self.dataframe[col].median()
                avg_val = self.dataframe[col].mean()
                non_zero_num = (self.dataframe[col] != 0).sum()

            # Calculate TOP N unique values
            distinct_values = self.dataframe[col].dropna().value_counts()
            num_distinct_vals = len(distinct_values)
            top_N_unique = distinct_values.head(N).index.tolist()

            # Append dictionary to summary rows
            summary_rows.append({
                "column_name": column_name,
                "column_dtype": column_dtype,
                "null_num": null_num,
                "non_null_num": non_null_num,
                "min_val": min_val,
                "max_val": max_val,
                "median_val": median_val,
                "avg_val": avg_val,
                "distinct_values": distinct_values.index.tolist(),
                "num_distinct_vals": num_distinct_vals,
                "non_zero_num": non_zero_num,
                "top_N_unique": top_N_unique
            })
        
        # Convert dictionary to DataFrame
        summary_df = pd.DataFrame(summary_rows)
        return summary_df

    def drop_cols_missing_data(self, threshold=0.5):
        """
        Drop columns where proportion of missing data exceeds threshold.
        If no columns need to be removed, the dataframe will not undergo a
        change.
        - Loss of information, though ... 
        ----------------------------------------
        INPUT:
            threshold: (float; default: 0.5) Proportion of missing data required to drop
            column, (0.5 = 50% missing)

        OUTPUT:
            None
        """
        self.dataframe = self.dataframe.loc[:, self.dataframe.isnull().mean() <= threshold]

        return self.dataframe # Method chaining

    def drop_rows_missing_data(self):
        """
        Drop rows with any missing data, where there's at least one NULL value
        - This isn't best practice, since it might delete valueable
        information.
        - pd.api.types.is_numeric_dtype is a function in the 
        Pandas library that is used to check if a given array 
        or dtype is of a numeric type.
        ----------------------------------------
        INPUT:
            None

        OUTPUT:
            None
        """
        self.dataframe = self.dataframe.dropna(axis=0)

        return self.dataframe

    def imputing_vals_mean(self, columns):
        """
        Imputes missing values in a numerical colum with the arithmetic mean.

        Mathematical definition:
        For a vector x = [x₁, ..., xₙ] with missing values, let:
        - I = {i | xᵢ is not missing}
        - μ = (1/|I|) * ∑ᵢ∈I xᵢ
        Then for each missing value j: xⱼ ← μ

        Properties:
        - Preserves the sample mean: E[X] remains unchanged
        - Reduces variance: Var(X_imputed) ≤ Var(X_original)
        - Maintains sample size: n_imputed = n_original
        -------------------------------------------------------
        INPUT:
            column: (str) Name of column to impute

        OUTPUT:
            self.dataframe: (pd.DataFrame) Dataframe with imputed values in
            specific column
        """
        # Validate column existence
        if column not in self.dataframe.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        # Validate column type
        if not pd.api.types.is_numeric.dtype(self.dataframe[column]):
            raise ValueError(f"""Column '{column}' must be numeric, got
                             {self.dataframe[column].dtype}""")

        # Create a copy to esnure we're working with original data
        df_copy = self.dataframe.copy()

        # Calculate mean of non-missing data
        column_mean = df_copy[column].mean()

        # Impute missing values
        df_copy[column] = df_copy[column].fillna(column_mean)

        # update the instance dataframe
        self.dataframe = df_copy

        return self.dataframe

    def imputing_vals_median(self, column):
        """
        Imputes missing values in a numerical column with the Median.
        - Cannot work on Categorical data
        --------------------------------------------------------
        INPUT:
            column: (str) Column name to impute

        OUTPUT:
        """
        if (column in self.dataframe.columns and
            pd.api.is_numeric.dtype(self.dataframe[column])) :
            self.dataframe[column].fillna(self.dataframe[column].median(),
                                          inplace=True)

        return self.dataframe

    def imputing_group(self, group_via_col, target_col, avg=True):
        """
        Imputes missing values in a target column by group
        --------------------------------------------------------
        INPUT:
            group_via_col: (str) Column to group by
            target_col: (str) Column with missing values
            method: (str) "Mean" or "Median"

        OUTPUT:

        """
        # Ensure target column exists and is numerical
        if (target_col in self.dataframe.columns and
            pd.api.is_numeric.dtype(self.dataframe[target_col])):

            # Determine whether group filled in via mean or median
            if avg:
                # Mean
                self.dataframe[target_col] = \
                self.dataframe[target_col].fillna(self.dataframe.groupby(group_via_col)[target_col].transform("mean"))

            else:
                # Median
                self.dataframe[target_col] = \
                self.dataframe[target_col].fillna(self.dataframe.groupby(group_via_col)[target_col].transform("median"))

            return self.dataframe

    def imputing_categorical_cols(self, column):
        """
        Imputes missing values in a categorical column with the MODE (most
        frequent value).
        ------------------------------------------------
        INPUT:
            column: (str) Column name to impute.

        OUTPUT:
        """
        if (column in self.dataframe.columns and not
            pd.api.is_numeric.dtype(self.dataframe[column])):
            # Most frequent value
            mode_value = self.dataframe[column].mode()[0]

            self.dataframe[column].fillna(mode_value, inplace=True)

        return self.dataframe

    def foward_fill(self):
        """
        Method to fill missing values by carrying forward the last observed
        non-missing value (ffil).
        Useful for;
            - Time series data
        -----------------------------------------------------
        INPUT:

        OUTPUT:
        """
        self.dataframe.fillna(method="ffill", inplace=True)

        return self.dataframe

    def backward_fill(self):
        """
        Method for filling in missing values via carrying backward the next
        observed non-missing value.
        --------------------------------------------------
        INPUT:

        OUTPUT:
        """
        self.dataframe.fillna(method="bfill", inplace=True)

        return self.dataframe


# Example usage
df = pd.read_csv("data/raw/sample_data.csv")
dc = DataCleaning(df)

# Generate summary
summary_df = dc.column_summary(10)

# Handle missing data
df_cleaned = dc.imputing_vals_mean("value")
df_cleaned = dc.imputing_vals_median("number_of_accounts")
df_cleaned = dc.imputing_categorical_cols("state_2")

# Apply forward and backward fill
df_cleaned = dc.foward_fill()
df_cleaned = dc.backward_fill()

# Display the cleaned dataset
import ace_tools as tools
tools.display_dataframe_to_user(name="Cleaned DataFrame", dataframe=df_cleaned)

