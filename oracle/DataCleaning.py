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
        summary_df = pd.DataFrame(columns=[
            "column_name",
            "column_dtype",
            "null_num",
            "non_null_num",
            "min_val",
            "max_val",
            "median_val",
            "avg_val",
            "distinct_values",
            "num_distinct_vals",
            "non_zero_num",
            "top_N_unique"
        ])

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
            num_distinct_vals = distinct_values.shape[0]
            top_N_unique = distinct_values.head(N).index.tolist()

            # Append column summary to summary df
            summary_df = pd.concat([summary_df, pd.DataFrame([{
                "column_name": column_name,
                "column_dtype": column_dtype,
                "null_num": null_num,
                "non_null_num": non_null_num,
                "min_val": min_val,
                "max_val": max_val,
                "median_val": median_val,
                "avg_val": avg_val,
                "distinct_values": distinct_values,
                "num_distinct_vals": num_distinct_vals,
                "non_zero_num": non_zero_num,
                "top_N_unique": top_N_unique
            }])], ignore_index=True)
        
        return summary_df

    def drop_cols_missing_data(self, threshold=0.5):
        """
        Drop columns where proportion of missing data exceeds threshold
        ----------------------------------------
        INPUT:
            threshold: (float; default: 0.5) Proportion of missing data required to drop
            column, (0.5 = 50% missing)

        OUTPUT:
            new_df: (pd.DataFrame) Dataframe with dropped columns
        """
        self.dataframe = self.dataframe.loc[:, self.dataframe.isnull().mean()
                                            <= threshold]

        breakpoint()

    def drop_rows_missing_data(self):
        """
        Drop rows with any missing data
        ----------------------------------------
        INPUT:

        OUTPUT:
        """
        pass

    def imputing_vals_mean(self, column):
        """
        Imputes missing values in a numerical column with the mean.
        -----------------------------------------------------------
        INPUT:
            column: (str) Column name to impute.

        OUTPUT:
            new_col: 
        """
        pass

    def imputing_vals_median(self, column):
        """
        Imputes missing values in a numerical column with the Median.
        --------------------------------------------------------
        INPUT:
            column: (str) Column name to impute

        OUTPUT:

        """
        pass

    def imputing_group(self, group_via_col, target_col, method="mean"):
        """
        Imputes missing values in a target column by group
        --------------------------------------------------------
        INPUT:
            group_via_col: (str) Column to group by
            target_col: (str) Column with missing values
            method: (str) "Mean" or "Median"

        OUTPUT:

        """
        pass

    def imputing_categorical_cols(self, column):
        """
        Imputes missing values in a categorical column with the MODE (most
        frequent value).
        ------------------------------------------------
        INPUT:
            column: (str) Column name to impute.

        OUTPUT:
        """
        pass

    def foward_fill(self):
        pass

    def backward_fill(self):
        pass


# Example usage
df = pd.read_csv("data/sample_data.csv")
dc = DataCleaning(df)
summary = dc.column_summary(10)

# Drop columns of mising data
missing = dc.drop_cols_missing_data()

