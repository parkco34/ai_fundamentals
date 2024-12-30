#!/usr/bin/env python
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

    def column_summary(df):
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
            df: (pd.DataFrame) Dataframe representing data

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
            "num_unique",
            "min_val",
            "max_val",
            "median_val",
            "avg_val",
            "non_zero_num",
            "top_N_unique"
        ])

        for col in df.columns:
            column_name = col
            column_dtype = df[col].dtype
            null_num = df[column_name].isnull().sum()
            non_null_num = df[column_name].notnull().sum()
            num_unique = len(df[column_name].unique())
            min_val = min(df[column_name])
            max_val = max(df[column_name])
            avg_val = mean(df[column_name])
            non_zero_num = len([v for v in df[column].unique() if v > 0])
            # Get value counts and sort, obtaining top N values
            value_counts = df[column_name].value_counts()

    def drop_cols_missing_data(self):
        """
        Drop columns where proportion of missing data exceeds threshold
        ----------------------------------------
        INPUT:
            threshold: (float) Proportion of missing data required to drop
            column, (0.5 = 50% missing)

        OUTPUT:
            new_df: (pd.DataFrame) Dataframe with dropped columns
        """
        self.dataframe = self.dataframe.loc[:, self.dataframe.isnull().mean()
                                            <= threshold]

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



