#!/usr/bin/env python
import numpy as np


class DataCleaning(object):
    """
    Data Wrangling Class for handling missing data and categorical encodings.
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.rows = dataframe.shape[0]
        self.columns = dataframe.shape[1]

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
        pass

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



