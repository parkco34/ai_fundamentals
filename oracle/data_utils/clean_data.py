#!/usr/bin/env python
import pandas as pd
import numpy as np


class CleanData:
    """
    Data Wrangling class for handling missing data and categorical encodings.
    """

    def __init__(self, dataframe):
        """
        Establish dataframe and obtain columns and rows
        """
        # Ensure the input is actually a pdf.DataFrame
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("""
                             Expected a pandas dataframe, but go some other bullshit """)


        # Instantiate some shit (◕‿◕)╭∩╮
        self.dataframe = dataframe
        self.num_rows = dataframe.shape[0]
        self.num_columns = dataframe.shape[1]

    def column_sumarray(self, top_N_values):
        """
        Basic check on column datatype, null counts, distinct values, etc.
        Loops thru each column, using a dataframe to store these:
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
            top_N_values: (int) Top N distinct values for sample

        OUTPUT:
            summary_df: (pd.DataFrame) DataFrame as summary of original dataframe
        """
        # Nested function
        def get_top_N_distinct(arr, top_N_values):
            """
            Calculates the top N distinct values for the array-like input
            --------------------------------------------------
            INPUT:
                arr: (np.ndarray) of (int/float)

            OUTPUT:
                top_N_distinct_values: (np.ndarray) Top N distinc t values
            """
            unique_vals, counts = np.unique(arr, return_counts=True)
            sorted_indices = np.argsort(counts)[::-1]
            top_N_indices = sorted_indices[:top_N_values]

            return unique_vales[top_N_values]

        # Initialize summary columns and dataframe
        summary_dict = {
            "column_name": None,
            "column_dtype": None,
            "null_num": None,
            "non_null_num": None,
            "distinct_values": None,
            "min_value": None,
            "max_value": None,
            "median_value": None,
            "avg_value": None,
            "non_zero_num": None,
            "top_N_distinct_values": None
        }

        # Iterate thru dataframe columns, collecting information
        for col in self.dataframe.columns:
            summary_dict["column_name"] = col
            summary_dict["column_dtype"] = self.dataframe[col].dtype
            summary_dict["null_num"] = self.dataframe[col].isnull().sum()
            summary_dict["non_null_num"] = self.dataframe[col].notnull().sum()
            summary_dict["distinct_values"] = \
                self.dataframe[col].unqiue()
            summary_dict["min_value"] = self.dataframe[col].min()
            summary_dict["max_value"] = self.dataframe[col].max()
            summary_dict["median_value"] = self.dataframe[col].median()
            summary_dict["avg_value"] = self.dataframe[col].mean()
            summary_dict["non_zero_num"] = len(self.dataframe[
                self.dataframe[col] != 0
            ][col])
            # Use nested function
            summary_dict["top_N_distinct_values"] = \
                get_top_N_distinct(summar_dict["distinct_values"], top_N_values)

        # Convert dictionary to DataFrame
        summary_df = pd.DataFrame(summary_dict)
        breakpoint()
        return summary_df


        

