#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get dataframe
df = pd.read_csv(
    "data/Utility_Energy_Registry_Monthly_County_Energy_Use__Beginning_2021_20241208.csv"
)

# Pure understanding of original data
def column_summary(df, n_counts, summary_data=[]):
    """
    Basic check on the column datatype, null n_counts, distinct values, to get a better understanding of the data. I also created a distinct values count dictionary where I go the top 10 n_counts and their distinct values displayed so I could roughly gauge how significant the distinct values are in the dataset.
    ---------------------------------------------
    INPUT:
        df: (pd.DataFrame)
        n_counts: (int) Number of top N counts and their distinct values
        summary_data: (list) Summary of provided dataset
    
    OUTPUT:
        summary_df: (pd.DataFrame) Summary of the given dataframe
    """
    for col_name in df.columns:
        col_dtype = df[col_name].dtype
        num_of_nulls = df[col_name].isnull().sum()
        num_of_non_nulls = df[col_name].notnull().sum()
        num_of_distinct_values = df[col_name].nunique()

        if num_of_distinct_values <= n_counts:
            distinct_values_counts = df[col_name].value_counts().to_dict()

        else:
            top_n_value_counts = \
            df[col_name].value_counts().head(n_counts).to_dict()
            distinct_values_counts = {k: v for k, v in
                                      sorted(top_n_value_counts.items(),
                                             key=lambda item: item[1],
                                             reverse=True)}

        summary_data.append({
            "col_name": col_name,
            "col_dtype": col_dtype,
            "num_of_nulls": num_of_nulls,
            "num_of_non_nulls": num_of_non_nulls,
            "num_of_distinct_values": num_of_distinct_values,
            "distinct_values_counts": distinct_values_counts
        })

    # Summary
    summary_df = pd.DataFrame(summary_data)

    return summary_df

def column_summary_plus(df, n_counts):
    # Initialize empty list
    results = []

    # Loop thru each column in the dataframe
    for column in df.columns:
        print(f"Start processing {column} col with {df[column].dtype} dtype")

        # GEt column dtype
        col_dtype = df[column].dtype

        # Get distinct values and their couunts
        value_counts = df[column].value_counts()
        distinct_values = value_counts.index.tolist()
        # Get number of distinct values
        num_distinct_values = len(distinct_values)
        # Get min and max values
        sorted_values = sorted(distinct_values)
        min_value = sorted_values[0] if sorted_values else None
        max_value = sorted_values[-1] if sorted_values else None

        # Get median value
        non_distinct_val_list = sorted(df[column].dropna().tolist())
        len_non_d_list = len(non_distinct_val_list)
        # Check length isn't zero
        if len(non_distinct_val_list) == 0:
            median = None

        else:
            median = non_distinct_val_list[len_non_d_list // 2]
            
        # Get average value if value is number
        if np.issubdtype(df[column].dtype, np.number):
            # If duplicate values list is not empty, 
            if len(non_distinct_val_list) > 0:
                average = sum(non_distinct_val_list) / len_non_d_list
                non_zero_val_list = [v for v in non_distinct_val_list if v > 0]
                average_non_zero = sum(non_zero_val_list) / len_non_d_list

            else:
                average = None
                average_non_zero = None

        else:
            average = None
            average_non_zero = None

        # Check if null values are present
        null_present = 1 if df[column].isnull().any() else 0

        # Get numver of nulls and non-nulls
        num_nulls = df[column].isnull().sum()
        num_non_nulls = df[column].notnull().sum()

        # Distinct values only take top n_count distinct values count
        top_n_d_v = value_counts.head(n_counts).index.tolist()
        top_n_c = value_counts.head(n_counts).tolist()
        top_n_d_v_dict = dict(zip(top_n_d_v, top_n_c))

        # Append information to result dataframe
        results.append({
            "col_name": column,
            "col_dtype": col_dtype,
            "num_disctinct_values": num_distinct_values,
            "min_value": min_value,
            "max_value": max_value,
            "median_no_na": median,
            "max_age_no_na": average,
            "average_non_zero": average_non_zero,
            "nulls_num": num_nulls,
            "distinct_values": top_n_d_v_dict, 
        })

        return pd.DataFrame(results)


# Example usage
summary_df = column_summary_plus(df, 10)
print(summary_df)



