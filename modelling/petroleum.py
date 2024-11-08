#!/usr/bin/env python
import pandas as pd

def get_data(filename):
    """
    Reads data and stores in DataFrame.
    ---------------------------------------------
    INPUT:
        filename: (str) Filename, duh.

    OUTPUT:
        dataframe: (pd.DataFrame)
    """
    return pd.read_csv(filename)

def clean_data(dataframe):
    """
    Take care of missing values and non-numerical inputs.
    --------------------------------------------------
    INPUT:
        dataframe: (pd.DataFrame)

    OUTPUT:
        new_dataframe: (pd.DataFrame)
    """
    # Output dataframe information
    print(f"INFO: {dataframe.info()}")
    print(f"Statistics: {dataframe.describe(include='all')}")
    
    # Drop any completely empty rows
    dataframe = dataframe.dropna(how="all").reset_index(drop=True)
    
    # Handling non-numerical constraint rows and identifying sections of data
    # based on unique colun name or row patterns
    # separate each table based on context
    oil_data = dataframe.iloc[0:3].reset_index(drop=True)
    oil_data.columns = ["Type", "Cost/Barrel", "Octane Rating", "Max Available"]

    gas_data = dataframe.iloc[4:11].reset_index(drop=True)
    gas_data.columns = ["Gasoline Type", "Type 1", "Type 2", "Type 3"]

    sales_data = dataframe.iloc[13:15].reset_index(drop=True)
    sales_data.columns = ["Gasoline Type", "Octane Rating", "Sales Price", "Minimum Demand"]

    # Clean data by converting constraints to numerical values
    for col in gas_data.columns[1:]:
        gas_data[col] = gas_data[col].replace({
            ">= 25%": 0.25, ">= 35%": 0.35, "<= 30%": 0.3,
            ">= 30%": 0.3, ">= 45%": 0.45, "<= 20%": 0.2
        })

    # Convert data to numerical types
    oil_data[["Cost/Barrel", "Octane Rating", "Max Available"]] = \
    oil_data[["Cost/Barrel", "Octane Rating", "Max Available"]].apply(pd.to_numeric)

    sales_data[["Octane Rating", "Sales Price", "Minimum Demand"]] = \
    sales_data[["Octane Rating", "Sales Price", 
                "Minimum Demand"]].apply(pd.to_numeric)

    # Display cleaned data for verification
    print("\nCleaned Oil Data:")
    print(oil_data)
    print("\nCleaned Gasoline Data:")
    print(gas_data)
    print("\nCleaned Sales Data:")
    print(sales_data)

    return oil_data, gas_data, sales_data

dframe = get_data("petroleum.csv")
oil_df, gas_df, sales_df = clean_data(dframe)

breakpoint()

