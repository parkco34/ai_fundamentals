#!/usr/bin/env python
def clean_data(dataframe):
    """
    Cleans the data by handling missing values, removing unnecessary rows,
    and converting percentage constraints to numerical values.
    --------------------------------------------------
    INPUT:
        dataframe: (pd.DataFrame)

    OUTPUT:
        cleaned_dataframe: (pd.DataFrame)
    """
    # Display dataframe information
    print("Original Data Info:")
    print(dataframe.info())
    print("Statistics:")
    print(dataframe.describe(include='all'))

    # Drop any completely empty rows
    dataframe = dataframe.dropna(how='all').reset_index(drop=True)

    # Handling non-numerical constraint rows
    # Identify the sections of the data based on unique column names or row patterns

    # Separate each table based on context:
    oil_data = dataframe.iloc[0:3].reset_index(drop=True)
    oil_data.columns = ["Type", "Cost/Barrel", "Octane Rating", "Max Available"]

    gasoline_requirements = dataframe.iloc[4:11].reset_index(drop=True)
    gasoline_requirements.columns = ["Gasoline Type", "Type 1", "Type 2", "Type 3"]

    sales_data = dataframe.iloc[13:15].reset_index(drop=True)
    sales_data.columns = ["Gasoline Type", "Average Octane Rating", "Sales Price", "Minimum Demand"]

    # Clean data by converting constraints to numerical values where possible
    for col in ["Type 1", "Type 2", "Type 3"]:
        gasoline_requirements[col] = gasoline_requirements[col].replace({
            ">= 25%": 0.25, ">= 35%": 0.35, "<= 30%": 0.30,
            ">= 30%": 0.30, ">= 45%": 0.45, "<= 20%": 0.20
        })

    # Convert data to numeric types where appropriate
    oil_data[["Cost/Barrel", "Octane Rating", "Max Available"]] = oil_data[["Cost/Barrel", "Octane Rating", "Max Available"]].apply(pd.to_numeric)
    sales_data[["Average Octane Rating", "Sales Price", "Minimum Demand"]] = sales_data[["Average Octane Rating", "Sales Price", "Minimum Demand"]].apply(pd.to_numeric)

    # Display cleaned data for verification
    print("\nCleaned Oil Data:")
    print(oil_data)
    print("\nCleaned Gasoline Requirements:")
    print(gasoline_requirements)
    print("\nCleaned Sales Data:")
    print(sales_data)

    return oil_data, gasoline_requirements, sales_data

