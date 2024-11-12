#!/usr/bin/env python
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

    # Separate each table based on context, adjusting the indices
    oil_data = dataframe.iloc[0:3].reset_index(drop=True)
    oil_data.columns = ["Type", "Cost/Barrel", "Octane Rating", "Max Available"]

    # Find the index where Regular gasoline starts
    regular_start = dataframe[dataframe['Type'] == 'Regular'].index[0]
    # Find the index where Sales data starts
    sales_start = dataframe[dataframe['Type'].str.contains('Regular gasoline', na=False)].index[0]

    gas_data = dataframe.iloc[regular_start:sales_start].reset_index(drop=True)
    gas_data.columns = ["Gasoline Type", "Type 1", "Type 2", "Type 3"]

    sales_data = dataframe.iloc[sales_start:sales_start+2].reset_index(drop=True)
    sales_data.columns = ["Gasoline Type", "Octane Rating", "Sales Price", "Minimum Demand"]

    # Clean data by converting constraints to numerical values
    for col in gas_data.columns[1:]:
        gas_data[col] = gas_data[col].str.extract(r'(\d+)').astype(float) / 100

    # Convert data to numerical types
    oil_data[["Cost/Barrel", "Octane Rating", "Max Available"]] = \
        oil_data[["Cost/Barrel", "Octane Rating", "Max Available"]].apply(pd.to_numeric)

    # Clean sales data - remove 'barrels' from Minimum Demand and convert to numeric
    sales_data['Minimum Demand'] = sales_data['Minimum Demand'].str.extract(r'(\d+,?\d*)').str.replace(',', '').astype(float)
    sales_data[["Octane Rating", "Sales Price"]] = \
        sales_data[["Octane Rating", "Sales Price"]].apply(lambda x: pd.to_numeric(x.str.replace('$', '')))

    # Display cleaned data for verification
    print("\nCleaned Oil Data:")
    print(oil_data)
    print("\nCleaned Gasoline Data:")
    print(gas_data)
    print("\nCleaned Sales Data:")
    print(sales_data)

    return oil_data, gas_data, sales_data

