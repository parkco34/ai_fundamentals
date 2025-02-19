#!/usr/bin/env python
"""
Oracle project:


*--> It is important to note that the data are subject to privacy screening and fields that fail the privacy screen are withheld
"""
from data_utils.data_cleaning import DataCleaning # Import certain class
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from nasa_power_api import NASAPowerAPI
from functools import reduce

def merge_weather_dataframes(dfs):
    """
    Merges weather dataframes on the date column
    ------------------------------------------------
    INPUT:
        dfs: (list) List of dataframes

    OUTPUT:
        proper_df: (pd.DataFrame)
    """
    # Initialize new dataframe to only include the date column
    proper_df = dfs[0].copy()

    for i in range(1, len(dfs)):
        proper_df = pd.merge(proper_df, dfs[i], on="date", how="outer")

    return proper_df


def display_cleaned_dataset(dataframe, step_name):
    """
    Shows the progress of our data cleaning by displaying key metrics art
    each step
    ------------------------------------------------------
    INPUT:

    OUTPUT:
    """
    print(f"\n=== After {step_name} ===")
    print(f"Rows: {dataframe.shape[0]}")
    print(f"Columns: {dataframe.shape[1]}")
    print(f"Missing values: {dataframe.isnull().sum().sum()}")

    if "value" in dataframe.columns:
        print(f"Vlaue column stats:")
        print(f"    - Negative values: {(dataframe['value'] < 0)}.sum()")
        print(f"    - Zero values: {(dataframe['value'] == 0).sum()}")
        print(f"    - Positive values: {(dataframe['value'] > 0).sum()}")

def read_data(file_path):
    """
    Reads the ( ͡° ͜ʖ ͡°  ) data
    """
    return pd.read_csv(file_path)

def export_county_fips(df_cleaned):
    """
    Exports the county FIPS codes to be selectable via the Java GUI 
    in a JSON file.
    -----------------------------------------------------------
    INPUT:
        df_cleaned: (pd.DataFrame)

    OUTPUT:
        None
    """
    county_fips_dict = df_cleaned[["county_name",
                                   "full_fips"]].drop_duplicates().to_dict(orient="records")

    # Write to json file
    with open("county_fips.json", "w") as f:
        json.dump(county_fips_dict, f)

def main():
    # Read Utility Energy data
    df = read_data("data/raw/Utility_Energy_Registry_Monthly_County_Energy_Use__Beginning_2021_20241208.csv")

    # FIPS code of interest ? (function for this)
    target_fips = [36051, 36053, 36055] # Generalize ?
    filtered_df = df["full_fips"].isin(target_fips)

    print("Columns after cleaning:", df.columns.tolist())

    # Now let's modify the filtering code to be more robust
    # First, find the actual FIPS column name if it exists
    fips_column = None
    for col in df.columns:
        if 'fips' in col.lower():
            fips_column = col
            break

    if fips_column is None:
        raise ValueError("No FIPS column found in the DataFrame")

    # Use the correct column name for filtering
    filtered_df = df[df[fips_column].isin(target_fips)]

    # Analyze cleaned and filtered data
    print("\n== Analysis of Target Counties ===")
    for fips in target_fips:
        county_data = filtered_df[filtered_df[fips_column] == fips]
        print(f"\nCounty FIPS: {fips}")
        print(f"Total records: {len(county_data)}")
        if len(county_data) > 0:
            print(f"Date range: {county_data['year'].min()}-{county_data['month'].min()} to {county_data['year'].max()}-{county_data['month'].max()}")
        else:
            print("No data found for this county")

    lat, lon = 43.1566, -77.6088  # Rochester, NY
    start_year = "2021"
    end_year = "2024"

    # API call
    nasa_api = NASAPowerAPI()
    weather_data = nasa_api.get_weather_data(lat, lon, start_year, end_year)

    # Obtaining appropriate dictionaries
    coordinates = weather_data["geometry"]["coordinates"] # Never used
    humidity = weather_data["properties"]["parameter"]["RH2M"]
    temp = weather_data["properties"]["parameter"]["T2M"]
    wind_speed = weather_data["properties"]["parameter"]["WS10M"]
    solar_radiation = weather_data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]

    # Converting dictionaries into DataFrames 
    # ? Fix the column names, pandas Series, etc.
    humidity_df = pd.DataFrame(list(humidity.items()), columns=["date",
                                                                "humidity%"])
    temp_df = pd.DataFrame(list(temp.items()), columns=["date", "temp (°C)"])
    wind_speed_df = pd.DataFrame(list(wind_speed.items()), columns=["date",
                                                                    "speed (m/s)"])
    solar_radiation_df = pd.DataFrame(list(solar_radiation.items()),
                                      columns=["date", "W/m²"])

    # List of dataframes
    dfs = [humidity_df, temp_df, solar_radiation_df, wind_speed_df]

    # Initialize with same columns as source DataFrames
    weather_df = pd.DataFrame(columns=['date'] + ['humidity%', 'temp (°C)', '', 'speed (m/s)'])

    # Merge energy dataframe with weather dataframe ?
    proper_weather_df = merge_weather_dataframes(dfs)

    breakpoint()

# Example Usage
if __name__ == "__main__":
    main()



