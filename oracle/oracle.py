#!/usr/bin/env python
"""
Oracle project:
    ? ->> Include error handling!

*--> It is important to note that the data are subject to privacy screening and fields that fail the privacy screen are withheld
"""
from data_utils.data_cleaning import DataCleaning # Import certain class
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from nasa_power_api import NASAPowerAPI

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

def export_county_fips(dataframe):
    """
    Exports the county FIPS codes to be selectable via the Java GUI 
    in a JSON file.
    -----------------------------------------------------------
    INPUT:
        dataframe: (pd.DataFrame)

    OUTPUT:
        None
    """
    county_fips_dict = df_cleaned[["county_name",
                                   "full_fips"]].drop_duplicates().to_dict(orient="records")

    # Write to json file
    with open("county_fips.json", "w") as f:
        json.dump(county_fips_dict, f)

def get_fips(df, target_fips):
    """
    Verify the FIPS code column exists and then filter dataset to include only
    those FIPS codes are relevant to user's location preferences
    ---------------------------------------------------------------
    INPUT:
        df: (pd.DataFrame) original dataframe
        target_fips: (list) FIPS codes of interest

    OUTPUT:
        filtered_df: (pd.DataFrame) Filtered dataframe w/ proper fips codes.
    """
    # Check for FIPS column
    fips_column = None
    for col in df.columns:
        if "fips" in col.lower():
            fips_column = col
            break

    if fips_column is None:
        raise ValueError("No FIPS column! Using entire dataframe")

    # Use correct column name for filtering
    filtered_df = df[df[fips_column].isin(target_fips)]
    
    # Analyze filtered data
    print(f"++ Analysis fo Target Counties ++")
    for fips in target_fips:
        county_data = filtered_df[filtered_df[fips_column] == fips]
        print(f"\nCounty FIPS: {fips}")
        print(f"Total Records: {len(county_data)}")

        if len(county_data) > 0:
            print(f"""Date range:
                  {county_data['year'].min()}-{county_data['month'].min()} to
                  {county_data['year'].max()}-{county_data['month'].max()}""")

        else:
            print("No data found for this county")

    return filtered_df

def gets_weather_data(latitude, longitude, start_year, end_year):
    """
    Call to NASA POWER API for weather data
    ------------------------------------------
    INPUT:
        latitude: (float)
        longitude: (float)
        start_year: (str) Date range
        end_year: (str)

    OUTPUT:
        data: (dict) Dictionary of the raw weather data
    """
    # Call API
    nasa_api = NASAPowerAPI()
    # Get weather data
    data = nasa_api.get_weather_data(latitude, longitude, start_year,
                                             end_year)
    
    return data

def get_weather_df(weather_data):
    """
    Takes the raw weather data and configures it such that a proper dataframe
    is outputn with the appropriate information.
    ------------------------------------------------------
    INPUT:
        weather_data: (dict) Dictionary of information from the NASA Power API

    OUTPUT:
        weather_df: (pd.DataFrame) Weather dataframe
    """
    # Getting the appropriate dictionaries
    humidity = weather_data["properties"]["parameter"]["RH2M"]
    temp = weather_data["properties"]["parameter"]["T2M"]
    wind_speed = weather_data["properties"]["parameter"]["WS10M"]
    solar_radiation = weather_data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]

    # Getting proper weather dataframes
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
    weather_df = pd.DataFrame(columns=['date'] + ['humidity%', 'temp (°C)', 'speed (m/s)', 'solar radiation (W/m²)'])

    # Using merge_weather_dataframes function for combining dataframes
    proper_weather_df = merge_weather_dataframes(dfs)

    return proper_weather_df

def energy_date_column(df):
    """
    Creates a "date" column from "year" and "month" columns for mothly data
    ---------------------------------------------------------------
    INPUT:
        df: (pd.DataFrame) Energy dataframe

    OUTPUT:
        new_df: (pd.DataFrame) New dataframe
    """
    # Copy dataframe
    new_df = df.copy()

    # Check required columns
    if "year" not in df.columns or "month" not in df.columns:
        raise ValueError("Energy dataframe must have 'year' and 'month' columns")

    # Convert to datetime (YYYY-MM-01)
    try:
        # zfill(N) used to insert a N-1 number of zeros in front of string
        new_df["date"] = pd.to_datetime(new_df["year"].astype(str) +
                                    new_df["month"].astype(str).str.zfill(2) + "01", format="%Y%m%d")

    except Exception as e:
        raise ValueError(f"Error converting year/month to datetime: {e}")

    return new_df

def main():
    # Read Energy dataset
    df = read_data("data/raw/Utility_Energy_Registry_Monthly_County_Energy_Use__Beginning_2021_20241208.csv")
    
    # Filter dataframe via FIPS
    target_fips = [36051, 36053, 36055]
    # Energy dataframe
    filtered_df = get_fips(df, target_fips)

    # Get weather dataframe
    lat, lon = 43.1566, -77.6088  # Rochester, NY
    start_year = "2021"
    end_year = "2024"
    weather_data = gets_weather_data(lat, lon, start_year, end_year)
    # Weather dataframe
    weather_df = get_weather_df(weather_data)

    # Merge the two dataframes: energy and weather

    breakpoint()

# Example Usage
if __name__ == "__main__":
    main()

