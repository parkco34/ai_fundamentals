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

def convert_weather_date_col(weather_df):
    """
    Convert NASA POWER'S string date (YYYYMMDD) to datetime
    -----------------------------------------------------------------
    INPUT:
        weather_df: (pd.DataFrame) Weather dataframe

    OUTPUT:
        new_weather_df: (pd.DataFrame) Updated weather dataframe
    """
    # Copy dataframe
    new_weather_df = weather_df.copy()

    try:
        new_weather_df["date"] = pd.to_datetime(new_weather_df["date"], format="%Y%m%d")

    except Exception as e:
        raise ValueError(f"Error converting weather date column: {e}")

    return new_weather_df

def aggregate_weather_monthly(weather_df):
    """
    Groups daily NASA weather data to monthnly averages (or sum)
    in order to match the energy dataframe for merging appropriately
    ------------------------------------------------------------
    INPUT:
        weather_df: (pd.DataFrame) Weather dataframe

    OUTPUT:
        monthly_weather: (pd.DataFrame) Matches the energy dataframe by
        incluing the first day of each month
    """
    # Extract year/month
    weather_df["year"] = weather_df["date"].dt.year
    weather_df["month"] = weather_df["date"].dt.month

    # Group by year, month: Groupby DataFrame using a mapper or by a Series of columns
    # .agg() Aggregates using one or more operations over the specified axis
    monthly_weather = weather_df.groupby(["year", "month"], as_index=False).agg({
        "humidity%": "mean",
        "temp (°C)": "mean",
        "speed (m/s)": "mean",
        "W/m²": "sum"   # or "mean
    })

    # Create monthly date, first day of each month matching energy
    monthly_weather["date"] = pd.to_datetime(
        monthly_weather["year"].astype(str)
        + monthly_weather["month"].astype(str)
        + "01",
        format="%Y%m%d"
    )

    return monthly_weather

def merge_energy_weather(energy_df, weather_df):
    """
    Merges the monthly energy dataframe with aggregated monthly weather
    --------------------------------------------------
    INPUT:
        energy_df: (pd.DataFrame) Energy dataframe
        weather_df: (pd.DataFrame) Weather dataframe

    OUTPUT:
        combined_df: (pd.DataFrame) energy_weather dataframe
    """
    # Esnure "data" is an actual column
    if "date" not in energy_df.columns:
        raise ValueError("Energy dataframe has no 'date' column; call energy_date_column first!")

    if "date" not in weather_df.columns:
        raise ValueError("Weather dataframe has no 'date' column; ensure convert_weather_date_col + aggregation ran")

    # merge dataframes
    combined_df = pd.merge(
        energy_df,
        weather_df,
        on="date",
        how="left" # or "inner", "right", "outer" as needed
    )

    return combined_df

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

    # Create datetime column for the dataframes
    energy_df = energy_date_column(filtered_df)
    weather_df = convert_weather_date_col(weather_df)

    # Ensure weather dataframe matches form of energy dataframe
    weather_df = aggregate_weather_monthly(weather_df)

    # Merge energy and weather dataframes
    combined_df = merge_energy_weather(energy_df, weather_df) 

    breakpoint()

# Example Usage
if __name__ == "__main__":
    main()

