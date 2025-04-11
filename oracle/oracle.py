#!/usr/bin/env python
"""
Oracle project:
    ? ->> Include error handling!
    - Plotting various data points
    - GUI for location selection
    - Interactive plotting, eventually
    - DataBase access and storage
    - API (NASA) connection
    - Generalize!!

*--> It is important to note that the data are subject to privacy screening and fields that fail the privacy screen are withheld
"""
from textwrap import dedent
from data_utils.data_cleaning import DataCleaning # Import certain class
from textwrap import dedent
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from nasa_power_api import NASAPowerAPI
import os
import traceback
import matplotlib

def read_data(filename):
    """
    Reads CSV file, returning a Pandas DataFrame.
    --------------------------------------------
    INPUT:
        filename: (str) FIlename of csv file

    OUTPUT:
        df: (pd.DataFrame) Pandas DataFrame
    """
    # Ensure filename exists in current directory
    if os.path.exists(filename):
        return pd.read_csv(filename)
    
    else:
        print(f"{filename} Doesn't seem to exist in the given path")
        return None

def get_weather_data(latitude, longitude, start_year, end_year):
    """
    Call to NASA POWER API for weather data.
    -------------------------------------------------
    INPUT:
        latitude: (float)
        longitude: (float)
        start_year: (str) Date range
        end_year: (str)

    OUTPUT:
        data: (dict) Dictionary of the raw weather data
    """
    # Call API
    try:
        nasa_api = NASAPowerAPI()
        # Get weather data
        data = nasa_api.get_weather_data(latitude, longitude, start_year, end_year)
        return data

    except Exception as e:
        print(f"Lo Siento ... NASA's Weather API call shit the bed with {str(e)}")
        return None 

def get_weather_dataframe(weather_data):
    """
    ? -> HAS TO BE A BETTER WAY THAN BEFORE ... ?
    Loops through a dictionary of dictionaries extracting the specific keys to store in a pandas dataframe.
    --------------------------------------------------
    INPUT:
        weather_data: (dict) Dictionary of information from the NASA Power API

    OUTPUT:
        (pd.DataFrame) Datetime converted dataframe merged with weather dataframe
    """
    # Initialize some parameters
    param_mapping = {
        "RH2M": "humidity%",
        "T2M": "temp (°C)",
        "WS10M": "speed (m/s)",
        "ALLSKY_SFC_SW_DWN": "W/m²"
    }

    # Initialize an empty DataFrame to store all values
    result_df = None

    # Extract parameters dictionary (only relevant info)
    parameters = weather_data["propertiesd"]["parameter"]

    # Loop thru each parameter, adding to dataframe
    # api_param for ['T2M', 'RH2M', 'WS10M', 'ALLSKY_SFC_SW_DWN']
    # and columns_name for the dataframe's column  names
    for api_param, columns_name in param_mapping.items():
        # Get the date-value dictionary for this parameter,
        # .get(key, what to fill in with {} if key not available)
        param_dict = parameters.get(api_param, {})

        # Create temporary DataFrame from dictionary items
        temp_df = pd.DataFrame(list(param_dict.items()), columns=["date", column_name])

        if result_df is None:
            # First parameter - create initial dataframe
            result_df = temp_df

        else:
            # Merge with existing dataframe on date column
            result_df = pd.merge(result_df, temp_df, on="date", how="outer")

    # Convert date column to datetime and ensure proper sorting
    return datetime_conversion(result_df)

def datetime_conversion(dataframe, sort_by_date=True):
    """
    Looks for a 'date' type column and converts it to datetime.
    If there's separate month, year columns, it properly combines them into one
    datetime column, removing the original columns.
    ------------------------------------------------------------
    INPUT:
        dataframe: (pd.DataFrame) Original dataframe
        sort_by_date: (bool) Whether to sort the dataframe by date or leave it as it is.

    OUTPUT:
        dataframe: (pd.DataFrame) Datetime converted dataframe
    """
    # Create a copy of dataframe
    df = dataframe.copy()

    # List of common date-related keywords
    date_keywords = ['date', 'time', 'day', 'month', 'year']

    # Track columns that were converted
    converted_columns = []

    # Check if separate year, month, day columns
    has_year = any("year" in col.lower() for col in df.columns)
    has_month = any("month" in col.lower() for col in df.columns)

    # If separate year and month columns, try to create a datetime columns
    if has_year and has_month:

        try:
            # Get correct column names via list comprehension, extracting first
            # element from the list
            year_col = [col for col in df.columns if col == "year"][0]
            month_col = [col for col in df.columns if col == "month"][0]
            # Create new "date" column if none exists
            if "date" not in df.columns:

                # If there's a 'day' column, use that shit!
                if any("day" in col.lower() for col in df.columns):
                    day_col = [col for col in df.columns if "day" in col.lower()][0]
                    df["date"] = pd.to_datetime(
                        df[year_col].astype(str) 
                        + "-" + 
                        df[month_col].astype(str).str.zfill(2)
                        + "-" +
                        df[day_col].astype(str).str.zfill(2),
                        errors="coerce"
                    )

                else:
                    # If no day column, use the 8th day of the month
                    df["date"] = pd.to_datetime(
                        df[year_col].astype(str)
                        + "-" +
                        df[month_col].astype(str).str.zfill(2)
                        + "-08",
                        errors="coerce"
                    )
                
                # Append converted columns to the list
                converted_columns.append("date")
                print(f"\nCreated new 'date' column from {year_col} and {month_col}")

        except Exception as e:
            print(f"\nFailed to create date column from year and month: {str(e)}")
    
    # Process each column that might contain date information
    for col in df.columns:
        col_lower = col.lower()
        
        # Skip columns we've already processed
        if col in converted_columns:
            continue
        
        # Check if any date keywords appear in the column name
        if any(keyword in col_lower for keyword in date_keywords):
            
            try:
                # Get a sample of non-null values o determine the format
                sample_values = df[col].dropna().astype(str).head().tolist()
                
                # Make user aware that there are no non-null values for the
                # column
                if len(sample_values) == 0:
                    print(f"Column '{col}' has no non-null values to convert")
                    continue

                # Check for different date formats, like if the value has all
                # date components
                if all(len(str(val)) == 8 and str(val).isdigit() for val in sample_values):
                    # Format: "20250608"
                    df[col] = pd.to_datetime(df[col].astype(str),
                                             format="%Y%m%d", yearfirst=True)
                    converted_columns.append(col)
                    print(f"Column '{col}' converted from YYYYMMDD format")

                elif all(len(str(val)) == 6 and str(val).isdigit() for val in
                         sample_values):

                    # Format like "202568," converting it to "20250608"
                    df[col] = pd.to_datetime(df[col].astype(str).zfill(2),
                                             format="%Y%m%d", yearfirst=True, errors="coerce")
                    converted_columns.append(col)
                    print(f"Column '{col}' converted from YYYYMM format")

                elif all(len(str(val)) == 4 and str(val).isdigit() for val in
                         sample_values):
                    # For only the 4-digit year, convert to 8-digits
                    df[col] = pd.to_datetime(df[col].astype(str) + "0608",
                                             format="%Y%m%d", yearfirst=True)
                
                elif any('-' in str(val) or '/' in str(val) for val in sample_values):
                    # Format with separators like "2021-01-01" or "01/01/2021"
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    converted_columns.append(col)
                    print(f"Column '{col}' converted with pandas automatic format detection")

                else:
                    # Try pandas' automatic parsing as a last resort
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if df[col].notna().any():
                        converted_columns.append(col)
                        print(f"Column '{col}' converted with pandas automatic parsing")
                    else:
                        print(f"Could not identify a date format for column '{col}'")

            except Exception as e:
                print(f"Could not convert column: '{col}' into datetime :  {str(e)}")

    # Print summary
    if converted_columns:
        print(f"\nSuccessfully converted {len(converted_columns)} date-related columns: {', '.join(converted_columns)}")

    else:
        print("\nNo date columns were converted")

    # Remove "date" from date_keywords list
    if "date" in date_keywords:
        date_keywords.remove("date")

    # Eliminate redundant date-related columns after establish 'date'
    if "date" in df.columns:
        redundant_cols = [col for col in df.columns if col in ["year", "month",
                                                              "day"]]
        df.drop(columns=redundant_cols, inplace=True)

    # Ensure the 'date' column is the first column of dataframe
    date_column = df.pop("date")

    # Move appropriate column to front of dataframe
    df.insert(0, "date", date_column)

    # Sort the dataframe by "date", ascending
    if sort_by_date and "date" in df.columns:
        df = df.sort_values(by="date").reset_index(drop=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df

def merge_energy_weather_location(energy_df, weather_df):
    """
    Merges energy consumption data with weather data and ensures location info is preseved.
    --------------------------------------------------------
    INPUT:
        energy_df: (pd.DataFrame) Energy consumption dataframe
        weather_df: (pd.DataFrame) Weather dataframe

    OUTPUT:
        merged_df: (pd.DataFrame) Merged dataframe with energy weather and location data.
    """
    # Ensure Both dataframes have a 'date' column
    if "date" not in energy_df.columns:
        energy_df = datetime_conversion(energy_df)

    if "date" not in weather_df.columns:
        weather_df = datetime__conversion(weather_df)

    # Merge dataframes on date
    merged_df = pd.merge(energy_df, weather_df, on="date", how="left")

    # Ensure location columns are preserved
    location_columns = ["county_name", "full_fips"]
    if all(col in merged_df.columns for col in location_columns):
        print(f"Location information preserved: {location_columns}")

    else:
        print("Warning: Some location columns are missing")

    return merged_df

def plot_energy_by_location(merged_df):
    """
    Visualization of energy, weather, and location (filtered by FIPS codes)
    ---------------------------------------------------------------
    INPUT:
        merged_df: (pd.DataFrame) 

    OUTPUT:
        None
    """
    # Get unique location
    locations = merged_df["county_name"]
    
    # Create new figure for the main plot
    fig, axs = plt.subsplots(
        len(locations),
        1,
        figsize=(16, 3*len(locations)-1), sharex=True
    )

    # Ensure axs is always list
    if len(locations) == 1:
        axs = [axs]

    # Loop thru each location
    for i, location in enumerate(locations):
        # Create a copy to avoid SettingWithCopyWarning
        location_data = merged_df[merged_df["county_name"] == location]

        # p Diagnostic info
        print(f"\nLocation: {location}")
        print(f"Records: {len(location_data)}")

        if location_data.empty:
            print(f"There's nothing in this bitch!: {location_data}")
            continue

        # Sort values for time series
        location_data = location_data.sort_values(by="date")
        
        # Verify value in column for existence
        if "value" not in location_data.columns:
            print(f"Error: No 'value' column for the location: {location}")
            continue

        # PLot data
        try:
            # Use explicit x and y data
            pass

df = read_data("data/raw/Utility_Energy_Registry_Monthly_County_Energy_Use__Beginning_2021_20241208.csv")
lat, lon = 43.1566, -77.6088  # Rochester, NY
start_year = "2021"
end_year = "2024"
weather_data = get_weather_data(lat, lon, start_year, end_year)
parameters = weather_data["properties"]["parameter"]
breakpoint()

