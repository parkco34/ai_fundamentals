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
        # .get(key, what to fill in with if key not available)
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
    pass



df = read_data("data/raw/Utility_Energy_Registry_Monthly_County_Energy_Use__Beginning_2021_20241208.csv")
lat, lon = 43.1566, -77.6088  # Rochester, NY
start_year = "2021"
end_year = "2024"
weather_data = get_weather_data(lat, lon, start_year, end_year)
breakpoint()

