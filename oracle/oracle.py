#!/usr/bin/env python
"""
Oracle project:
    ? ->> Include error handling!

*--> It is important to note that the data are subject to privacy screening and fields that fail the privacy screen are withheld
"""
from data_utils.data_cleaning import DataCleaning # Import certain class
from textwrap import dedent
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
        print(f"    - Negative values: {(dataframe['value'] < 0).sum()}")
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
    county_fips_dict = dataframe[["county_name",
                                   "full_fips"]].drop_duplicates().to_dict(orient="records")

    # Write to json file
    with open("county_fips.json", "w") as f:
        json.dump(county_fips_dict, f)

def fips_df(df, filter_by_fips):
    """
    Verify the FIPS code column exists and then filter dataset to include only
    those FIPS codes are relevant to user's location preferences
    ---------------------------------------------------------------
    INPUT:
        df: (pd.DataFrame) original dataframe
        filter_by_fips: (list) FIPS codes of interest

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
    filtered_df = df[df[fips_column].isin(filter_by_fips)]
    
    # Analyze filtered data
    print(f"=== Analysis fo Target Counties ===")
    for fips in filter_by_fips:
        # Get county data via relevant FIPS codes
        county_data = filtered_df[filtered_df[fips_column] == fips]
        print(f"\nCounty FIPS: {fips}")
        print(f"Total Records: {len(county_data)}")

        if not county_data.empty:
            print(f"""Date range:
                  {county_data['year'].min()}-{county_data['month'].min()} to
                  {county_data['year'].max()}-{county_data['month'].max()}""")

        else:
            print("No data found for this county")

    return filtered_df

def get_weather_data(latitude, longitude, start_year, end_year):
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
    data = nasa_api.get_weather_data(latitude, longitude, start_year, end_year)
    
    return data

def get_weather_df(weather_data):
    """
    Takes the raw weather data and configures it such that a proper dataframe
    is outputn with the appropriate information.
    ------------------------------------------------------
    INPUT:
        weather_data: (dict) Dictionary of information from the NASA Power API

    OUTPUT:
        (pd.DataFrame) Datetime converted dataframe merged with weather dataframe
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

    # Convert any date-related columns to datetime 

    return datetime_conversion(proper_weather_df)

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
            # Get correct column names
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

def preprocess_data(dataframe):
    """
    Handles missing values and outliers via DataCleaning class
    ------------------------------------------------------------
    INPUT:
        dataframe: (pd.DataFrame) Dataframe to be preprocessed

    OUTPUT:
        proper_df: (pd.DataFrame) Preprocessed dataframe.
    """
    # Initialize DataCleaning class
    dc = DataCleaning(dataframe)

    # Output dataframe info prior to preprocessing
    print("=== Before Preprocessing ===")
    print(f"""Rows: {dc.dataframe.shape[0]}\n
          Columns: {dc.dataframe.shape[1]}\n
          Missing: {dc.dataframe.isnull().sum()}
          """)

    # Drop missing columns via threshold
    dc.drop_cols_missing_data(threshold=0.5)

    # Remove invalid values like -999 or -999.0 (inplace)
    dc.replace_negative_values()
    
    # Display cleaned data
    display_cleaned_dataset(dc.dataframe,
                            "Drop columns with >50% Missing data")
    
    # Impute missing values
    for col in dc.dataframe.columns:

        # For count of null values greater than zero
        if dc.dataframe[col].isnull().sum() > 0:
            # Check for numeric data types
            if pd.api.types.is_numeric_dtype(dc.dataframe[col]):
                dc.imputing_vals_mean(col)

            else:
                dc.imputing_vals_categorical_cols(col)

    # Forward fill that shit
    dc.forward_fill()
    # Drop missing data
    dc.drop_rows_missing_data()
    # Display the cleaned dataset
    display_cleaned_dataset(dc.dataframe, "After preprocessing")

    return dc.dataframe

def main():
    # Read Energy dataset
    # ? --> Replace this with a C++ GUI for user to make selection ?
    df = \
    read_data("data/raw/Utility_Energy_Registry_Monthly_County_Energy_Use__Beginning_2021_20241208.csv")
    
    # Filter dataframe via FIPS
    filter_by_fips = [36051, 36053, 36055]

    # Energy dataframe for specific region
#    filtered_df = fips_df(df, filter_by_fips)

    # Get weather dataframe
    lat, lon = 43.1566, -77.6088  # Rochester, NY
    start_year = "2021"
    end_year = "2024"
    weather_data = get_weather_data(lat, lon, start_year, end_year)

    # Weather dataframe
    weather_df = get_weather_df(weather_data)
    
    # Consolidate "date-like" columns into one, converting it to pd.datetime
    datetime_df = datetime_conversion(df)

    # Instantiate DataCleaning Object (dc)
    dc = DataCleaning(datetime_df)

    # ? data preprocessing ?
#    proper_df = preprocess_data(df)
    proper_df = preprocess_data(datetime_df)
    
    # Summary with the top 10 
    summary = dc.column_summary(10)
    print(f"\nSummary DataFrame:\n {summary}")
    breakpoint()

    # Summary about dataframe
#    for col in proper_df.columns:
#        print(f"Column: '{col}'\n{proper_df[col].value_counts().unique()}") 


if __name__ == "__main__":
    main()
