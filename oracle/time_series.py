#!/usr/bin/env python
"""
Time Series Analysis
ACF: Quantifies similarity between observations of a RANDOM VARIABLE at
different points in time, in order to understand the behavior over time.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf # ?
from statsmodels.tsa.seasonal import seasonal_decompose # ?


class TimeSeries:

    def __init__(
        self,
        data,
        date_column=None,
        target_column=None,
        frequency=None
    ):
        """
        Initializes the TimeSeries object, taking in the dataframe, target and date columns.
        ------------------------------------------------
        INPUT:
            data: (pd.DataFrame) Time Series data
            date_column: (str) Name of date column
            target_column: (str) Name of target column
            frequency: (str) Frequency of time series ("D" for daily, "M" for monthly, etc.)

        OUTPUT:
        """
        # Assign basic attributes
        self.time_series = None
        self.original_data = None
        self.date_column = date_column
        self.target_column = target_column
        self.frequency = frequency

        if data is not None:
            self.load_data(data, date_column, target_column, frequency)
            print(f"\n\nLoading data first!\n\nSummary: {data}")

    def load_data(
        self, 
        data, 
        date_column=None, 
        target_column=None, 
        frequency=None
    ):
        """
        Load time series data into the object.
        ---------------------------------------------
        INPUT:
        data: (pd.DataFrame or CSV file) Time Series data
        date_column: (str) Name of date column
        target_column: (str) Name of target column
        frequency: (str) Frequency of time series ("D" for daily, "M" for monthly, etc.)

        OUTPUT:
            self: (TimeSeries) Updated object.
        """
        # Load data from file if path is provided
        if isinstance(data, str):
            data = pd.read_csv(data)

        # Store original data
        self.original_data = data

        # Convert DataFrame to Series if needed
        if isinstance(data, pd.DataFrame) and date_column is not None and target_column is not None:

            # Check if index is already datetime, converting it if needed
            if not isinstance(data.index, pd.DatetimeIndex):
                data = self.datetime_conversion(data)

            # Obtain pandas Series after converting to DatetimeIndex
            data =  data.set_index(date_column)[target_column]

        # Ensure data is a Series with DatetimeIndex
        if isinstance(data, pd.Series):

            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("Data index must be a DatetimeIndex")
            # We now haz time series!
            self.time_series = data

        else:
            raise ValueError("Data must be a pandas Series with DatetimeIndex")

        # Establish frequency
        if frequency is not None:
            # DataFrame.asfreq(freq) Converts time series to sepcified frequency
            self.time_series = self.time_series.asfreq(frequency)
            self.frequency = frequency

        else:
            # Infer the most likely frquency given input index
            # Uses value of the series and NOT the index
            self.frequency = pd.infer_freq(self.time_series.index)

        return self

    def datetime_conversion(self, dataframe):
        """
        Looks for a 'date' type column and converts it to datetime.
        If there's separate month, year columns, it properly combines them into one
        datetime column, removing the original columns.
        ------------------------------------------------------------
        INPUT:
            dataframe: (pd.DataFrame) Original dataframe

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
        if "date" in df.columns:
            df = df.sort_values(by="date").reset_index(drop=True)

        # Reset index
        df.reset_index(drop=True, inplace=True)

        return df

    def simple_time_series_plot(self):
        """
        Simple time series plot for easy visualization.
        ---------------------------------------------
        INPUT:
            None

        OUTPUT:
            None
        """
        plt.figure(figsize=(14, 7))
        plt.plot(self.time_series.index, self.time_series.values)
        plt.title("Energy Consumption Over Time")
        plt.xlabel("Date")
        plt.ylabel("Energy Usage")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    def visualize_data(
        self,
        figsize=(14, 8),
        show_rolling=False,
        window=12
    ):
        """
        Visualize the time series data.
        ------------------------------------------------
        INPUT:
            figsize: (width, height; default=(14, 8))
            show_rolling: (bool) Whether to show the rolling average (default=False)
            window: (int; default=12) Window size for the rolling average.

        OUTPUT:
            fig: (matplotlib.pyplot) Matplotlib figure object.
        """
        if self.time_series is None:
            raise ValueError("No data provided. Load data first.")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot the time series
        ax.plot(self.time_series.index, self.time_series.values, label="Time Series")

        # Plotting rolling average if requested
        if show_rolling:
            rolling_avg = self.time_series.rolling(window=window).mean()
            ax.plot(rolling_avg.index, rolling_avg.values,
                    label=f"{window}-period Rolling Average",
                    color="red")

            # Add labels and title
            ax.set_xlabel("Date")
            ax.set_ylabe("Value")
            ax.set_title("Time Series Plot")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()

            return fig
