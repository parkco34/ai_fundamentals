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

    def decompose(self, model="additive", period=None, figsize=(14, 12)):
        """
        Decompose the time series into trend, seasonal, and residual components.
        --------------------------------------------------------
        INPUT:
            model: (str) Type of decomposition "additive", or "multiplcative"
            (default="additive")
            period: (int) Period of seasonality.
            figsize: (tuple of integers) Figure size in inches (width, height) (default=(14 12))

        OUTPUT:
            decomposition: (dict) Dictionary containing the ndecomposed components and the figure.
        """
        # Ensure data is properly loaded
        if self.time_series is None:
            raise ValueError("No data loaded.  Please load the data!")

        # Determine period if not specified
        if period is None:
            # Frequency = monthly
            if self.frequency in ["M", "MS"]:
                period = 12
            
            # Quarterly data
            elif self.frequency in ["Q", "QS"]:
                period = 4

            # Daily data
            elif self.frequency in ["D"]:
                period = 7

            # Default setting to annually
            else:
                period = 12

        # Performance decomposition
        decomposition_result = seasonal_decompose(
            self.time_series, 
            model=model, 
            period=period
        )

        # Store results
        components = {
            "trend": decomposition_result.trend,
            "seasonal": decomposition_result.seasonal,
            "residual": decomposition_result.resid,
            "observed": decomposition_result.observed
        }

        # Visualize decomposition
        fig, axes = plt.subplots(
            4,
            1,
            figsize=figsize,
            sharex=True
        )

        # Plot observed data
        axes[0].plot(components["observed"].index,
                     components["observed"].values)
        axes[0].set_title("Observed Time Series")
        axes[0].grid(True)

        # Plot observed data
        axes[1].plot(components["trend"].index, components["trend"].values, color="orange")
        axes[1].set_title("Trend Component")
        axes[1].grid(True)

        # Plot seasonal component
        axes[2].plot(components["seasonal"].index,
                     components["seasonal"].values, color="green")
        axes[2].set_title("Seasonal Component")
        axes[2].grid(True)

        # Plot residual component
        axes[3].plot(components["residual"].index,
                     components["residual"].values,
                    color="red")

        plt.tight_layout()

        return {"components": components,
                "figure": fig}

    def check_stationary(self, figsize=(14, 8), window=12):
        """
        Check the stationary of the time series using Augmented Dickey-Fuller test and visual inspection of rolling statistics.
        --------------------------------------------------------
        INPUT:
            figsize: (width, height) In inches (Defualt: (14, 8))
            window: (int) Window size for rolling statistics (default: 12).

        OUTPUT:
            result: (dict) Dictionary containing test results and the figure.
        """
        # Ensure data loaded properly
        if self.time_series is None:
            raise ValueError("No data loaded.  Please figure it out, dude!")

        # Calculating rolling statistics
        rolling_mean = self.time_series.rolling(window=window).mean()
        rolling_std = self.time_series.rolling(window=window).std()
        
        # Perform ADF test
        adf_result = adfuller(self.time_series.dropna())
        adf_output = {
            "test_statistic": adf_result[0],
            "p_value": adf_result[1],
            "lags_used": adf_result[2],
            "num_observation": adf_result[3],
            "critical_values": adf_result[4]
        }

        # Is the series stationary?
        is_stationary = adf_result[1] < 0.05

        # Visualize rolling statistics
        fig, ax = plt.subplots(figsize=figsize)

        # Plot time series
        ax.plot(self.time_series.index, self.time_series.values, label="Time Series")

        # Plot rolling mean
        ax.plot(rolling_mean.index, rolling_mean.values,
                label=f"{window}-period Rolling Mean", color="red")
        
        # Plot rolling standard deviation
        ax.plot(rolling_std.index, rolling_std.values, label=f"{window}-period Rolling Standard Deviation", color="green")

        # Add labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.set_title("Rolling Statistics")
        ax.legend()
        ax.grid(True)

        # add adf test results as text
        stationary_text = "stationary (p < 0.05)" if is_stationary else "non-stationary (p >= 0.05)"
        ax.text(0.01, 0.05,
                f"adf test statistic: {adf_output['test_statistic']:.4f}\n"
                f"p-value: {adf_output['p_value']:.4f}\n"
                f"result: {stationary_text}",
                transform=ax.transaxes, fontsize=12, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        return {
            "adf_output": adf_output,
            "is_stationary": is_stationary,
            "figure": fig
        }

    def make_stationary(self, method="difference", order=1):
        """
        Transform the time series to make it stationary.
        -------------------------------------------------------
        INPUT:
            method: (str) Method to use: "difference", "log", or
            "log_difference" (default: "difference")
            order: (int) Order of differencing (default: 1)

        0UTPUT:
            self: (TimeSeries) Updated object with stationary time series
        """
        # Ensure data loaded properly
        if self.time_series is None:
            raise ValueError("No data loaded. Get your life together, please!")

        # Store original data if not already stored
        if not hasattr(self, "original_time_series"):
            self.original_time_series = self.time_series.copy()

            # Apply transformation
            if method == "difference":
                self.time_series = self.time_series.diff(order).dropna()
                self.transformation = {"method": "difference", "order": order}

            elif method == "log":
                # Check for non-positive values
                if (self.time_series <= 0).any():
                    raise ValueError("Cannot take log on non-positive values!")
                self.time_series = np.log(self.time_series)
                self.transformation = {"method": "log"}

            elif method == "log_difference":
                # Check for non-positive values
                if (self.time_series <= 0).any():
                    raise ValueError("Cannot take log on non-positive values!")
                self.time_series = \
                np.log(self.time_series).diff(order).dropna()
                self.transformation = {"method": "log_difference", "order": order}

            else:
                raise ValueError("Invalid transformation method")

    def restore_original(self):
        """
        Restore the original time series data.
        -------------------------------------------------------
        INPUT:
            None

        OUTPUT:
            self: (TimeSeries) UPdated object with original time series data.
        """
        if hasattr(self, "original_time_series"):
            self.time_series = self.original_time_series.copy()

            if hasattr(self, "tranformation"):
                delattr(self, "transformation")

        return self

    def plot_acf_pacf(self, lags=40, figsize=(14, 8)):
        """
        Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).
        ----------------------------------------------------
        INPUT:
            lags: (int) Number of lags to include (default: 40)
            figsize: (tuple: width, height) in inches (default: (14, 8))

        OUTPUT:
            fig: (matplotlib.pyplot) Figure object
        """
        # Ensure data loaded proplery
        if self.time_series is None:
            raise ValueError("Data not loaded... FIGURE IT OUT!")

        # Calculate ACF and PACF
        acf_values = acf(self.time_series.dropna(), nlags=lags)
        pacf_values = pacf(self.time_series.dropna(), nlags=lags)

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)

        # PLot ACF
        # Stem plot draws lines perpendicular to a baseline at each location
        # (locs) from baseline to (heads), and places marker there.
        axes[0].stem(range(len(acf_values)), acf_values, linefmt='b-', markerfmt="bo", basefmt="r-")
        axes[0].set_title("Autocorrelation Function (ACF)")
        axes[0].set_xlabel("Lag")
        axes[0].set_ylabel("Correlation")
        axes[0].axhline(y=0, linestyle="--", color="black")

        # Confidence intervals
        confidence = 1.96 / np.sqrt(len(self.time_series))
        axes[0].axhline(y=confidence, linestyle="--", color="gray")
        axes[0].axhline(y=-confidence, linestyle="--", color="gray")

        # Plot PACF
        axes[1].stem(range(len(pacf_values)), pacf_values, linefmt="b-", markerfmt="bo", basefmt="r-")
        axes[1].set_title("Partial Autocorrelation Function (PACF)")
        axes[1].set_xlabel("Lag") 
        axes[1].set_ylabel("Correlation")
        axes[1].axhline(y=0, linestyle="--", color="black")

        # Add confidence intervals
        axes[1].axhline(y=confidence, linestyle="--", color="gray")
        axes[1].axhline(y=-confidence, linestyle="--", color="gray")

        plt.tight_layout()

        return fig

    def custom_arima(
        self,
        p=1,
        d=1,
        q=1,
        exog=None,
        transform_back=True
    ):
        """
        Custom implementation of ARIMA model from scratch.
        Simplified version that uses statsmodels for parameter estimation but implements the forecasting logic manually.
        --------------------------------------------------
        INPUT:
            p: (int) Order of the AR term (default: 1).
            d: (int) Order of differencing (default: 1).
            q: (int) Order of the MA term (default: None)
            exog: (pd.DataFrame) Exogenou variables (default: None).
            transform_back: (bool) Whether to transform predictions back to original scale (default: True).

        OUTPUT:
            model_result: (dict) Dictionary containing the model and its results.
        """
        pass


