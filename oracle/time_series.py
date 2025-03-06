#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class TimeSeries:

    def __init__(
        self,
        data=None,
        date_column=None,
        target_column=None
    ):
        """
        Initializes the TimeSeries object.
        ------------------------------------------------
        INPUT:

        OUTPUT:
        """
        pass

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




