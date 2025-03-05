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
        pass

