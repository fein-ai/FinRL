from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

from abc import ABC, abstractmethod
from stockstats import StockDataFrame as Sdf

import numpy as np
import pandas as pd
import pytz
import logbook

class AbstractProcessor(ABC):

    def __init__(self):
        self.logger = logbook.Logger(self.__class__.__name__)
    
    def add_technical_indicator(
        self,
        df,
        tech_indicator_list=[
            "macd",
            "boll_ub",
            "boll_lb",
            "rsi_30",
            "dx_30",
            "close_30_sma",
            "close_60_sma",
        ],
    ):
        self.logger.info("Started adding Indicators")

        print ( df.dtypes)
        self.logger.info ( df["timestamp"])
        # Store the original data type of the 'timestamp' column
        original_timestamp_dtype = df["timestamp"].dtype
        self.logger.info ( 'hihihi')

        # Convert df to stock data format just once
        stock = Sdf.retype(df)
        unique_ticker = stock.tic.unique()
        self.logger.info ( f"unique_ticker: {unique_ticker}")

        # Convert timestamp to a consistent datatype (timezone-naive) before entering the loop
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)
        self.logger.info ( f"after convert")

        # Create a dictionary to store the intermediate results
        indicator_data = {}
        
        with ProcessPoolExecutor() as executor:
            for indicator in tech_indicator_list:
                indicator_dfs = list(executor.map(self.calculate_indicator, [(stock, df, tic, indicator) for tic in unique_ticker]))

                # Concatenate all intermediate dataframes at once
                indicator_data[indicator] = pd.concat(indicator_dfs, ignore_index=True)

        # Merge the indicator data frames
        for indicator, indicator_df in indicator_data.items():
            df = df.merge(
                indicator_df[["tic", "date", indicator]],
                left_on=["tic", "timestamp"],
                right_on=["tic", "date"],
                how="left"
            ).drop(columns="date")

        self.logger.info("Restore Timestamps")
        # Restore the original data type of the 'timestamp' column
        if isinstance(original_timestamp_dtype, pd.DatetimeTZDtype):
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            df["timestamp"] = df["timestamp"].dt.tz_convert(original_timestamp_dtype.tz)
        else:
            df["timestamp"] = df["timestamp"].astype(original_timestamp_dtype)

        self.logger.info("Finished adding Indicators")
        return df

    def calculate_indicator(self, args):
        stock, df, tic, indicator = args
        tic_data = stock[stock.tic == tic]
        indicator_series = tic_data[indicator]

        tic_timestamps = df.loc[df.tic == tic, "timestamp"]

        indicator_df = pd.DataFrame({
            "tic": tic,
            "date": tic_timestamps.values,
            indicator: indicator_series.values
        })
        return indicator_df

    def calculate_turbulence(self, data, time_period=252) -> pd.DataFrame:
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="timestamp", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.timestamp.unique()
        # start after a fixed timestamp period
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"timestamp": df_price_pivot.index, "turbulence": turbulence_index}
        )

        # self.logger.info("turbulence_index\n", turbulence_index)

        return turbulence_index

    def add_turbulence(self, data, time_period=252):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def df_to_array(self, df, tech_indicator_list, if_vix):
        df = df.copy()
        unique_ticker = df.tic.unique()
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["close"]].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic == tic]["VIXY"].values
                else:
                    turbulence_array = df[df.tic == tic]["turbulence"].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][tech_indicator_list].values]
                )
        #        self.logger.info("Successfully transformed into array")
        return price_array, tech_array, turbulence_array
    
    @staticmethod
    def clean_individual_ticker(args):
        tic, df, times = args
        tmp_df = pd.DataFrame(index=times)
        tic_df = df[df.tic == tic].set_index("timestamp")

        # Step 1: Merging dataframes to avoid loop
        tmp_df = tmp_df.merge(
            tic_df[["open", "high", "low", "close", "volume"]],
            left_index=True,
            right_index=True,
            how="left",
        )

        # Step 2: Handling NaN values efficiently
        if pd.isna(tmp_df.iloc[0]["close"]):
            first_valid_index = tmp_df["close"].first_valid_index()
            if first_valid_index is not None:
                first_valid_price = tmp_df.loc[first_valid_index, "close"]
                logbook.info(
                    f"The price of the first row for ticker {tic} is NaN. It will be filled with the first valid price."
                )
                tmp_df.iloc[0] = [first_valid_price] * 4 + [0.0]  # Set volume to zero
            else:
                logbook.info(
                    f"Missing data for ticker: {tic}. The prices are all NaN. Fill with 0."
                )
                tmp_df.iloc[0] = [0.0] * 5

        for i in range(1, tmp_df.shape[0]):
            if pd.isna(tmp_df.iloc[i]["close"]):
                previous_close = tmp_df.iloc[i - 1]["close"]
                tmp_df.iloc[i] = [previous_close] * 4 + [0.0]

        # Setting the volume for the market opening timestamp to zero - Not needed
        # tmp_df.loc[tmp_df.index.time == pd.Timestamp("09:30:00").time(), 'volume'] = 0.0

        # Step 3: Data type conversion
        tmp_df = tmp_df.astype(float)

        tmp_df["tic"] = tic

        return tmp_df

    def download_and_clean_data(self, start, end, time_interval) -> pd.DataFrame:
        """
        Downloads and cleans the data for further processing.

        Returns:
            pandas.DataFrame: The cleaned data.
        """
        vix_df = self.download_data(["VIXY"], start, end, time_interval)
        return self.clean_data(vix_df)

    def add_vix(self, data, start, end, interval) -> pd.DataFrame:
        """
        Adds the VIX data to the given DataFrame.

        Parameters:
        - data (pandas.DataFrame): The DataFrame to which the VIX data will be added.

        Returns:
        - pandas.DataFrame: The DataFrame with the VIX data added.
        """
        with ThreadPoolExecutor() as executor:
            future = executor.submit(self.download_and_clean_data)
            cleaned_vix = future.result()

        vix = cleaned_vix[["timestamp", "close"]]

        merge_column = "date" if "date" in data.columns else "timestamp"

        vix = vix.rename(
            columns={"timestamp": merge_column, "close": "VIXY"}
        )  # Change column name dynamically

        data = data.copy()
        data = data.merge(
            vix, on=merge_column
        )  # Use the dynamic column name for merging
        data = data.sort_values([merge_column, "tic"]).reset_index(drop=True)

        return data

    @abstractmethod
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the data for further processing.

        Args:
            data (pd.DataFrame): The raw data to be cleaned.

        Returns:
            pd.DataFrame: The cleaned data.

        """
        pass
    
    @abstractmethod
    def download_data(
            self, ticker_list, start_date, end_date, time_interval
        ) -> pd.DataFrame:
            """
            Downloads financial data for the specified ticker list and time range.

            Args:
                ticker_list (list): A list of ticker symbols.
                start_date (str): The start date of the data range in the format 'YYYY-MM-DD'.
                end_date (str): The end date of the data range in the format 'YYYY-MM-DD'.
                time_interval (str): The time interval for the data, e.g., '1d' for daily data.

            Returns:
                pd.DataFrame: A pandas DataFrame containing the downloaded financial data.

            """
            pass

    @abstractmethod
    def fetch_latest_data(
            self, ticker_list, time_interval, tech_indicator_list, limit=100
        ) -> pd.DataFrame:
            """
            Fetches the latest data for the given ticker list, time interval, and technical indicator list.

            Parameters:
            ticker_list (list): A list of ticker symbols.
            time_interval (str): The time interval for the data (e.g., '1d' for daily, '1h' for hourly).
            tech_indicator_list (list): A list of technical indicators to include in the data.
            limit (int, optional): The maximum number of data points to fetch. Defaults to 100.

            Returns:
            pd.DataFrame: A DataFrame containing the fetched data.

            """
            pass


    @abstractmethod
    def close_conn(self) -> None:
            """
            Closes the connection to the data source.

            This method is responsible for closing the connection to the data source
            once the data processing is complete or when it is no longer needed.

            Parameters:
                None

            Returns:
                None
            """
            pass
