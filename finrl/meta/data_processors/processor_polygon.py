from __future__ import annotations

import numpy as np
import pandas as pd
import logbook

import alpaca_trade_api as tradeapi
import exchange_calendars as tc
import numpy as np
import pandas as pd
import pytz

from polygon import RESTClient
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from stockstats import StockDataFrame as Sdf
from finrl.meta.data_processors.schemas import DownloadDataSchema

class PolygonProcessor:
    def __init__(self, api_key=None):
        
        self.client = RESTClient(api_key=api_key)
        self.logger = logbook.Logger(type(self).__name__)
        self.tz = "America/New_York"

    def _time_interval_to_multiplier_timespan(self, time_interval:str):
        if time_interval.lower() == "1min":
            return 1, "minute"
        elif time_interval.lower() == "5min":
            return 5, "minute"
        elif time_interval.lower() == "15min":
            return 15, "minute"
        elif time_interval.lower() == "1h":
            return 1, "hour"
        elif time_interval.lower() == "1d":
            return 1, "day"
        else:
            raise ValueError("Invalid time interval")
        
        

    def _fetch_data_for_ticker(self, ticker, start_date, end_date, time_interval):
        self.logger.info(f"Fetching data for {ticker} from {start_date} to {end_date} with interval {time_interval}")
        multiplier, timespan = self._time_interval_to_multiplier_timespan(time_interval)
        limit = 50000
        # List Aggregates (Bars)
        aggs = []
        for a in self.client.list_aggs(ticker=ticker, multiplier=multiplier, timespan=timespan, from_=start_date, to=end_date, limit=limit):
            aggs.append(a) 

        all_data = pd.DataFrame( aggs)
        
        all_data['timestamp'] = pd.to_datetime( all_data['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert( self.tz)
        all_data['tic'] = ticker
        all_data['volume'] = all_data['volume'].astype(pd.Int64Dtype())

        
        # self.logger.info(f"Data fetched for {ticker}")
        return DownloadDataSchema.validate(all_data)


    def download_data(
        self, ticker_list, start_date, end_date, time_interval
    ) -> pd.DataFrame:
        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval

        
        # data = pd.DataFrame()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    self._fetch_data_for_ticker,
                    ticker,
                    start_date,
                    end_date,
                    time_interval,
                )
                for ticker in ticker_list
            ]
            data_list = [future.result() for future in futures]


        
        
        # Combine the data
        data_df = pd.concat(data_list, axis=0)
        data_df.set_index("timestamp", inplace=True)

        # Convert the timezone
        data_df = data_df.tz_convert( self.tz)
        

        # If time_interval is less than a day, filter out the times outside of NYSE trading hours
        if pd.Timedelta(time_interval) < pd.Timedelta(days=1):
            data_df = data_df.between_time("09:30", "15:59")


        # Reset the index and rename the columns for consistency
        data_df = data_df.reset_index().rename(
            columns={"index": "timestamp", "symbol": "tic"}
        )
        
        # Sort the data by both timestamp and tic for consistent ordering
        data_df = data_df.sort_values(by=["tic", "timestamp"])
        
        # Reset the index and drop the old index column
        data_df = data_df.reset_index(drop=True)

        
        validated_df = DownloadDataSchema.validate( data_df)
        return data_df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Data cleaning started")
        tic_list = np.unique(df.tic.values)
        n_tickers = len(tic_list)

        self.logger.info("align start and end dates")
        grouped = df.groupby("timestamp")
        filter_mask = grouped.transform("count")["tic"] >= n_tickers
        df = df[filter_mask]

        # ... (generating 'times' series, same as in your existing code)

        trading_days = self.get_trading_days(start=self.start, end=self.end)

        # produce full timestamp index
        self.logger.info("produce full timestamp index")
        # times = []
        # for day in trading_days:
        #     current_time = pd.Timestamp(day + " 09:30:00").tz_localize( self.tz)
        #     for i in range(390):
        #         times.append(current_time)
        #         current_time += pd.Timedelta(minutes=1)

        
        self.logger.info("Start processing tickers")
        
        future_results = []
        for tic in tic_list:
            result = self.clean_individual_ticker((tic, df.copy()))
            future_results.append(result)
        
        self.logger.info("ticker list complete")

        self.logger.info("Start concat and rename")
        new_df = pd.concat(future_results)
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        self.logger.info("Data clean finished!")

        return new_df

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
        
        # Store the original data type of the 'timestamp' column
        original_timestamp_dtype = df["timestamp"].dtype

        # Convert df to stock data format just once
        stock = Sdf.retype(df)
        unique_ticker = stock.tic.unique()
        

        # Convert timestamp to a consistent datatype (timezone-naive) before entering the loop
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)

        self.logger.info("Running Loop")
        for indicator in tech_indicator_list:
            indicator_dfs = []
            for tic in unique_ticker:
                tic_data = stock[stock.tic == tic]
                indicator_series = tic_data[indicator]

                tic_timestamps = df.loc[df.tic == tic, "timestamp"]

                indicator_df = pd.DataFrame(
                    {
                        "tic": tic,
                        "date": tic_timestamps.values,
                        indicator: indicator_series.values,
                    }
                )
                indicator_dfs.append(indicator_df)

            # Concatenate all intermediate dataframes at once
            indicator_df = pd.concat(indicator_dfs, ignore_index=True)
            
            # Merge the indicator data frame
            df = df.merge(
                indicator_df[["tic", "date", indicator]],
                left_on=["tic", "timestamp"],
                right_on=["tic", "date"],
                how="left",
            ).drop(columns="date")

        self.logger.info("Restore Timestamps")
        # Restore the original data type of the 'timestamp' column
        if isinstance(original_timestamp_dtype, pd.DatetimeTZDtype):
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            df["timestamp"] = df["timestamp"].dt.tz_convert(original_timestamp_dtype.tz)
        else:
            df["timestamp"] = df["timestamp"].astype(original_timestamp_dtype)

        df[tech_indicator_list] = df[tech_indicator_list].astype(float).round( 5)

        self.logger.info("Finished adding Indicators")

        return df


    def download_and_clean_data(self):
        vix_df = self.download_data(["VIXY"], self.start, self.end, self.time_interval)
        return self.clean_data(vix_df)

    def add_vix(self, data):
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

    def calculate_turbulence(self, data, time_period=252):
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
        df = df.sort_values(["timestamp", "tic"])
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

    def get_trading_days(self, start, end):
        nyse = tc.get_calendar("NYSE")
        df = nyse.sessions_in_range(
            pd.Timestamp(start).tz_localize(None), pd.Timestamp(end).tz_localize(None)
        )
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])

        return trading_days

    @staticmethod
    def clean_individual_ticker(args):
        tic, df = args
        df.set_index("timestamp", inplace=True)
        tmp_df = df

        # Step 1: Merging dataframes to avoid loop
        # tmp_df = tmp_df.merge(
        #     tic_df[["open", "high", "low", "close", "volume"]],
        #     left_index=True,
        #     right_index=True,
        #     how="left",
        # )

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
        tmp_df["tic"] = tic

        return tmp_df

    def fetch_latest_data(
        self, ticker_list, time_interval, tech_indicator_list, limit=100
    ) -> pd.DataFrame:
        pass

    def close_conn(self):
        self.logger.info ( 'close conn')