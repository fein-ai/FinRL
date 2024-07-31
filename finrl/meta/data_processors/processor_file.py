import os
import logbook
import pandas as pd
import numpy as np
import exchange_calendars as tc
from finrl.config import (
        DATA_SAVE_DIR
)
from finrl.meta.data_processors.processor import AbstractProcessor

class FileProcessor(AbstractProcessor):
    def __init__(self, directory_path: str) -> None:
        try:
            super().__init__()
            self.tz = "America/New_York"
            self.logger = logbook.Logger(self.__class__.__name__)
            self.directory_path = directory_path
        except Exception as e:
            self.logger.error(e)


    def download_data(self, ticker_list, start_date, end_date, time_interval ) -> pd.DataFrame:
        self.logger.info ( f"Loading data from {self.directory_path}")
        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval
    
        
        dfs = []
        # Loop through all files in the directory
        for filename in os.listdir(self.directory_path):
            if filename.endswith('.csv'):
                # Construct the full file path
                file_path = os.path.join(self.directory_path, filename)
                self.logger.info ( f"Reading file {file_path}")
                
                # Read the CSV file and append the DataFrame to the list
                dfs.append(pd.read_csv(file_path,
                    parse_dates=['timestamp'],
                    # infer_datetime_format=True,
                    index_col=0, 
                    date_parser=lambda x: pd.to_datetime(x, utc=True).tz_convert( self.tz)
                ))

        combined_df = pd.concat(dfs)
        return combined_df

    def clean_data(self, df):
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
        times = []
        for day in trading_days:
            
            current_time = pd.Timestamp(day + " 09:30:00").tz_localize( self.tz)
            for i in range(390):
                times.append(current_time)
                current_time += pd.Timedelta(minutes=1)

        self.logger.info("Start processing tickers")

        future_results = []
        for tic in tic_list:
            result = self.clean_individual_ticker((tic, df.copy(), times))
            future_results.append(result)

        self.logger.info("ticker list complete")

        self.logger.info("Start concat and rename")
        new_df = pd.concat(future_results)
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        self.logger.info("Data clean finished!")

        return new_df
    
    def add_vix(self, data: pd.DataFrame) -> pd.DataFrame:
        return super.add_vix(data, self.start, self.end, self.time_interval)
    
    
    
    def df_to_array(self, df: pd.DataFrame, tech_indicator_list: np.array, if_vix: bool) -> np.array:
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

    def close_conn(self) -> None:
        # do nothing
        pass

    def fetch_latest_data(self, ticker_list, time_interval, tech_indicator_list) -> pd.DataFrame:
        # do nothing
        pass
        