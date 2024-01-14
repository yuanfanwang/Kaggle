import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

# config
Local = True
if Local:
    data_path = "./data/"
else:
    data_path = "/kaggle/input/predict-energy-behavior-of-prosumers/"

# read all csv files
train_df              = pl.read_csv(data_path + 'train.csv', dtypes={'datetime': pl.Datetime()})
gas_prices_df         = pl.read_csv(data_path + 'gas_prices.csv')
client_df             = pl.read_csv(data_path + 'client.csv', dtypes={'date': pl.Datetime()})
electricity_prices_df = pl.read_csv(data_path + 'electricity_prices.csv')
forecast_weather_df   = pl.read_csv(data_path + 'forecast_weather.csv')
historical_weather_df = pl.read_csv(data_path + 'historical_weather.csv')

class Debug:
    def __init__(self):
        pass

    def display_time_range(self, df, time_col, id=2):
        print(
            df.filter(
                pl.col("data_block_id") == id
            ).select(
                pl.col(time_col).min().alias("min_" + time_col),
                pl.col(time_col).max().alias("max_" + time_col),
            )
        )
 
    def display_all(self, df, n_rows=100):
        df_len = df.shape[1] * (20 + 3)
        # shape
        print("shape: ", df.shape)

        # columns
        print("-" * df_len)
        print("| ", end="")
        for col in df.columns:
            print("{:<20}".format(col), end=" | ")
        print("")
        print("-" * df_len)

        # data
        for i in range(n_rows):
            print("| ", end="")
            for col in df.columns:
                print("{:<20}".format(df[col][i]), end=" | ")
            print("")
        print("-" * df_len)
        print("\n\n\n\n\n")

class Preprocess:
    def __init__(self):
        pass

    def get_data_block_id(self, df, datetime_col):
        """
        Find data_block_id from date
        """
        basedate = pl.datetime(2021, 9, 1)
        df = df.with_columns(
            (pl.col(datetime_col) - basedate).dt.days().alias("data_block_id")
        )
        return df

    # for train_df, test_df, revealed_targets_df
    def revealed_targets_feature(self, df, datetime_col):
        if "data_block_id" not in df.columns:
            df = self.get_data_block_id(df, datetime_col)
        return df

    def gas_prices_feature(self, df):
        df = df.with_columns(
            pl.col("forecast_date").cast(pl.Date).alias("date") + pl.duration(days=1)
        )
        if "data_block_id" not in df.columns:
            df = self.get_data_block_id(df, "date")
        return df

    def client_feature(self, df):
        df = df.with_columns(
            pl.col("date").cast(pl.Date).alias("date") + pl.duration(days=2)
        )
        if "data_block_id" not in df.columns:
            df = self.get_data_block_id(df, "date")
        return df

    def electricity_prices_feature(self, df):
        df = df.with_columns(
            pl.col("forecast_date").dt.hour().alias("hour"),
            pl.col("forecast_date").cast(pl.Date).alias("date") + pl.duration(days=1)
        )
        if "data_block_id" not in df.columns:
            df = self.get_data_block_id(df, "date")
        return df

    def forecast_weather_feature(self, df):
        df = df.with_columns(
            pl.col("origin_datetime").cast(pl.Date).alias("date") + pl.duration(days=1)
        )
        if "data_block_id" not in df.columns:
            df = self.get_data_block_id(df, "date")
        return df

    def historical_weather_feature(self, df):
        df = df.with_columns(
            pl.col("datetime").dt.hour().alias("hour"),
            pl.col("datetime") + pl.duration(hours=37).cast(pl.Date).alias("date"),
            pl.col("datetime") + pl.duration(hours=37).alias("key_datetime")
        )
        if "data_block_id" not in df.columns:
            df = self.get_data_block_id(df, "date")
        return df


debug = Debug()
preprocess = Preprocess()

"""
print("train_df datetime:")
debug.display_time_range(train_df, "datetime")                        # base
print("\n")

print("gas_prices_df origin_date:")
debug.display_time_range(gas_prices_df, "origin_date")                # 2 day before base
print("gas_prices_df forecast_date:")
debug.display_time_range(gas_prices_df, "forecast_date")              # 1 day before base
print("\n")

print("client_df date:")
debug.display_time_range(client_df, "date")                           # 2 day before base
print("electricity_prices_df origin_date:")
debug.display_time_range(electricity_prices_df, "origin_date")        # 2 day before base
print("electricity_prices_df forecast_date:")
debug.display_time_range(electricity_prices_df, "forecast_date")      # 1 day before base
print("\n")

print("forecast_weather_df origin_datetime:")
debug.display_time_range(forecast_weather_df, "origin_datetime")      # 1 day before base
print("forecast_weather_df forecast_datetime:")
debug.display_time_range(forecast_weather_df, "forecast_datetime")    # From 3:00 a.m. 1 day before to 2:00 a.m. 1 day after, after origin_date
print("\n")

print("historical_weather_df datetime:")
debug.display_time_range(historical_weather_df, "datetime")           # 1 day before base
print("\n")
"""

# debug.display_all(forecast_weather_df, 100)
# train_df = preprocess.revealed_targets_feature(train_df, "datetime")

