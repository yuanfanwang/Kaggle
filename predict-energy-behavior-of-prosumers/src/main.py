import polars as pl
import lightgbm as lgb

Local = True

if Local:
    data_path = "./data/"
else:
    data_path = "/kaggle/input/predict-energy-behavior-of-prosumers/"

train_df              = pl.read_csv(data_path + 'train.csv')
gas_prices_df         = pl.read_csv(data_path + 'gas_prices.csv')
client_df             = pl.read_csv(data_path + 'client.csv')
electricity_prices_df = pl.read_csv(data_path + 'electricity_prices.csv')
forecast_weather_df   = pl.read_csv(data_path + 'forecast_weather.csv')
historical_weather_df = pl.read_csv(data_path + 'historical_weather.csv')

def display_all(df, n_rows=100):
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
    for i in range(100):
        print("| ", end="")
        for col in df.columns:
            print("{:<20}".format(df[col][i]), end=" | ")
        print("")
    print("-" * df_len)
    print("\n\n\n\n\n")

train_client_df = train_df.filter(pl.col("prediction_unit_id") == 2) \
                          .join(client_df, on=["county", "is_business", "product_type", "data_block_id",])

"""
train_2_df = train_df.filter(pl.col("prediction_unit_id") == 2)
client_2_df = client_df.filter(pl.col("county") == 0).filter(pl.col("is_business") == 0).filter(pl.col("product_type") == 3) \
                       .select([
                           pl.col("eic_count"),
                           pl.col("installed_capacity"),
                           pl.col("is_business"),
                           pl.col("data_block_id"),
                           pl.col("date")
                       ])
print(train_2_df)
print(client_2_df)
display_all(client_2_df, 50)
"""

