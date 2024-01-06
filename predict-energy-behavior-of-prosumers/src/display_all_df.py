import polars as pl

local = True

if local:
    data_path = "./data/"
else:
    data_path = "/kaggle/input/predict-energy-behavior-of-prosumers/"

train_df              = pl.read_csv(data_path + 'train.csv')
gas_prices_df         = pl.read_csv(data_path + 'gas_prices.csv')
client_df             = pl.read_csv(data_path + 'client.csv')
electricity_prices_df = pl.read_csv(data_path + 'electricity_prices.csv')
forecast_weather_df   = pl.read_csv(data_path + 'forecast_weather.csv')
historical_weather_df = pl.read_csv(data_path + 'historical_weather.csv')

print("train_df: \n", train_df, "\n\n")
print("gas_prices_df: \n", gas_prices_df, "\n\n")
print("client_df: \n", client_df, "\n\n")
print("electricity_prices_df: \n", electricity_prices_df, "\n\n")
print("forecast_weather_df: \n", forecast_weather_df, "\n\n")
print("historical_weather_df: \n", historical_weather_df, "\n\n")