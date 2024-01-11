import polars as pl
import datetime as dt

data = {'column1': [1, 2, 3], 'column2': ['A', 'B', 'C']}
df = pl.DataFrame(data)

# Use with_columns to add a new column
new_df = df.with_columns([
    pl.col('column1').alias("new_column1")
])

print(new_df)