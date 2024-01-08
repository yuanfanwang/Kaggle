import polars as pl
import datetime as dt

basetime = pl.datetime(2021,9,1)
now = pl.datetime(2021, 9, 10)
id = (now - basetime).dt.days()
print(id)