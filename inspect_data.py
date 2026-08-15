import pandas as pd


PATH = "data/raw/yahoo/AAPL.parquet"

df = pd.read_parquet(PATH)

print("\n=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== DTYPES ===")
print(df.dtypes)

print("\n=== FIRST 5 ===")
print(df.head().to_string(index=False))

print("\n=== LAST 5 ===")
print(df.tail().to_string(index=False))

print("\n=== MISSING VALUES ===")
print(df.isna().sum())

print("\n=== DUPLICATE DATES ===")
print(df["date"].duplicated().sum())

print("\n=== DATE RANGE ===")
print(df["date"].min(), "->", df["date"].max())

print("\n=== OHLC CHECKS ===")
print("High < Low:", (df["high"] < df["low"]).sum())
print("Open > High:", (df["open"] > df["high"]).sum())
print("Open < Low:", (df["open"] < df["low"]).sum())
print("Close > High:", (df["close"] > df["high"]).sum())
print("Close < Low:", (df["close"] < df["low"]).sum())

if "adj_close" in df.columns:
    print("\n=== CLOSE VS ADJ_CLOSE ===")
    print(
        df[["date", "close", "adj_close"]]
        .head(10)
        .to_string(index=False)
    )

    print("\nNumber of rows where Close != Adj Close:")
    print((df["close"] != df["adj_close"]).sum())