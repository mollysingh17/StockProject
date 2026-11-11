import pandas as pd
from sqlalchemy import text

from backend.db import engine


def load_close_matrix() -> pd.DataFrame:
    df = pd.read_sql_query(
        text("SELECT date, ticker, close FROM price_history"),
        engine,
        parse_dates=["date"],
    )
    return df.pivot(index="date", columns="ticker", values="close").sort_index()


if __name__ == "__main__":
    closes = load_close_matrix()
    print(closes.shape)
    print(closes[["AAPL", "MSFT", "SPY"]].tail())
    print(closes.isna().sum().sort_values(ascending=False).head(10))