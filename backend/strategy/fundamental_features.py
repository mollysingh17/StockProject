import pandas as pd
from sqlalchemy import bindparam, text

from backend.db import engine

MAX_AGE_DAYS = 450


def load_fundamentals(items: list[str]) -> pd.DataFrame:
    query = text(
        'SELECT ticker, item, "end", filed, val FROM fundamentals WHERE item IN :items'
    ).bindparams(bindparam("items", expanding=True))
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn, params={"items": items}, parse_dates=["end", "filed"])


def point_in_time(fund: pd.DataFrame, item: str, dates: pd.DatetimeIndex, tickers: pd.Index) -> pd.DataFrame:
    df = fund[fund["item"] == item].sort_values(["ticker", "filed", "end"])
    df = df[df["end"] == df.groupby("ticker")["end"].cummax()]
    df = df.assign(available=df["filed"] + pd.Timedelta(days=1))
    df = df.drop_duplicates(subset=["ticker", "available"], keep="last")

    values = df.pivot(index="available", columns="ticker", values="val")
    available = df.pivot(index="available", columns="ticker", values="available")
    full_index = values.index.union(dates)
    values = values.reindex(full_index).ffill().reindex(dates)
    available = available.reindex(full_index).ffill().reindex(dates)

    age = available.rsub(pd.Series(dates, index=dates), axis=0)
    return values.where(age <= pd.Timedelta(days=MAX_AGE_DAYS)).reindex(columns=tickers)


def leverage(fund: pd.DataFrame, dates: pd.DatetimeIndex, tickers: pd.Index) -> pd.DataFrame:
    assets = point_in_time(fund, "assets", dates, tickers)
    equity = point_in_time(fund, "equity", dates, tickers)
    return (assets - equity) / assets


if __name__ == "__main__":
    from backend.strategy.portfolio import month_end_dates
    from backend.strategy.prices import load_close_matrix

    closes = load_close_matrix().drop(columns=["SPY"])
    dates = month_end_dates(closes.index)
    fund = load_fundamentals(["assets", "equity"])
    lev = leverage(fund, dates, closes.columns)

    print("Stocks with a leverage value, each December:")
    december = lev[lev.index.month == 12]
    print(december.notna().sum(axis=1).set_axis(december.index.year).to_string())
    print("\nLatest leverage:")
    print(lev.iloc[-1][["AAPL", "JPM", "MSFT", "XOM"]].round(3).to_string())