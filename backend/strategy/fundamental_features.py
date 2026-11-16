import numpy as np
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


def load_splits() -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(text("SELECT ticker, date, ratio FROM splits"), conn, parse_dates=["date"])


def split_adjust_shares(fund: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    shares = fund[fund["item"] == "shares"].sort_values("end")
    later = splits.sort_values(["ticker", "date"], ascending=[True, False])
    later = later.assign(factor=later.groupby("ticker")["ratio"].cumprod()).sort_values("date")
    merged = pd.merge_asof(
        shares, later[["ticker", "date", "factor"]],
        left_on="end", right_on="date", by="ticker",
        direction="forward", allow_exact_matches=False,
    )
    merged["val"] = merged["val"] * merged["factor"].fillna(1.0)
    return merged[fund.columns]


def market_value_features(
    fund: pd.DataFrame, splits: pd.DataFrame, closes: pd.DataFrame, dates: pd.DatetimeIndex
) -> dict[str, pd.DataFrame]:
    tickers = closes.columns
    shares = point_in_time(split_adjust_shares(fund, splits), "shares", dates, tickers)
    equity = point_in_time(fund, "equity", dates, tickers)
    market_cap = (closes.loc[dates] * shares).where(lambda x: x > 0)
    return {
        "market_cap": market_cap,
        "book_to_market": equity / market_cap,
        "size": np.log(market_cap),
    }


if __name__ == "__main__":
    from backend.strategy.portfolio import month_end_dates
    from backend.strategy.prices import load_close_matrix

    closes = load_close_matrix().drop(columns=["SPY"])
    dates = month_end_dates(closes.index)
    fund = load_fundamentals(["equity", "shares"])
    features = market_value_features(fund, load_splits(), closes, dates)

    cap = features["market_cap"] / 1e9
    for date in [dates[dates.year == 2019][-1], dates[-1]]:
        print(f"Market value in $ billions on {date.date()}:")
        print(cap.loc[date, ["AAPL", "NVDA", "JPM", "TSLA"]].round(0).to_string())
        print()
    print("Book-to-market, latest:")
    print(features["book_to_market"].iloc[-1][["AAPL", "NVDA", "JPM", "TSLA"]].round(3).to_string())
    print(f"\nStocks with a book-to-market value, latest month: {features['book_to_market'].iloc[-1].notna().sum()}")