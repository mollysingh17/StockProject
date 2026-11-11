import sys

import pandas as pd
from sqlalchemy import bindparam, text

from backend.db import engine

TOLERANCE_DAYS = 10


def load_spans() -> pd.DataFrame:
    with engine.connect() as conn:
        prices = pd.read_sql_query(text("""
            SELECT ticker, MIN(date) AS price_start, MAX(date) AS price_end
            FROM price_history
            WHERE ticker NOT IN (SELECT ticker FROM tickers) AND ticker <> 'SPY'
            GROUP BY ticker
        """), conn, parse_dates=["price_start", "price_end"])
        members = pd.read_sql_query(text("""
            SELECT ticker, MIN(date) AS member_start, MAX(date) AS member_end
            FROM index_membership
            GROUP BY ticker
        """), conn, parse_dates=["member_start", "member_end"])
        data_start = pd.to_datetime(conn.execute(text("SELECT MIN(date) FROM price_history")).scalar())

    spans = prices.merge(members, on="ticker", how="left")
    spans["needed_start"] = spans["member_start"].clip(lower=data_start)
    return spans


def delete_tickers(tickers: list[str]) -> None:
    query = text("DELETE FROM price_history WHERE ticker IN :tickers").bindparams(
        bindparam("tickers", expanding=True)
    )
    with engine.begin() as conn:
        conn.execute(query, {"tickers": tickers})


def main(delete: bool) -> None:
    spans = load_spans()
    late = spans["price_start"] > spans["needed_start"] + pd.Timedelta(days=TOLERANCE_DAYS)
    bad = spans[late].sort_values("ticker")

    print(f"Checked {len(spans)} removed tickers")
    print(f"{len(bad)} have prices that start after they joined the index:\n")
    print(bad[["ticker", "needed_start", "price_start", "member_end"]].to_string(index=False))

    if delete and not bad.empty:
        delete_tickers(bad["ticker"].tolist())
        print(f"\nDeleted prices for {len(bad)} tickers")


if __name__ == "__main__":
    main(delete="--delete" in sys.argv)