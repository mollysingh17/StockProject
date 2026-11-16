import pandas as pd
from sqlalchemy import text

from backend.data.membership import current_tickers, load_membership_matrix
from backend.db import engine
from backend.strategy.portfolio import month_end_dates
from backend.strategy.prices import load_close_matrix


def all_member_tickers() -> pd.Index:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT ticker FROM index_membership")).fetchall()
    return pd.Index(sorted({r[0] for r in rows} | set(current_tickers())))


def main():
    closes = load_close_matrix().drop(columns=["SPY"])
    dates = month_end_dates(closes.index)

    total = load_membership_matrix(dates, all_member_tickers()).sum(axis=1)
    members = load_membership_matrix(dates, closes.columns)
    covered = (members & closes.loc[dates].notna()).sum(axis=1)

    by_year = pd.DataFrame({"members": total, "with_prices": covered}).groupby(dates.year).mean()
    by_year["coverage"] = by_year["with_prices"] / by_year["members"]
    print(by_year.round({"members": 0, "with_prices": 0, "coverage": 3}).to_string())


if __name__ == "__main__":
    main()