import pandas as pd
from sqlalchemy import bindparam, text

from backend.db import engine
from backend.data.membership import RENAMES


def main():
    targets = sorted(set(RENAMES.values()))
    query = text("""
        SELECT p.ticker, t.name, MIN(p.date) AS price_start
        FROM price_history p
        LEFT JOIN tickers t ON t.ticker = p.ticker
        WHERE p.ticker IN :targets
        GROUP BY p.ticker, t.name
    """).bindparams(bindparam("targets", expanding=True))
    with engine.connect() as conn:
        found = pd.read_sql_query(query, conn, params={"targets": targets})

    table = pd.DataFrame({"old": list(RENAMES), "new": list(RENAMES.values())})
    table = table.merge(found, left_on="new", right_on="ticker", how="left").drop(columns="ticker")
    print(table.to_string(index=False))

if __name__ == "__main__":
    main()