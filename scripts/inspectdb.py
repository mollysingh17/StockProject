import pandas as pd
from sqlalchemy import text
from backend.db import engine

def main():
    with engine.connect() as conn:
        # total rows
        total = conn.execute(text("SELECT COUNT(*) FROM price_history")).scalar()
        print(f"Total rows in price_history: {total}")

        # per-ticker summary
        q = text("""
            SELECT
                ticker,
                COUNT(*) AS n_rows,
                MIN(date) AS min_date,
                MAX(date) AS max_date
            FROM price_history
            GROUP BY ticker
            ORDER BY ticker
        """)
        rows = conn.execute(q).fetchall()

    if not rows:
        print("No rows found.")
        return

    df = pd.DataFrame(rows, columns=["ticker", "n_rows", "min_date", "max_date"])
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()