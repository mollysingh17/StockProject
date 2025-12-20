import pandas as pd
from db import engine


def show_sample_rows(table_name: str = "price_history", limit: int = 5):
    query = f"SELECT * FROM {table_name} LIMIT {limit};"
    df = pd.read_sql(query, engine)
    print(df)


if __name__ == "__main__":
    show_sample_rows()