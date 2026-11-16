import os
import time

import pandas as pd
import requests
from sqlalchemy import Date

from backend.data.membership import current_tickers
from backend.db import engine

REQUEST_PAUSE = 0.15

ITEMS = {
    "equity": [("us-gaap", "StockholdersEquity", "USD")],
    "assets": [("us-gaap", "Assets", "USD")],
    "liabilities": [("us-gaap", "Liabilities", "USD")],
    "net_income": [("us-gaap", "NetIncomeLoss", "USD")],
    "revenue": [
        ("us-gaap", "Revenues", "USD"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
    ],
    "shares": [("dei", "EntityCommonStockSharesOutstanding", "shares")],
}
COLUMNS = ["ticker", "item", "start", "end", "filed", "val", "form"]


def sec_headers() -> dict[str, str]:
    return {"User-Agent": os.environ["SEC_USER_AGENT"]}


def load_cik_map() -> dict[str, int]:
    data = requests.get("https://www.sec.gov/files/company_tickers.json", headers=sec_headers(), timeout=30).json()
    return {row["ticker"]: row["cik_str"] for row in data.values()}


def fetch_company_facts(cik: int) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    response = requests.get(url, headers=sec_headers(), timeout=30)
    response.raise_for_status()
    return response.json()["facts"]


def extract_items(facts: dict, ticker: str) -> pd.DataFrame:
    frames = []
    for item, sources in ITEMS.items():
        for group, concept, unit in sources:
            rows = facts.get(group, {}).get(concept, {}).get("units", {}).get(unit)
            if rows:
                frames.append(pd.DataFrame(rows).assign(item=item))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    if "start" not in df.columns:
        df["start"] = None
    df["ticker"] = ticker
    df = df[COLUMNS].sort_values("filed")
    return df.drop_duplicates(subset=["ticker", "item", "start", "end"], keep="first")


def store_fundamentals(df: pd.DataFrame) -> None:
    df = df.copy()
    for col in ["start", "end", "filed"]:
        df[col] = pd.to_datetime(df[col]).dt.date
    df.to_sql(
        "fundamentals", engine, if_exists="replace", index=False,
        dtype={"start": Date(), "end": Date(), "filed": Date()},
    )


def main():
    cik_map = load_cik_map()
    tickers = current_tickers()
    frames, missing, failed = [], [], []

    for i, ticker in enumerate(tickers, start=1):
        cik = cik_map.get(ticker)
        if cik is None:
            missing.append(ticker)
            continue
        try:
            frames.append(extract_items(fetch_company_facts(cik), ticker))
        except Exception as e:
            failed.append(f"{ticker} ({e})")
        time.sleep(REQUEST_PAUSE)
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)} done")

    df = pd.concat(frames, ignore_index=True)
    store_fundamentals(df)

    print(f"\nStored {len(df)} rows for {df['ticker'].nunique()} tickers")
    print(df.groupby("item")["ticker"].nunique().rename("tickers").to_string())
    print(f"\nNot in SEC ticker list ({len(missing)}): {missing}")
    print(f"Failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()