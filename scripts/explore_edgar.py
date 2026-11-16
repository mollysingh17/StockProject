import sys
import os
import pandas as pd
import requests

USER_AGENT = os.environ["SEC_USER_AGENT"]
HEADERS = {"User-Agent": USER_AGENT}

CONCEPTS = [
    "StockholdersEquity",
    "NetIncomeLoss",
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Assets",
    "Liabilities",
]

SHARE_CONCEPTS = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
]


def ticker_to_cik(ticker: str) -> int:
    data = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=30).json()
    for row in data.values():
        if row["ticker"] == ticker:
            return row["cik_str"]
    raise ValueError(f"{ticker} not found in SEC ticker list")


def company_facts(cik: int) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    return requests.get(url, headers=HEADERS, timeout=30).json()


def show(rows: pd.DataFrame, name: str) -> None:
    columns = [c for c in ["start", "end", "val", "fp", "form", "filed"] if c in rows.columns]
    print(f"{name}: {len(rows)} values, filed {rows['filed'].min()} to {rows['filed'].max()}")
    print(rows[columns].tail(4).to_string(index=False))
    print()


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    cik = ticker_to_cik(ticker)
    facts = company_facts(cik)
    gaap = facts["facts"]["us-gaap"]
    print(f"{facts['entityName']} (CIK {cik}): {len(gaap)} reported items\n")

    for concept in CONCEPTS:
        if concept not in gaap:
            print(f"{concept}: not reported\n")
            continue
        show(pd.DataFrame(gaap[concept]["units"]["USD"]), concept)

    for group_name, concept in SHARE_CONCEPTS:
        group = facts["facts"].get(group_name, {})
        if concept not in group:
            print(f"{concept}: not reported\n")
            continue
        show(pd.DataFrame(group[concept]["units"]["shares"]), concept)