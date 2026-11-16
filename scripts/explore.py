import sys

import pandas as pd
import requests

USER_AGENT = "Malvina Singh malvinasingh17@gmail.com"
HEADERS = {"User-Agent": USER_AGENT}

CONCEPTS = [
    "StockholdersEquity",
    "NetIncomeLoss",
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Assets",
    "Liabilities",
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
        rows = pd.DataFrame(gaap[concept]["units"]["USD"])
        print(f"{concept}: {len(rows)} values, filed {rows['filed'].min()} to {rows['filed'].max()}")
        print(rows[["end", "val", "fp", "form", "filed"]].tail(4).to_string(index=False))
        print()