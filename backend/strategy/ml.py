import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from backend.strategy.features import FEATURES

FIRST_TEST_YEAR = 2017


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, random_state=42
    )


def walk_forward_predictions(
    data: pd.DataFrame, first_test_year: int = FIRST_TEST_YEAR, features: list[str] = FEATURES
) -> pd.Series:
    dates = data.index.get_level_values("date")
    predictions = []
    for year in range(first_test_year, dates.max().year + 1):
        test_start = pd.Timestamp(f"{year}-12-01")
        test_end = pd.Timestamp(f"{year + 1}-12-01")
        train = data[(dates < test_start) & data["label"].notna()]
        test = data[(dates >= test_start) & (dates < test_end)]
        if test.empty:
            continue
        model = make_model()
        model.fit(train[features], train["label"].astype(int))
        predictions.append(pd.Series(model.predict_proba(test[features])[:, 1], index=test.index))
    return pd.concat(predictions).rename("score")


def information_coefficient(scores: pd.Series, fwd_ret: pd.Series) -> pd.Series:
    both = pd.concat([scores, fwd_ret], axis=1).dropna()
    return both.groupby(level="date").apply(
        lambda g: g.iloc[:, 0].rank().corr(g.iloc[:, 1].rank())
    )


if __name__ == "__main__":
    from backend.data.membership import load_membership_matrix
    from backend.strategy.features import FUNDAMENTAL_FEATURES, build_dataset
    from backend.strategy.fundamental_features import (
        leverage,
        load_fundamentals,
        load_splits,
        market_value_features,
    )
    from backend.strategy.portfolio import month_end_dates
    from backend.strategy.prices import load_close_matrix

    closes = load_close_matrix().drop(columns=["SPY"])
    dates = month_end_dates(closes.index)
    eligible = load_membership_matrix(closes.index, closes.columns)

    fund = load_fundamentals(["assets", "equity", "shares"])
    market = market_value_features(fund, load_splits(), closes, dates)
    extra = {
        "leverage": leverage(fund, dates, closes.columns),
        "book_to_market": market["book_to_market"],
        "size": market["size"],
    }
    data = build_dataset(closes, eligible, dates, extra)

    complete = data.dropna(subset=FUNDAMENTAL_FEATURES)
    print(f"Rows with all fundamentals: {len(complete) / len(data):.0%}\n")

    runs = [
        ("price only", data, FEATURES),
        ("price + fundamentals", data, FEATURES + FUNDAMENTAL_FEATURES),
        ("price only, complete", complete, FEATURES),
        ("price + fund., complete", complete, FEATURES + FUNDAMENTAL_FEATURES),
    ]
    results = {}
    for name, frame, features in runs:
        preds = walk_forward_predictions(frame, features=features)
        results[name] = information_coefficient(preds, frame["fwd_ret"])

    by_year = pd.DataFrame({name: ic.groupby(ic.index.year).mean() for name, ic in results.items()})
    print("IC by year (mean of monthly ICs):")
    print(by_year.round(3).to_string())
    print()
    for name, ic in results.items():
        print(f"{name:>22}: mean IC {ic.mean():.3f}, positive in {(ic > 0).mean():.0%} of months")