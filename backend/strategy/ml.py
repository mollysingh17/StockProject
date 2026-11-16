import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from backend.strategy.features import FEATURES

FIRST_TEST_YEAR = 2017


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, random_state=42
    )


def walk_forward_predictions(data: pd.DataFrame, first_test_year: int = FIRST_TEST_YEAR) -> pd.Series:
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
        model.fit(train[FEATURES], train["label"].astype(int))
        predictions.append(pd.Series(model.predict_proba(test[FEATURES])[:, 1], index=test.index))
    return pd.concat(predictions).rename("score")


def information_coefficient(scores: pd.Series, fwd_ret: pd.Series) -> pd.Series:
    both = pd.concat([scores, fwd_ret], axis=1).dropna()
    return both.groupby(level="date").apply(
        lambda g: g.iloc[:, 0].rank().corr(g.iloc[:, 1].rank())
    )


if __name__ == "__main__":
    from backend.data.membership import load_membership_matrix
    from backend.strategy.features import build_dataset
    from backend.strategy.portfolio import month_end_dates
    from backend.strategy.prices import load_close_matrix

    closes = load_close_matrix().drop(columns=["SPY"])
    eligible = load_membership_matrix(closes.index, closes.columns)
    data = build_dataset(closes, eligible, month_end_dates(closes.index))

    preds = walk_forward_predictions(data)
    ic = information_coefficient(preds, data["fwd_ret"])

    print(f"Predictions for {preds.index.get_level_values('date').nunique()} months\n")
    print("IC by year (mean of monthly ICs):")
    print(ic.groupby(ic.index.year).mean().round(3).to_string())
    print(f"\nOverall mean IC: {ic.mean():.3f}")
    print(f"Months with positive IC: {(ic > 0).mean():.0%}")