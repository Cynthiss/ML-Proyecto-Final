"""Pruebas de los nodos de ``feature_engineering``."""

import pandas as pd
import pytest

from sales_performance.pipelines.feature_engineering.nodes import (
    build_features,
    build_seller_daily,
    split_train_test,
)

PARAMS = {
    "lags": [1, 2],
    "rolling_windows": [2],
    "behaviour_window": 2,
    "test_days": 2,
    "target": "Revenue",
    "profile_features": ["Age", "TenureYears", "AgeAtHire"],
}


@pytest.fixture
def sales_primary():
    # el vendedor 1 no vende el día 3; el vendedor 2 vende los 7 días
    rows = []
    for seller, days in {1: [1, 2, 4, 5, 6, 7], 2: [1, 2, 3, 4, 5, 6, 7]}.items():
        for day in days:
            rows.append(
                {
                    "SalesID": len(rows),
                    "SalesPersonID": seller,
                    "SalesDate": pd.Timestamp(2018, 1, day, 10),
                    "Revenue": 10.0 * day,
                    "GrossRevenue": 10.0 * day,
                    "DiscountAmount": 0.0,
                    "Quantity": 1,
                    "Discount": 0.1 if day % 2 else 0.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def employees():
    return pd.DataFrame(
        {
            "EmployeeID": [1, 2],
            "Age": [40.0, 30.0],
            "TenureYears": [5.0, 2.0],
            "AgeAtHire": [35.0, 28.0],
            "Gender": ["F", "M"],
        }
    )


def test_build_seller_daily_fills_missing_days(sales_primary):
    daily = build_seller_daily(sales_primary)
    assert len(daily) == 2 * 7
    day3 = daily[(daily["SalesPersonID"] == 1) & (daily["Date"].dt.day == 3)]
    assert day3["Revenue"].item() == 0
    assert day3["AvgTicket"].item() == 0


def test_build_features_has_no_leakage(sales_primary, employees):
    feats = build_features(build_seller_daily(sales_primary), employees, PARAMS)
    seller2 = feats[feats["SalesPersonID"] == 2]
    seller2 = seller2.set_index(seller2["Date"].dt.day)
    # el lag 1 del día 5 es la venta del día 4, nunca la del mismo día
    assert seller2.loc[5, "Revenue_lag1"] == 40.0
    assert seller2.loc[5, "Revenue_rmean2"] == pytest.approx(35.0)
    assert "NumSales" not in feats.columns
    assert feats.isna().sum().sum() == 0
    # el género nunca entra al modelo
    assert not {"Gender", "IsFemale"} & set(feats.columns)
    assert {"Age", "TenureYears", "AgeAtHire"} <= set(feats.columns)


def test_split_train_test_is_temporal(sales_primary, employees):
    feats = build_features(build_seller_daily(sales_primary), employees, PARAMS)
    train, test = split_train_test(feats, PARAMS)
    assert train["Date"].max() < test["Date"].min()
    assert test["Date"].nunique() == 2
    assert len(train) + len(test) == len(feats)
