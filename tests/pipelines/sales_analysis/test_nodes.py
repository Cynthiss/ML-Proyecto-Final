"""Pruebas de los nodos de ``sales_analysis``."""

import numpy as np
import pandas as pd
import pytest

from sales_performance.pipelines.sales_analysis.nodes import (
    build_seller_summary,
    compute_insights,
)

PARAMS = {
    "alpha": 0.05,
    "profile_features": ["Age", "TenureYears", "AgeAtHire"],
    "performance_band_pct": 0.5,
}
N_SELLERS = 4


@pytest.fixture
def seller_daily():
    # 4 semanas completas (lunes 2018-01-01 a domingo 2018-01-28)
    dates = pd.date_range("2018-01-01", periods=28, freq="D")
    rows = []
    for seller in range(1, N_SELLERS + 1):
        for t, date in enumerate(dates):
            # el vendedor 1 crece día a día; los demás son constantes
            revenue = 1000.0 * seller + (10.0 * t if seller == 1 else 0.0)
            rows.append(
                {
                    "SalesPersonID": seller,
                    "Date": date,
                    "Revenue": revenue,
                    "GrossRevenue": revenue,
                    "DiscountAmount": 0.0,
                    "NumSales": 10 + seller,
                    "Units": 100,
                    "AvgTicket": revenue / 10,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def employees():
    return pd.DataFrame(
        {
            "EmployeeID": range(1, N_SELLERS + 1),
            "EmployeeName": [f"V{i}" for i in range(1, N_SELLERS + 1)],
            "Gender": ["F", "M", "F", "M"],
            "Age": [30.0, 40.0, 50.0, 60.0],
            "TenureYears": [1.0, 2.0, 3.0, 4.0],
            "AgeAtHire": [29.0, 38.0, 47.0, 56.0],
            "CityName": ["A"] * N_SELLERS,
        }
    )


def test_seller_summary_shares_and_trend(seller_daily, employees):
    summary = build_seller_summary(seller_daily, employees, PARAMS)
    assert summary["SharePct"].sum() == pytest.approx(100.0)
    assert summary.iloc[0]["SalesPersonID"] == N_SELLERS  # el que más vende
    assert summary["Rank"].tolist() == list(range(1, N_SELLERS + 1))
    trend = summary.set_index("SalesPersonID")["WeeklyTrendPct"]
    assert trend[1] > 0
    assert trend[2] == pytest.approx(0.0, abs=1e-9)
    assert set(summary["Band"]) <= {"Por encima", "Por debajo", "En el promedio"}


def test_insights_detect_profile_and_discount(seller_daily, employees):
    summary = build_seller_summary(seller_daily, employees, PARAMS)
    rng = np.random.default_rng(0)
    sales = pd.DataFrame(
        {
            "SalesID": range(300),
            "Discount": np.repeat([0.0, 0.1, 0.2], 100),
            "Quantity": rng.integers(1, 25, 300),
            "GrossRevenue": 100.0,
            "DiscountAmount": np.repeat([0.0, 10.0, 20.0], 100),
        }
    )
    insights = compute_insights(seller_daily, sales, summary, PARAMS)
    # la edad sube junto con el ingreso en los datos de prueba
    assert insights["profile"]["Age"]["spearman_rho"] == pytest.approx(1.0)
    assert insights["discounts"]["discount_pct_of_gross"] == pytest.approx(10.0)
    assert insights["period"]["complete_weeks"] == 4  # noqa: PLR2004
    assert insights["salespeople"]["growing"] == 1
