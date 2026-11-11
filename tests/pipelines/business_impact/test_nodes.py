"""Pruebas de los nodos de ``business_impact``."""

import pandas as pd
import pytest

from sales_performance.pipelines.business_impact.nodes import (
    estimate_discount_impact,
    estimate_goal_accuracy,
    estimate_upsell_impact,
    summarize_impact,
)

PARAMS = {
    "days_per_year": 10,
    "discount_reduction_scenarios": [0.5, 1.0],
    "high_value_price_quantile": 0.5,
    "upsell_target_quantile": 0.5,
    "naive_baseline": "Ayer",
}


@pytest.fixture
def sales():
    # 5 días; vendedor 1 vende 10 unidades en productos caros, vendedor 2 solo 6
    rows = []
    for day in range(1, 6):
        for seller, units in [(1, 10), (2, 6)]:
            for price, qty in [(100.0, units), (1.0, 5)]:
                rows.append(
                    {
                        "SalesPersonID": seller,
                        "SalesDate": pd.Timestamp(2018, 1, day),
                        "Price": price,
                        "Quantity": qty,
                        "Discount": 0.0,
                        "Revenue": price * qty,
                        "DiscountAmount": 0.0,
                    }
                )
    df = pd.DataFrame(rows)
    df.loc[0, "DiscountAmount"] = 50.0  # un descuento de 50 en el periodo
    return df


def test_discount_scenarios(sales):
    impact = estimate_discount_impact(sales, PARAMS)
    assert impact["discount_given_period"] == pytest.approx(50.0)
    half = impact["scenarios"][0]
    assert half["recovered_period"] == pytest.approx(25.0)
    # anualizado: 10 días por año / 5 días observados = x2
    assert half["recovered_annual"] == pytest.approx(50.0)


def test_upsell_only_lifts_sellers_below_target(sales):
    summary, table = estimate_upsell_impact(sales, PARAMS)
    # meta = mediana entre 10 y 6 = 8 unidades; solo el vendedor 2 sube 2 unidades
    assert summary["target_units_per_high_value_sale"] == pytest.approx(8.0)
    assert summary["sellers_below_target"] == 1
    # 2 unidades × 5 ventas caras × 100 por unidad = 1000
    assert summary["uplift_period"] == pytest.approx(1000.0)
    assert table.set_index("SalesPersonID").loc[1, "UpliftPeriod"] == 0


def test_goal_accuracy_and_summary(sales):
    metrics = pd.DataFrame(
        {"model": ["Promedio", "Ayer"], "MAE": [20.0, 40.0], "MAPE_%": [2.0, 4.0]}
    )
    goals = estimate_goal_accuracy(metrics, {"selected_model": "Promedio"}, PARAMS)
    assert goals["error_reduction_pct"] == pytest.approx(50.0)

    discount = estimate_discount_impact(sales, PARAMS)
    upsell, _ = estimate_upsell_impact(sales, PARAMS)
    report, fig = summarize_impact(discount, upsell, goals)
    assert set(report) == {"discounts", "upselling", "goals"}
    assert fig is not None
