"""Nodos del pipeline ``business_impact``.

Traduce los hallazgos a cifras de negocio con escenarios explícitos:

1. **Descuentos:** ingreso que se recupera al reducir el monto otorgado.
2. **Venta sugestiva:** ingreso adicional si los vendedores por debajo de la
   mediana del equipo alcanzan su nivel de unidades en las ventas de mayor valor.
3. **Metas por vendedor:** precisión de las metas diarias con el pronóstico
   elegido frente a una regla ingenua (usar la venta de ayer).

Cada escenario reporta sus supuestos, porque son estimaciones, no garantías.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _period_days(sales_primary: pd.DataFrame) -> int:
    """Número de días con ventas en el periodo observado."""
    return int(sales_primary["SalesDate"].dt.normalize().nunique())


def estimate_discount_impact(
    sales_primary: pd.DataFrame, params: dict[str, Any]
) -> dict[str, Any]:
    """Estima el ingreso recuperable al reducir los descuentos.

    Supuesto: las unidades vendidas no cambian al reducir el descuento, porque
    en los datos se venden las mismas unidades con 0%, 10% o 20% de descuento.

    Args:
        sales_primary: Ventas a nivel transacción.
        params: Parámetros ``business_impact``.

    Returns:
        Monto otorgado y recuperación por escenario (periodo y anualizada).
    """
    days = _period_days(sales_primary)
    annual = params["days_per_year"] / days
    revenue = sales_primary["Revenue"].sum()
    given = sales_primary["DiscountAmount"].sum()
    units = sales_primary.groupby("Discount")["Quantity"].mean()

    scenarios = [
        {
            "reduction_pct": 100 * share,
            "recovered_period": round(float(given * share), 2),
            "recovered_annual": round(float(given * share * annual), 2),
            "revenue_increase_pct": round(float(100 * given * share / revenue), 2),
        }
        for share in params["discount_reduction_scenarios"]
    ]
    return {
        "discount_given_period": round(float(given), 2),
        "discount_given_annual": round(float(given * annual), 2),
        "units_per_sale_by_discount": {
            f"{100 * k:.0f}%": round(float(v), 2) for k, v in units.items()
        },
        "scenarios": scenarios,
        "assumption": (
            "Las unidades por venta no cambian al reducir el descuento (en los datos "
            "son iguales con 0%, 10% y 20%). No se puede medir si algún cliente "
            "dejaría de comprar."
        ),
    }


def estimate_upsell_impact(
    sales_primary: pd.DataFrame, params: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Estima el ingreso adicional por venta sugestiva en ventas de mayor valor.

    Para cada vendedor por debajo de la meta (cuantil ``upsell_target_quantile``
    del equipo en unidades por venta de mayor valor), calcula:
    ``(meta - sus unidades) × sus ventas de mayor valor × ingreso por unidad``.

    Args:
        sales_primary: Ventas a nivel transacción.
        params: Parámetros ``business_impact``.

    Returns:
        Resumen del escenario y tabla por vendedor.
    """
    days = _period_days(sales_primary)
    annual = params["days_per_year"] / days
    threshold = sales_primary["Price"].quantile(params["high_value_price_quantile"])
    high = sales_primary[sales_primary["Price"] > threshold]

    by_seller = high.groupby("SalesPersonID").agg(
        HighValueSales=("Quantity", "size"),
        UnitsPerSale=("Quantity", "mean"),
        Units=("Quantity", "sum"),
        Revenue=("Revenue", "sum"),
    )
    by_seller["RevenuePerUnit"] = by_seller["Revenue"] / by_seller["Units"]
    target = by_seller["UnitsPerSale"].quantile(params["upsell_target_quantile"])
    by_seller["UnitGap"] = (target - by_seller["UnitsPerSale"]).clip(lower=0)
    by_seller["UpliftPeriod"] = (
        by_seller["UnitGap"] * by_seller["HighValueSales"] * by_seller["RevenuePerUnit"]
    )
    by_seller["UpliftAnnual"] = by_seller["UpliftPeriod"] * annual

    uplift = by_seller["UpliftPeriod"].sum()
    summary = {
        "high_value_price_threshold": round(float(threshold), 2),
        "target_units_per_high_value_sale": round(float(target), 3),
        "sellers_below_target": int((by_seller["UnitGap"] > 0).sum()),
        "uplift_period": round(float(uplift), 2),
        "uplift_annual": round(float(uplift * annual), 2),
        "revenue_increase_pct": round(
            float(100 * uplift / sales_primary["Revenue"].sum()), 2
        ),
        "assumption": (
            "Las unidades adicionales se suman a la venta (no reemplazan otras) y el "
            "número de ventas de cada vendedor no cambia. La meta es el nivel que ya "
            "alcanza la mitad del equipo, por lo que es realista."
        ),
    }
    table = by_seller.reset_index().sort_values("UpliftPeriod", ascending=False)
    return summary, table


def estimate_goal_accuracy(
    model_metrics: pd.DataFrame,
    model_selection: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Compara la precisión de las metas diarias con el pronóstico elegido.

    Args:
        model_metrics: Métricas por modelo y línea base en el periodo de prueba.
        model_selection: Modelo elegido por ``model_evaluation``.
        params: Parámetros ``business_impact`` (``naive_baseline``).

    Returns:
        Error de la meta diaria por vendedor con el pronóstico elegido y con la
        regla ingenua, y la reducción del error.
    """
    metrics = model_metrics.set_index("model")
    chosen = metrics.loc[model_selection["selected_model"]]
    naive = metrics.loc[params["naive_baseline"]]
    return {
        "selected_forecast": model_selection["selected_model"],
        "daily_goal_error_pct": round(float(chosen["MAPE_%"]), 2),
        "daily_goal_error_abs": round(float(chosen["MAE"]), 2),
        "naive_rule": params["naive_baseline"],
        "naive_error_pct": round(float(naive["MAPE_%"]), 2),
        "error_reduction_pct": round(
            float(100 * (1 - chosen["MAE"] / naive["MAE"])), 1
        ),
    }


def summarize_impact(
    discount: dict[str, Any], upsell: dict[str, Any], goals: dict[str, Any]
) -> tuple[dict[str, Any], plt.Figure]:
    """Reúne los escenarios en un solo reporte y una gráfica anualizada.

    Args:
        discount: Salida de ``estimate_discount_impact``.
        upsell: Resumen de ``estimate_upsell_impact``.
        goals: Salida de ``estimate_goal_accuracy``.

    Returns:
        Reporte de impacto y gráfica del ingreso anual adicional por escenario.
    """
    bars = {
        f"Descuentos −{s['reduction_pct']:.0f}%": s["recovered_annual"]
        for s in discount["scenarios"]
    }
    bars["Venta sugestiva (meta = mediana)"] = upsell["uplift_annual"]

    fig, ax = plt.subplots(figsize=(8, 4))
    labels, values = list(bars), [v / 1e6 for v in bars.values()]
    colors = ["#4C72B0"] * len(discount["scenarios"]) + ["#55A868"]
    ax.barh(labels, values, color=colors)
    for y, v in enumerate(values):
        ax.text(v, y, f" Q{v:,.1f}M", va="center")
    ax.set_xlim(0, max(values) * 1.25)
    ax.invert_yaxis()
    ax.set_xlabel("Ingreso adicional estimado por año (millones)")
    ax.set_title("Impacto anual estimado de las recomendaciones")
    fig.tight_layout()

    report = {"discounts": discount, "upselling": upsell, "goals": goals}
    return report, fig
