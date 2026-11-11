"""Nodos del pipeline ``sales_analysis``.

Responde las preguntas descriptivas del Reto 1:

- **Tendencia**: ¿cómo evolucionan las ventas por semana y por día de la semana?
- **Generación y contribución**: ¿cuánto aporta cada vendedor y en qué se diferencian?
- **Tendencia por vendedor**: ¿quién crece y quién cae?
- **Descuentos**: ¿los descuentos generan más volumen?
- **Perfil para contratación**: ¿la edad, antigüedad o género explican el desempeño?
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

DAYS_PER_WEEK = 7
SLOPE_DECIMALS = 6
WEEKDAYS_ES = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
PROFILE_LABELS = {
    "Age": "Edad (años)",
    "TenureYears": "Antigüedad (años)",
    "AgeAtHire": "Edad al ser contratado",
}
GREY, BLUE, GREEN, RED = "#BBBBBB", "#4C72B0", "#55A868", "#C44E52"


def _complete_weeks(seller_daily: pd.DataFrame) -> pd.DataFrame:
    """Ventas totales por semana, solo semanas con los 7 días registrados."""
    weekly = (
        seller_daily.assign(Week=seller_daily["Date"].dt.to_period("W-SUN"))
        .groupby("Week")
        .agg(Revenue=("Revenue", "sum"), Days=("Date", "nunique"))
    )
    return weekly[weekly["Days"] == DAYS_PER_WEEK]


def _weekly_slope_pct(seller_daily: pd.DataFrame) -> pd.Series:
    """Pendiente semanal de ingreso por vendedor como % de su promedio semanal.

    Una pendiente de +0.1 significa que el vendedor crece 0.1% por semana.
    """
    weeks = _complete_weeks(seller_daily).index
    weekly = (
        seller_daily.assign(Week=seller_daily["Date"].dt.to_period("W-SUN"))
        .pipe(lambda d: d[d["Week"].isin(weeks)])
        .groupby(["SalesPersonID", "Week"])["Revenue"]
        .sum()
        .reset_index()
    )
    weekly["t"] = weekly.groupby("SalesPersonID").cumcount()

    def slope(g: pd.DataFrame) -> float:
        return 100 * np.polyfit(g["t"], g["Revenue"], 1)[0] / g["Revenue"].mean()

    # se redondea para que el ruido numérico (p. ej. 1e-13 en vez de 0) no cuente
    # como crecimiento o caída; depende de la librería de álgebra de cada sistema
    return (
        weekly.groupby("SalesPersonID")[["t", "Revenue"]]
        .apply(slope)
        .round(SLOPE_DECIMALS)
    )


def build_seller_summary(
    seller_daily: pd.DataFrame,
    employees_clean: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Construye la ficha de desempeño de cada vendedor.

    Incluye ingreso total, participación, ventas, ticket promedio, uso de
    descuentos, tendencia semanal y una banda de desempeño respecto al equipo.

    Args:
        seller_daily: Ventas por vendedor-día.
        employees_clean: Vendedores con perfil.
        params: Parámetros ``sales_analysis``.

    Returns:
        Una fila por vendedor, ordenada por ingreso.
    """
    agg = seller_daily.groupby("SalesPersonID").agg(
        Revenue=("Revenue", "sum"),
        NumSales=("NumSales", "sum"),
        Units=("Units", "sum"),
        DiscountAmount=("DiscountAmount", "sum"),
        GrossRevenue=("GrossRevenue", "sum"),
    )
    agg["SharePct"] = 100 * agg["Revenue"] / agg["Revenue"].sum()
    agg["AvgTicket"] = agg["Revenue"] / agg["NumSales"]
    agg["UnitsPerSale"] = agg["Units"] / agg["NumSales"]
    agg["DiscountPct"] = 100 * agg["DiscountAmount"] / agg["GrossRevenue"]
    agg["WeeklyTrendPct"] = _weekly_slope_pct(seller_daily)
    agg["VsTeamPct"] = 100 * (agg["Revenue"] / agg["Revenue"].mean() - 1)

    band = params["performance_band_pct"]
    agg["Band"] = np.select(
        [agg["VsTeamPct"] >= band, agg["VsTeamPct"] <= -band],
        ["Por encima", "Por debajo"],
        default="En el promedio",
    )
    profile = employees_clean.set_index("EmployeeID")[
        ["EmployeeName", "Gender", "Age", "TenureYears", "AgeAtHire", "CityName"]
    ]
    summary = agg.join(profile).sort_values("Revenue", ascending=False)
    summary.insert(0, "Rank", range(1, len(summary) + 1))
    return summary.reset_index()


def compute_insights(
    seller_daily: pd.DataFrame,
    sales_primary: pd.DataFrame,
    seller_summary: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Calcula las cifras clave y pruebas estadísticas del análisis.

    Args:
        seller_daily: Ventas por vendedor-día.
        sales_primary: Ventas a nivel transacción.
        seller_summary: Ficha por vendedor.
        params: Parámetros ``sales_analysis``.

    Returns:
        Diccionario con tendencia, diferencias entre vendedores, efecto de los
        descuentos y relación perfil-desempeño (con p-values).
    """
    alpha = params["alpha"]
    weekly = _complete_weeks(seller_daily)["Revenue"]
    daily_total = seller_daily.groupby("Date")["Revenue"].sum()

    def anova(col: str) -> float:
        groups = [g[col].to_numpy() for _, g in seller_daily.groupby("SalesPersonID")]
        return float(stats.f_oneway(*groups).pvalue)

    by_discount = sales_primary.groupby("Discount").agg(
        sales=("SalesID", "size"), units_per_sale=("Quantity", "mean")
    )
    units_by_level = [
        g["Quantity"].to_numpy() for _, g in sales_primary.groupby("Discount")
    ]

    profile_tests = {}
    for feat in params["profile_features"]:
        rho, p = stats.spearmanr(seller_summary[feat], seller_summary["Revenue"])
        profile_tests[feat] = {
            "spearman_rho": round(float(rho), 3),
            "p_value": round(float(p), 4),
            "significant": bool(p < alpha),
        }
    women = seller_summary.loc[seller_summary["Gender"] == "F", "Revenue"]
    men = seller_summary.loc[seller_summary["Gender"] == "M", "Revenue"]
    gender_p = float(stats.mannwhitneyu(women, men).pvalue)
    profile_tests["Gender"] = {
        "mean_revenue_F": round(float(women.mean()), 2),
        "mean_revenue_M": round(float(men.mean()), 2),
        "p_value": round(gender_p, 4),
        "significant": bool(gender_p < alpha),
    }
    ticket_rho = stats.spearmanr(seller_summary["AvgTicket"], seller_summary["Revenue"])
    volume_rho = stats.spearmanr(seller_summary["NumSales"], seller_summary["Revenue"])

    return {
        "period": {
            "start": str(seller_daily["Date"].min().date()),
            "end": str(seller_daily["Date"].max().date()),
            "days": int(seller_daily["Date"].nunique()),
            "complete_weeks": len(weekly),
        },
        "totals": {
            "revenue": round(float(seller_daily["Revenue"].sum()), 2),
            "sales": int(seller_daily["NumSales"].sum()),
            "salespeople": int(seller_daily["SalesPersonID"].nunique()),
            "avg_daily_revenue": round(float(daily_total.mean()), 2),
            "daily_revenue_cv_pct": round(
                float(100 * daily_total.std() / daily_total.mean()), 2
            ),
        },
        "trend": {
            "weekly_slope_pct": round(
                float(
                    100
                    * np.polyfit(range(len(weekly)), weekly.to_numpy(), 1)[0]
                    / weekly.mean()
                ),
                3,
            ),
            "first_week_revenue": round(float(weekly.iloc[0]), 2),
            "last_week_revenue": round(float(weekly.iloc[-1]), 2),
            "weekday_anova_p": round(
                float(
                    stats.f_oneway(
                        *[
                            g.to_numpy()
                            for _, g in daily_total.groupby(
                                daily_total.index.day_of_week
                            )
                        ]
                    ).pvalue
                ),
                4,
            ),
        },
        "salespeople": {
            "share_min_pct": round(float(seller_summary["SharePct"].min()), 3),
            "share_max_pct": round(float(seller_summary["SharePct"].max()), 3),
            "top_vs_bottom_gap_pct": round(
                float(
                    100
                    * (
                        seller_summary["Revenue"].max()
                        / seller_summary["Revenue"].min()
                        - 1
                    )
                ),
                2,
            ),
            "anova_p_revenue": anova("Revenue"),
            "anova_p_num_sales": anova("NumSales"),
            "anova_p_avg_ticket": anova("AvgTicket"),
            "spearman_revenue_vs_ticket": round(float(ticket_rho.statistic), 3),
            "spearman_revenue_vs_num_sales": round(float(volume_rho.statistic), 3),
            "bands": seller_summary["Band"].value_counts().to_dict(),
            "growing": int((seller_summary["WeeklyTrendPct"] > 0).sum()),
            "declining": int((seller_summary["WeeklyTrendPct"] < 0).sum()),
        },
        "discounts": {
            "discount_amount": round(float(sales_primary["DiscountAmount"].sum()), 2),
            "discount_pct_of_gross": round(
                float(
                    100
                    * sales_primary["DiscountAmount"].sum()
                    / sales_primary["GrossRevenue"].sum()
                ),
                2,
            ),
            "sales_with_discount_pct": round(
                float(100 * (sales_primary["Discount"] > 0).mean()), 2
            ),
            "units_per_sale_by_discount": {
                str(k): round(float(v), 3)
                for k, v in by_discount["units_per_sale"].items()
            },
            "anova_p_units_by_discount": round(
                float(stats.f_oneway(*units_by_level).pvalue), 4
            ),
        },
        "profile": profile_tests,
    }


def plot_weekly_trend(seller_daily: pd.DataFrame) -> plt.Figure:
    """Venta total por semana completa, con su línea de tendencia."""
    weekly = _complete_weeks(seller_daily)["Revenue"] / 1e6
    x = np.arange(len(weekly))
    fit = np.poly1d(np.polyfit(x, weekly.to_numpy(), 1))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, weekly.to_numpy(), marker="o", color=BLUE, label="Venta semanal")
    ax.plot(x, fit(x), ls="--", color=GREY, label="Tendencia")
    ax.set_xticks(x, [str(w.start_time.date()) for w in weekly.index], rotation=45)
    ax.set_ylim(weekly.min() * 0.98, weekly.max() * 1.02)
    ax.set_ylabel("Millones")
    ax.set_title("Venta total por semana (solo semanas completas)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_weekday_pattern(seller_daily: pd.DataFrame) -> plt.Figure:
    """Venta total promedio por día de la semana."""
    daily = seller_daily.groupby("Date")["Revenue"].sum()
    by_day = daily.groupby(daily.index.day_of_week).mean() / 1e6
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(WEEKDAYS_ES, by_day.to_numpy(), color=BLUE)
    ax.set_ylim(by_day.min() * 0.99, by_day.max() * 1.01)
    ax.set_ylabel("Millones (promedio por día)")
    ax.set_title("Venta promedio por día de la semana")
    fig.tight_layout()
    return fig


def plot_seller_contribution(seller_summary: pd.DataFrame) -> plt.Figure:
    """Participación de cada vendedor en la venta total, coloreada por banda."""
    data = seller_summary.sort_values("SharePct")
    colors = data["Band"].map(
        {"Por encima": GREEN, "Por debajo": RED, "En el promedio": GREY}
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(data["EmployeeName"], data["SharePct"], color=colors)
    ax.axvline(100 / len(data), color="black", ls="--", lw=1, label="Reparto igual")
    ax.set_xlim(data["SharePct"].min() * 0.98, data["SharePct"].max() * 1.01)
    ax.set_xlabel("% de la venta total")
    ax.set_title("Contribución por vendedor (verde: por encima, rojo: por debajo)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_seller_trend(seller_summary: pd.DataFrame) -> plt.Figure:
    """Tendencia semanal de cada vendedor (% de cambio por semana)."""
    data = seller_summary.sort_values("WeeklyTrendPct")
    colors = np.where(data["WeeklyTrendPct"] >= 0, GREEN, RED)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(data["EmployeeName"], data["WeeklyTrendPct"], color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Cambio promedio por semana (%)")
    ax.set_title("Tendencia de venta por vendedor")
    fig.tight_layout()
    return fig


def plot_volume_vs_ticket(seller_summary: pd.DataFrame) -> plt.Figure:
    """Número de ventas vs. ticket promedio; el tamaño es el ingreso."""
    data = seller_summary
    size = 50 + 400 * (data["Revenue"] - data["Revenue"].min()) / (
        data["Revenue"].max() - data["Revenue"].min()
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data["NumSales"] / 1e3, data["AvgTicket"], s=size, color=BLUE, alpha=0.7)
    for _, row in data.head(3).iterrows():
        ax.annotate(row["EmployeeName"], (row["NumSales"] / 1e3, row["AvgTicket"]))
    for _, row in data.tail(3).iterrows():
        ax.annotate(row["EmployeeName"], (row["NumSales"] / 1e3, row["AvgTicket"]))
    ax.set_xlabel("Número de ventas (miles)")
    ax.set_ylabel("Ticket promedio")
    ax.set_title("Lo que separa a los vendedores es el ticket, no el volumen")
    fig.tight_layout()
    return fig


def plot_discount_effect(sales_primary: pd.DataFrame) -> plt.Figure:
    """Unidades por venta según el nivel de descuento."""
    by_discount = sales_primary.groupby("Discount")["Quantity"].mean()
    labels = [f"{int(100 * d)}%" for d in by_discount.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, by_discount.to_numpy(), color=[GREY, BLUE, BLUE])
    ax.bar_label(bars, fmt="%.2f")
    ax.set_ylim(0, by_discount.max() * 1.2)
    ax.set_xlabel("Descuento aplicado")
    ax.set_ylabel("Unidades promedio por venta")
    ax.set_title("El descuento no aumenta las unidades vendidas")
    fig.tight_layout()
    return fig


def plot_profile_vs_performance(
    seller_summary: pd.DataFrame, params: dict[str, Any]
) -> plt.Figure:
    """Perfil del vendedor (edad, antigüedad, género) vs. su ingreso total."""
    features = params["profile_features"]
    fig, axes = plt.subplots(1, len(features) + 1, figsize=(4 * (len(features) + 1), 4))
    revenue = seller_summary["Revenue"] / 1e6
    for ax, feat in zip(axes, features, strict=False):
        rho, p = stats.spearmanr(seller_summary[feat], revenue)
        ax.scatter(seller_summary[feat], revenue, color=BLUE)
        ax.set_xlabel(PROFILE_LABELS.get(feat, feat))
        ax.set_title(f"rho = {rho:.2f} (p = {p:.2f})")
    axes[0].set_ylabel("Ingreso total (millones)")
    by_gender = seller_summary.groupby("Gender")["Revenue"]
    axes[-1].boxplot(
        [g / 1e6 for _, g in by_gender], tick_labels=list(by_gender.groups)
    )
    axes[-1].set_xlabel("Género")
    axes[-1].set_title("Ingreso por género")
    fig.suptitle("¿El perfil del vendedor explica su desempeño?")
    fig.tight_layout()
    return fig
