"""Nodos del pipeline ``feature_engineering``.

La unidad de análisis es **vendedor-día**: cuánto vendió cada vendedor cada día.
Todas las variables históricas se calculan con ``shift(1)`` para usar solo
información disponible antes del día que se quiere pronosticar (sin fuga de datos).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

KEYS = ["SalesPersonID", "Date"]


def build_seller_daily(sales_primary: pd.DataFrame) -> pd.DataFrame:
    """Agrega las ventas a nivel vendedor-día.

    Los días en que un vendedor no registró ventas se completan con 0 para que
    los rezagos representen días calendario reales.

    Args:
        sales_primary: Ventas limpias y enriquecidas.

    Returns:
        Tabla vendedor-día con ingreso, número de ventas, unidades, ticket
        promedio y proporción de ventas con descuento.
    """
    df = sales_primary.assign(
        Date=sales_primary["SalesDate"].dt.normalize(),
        HasDiscount=sales_primary["Discount"] > 0,
    )
    daily = (
        df.groupby(KEYS)
        .agg(
            Revenue=("Revenue", "sum"),
            GrossRevenue=("GrossRevenue", "sum"),
            DiscountAmount=("DiscountAmount", "sum"),
            NumSales=("SalesID", "size"),
            Units=("Quantity", "sum"),
            DiscountShare=("HasDiscount", "mean"),
        )
        .reset_index()
    )

    full_index = pd.MultiIndex.from_product(
        [
            daily["SalesPersonID"].unique(),
            pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D"),
        ],
        names=KEYS,
    )
    daily = daily.set_index(KEYS).reindex(full_index, fill_value=0).reset_index()
    daily["AvgTicket"] = (daily["Revenue"] / daily["NumSales"]).fillna(0.0)
    return daily.sort_values(KEYS).reset_index(drop=True)


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables de calendario a partir de ``Date``."""
    df["DayOfWeek"] = df["Date"].dt.day_of_week
    df["DayOfMonth"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Month"] = df["Date"].dt.month
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)  # noqa: PLR2004
    return df


def _add_history_features(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Agrega rezagos y estadísticas móviles por vendedor (solo con días previos)."""
    target = params["target"]
    by_seller = df.groupby("SalesPersonID")
    past = by_seller[target].shift(1)

    for lag in params["lags"]:
        df[f"{target}_lag{lag}"] = by_seller[target].shift(lag)

    past_by_seller = past.groupby(df["SalesPersonID"])
    for window in params["rolling_windows"]:
        df[f"{target}_rmean{window}"] = past_by_seller.transform(
            lambda s, w=window: s.rolling(w).mean()
        )
        df[f"{target}_rstd{window}"] = past_by_seller.transform(
            lambda s, w=window: s.rolling(w).std()
        )
    # promedio histórico acumulado del vendedor: su "nivel" de venta
    df[f"{target}_expanding_mean"] = past_by_seller.transform(
        lambda s: s.expanding().mean()
    )
    # comportamiento reciente: volumen, ticket y uso de descuentos
    window = params["behaviour_window"]
    for col in ["NumSales", "AvgTicket", "DiscountShare"]:
        past_col = by_seller[col].shift(1)
        df[f"{col}_lag1"] = past_col
        df[f"{col}_rmean{window}"] = past_col.groupby(df["SalesPersonID"]).transform(
            lambda s, w=window: s.rolling(w).mean()
        )
    return df


def build_features(
    seller_daily: pd.DataFrame, employees_clean: pd.DataFrame, params: dict[str, Any]
) -> pd.DataFrame:
    """Construye la tabla de modelado vendedor-día.

    Combina calendario, historial de ventas del vendedor y su perfil
    (edad, antigüedad, edad al ser contratado y género). Se descartan los primeros
    días de cada vendedor, donde la ventana móvil más larga aún no está completa.

    Args:
        seller_daily: Ventas agregadas por vendedor-día.
        employees_clean: Vendedores con variables de perfil.
        params: Parámetros ``feature_engineering``.

    Returns:
        Tabla con llaves, variables explicativas y la variable objetivo.
    """
    target = params["target"]
    base_cols = [*KEYS, target]
    df = seller_daily[base_cols + ["NumSales", "AvgTicket", "DiscountShare"]].copy()
    df = _add_calendar_features(df)
    df = _add_history_features(df, params)

    profile = employees_clean[
        ["EmployeeID", "Age", "TenureYears", "AgeAtHire", "Gender"]
    ].rename(columns={"EmployeeID": "SalesPersonID"})
    profile["IsFemale"] = (profile.pop("Gender") == "F").astype(int)
    df = df.merge(profile, on="SalesPersonID", how="left")

    # las variables del mismo día (NumSales, AvgTicket, DiscountShare) se
    # conocen solo al cierre del día: se eliminan para evitar fuga de datos
    df = df.drop(columns=["NumSales", "AvgTicket", "DiscountShare"])

    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(
        "Features: %s filas (%s descartadas por ventanas)", len(df), n_before - len(df)
    )
    return df


def split_train_test(
    features: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa entrenamiento y prueba de forma temporal.

    Los últimos ``test_days`` días quedan como prueba: simula pronosticar
    semanas futuras con lo aprendido del pasado (no se mezclan fechas).

    Args:
        features: Tabla de modelado vendedor-día.
        params: Parámetros ``feature_engineering`` (``test_days``).

    Returns:
        Conjuntos de entrenamiento y prueba.
    """
    cutoff = features["Date"].max() - pd.Timedelta(days=params["test_days"] - 1)
    train = features[features["Date"] < cutoff].reset_index(drop=True)
    test = features[features["Date"] >= cutoff].reset_index(drop=True)
    logger.info(
        "Train: %s filas hasta %s | Test: %s filas",
        len(train),
        cutoff.date(),
        len(test),
    )
    return train, test
