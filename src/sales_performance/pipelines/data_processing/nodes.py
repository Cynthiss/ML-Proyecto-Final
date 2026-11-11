"""Nodos del pipeline ``data_processing``.

Cada nodo limpia una tabla cruda o une las tablas limpias. Las decisiones de
limpieza se registran en ``data_quality_report`` para documentarlas en el reporte.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DAYS_PER_YEAR = 365.25


def _check_no_orphans(
    child: pd.DataFrame, parent: pd.DataFrame, key: str, child_key: str | None = None
) -> int:
    """Cuenta las llaves de ``child`` que no existen en ``parent``.

    Args:
        child: Tabla que contiene la llave foránea.
        parent: Tabla que contiene la llave primaria.
        key: Nombre de la llave primaria en ``parent``.
        child_key: Nombre de la llave foránea en ``child`` (por defecto ``key``).

    Returns:
        Número de filas de ``child`` sin correspondencia en ``parent``.
    """
    child_key = child_key or key
    return int((~child[child_key].isin(parent[key])).sum())


def clean_sales(
    raw_sales: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Limpia la tabla de ventas.

    - Elimina ``TotalPrice``: viene en 0 en todas las filas y se recalcula después.
    - Convierte ``SalesDate`` a fecha y descarta las ventas sin fecha (no se
      pueden ubicar en el tiempo, que es la base del análisis de tendencias).
    - Elimina duplicados por ``SalesID`` y por ``TransactionNumber``.
    - Descarta cantidades y descuentos fuera de los rangos válidos.

    Args:
        raw_sales: Tabla ``sales`` cruda.
        params: Parámetros ``data_processing`` (``quantity_range``, ``valid_discounts``).

    Returns:
        Ventas limpias y un diccionario con el conteo de filas afectadas.
    """
    df = raw_sales.copy()
    report: dict[str, Any] = {"rows_raw": len(df)}

    report["total_price_all_zero"] = bool((df["TotalPrice"] == 0).all())
    df = df.drop(columns=["TotalPrice"])

    df["SalesDate"] = pd.to_datetime(df["SalesDate"], errors="coerce")
    missing_date = df["SalesDate"].isna()
    report["rows_missing_date"] = int(missing_date.sum())
    df = df[~missing_date]

    n_before = len(df)
    df = df.drop_duplicates(subset="SalesID").drop_duplicates(
        subset="TransactionNumber"
    )
    report["rows_duplicated"] = n_before - len(df)

    q_min, q_max = params["quantity_range"]
    bad_quantity = ~df["Quantity"].between(q_min, q_max)
    report["rows_invalid_quantity"] = int(bad_quantity.sum())
    df = df[~bad_quantity]

    bad_discount = ~df["Discount"].round(2).isin(params["valid_discounts"])
    report["rows_invalid_discount"] = int(bad_discount.sum())
    df = df[~bad_discount]

    report["rows_clean"] = len(df)
    report["date_min"] = str(df["SalesDate"].min())
    report["date_max"] = str(df["SalesDate"].max())
    logger.info("Ventas limpias: %s de %s filas", len(df), report["rows_raw"])
    return df.reset_index(drop=True), report


def clean_products(
    raw_products: pd.DataFrame, raw_categories: pd.DataFrame
) -> pd.DataFrame:
    """Limpia productos y les agrega el nombre de la categoría.

    Args:
        raw_products: Tabla ``products`` cruda.
        raw_categories: Tabla ``categories`` cruda.

    Returns:
        Productos con ``CategoryName`` y tipos corregidos.

    Raises:
        ValueError: Si algún producto tiene una categoría inexistente o precio <= 0.
    """
    orphans = _check_no_orphans(raw_products, raw_categories, "CategoryID")
    if orphans:
        raise ValueError(f"{orphans} productos con CategoryID inexistente")
    if (raw_products["Price"] <= 0).any():
        raise ValueError("Hay productos con precio menor o igual a 0")

    df = raw_products.merge(raw_categories, on="CategoryID", how="left")
    df["ModifyDate"] = pd.to_datetime(df["ModifyDate"], errors="coerce")
    df["IsAllergic"] = df["IsAllergic"].astype(str)
    return df


def clean_employees(
    raw_employees: pd.DataFrame, raw_cities: pd.DataFrame, params: dict[str, Any]
) -> pd.DataFrame:
    """Limpia vendedores y calcula variables de perfil.

    Calcula ``Age`` y ``TenureYears`` a la fecha de corte ``reference_date``,
    que son las variables de perfil para la pregunta de contratación.

    Args:
        raw_employees: Tabla ``employees`` cruda.
        raw_cities: Tabla ``cities`` cruda.
        params: Parámetros ``data_processing`` (``reference_date``).

    Returns:
        Vendedores con nombre completo, ciudad, edad y antigüedad.
    """
    ref = pd.Timestamp(params["reference_date"])
    df = raw_employees.copy()
    df["BirthDate"] = pd.to_datetime(df["BirthDate"])
    df["HireDate"] = pd.to_datetime(df["HireDate"])
    df["EmployeeName"] = df["FirstName"] + " " + df["LastName"]
    df["Age"] = ((ref - df["BirthDate"]).dt.days / DAYS_PER_YEAR).round(1)
    df["TenureYears"] = ((ref - df["HireDate"]).dt.days / DAYS_PER_YEAR).round(1)
    df["AgeAtHire"] = (
        (df["HireDate"] - df["BirthDate"]).dt.days / DAYS_PER_YEAR
    ).round(1)
    df = df.merge(raw_cities[["CityID", "CityName"]], on="CityID", how="left")
    return df.drop(columns=["FirstName", "MiddleInitial", "LastName"])


def build_sales_primary(
    sales_clean: pd.DataFrame,
    products_clean: pd.DataFrame,
    employees_clean: pd.DataFrame,
    raw_customers: pd.DataFrame,
    quality_report: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Une ventas con productos y clientes y calcula el ingreso de cada venta.

    ``Revenue = Price * Quantity * (1 - Discount)`` reemplaza al ``TotalPrice``
    original, que venía vacío.

    Args:
        sales_clean: Ventas limpias.
        products_clean: Productos limpios.
        employees_clean: Vendedores limpios (solo para validar llaves).
        raw_customers: Tabla ``customers`` cruda (para la ciudad del cliente).
        quality_report: Reporte de calidad generado por ``clean_sales``.

    Returns:
        Tabla de ventas enriquecida y el reporte de calidad completo.

    Raises:
        ValueError: Si existen ventas con producto, cliente o vendedor inexistente.
    """
    orphans = {
        "orphan_products": _check_no_orphans(sales_clean, products_clean, "ProductID"),
        "orphan_customers": _check_no_orphans(sales_clean, raw_customers, "CustomerID"),
        "orphan_salespeople": _check_no_orphans(
            sales_clean, employees_clean, "EmployeeID", "SalesPersonID"
        ),
    }
    if any(orphans.values()):
        raise ValueError(f"Llaves huérfanas en ventas: {orphans}")

    df = sales_clean.merge(
        products_clean[["ProductID", "Price", "CategoryID", "CategoryName", "Class"]],
        on="ProductID",
        how="left",
    ).merge(
        raw_customers[["CustomerID", "CityID"]].rename(
            columns={"CityID": "CustomerCityID"}
        ),
        on="CustomerID",
        how="left",
    )
    df["GrossRevenue"] = df["Price"] * df["Quantity"]
    df["Revenue"] = df["GrossRevenue"] * (1 - df["Discount"])
    df["DiscountAmount"] = df["GrossRevenue"] - df["Revenue"]

    report = {
        **quality_report,
        **orphans,
        "total_revenue": round(float(df["Revenue"].sum()), 2),
    }
    return df, report
