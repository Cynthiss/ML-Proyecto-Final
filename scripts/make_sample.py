"""Genera la muestra pequeña de datos que usa el entorno ``ci``.

Las tablas completas (``sales.csv`` pesa ~500 MB) no se suben a GitHub, así que
GitHub Actions ejecuta el pipeline con una muestra aleatoria de las ventas que
conserva todo el periodo (ene-may 2018), todos los vendedores y las filas sin
fecha, para que cada paso de limpieza se pruebe con casos reales.

Uso (desde la raíz del proyecto, con los CSV completos en data/01_raw)::

    python scripts/make_sample.py
"""

from pathlib import Path

import pandas as pd

RAW = Path("data/01_raw")
SAMPLE = Path("data/00_sample")
FRACTION = 0.005
SEED = 42
SMALL_TABLES = ["products", "categories", "employees", "cities", "countries"]


def main() -> None:
    """Crea los CSV de muestra en ``data/00_sample``."""
    SAMPLE.mkdir(parents=True, exist_ok=True)
    sales = pd.read_csv(RAW / "sales.csv").sample(frac=FRACTION, random_state=SEED)
    sales.sort_values("SalesID").to_csv(SAMPLE / "sales.csv", index=False)

    customers = pd.read_csv(RAW / "customers.csv")
    customers[customers["CustomerID"].isin(sales["CustomerID"])].to_csv(
        SAMPLE / "customers.csv", index=False
    )
    for name in SMALL_TABLES:
        # keep_default_na=False conserva el código de país "NA" tal cual
        pd.read_csv(RAW / f"{name}.csv", keep_default_na=False, dtype=str).to_csv(
            SAMPLE / f"{name}.csv", index=False
        )


if __name__ == "__main__":
    main()
