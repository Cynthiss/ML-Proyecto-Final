"""Pruebas de los nodos de ``data_processing`` con datos pequeños de ejemplo."""

import pandas as pd
import pytest

from sales_performance.pipelines.data_processing.nodes import (
    build_sales_primary,
    clean_employees,
    clean_products,
    clean_sales,
)

PARAMS = {
    "reference_date": "2018-05-10",
    "quantity_range": [1, 1000],
    "valid_discounts": [0.0, 0.1, 0.2],
}


@pytest.fixture
def raw_sales():
    return pd.DataFrame(
        {
            "SalesID": [1, 2, 3, 4, 5, 5],
            "SalesPersonID": [1, 1, 2, 2, 1, 1],
            "CustomerID": [10, 11, 10, 11, 10, 10],
            "ProductID": [100, 101, 100, 101, 100, 100],
            "Quantity": [2, 3, 0, 1, 4, 4],
            "Discount": [0.0, 0.1, 0.0, 0.5, 0.2, 0.2],
            "TotalPrice": [0.0] * 6,
            "SalesDate": [
                "2018-01-01 10:00:00",
                None,
                "2018-01-02 11:00:00",
                "2018-01-03 12:00:00",
                "2018-01-04 13:00:00",
                "2018-01-04 13:00:00",
            ],
            "TransactionNumber": ["A", "B", "C", "D", "E", "E"],
        }
    )


@pytest.fixture
def products():
    return clean_products(
        pd.DataFrame(
            {
                "ProductID": [100, 101],
                "ProductName": ["Pan", "Leche"],
                "Price": [10.0, 5.0],
                "CategoryID": [1, 2],
                "Class": ["Low", "High"],
                "ModifyDate": ["2018-01-01", "2018-01-01"],
                "Resistant": ["Weak", "Durable"],
                "IsAllergic": ["True", "Unknown"],
                "VitalityDays": [0.0, 10.0],
            }
        ),
        pd.DataFrame({"CategoryID": [1, 2], "CategoryName": ["Cereals", "Dairy"]}),
    )


@pytest.fixture
def employees():
    return clean_employees(
        pd.DataFrame(
            {
                "EmployeeID": [1, 2],
                "FirstName": ["Ana", "Luis"],
                "MiddleInitial": ["A", "B"],
                "LastName": ["Paz", "Gil"],
                "BirthDate": ["1980-05-10", "1990-05-10"],
                "Gender": ["F", "M"],
                "CityID": [1, 1],
                "HireDate": ["2010-05-10", "2015-05-10"],
            }
        ),
        pd.DataFrame({"CityID": [1], "CityName": ["Dayton"], "CountryID": [32]}),
        PARAMS,
    )


def test_clean_sales_removes_bad_rows(raw_sales):
    clean, report = clean_sales(raw_sales, PARAMS)
    # quedan: venta 1 y venta 5 (sin duplicado); se van sin fecha, qty 0 y desc 0.5
    assert clean["SalesID"].tolist() == [1, 5]
    assert "TotalPrice" not in clean.columns
    assert report["rows_missing_date"] == 1
    assert report["rows_duplicated"] == 1
    assert report["rows_invalid_quantity"] == 1
    assert report["rows_invalid_discount"] == 1
    assert report["total_price_all_zero"] is True


def test_clean_products_rejects_unknown_category(products):
    assert products["CategoryName"].tolist() == ["Cereals", "Dairy"]
    bad = pd.DataFrame({"ProductID": [1], "Price": [1.0], "CategoryID": [99]})
    with pytest.raises(ValueError):
        clean_products(bad, pd.DataFrame({"CategoryID": [1], "CategoryName": ["X"]}))


def test_clean_employees_profile(employees):
    ana = employees.set_index("EmployeeID").loc[1]
    assert ana["Age"] == pytest.approx(38.0, abs=0.1)
    assert ana["TenureYears"] == pytest.approx(8.0, abs=0.1)
    assert ana["AgeAtHire"] == pytest.approx(30.0, abs=0.1)
    assert ana["CityName"] == "Dayton"
    assert "LastName" not in employees.columns


def test_build_sales_primary_revenue(raw_sales, products, employees):
    clean, report = clean_sales(raw_sales, PARAMS)
    customers = pd.DataFrame({"CustomerID": [10, 11], "CityID": [5, 6]})
    primary, full_report = build_sales_primary(
        clean, products, employees, customers, report
    )
    # venta 1: 10 * 2 * (1 - 0) = 20 ; venta 5: 10 * 4 * (1 - 0.2) = 32
    assert primary["Revenue"].tolist() == pytest.approx([20.0, 32.0])
    assert primary["DiscountAmount"].tolist() == pytest.approx([0.0, 8.0])
    assert full_report["total_revenue"] == pytest.approx(52.0)


def test_build_sales_primary_detects_orphans(raw_sales, products, employees):
    clean, report = clean_sales(raw_sales, PARAMS)
    customers = pd.DataFrame({"CustomerID": [11], "CityID": [6]})  # falta el 10
    with pytest.raises(ValueError, match="huérfanas"):
        build_sales_primary(clean, products, employees, customers, report)
