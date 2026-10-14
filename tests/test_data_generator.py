"""Unit tests for synthetic e-commerce data generation."""

from pathlib import Path

import pytest

from src.data_generator import generate_customers, generate_dataset, generate_orders, generate_products


def test_generated_orders_reference_existing_entities() -> None:
    """Ensure generated order customer and product IDs exist in their source lists."""
    customers = generate_customers(5, 10)
    products = generate_products(4, 10)
    orders = generate_orders(12, customers, products, 10)
    customer_ids = {customer["customer_id"] for customer in customers}
    product_ids = {product["product_id"] for product in products}
    assert all(order["customer_id"] in customer_ids for order in orders)
    assert all(order["product_id"] in product_ids for order in orders)


def test_generate_dataset_writes_three_csvs(tmp_path: Path) -> None:
    """Ensure a dataset run produces the three expected source CSV files."""
    generate_dataset(3, 2, 4, 7, tmp_path)
    assert {path.name for path in tmp_path.glob("*.csv")} == {
        "customers.csv", "products.csv", "orders.csv"
    }


def test_generate_customers_requires_positive_count() -> None:
    """Reject an invalid zero customer count with a helpful exception."""
    with pytest.raises(ValueError, match="at least 1"):
        generate_customers(0, 1)
