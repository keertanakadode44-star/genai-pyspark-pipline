"""Generate synthetic e-commerce data as pandas DataFrames."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from faker import Faker
from tqdm.auto import tqdm

from src import config

LOGGER = logging.getLogger(__name__)
CATEGORIES: tuple[str, ...] = ("Electronics", "Clothing", "Home", "Sports", "Books")


class SyntheticDataGenerator:
    """Create realistic, reproducible synthetic e-commerce DataFrames.

    By default, the generator creates 100,000 customers, 10,000 products, and
    1,000,000 orders. NumPy's random generator controls numeric and sampling
    choices; Faker uses the same seed for generated customer names and emails.
    Approximately 80% of orders are assigned to the top 20% of customers,
    whose individual order propensities follow a Pareto distribution.
    """

    def __init__(
        self,
        n_customers: int = 100_000,
        n_products: int = 10_000,
        n_orders: int = 1_000_000,
        seed: int = 42,
        progress: bool = True,
    ) -> None:
        """Initialize dataset sizes, seeded random sources, and progress settings."""
        for label, value in (
            ("n_customers", n_customers),
            ("n_products", n_products),
            ("n_orders", n_orders),
        ):
            if value < 1:
                raise ValueError(f"{label} must be at least 1")
        self.n_customers = n_customers
        self.n_products = n_products
        self.n_orders = n_orders
        self.seed = seed
        self.progress = progress
        self.rng = np.random.default_rng(seed)
        self.fake = Faker()
        self.fake.seed_instance(seed)
        LOGGER.debug(
            "Initialized generator: customers=%d products=%d orders=%d seed=%d",
            n_customers,
            n_products,
            n_orders,
            seed,
        )

    def generate_customers(self) -> pd.DataFrame:
        """Generate customers with IDs, Faker identity/location, and normal ages."""
        LOGGER.info("Generating %d customers", self.n_customers)
        records: list[dict[str, object]] = []
        iterator = tqdm(
            range(self.n_customers),
            desc="Customers",
            disable=not self.progress,
            unit="customer",
        )
        for index in iterator:
            records.append(
                {
                    "customer_id": f"C{index + 1:07d}",
                    "name": self.fake.name(),
                    "email": self.fake.unique.email(),
                    "age": int(np.clip(round(self.rng.normal(loc=35, scale=10)), 18, 90)),
                    "city": self.fake.city(),
                    "country": self.fake.country(),
                    "registration_date": self.fake.date_between(
                        start_date="-5y", end_date="today"
                    ).isoformat(),
                }
            )
        dataframe = pd.DataFrame.from_records(records)
        LOGGER.info("Generated customer DataFrame with %d rows", len(dataframe))
        return dataframe

    def generate_products(self) -> pd.DataFrame:
        """Generate product IDs, names, categories, prices, stock, and ratings."""
        LOGGER.info("Generating %d products", self.n_products)
        iterator = tqdm(
            range(self.n_products),
            desc="Products",
            disable=not self.progress,
            unit="product",
        )
        records: list[dict[str, object]] = []
        for index in iterator:
            records.append(
                {
                    "product_id": f"P{index + 1:06d}",
                    "name": self.fake.unique.catch_phrase(),
                    "category": str(self.rng.choice(CATEGORIES)),
                    "price": round(float(self.rng.uniform(10, 500)), 2),
                    "stock": int(self.rng.integers(0, 1001)),
                    "rating": round(float(self.rng.uniform(1, 5)), 1),
                }
            )
        dataframe = pd.DataFrame.from_records(records)
        LOGGER.info("Generated product DataFrame with %d rows", len(dataframe))
        return dataframe

    def _sample_customer_indexes(self, customers: pd.DataFrame) -> np.ndarray:
        """Sample customer row indexes with an 80/20 Pareto order split."""
        customer_count = len(customers)
        if customer_count < 2:
            LOGGER.warning("Only one customer exists; all orders will use that customer")
            return np.zeros(self.n_orders, dtype=np.int64)

        top_count = max(1, int(np.ceil(customer_count * 0.20)))
        top_indexes = np.arange(top_count)
        other_indexes = np.arange(top_count, customer_count)
        top_order_count = int(round(self.n_orders * 0.80))
        other_order_count = self.n_orders - top_order_count

        # Pareto-distributed weights make order frequency unequal within the top quintile.
        pareto_weights = self.rng.pareto(a=1.5, size=top_count) + 1.0
        top_probabilities = pareto_weights / pareto_weights.sum()
        top_sample = self.rng.choice(top_indexes, size=top_order_count, p=top_probabilities)
        other_sample = self.rng.choice(other_indexes, size=other_order_count, replace=True)
        customer_indexes = np.concatenate((top_sample, other_sample))
        self.rng.shuffle(customer_indexes)
        LOGGER.debug(
            "Assigned %d (%.1f%%) orders to top %d (%.1f%%) customers",
            top_order_count,
            100 * top_order_count / self.n_orders,
            top_count,
            100 * top_count / customer_count,
        )
        return customer_indexes

    def generate_orders(
        self, customers: pd.DataFrame, products: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate order lines referencing customers and products in the inputs."""
        if customers.empty or products.empty:
            raise ValueError("customers and products must both contain at least one row")
        if "customer_id" not in customers or "product_id" not in products:
            raise ValueError("customers require customer_id and products require product_id")

        LOGGER.info("Generating %d orders", self.n_orders)
        customer_indexes = self._sample_customer_indexes(customers)
        customer_ids = customers["customer_id"].to_numpy()
        product_ids = products["product_id"].to_numpy()
        batch_size = 50_000
        batch_starts = range(0, self.n_orders, batch_size)
        product_indexes = np.empty(self.n_orders, dtype=np.int64)
        quantities = np.empty(self.n_orders, dtype=np.int8)
        day_offsets = np.empty(self.n_orders, dtype=np.int16)
        for start in tqdm(
            batch_starts,
            desc="Orders",
            disable=not self.progress,
            unit="batch",
        ):
            stop = min(start + batch_size, self.n_orders)
            size = stop - start
            product_indexes[start:stop] = self.rng.integers(0, len(product_ids), size=size)
            quantities[start:stop] = self.rng.integers(1, 11, size=size)
            day_offsets[start:stop] = self.rng.integers(0, 730, size=size)
        today = np.datetime64("today", "D")
        dates = (today - day_offsets.astype("timedelta64[D]")).astype(str)
        order_ids = np.fromiter(
            (f"O{index:08d}" for index in range(1, self.n_orders + 1)),
            dtype=object,
            count=self.n_orders,
        )
        dataframe = pd.DataFrame(
            {
                "order_id": order_ids,
                "customer_id": customer_ids[customer_indexes],
                "product_id": product_ids[product_indexes],
                "quantity": quantities,
                "order_date": dates,
            }
        )
        LOGGER.info("Generated order DataFrame with %d rows", len(dataframe))
        return dataframe

    def generate_all(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate and return customer, product, and order DataFrames in order."""
        customers = self.generate_customers()
        products = self.generate_products()
        orders = self.generate_orders(customers, products)
        return customers, products, orders


def generate_dataset(
    customer_count: int = 100_000,
    product_count: int = 10_000,
    order_count: int = 1_000_000,
    seed: int = 42,
    output_dir: Path = config.RAW_DATA_DIR,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate all datasets, write them as CSV files, and return the DataFrames."""
    generator = SyntheticDataGenerator(
        n_customers=customer_count,
        n_products=product_count,
        n_orders=order_count,
        seed=seed,
        progress=progress,
    )
    customers, products, orders = generator.generate_all()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, dataframe in (
        ("customers.csv", customers),
        ("products.csv", products),
        ("orders.csv", orders),
    ):
        destination = output_dir / name
        dataframe.to_csv(destination, index=False)
        LOGGER.info("Wrote %d rows to %s", len(dataframe), destination)
    return customers, products, orders


def _parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for generating and saving synthetic data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=100_000)
    parser.add_argument("--products", type=int, default=10_000)
    parser.add_argument("--orders", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    """Run the data generator command-line application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    options = _parse_args(args)
    generate_dataset(
        customer_count=options.customers,
        product_count=options.products,
        order_count=options.orders,
        seed=options.seed,
        output_dir=options.output_dir,
        progress=not options.no_progress,
    )


if __name__ == "__main__":
    main()
