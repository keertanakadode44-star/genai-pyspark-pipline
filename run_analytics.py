"""Run all Spark sales analytics against the generated Parquet datasets."""

import logging
import time
from collections.abc import Callable

from pyspark.sql import DataFrame
from src import config
from src.spark_analytics import SalesAnalytics

LOGGER = logging.getLogger(__name__)


def _display_timed_result(operation_name: str, operation: Callable[[], DataFrame]) -> None:
    """Execute a zero-argument DataFrame operation, display it, and print elapsed time."""
    started_at = time.perf_counter()
    result = operation()
    result.show(truncate=False)
    elapsed = time.perf_counter() - started_at
    print(f"{operation_name} execution time: {elapsed:.2f} seconds")


def main() -> int:
    """Load the source Parquet files, run analytics, and stop the Spark session."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    analytics = SalesAnalytics()
    spark = analytics.create_spark_session()
    try:
        customers_df = analytics.load_parquet(config.CUSTOMERS_PARQUET_FILE)
        products_df = analytics.load_parquet(config.PRODUCTS_PARQUET_FILE)
        orders_df = analytics.load_parquet(config.ORDERS_PARQUET_FILE)
        LOGGER.info(
            "Loaded customers (%d columns), products (%d columns), and orders (%d columns)",
            len(customers_df.columns),
            len(products_df.columns),
            len(orders_df.columns),
        )

        print("Top customers by revenue")
        _display_timed_result(
            "Top customers by revenue",
            lambda: analytics.top_customers_by_revenue(orders_df, products_df),
        )

        print("Sales by category")
        _display_timed_result(
            "Sales by category",
            lambda: analytics.sales_by_category(orders_df, products_df),
        )

        print("Monthly revenue trends")
        _display_timed_result(
            "Monthly revenue trends",
            lambda: analytics.monthly_trends(orders_df, products_df),
        )
        return 0
    except Exception:
        LOGGER.exception("Analytics run failed")
        return 1
    finally:
        spark.stop()
        LOGGER.info("Spark session stopped")


if __name__ == "__main__":
    raise SystemExit(main())
