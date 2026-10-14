"""PySpark analytics for the synthetic e-commerce sales dataset."""

from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src import config

LOGGER = logging.getLogger(__name__)


class SalesAnalytics:
    """Load Parquet data and calculate customer, category, and monthly sales metrics."""

    def __init__(self, spark: SparkSession | None = None) -> None:
        """Initialize analytics, optionally with an existing Spark session."""
        self.spark = spark

    def create_spark_session(self) -> SparkSession:
        """Create a configured local Spark session with 4 GB memory and Kryo serialization."""
        self.spark = (
            SparkSession.builder.appName(config.APP_NAME)
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .getOrCreate()
        )
        LOGGER.info("Created local Spark session with adaptive execution and Kryo enabled")
        return self.spark

    def load_parquet(self, path: str | Path) -> DataFrame:
        """Load a Parquet file or Spark Parquet directory into a DataFrame."""
        if self.spark is None:
            self.create_spark_session()
        parquet_path = Path(path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet input does not exist: {parquet_path}")
        LOGGER.info("Loading Parquet data from %s", parquet_path)
        return self.spark.read.parquet(str(parquet_path))

    @staticmethod
    def _orders_with_revenue(orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Join order lines to product prices and add a calculated revenue column."""
        required_order_columns = {"product_id", "quantity"}
        required_product_columns = {"product_id", "price"}
        missing_orders = required_order_columns.difference(orders_df.columns)
        missing_products = required_product_columns.difference(products_df.columns)
        if missing_orders or missing_products:
            raise ValueError(
                "Missing required columns: "
                f"orders={sorted(missing_orders)}, products={sorted(missing_products)}"
            )
        priced_products = products_df.select("product_id", F.col("price").cast("double").alias("unit_price"))
        return (
            orders_df.join(priced_products, on="product_id", how="inner")
            .withColumn("revenue", F.col("quantity").cast("double") * F.col("unit_price"))
        )

    def top_customers_by_revenue(
        self, orders_df: DataFrame, products_df: DataFrame, n: int = 10
    ) -> DataFrame:
        """Return the top N customers ranked by total order revenue."""
        if n < 1:
            raise ValueError("n must be at least 1")
        if "customer_id" not in orders_df.columns:
            raise ValueError("orders_df must contain a customer_id column")
        return (
            self._orders_with_revenue(orders_df, products_df)
            .groupBy("customer_id")
            .agg(F.round(F.sum("revenue"), 2).alias("total_revenue"))
            .orderBy(F.desc("total_revenue"))
            .limit(n)
        )

    def sales_by_category(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Return revenue and units sold grouped by product category."""
        if "category" not in products_df.columns:
            raise ValueError("products_df must contain a category column")
        categorized_products = products_df.select(
            "product_id",
            "category",
            F.col("price").cast("double").alias("unit_price"),
        )
        joined = orders_df.join(categorized_products, on="product_id", how="inner")
        return (
            joined.withColumn("revenue", F.col("quantity").cast("double") * F.col("unit_price"))
            .groupBy("category")
            .agg(
                F.round(F.sum("revenue"), 2).alias("total_revenue"),
                F.sum("quantity").alias("units_sold"),
            )
            .orderBy(F.desc("total_revenue"))
        )

    def monthly_trends(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Calculate monthly revenue and month-over-month revenue growth percentage."""
        if "order_date" not in orders_df.columns:
            raise ValueError("orders_df must contain an order_date column")
        monthly_revenue = (
            self._orders_with_revenue(orders_df, products_df)
            .withColumn("month", F.date_format(F.to_date("order_date"), "yyyy-MM"))
            .groupBy("month")
            .agg(F.sum("revenue").alias("revenue"))
        )
        monthly_window = Window.orderBy("month")
        with_previous_month = monthly_revenue.withColumn(
            "previous_month_revenue", F.lag("revenue").over(monthly_window)
        )
        return (
            with_previous_month.withColumn(
                "revenue_growth_pct",
                F.when(
                    F.col("previous_month_revenue") > 0,
                    F.round(
                        (F.col("revenue") - F.col("previous_month_revenue"))
                        / F.col("previous_month_revenue")
                        * 100,
                        2,
                    ),
                ).otherwise(F.lit(None).cast("double")),
            )
            .select("month", F.round("revenue", 2).alias("revenue"), "revenue_growth_pct")
            .orderBy("month")
        )


def main() -> None:
    """Load configured Parquet datasets and print each sales analytics result."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    analytics = SalesAnalytics()
    spark = analytics.create_spark_session()
    try:
        orders = analytics.load_parquet(config.ORDERS_PARQUET_FILE)
        products = analytics.load_parquet(config.PRODUCTS_PARQUET_FILE)
        print("Top customers by revenue")
        analytics.top_customers_by_revenue(orders, products).show(truncate=False)
        print("Sales by category")
        analytics.sales_by_category(orders, products).show(truncate=False)
        print("Monthly revenue trends")
        analytics.monthly_trends(orders, products).show(truncate=False)
    except (OSError, ValueError) as error:
        LOGGER.exception("Analytics could not be completed: %s", error)
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
