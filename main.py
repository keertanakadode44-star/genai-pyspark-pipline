"""Generate synthetic e-commerce data and save it as Parquet files."""

import logging
import sys
from time import perf_counter

from src import config
from src.data_generator import SyntheticDataGenerator

LOGGER = logging.getLogger(__name__)


def _format_file_size(size_bytes: int) -> str:
    """Format a file size in bytes as a readable binary unit."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def main() -> int:
    """Generate the configured dataset, write Parquet files, and report timings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    started_at = perf_counter()
    try:
        config.ensure_data_directories()
        generator = SyntheticDataGenerator(
            n_customers=config.DEFAULT_CUSTOMER_COUNT,
            n_products=config.DEFAULT_PRODUCT_COUNT,
            n_orders=config.DEFAULT_ORDER_COUNT,
            seed=config.DEFAULT_RANDOM_SEED,
        )
        customers, products, orders = generator.generate_all()

        datasets = (
            ("customers", customers, config.CUSTOMERS_PARQUET_FILE),
            ("products", products, config.PRODUCTS_PARQUET_FILE),
            ("orders", orders, config.ORDERS_PARQUET_FILE),
        )
        for name, dataframe, output_path in datasets:
            dataframe.to_parquet(output_path, index=False)
            file_size = output_path.stat().st_size
            print(f"{name}: {output_path} ({_format_file_size(file_size)})")

        elapsed = perf_counter() - started_at
        print(f"Generation and Parquet writing completed in {elapsed:.2f} seconds.")
        LOGGER.info("Synthetic data pipeline completed successfully")
        return 0
    except (OSError, ValueError, ImportError) as error:
        LOGGER.exception("Could not generate or save synthetic data: %s", error)
        print(f"Data generation failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        LOGGER.exception("Unexpected error during synthetic data generation")
        print(f"Unexpected error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
