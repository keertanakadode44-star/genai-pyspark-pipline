"""Central paths and defaults for the e-commerce pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CUSTOMERS_FILE = RAW_DATA_DIR / "customers.csv"
PRODUCTS_FILE = RAW_DATA_DIR / "products.csv"
ORDERS_FILE = RAW_DATA_DIR / "orders.csv"

DEFAULT_CUSTOMER_COUNT = 100_000
DEFAULT_PRODUCT_COUNT = 10_000
DEFAULT_ORDER_COUNT = 1_000_000
DEFAULT_RANDOM_SEED = 42
APP_NAME = "EcommercePipelineAnalytics"

CUSTOMERS_PARQUET_FILE = RAW_DATA_DIR / "customers.parquet"
PRODUCTS_PARQUET_FILE = RAW_DATA_DIR / "products.parquet"
ORDERS_PARQUET_FILE = RAW_DATA_DIR / "orders.parquet"


def ensure_data_directories() -> None:
    """Create the raw and processed data directories if they do not exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
