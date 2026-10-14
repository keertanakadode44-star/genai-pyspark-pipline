# E-commerce Data Pipeline

A small Python project that generates seeded fake customer, product, and order data, then uses Apache Spark to produce useful business summaries. Random values are repeatable for a given seed; generated dates are relative to the day the generator runs.

## Project structure

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_generator.py
│   └── spark_analytics.py
├── data/
│   ├── raw/          # Generated CSV input
│   └── processed/    # Spark result CSV directories
├── tests/
│   └── test_data_generator.py
└── notebooks/        # Space for exploratory Jupyter notebooks
```

## Setup

Python 3.10 or newer and Java 17 or newer are recommended (Spark requires Java).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the pipeline

Generate the default large dataset as Parquet files (random values are repeatable for the same seed):

```bash
python main.py
```

Analyze the generated Parquet files with PySpark:

```bash
python run_analytics.py
```

For smaller CSV samples, the generator module can still be run directly:

```bash
python -m src.data_generator --customers 500 --products 100 --orders 2000 --seed 42
```

The Parquet entry point creates `customers.parquet`, `products.parquet`, and `orders.parquet` in `data/raw/`, then prints elapsed time and each file's size. Parquet output requires the `pyarrow` dependency in `requirements.txt`.

Spark writes `revenue_by_category`, `monthly_revenue`, `top_products`, and `customer_order_summary` under `data/processed/`. Spark writes each result as a directory containing one or more part files; this is normal Spark CSV output. Use `--input-dir` or `--output-dir` to override the defaults.

## Data model and insights

- `customers.csv`: customer ID, name, email, age, city, country, and registration date.
- `products.csv`: product ID, name, category, price, stock, and rating.
- `orders.csv`: order ID, customer/product IDs, quantity, and order date.
- Analytics summarize completed-order revenue by category and month, rank products by revenue, and aggregate order count and spend per customer.

The generator creates one product line per order, and assigns 80% of orders to 20% of customers using Pareto-distributed customer weights. Revenue is quantity multiplied by the product catalog price. If an input orders file has a `status` column, only `completed` orders contribute to revenue metrics. All generated records are synthetic.

## Tests

Run the generator tests with:

```bash
python -m pytest
```
