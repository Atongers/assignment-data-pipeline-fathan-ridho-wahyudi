"""ETL data processing pipeline for the Automobile EDA dataset."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "automobileEDA_dirty_training.csv"
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "automobileEDA_processed.csv"
)

CATEGORICAL_COLUMNS = [
    "make",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "engine-type",
    "num-of-cylinders",
    "fuel-system",
    "horsepower-binned",
]

SCALED_NUMERIC_COLUMNS = [
    "normalized-losses",
    "wheel-base",
    "height",
    "curb-weight",
    "engine-size",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price",
    "city-L/100km",
]

ONE_HOT_COLUMNS = [
    "aspiration",
    "body-style",
    "drive-wheels",
    "engine-location",
    "engine-type",
    "fuel-system",
]

DOOR_COUNT_MAP = {"two": 2, "four": 4}
CYLINDER_COUNT_MAP = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "eight": 8,
    "twelve": 12,
}
HORSEPOWER_BIN_MAP = {"Low": 0, "Medium": 1, "High": 2}

DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d-%b-%Y")


def _parse_mixed_date(value: Any) -> pd.Timestamp:
    """Parse a date string across known dataset formats without day/month swapping."""
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(text, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(text, errors="coerce")


def _derive_horsepower_bin(horsepower: float) -> str:
    """Map horsepower into the dataset's Low/Medium/High bin thresholds."""
    if horsepower <= 101:
        return "Low"
    if horsepower <= 155:
        return "Medium"
    return "High"


def _print_section(title: str) -> None:
    """Print a clearly formatted section header for terminal readability."""
    separator = "=" * 72
    print(f"\n{separator}\n{title}\n{separator}")


def extract_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Extract the raw CSV dataset without mutating the source file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {file_path}")
    df = pd.read_csv(file_path)
    _print_section("1. EXTRACT")
    print(f"Source file : {file_path}")
    print(f"Loaded shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def inspect_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Inspect initial dataset structure, types, missing values, duplicates, and categories."""
    _print_section("2. DATASET INSPECTION")

    print("\n--- First 5 Rows (head) ---")
    print(df.head(5).to_string())

    print(f"\n--- Shape ---\nRows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n--- Column Names ---")
    for idx, col in enumerate(df.columns, start=1):
        print(f"  {idx:2d}. {col}")

    print("\n--- Data Types ---")
    print(df.dtypes.to_string())

    missing_counts = df.isna().sum()
    missing_nonzero = missing_counts[missing_counts > 0]
    print("\n--- Missing Values per Column ---")
    print(missing_counts.to_string())
    print(f"Total missing cells: {int(missing_counts.sum())}")

    duplicate_count = int(df.duplicated().sum())
    print(f"\n--- Duplicate Records ---\nExact duplicate rows: {duplicate_count}")

    print("\n--- Unique Values in Relevant Categorical Columns ---")
    unique_summary: dict[str, list[Any]] = {}
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            unique_vals = df[col].dropna().unique().tolist()
            unique_summary[col] = unique_vals
            print(f"  {col} ({len(unique_vals)} unique): {unique_vals}")

    if "transaction_date" in df.columns:
        sample_dates = df["transaction_date"].dropna().head(8).tolist()
        print(f"\n  transaction_date (sample raw formats): {sample_dates}")

    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_per_column": missing_nonzero.to_dict(),
        "total_missing": int(missing_counts.sum()),
        "duplicate_count": duplicate_count,
        "unique_categorical": unique_summary,
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean duplicates, text inconsistencies, mixed dates, and missing values."""
    _print_section("3. DATA CLEANING")

    rows_before = len(df)
    missing_before = int(df.isna().sum().sum())
    cleaned = df.copy()
    modified_columns: set[str] = set()

    # Drop exact duplicates before imputation so duplicate rows do not skew statistics.
    duplicates_removed = int(cleaned.duplicated().sum())
    if duplicates_removed > 0:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Standardize text whitespace and casing in categorical columns.
    for col in CATEGORICAL_COLUMNS:
        if col not in cleaned.columns:
            continue
        original_series = cleaned[col].copy()
        non_null_mask = cleaned[col].notna()
        normalized = cleaned.loc[non_null_mask, col].astype(str).str.strip()
        if col == "horsepower-binned":
            normalized = normalized.str.title()
        else:
            normalized = normalized.str.lower()
        cleaned.loc[non_null_mask, col] = normalized
        if not original_series.equals(cleaned[col]):
            modified_columns.add(col)

    # Normalize mixed transaction_date formats and interpolate missing sequential dates.
    if "transaction_date" in cleaned.columns:
        parsed_dates = cleaned["transaction_date"].apply(_parse_mixed_date)
        # Interpolate in days since epoch to stay independent of pandas datetime resolution.
        epoch = pd.Timestamp("1970-01-01")
        days_since_epoch = (parsed_dates - epoch).dt.total_seconds() / 86400.0
        interpolated_days = days_since_epoch.interpolate(method="linear")
        cleaned["transaction_date"] = (
            epoch + pd.to_timedelta(interpolated_days.round(), unit="D")
        ).dt.strftime("%Y-%m-%d")
        modified_columns.add("transaction_date")

    # Drop rows with missing target variable (price) to avoid fabricating target labels.
    price_missing_count = int(cleaned["price"].isna().sum())
    if price_missing_count > 0:
        cleaned = cleaned.dropna(subset=["price"]).reset_index(drop=True)
        modified_columns.add("price")

    # Impute numeric features using median (robust to right-skewed engine/stroke distributions).
    for num_col in ("stroke", "horsepower"):
        if num_col in cleaned.columns and cleaned[num_col].isna().any():
            median_val = float(cleaned[num_col].median())
            cleaned[num_col] = cleaned[num_col].fillna(median_val)
            modified_columns.add(num_col)

    # Impute num-of-doors using mode per body-style, falling back to overall mode.
    if "num-of-doors" in cleaned.columns and cleaned["num-of-doors"].isna().any():
        overall_door_mode = cleaned["num-of-doors"].mode().iloc[0]
        group_modes = cleaned.groupby("body-style")["num-of-doors"].transform(
            lambda s: s.mode().iloc[0] if not s.mode().empty else overall_door_mode
        )
        cleaned["num-of-doors"] = cleaned["num-of-doors"].fillna(group_modes)
        modified_columns.add("num-of-doors")

    # Label missing make as 'unknown' rather than guessing an unverified manufacturer.
    if "make" in cleaned.columns and cleaned["make"].isna().any():
        cleaned["make"] = cleaned["make"].fillna("unknown")
        modified_columns.add("make")

    # Derive missing horsepower-binned from cleaned horsepower values using dataset boundaries.
    if (
        "horsepower-binned" in cleaned.columns
        and cleaned["horsepower-binned"].isna().any()
    ):
        missing_bin_mask = cleaned["horsepower-binned"].isna()
        cleaned.loc[missing_bin_mask, "horsepower-binned"] = cleaned.loc[
            missing_bin_mask, "horsepower"
        ].apply(_derive_horsepower_bin)
        modified_columns.add("horsepower-binned")

    rows_after = len(cleaned)
    missing_after = int(cleaned.isna().sum().sum())

    print(f"Rows before cleaning          : {rows_before}")
    print(f"Rows after cleaning           : {rows_after}")
    print(f"Duplicate rows removed        : {duplicates_removed}")
    print(f"Rows dropped (missing price)  : {price_missing_count}")
    print(f"Missing values before cleaning: {missing_before}")
    print(f"Missing values after cleaning : {missing_after}")
    print(f"Modified columns ({len(modified_columns)}): {sorted(modified_columns)}")

    return cleaned


def transform_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Min-Max scaling and categorical encoding while keeping comparisons clear."""
    _print_section("4. DATA TRANSFORMATION")
    transformed = df.copy()

    # A. Min-Max Scaling using scikit-learn MinMaxScaler -> saved into *_scaled columns.
    scaler = MinMaxScaler()
    scaled_cols = [f"{col}_scaled" for col in SCALED_NUMERIC_COLUMNS]
    transformed[scaled_cols] = scaler.fit_transform(transformed[SCALED_NUMERIC_COLUMNS])

    print("A. Min-Max Scaling (new *_scaled columns via MinMaxScaler):")
    for col, scaled_col in zip(SCALED_NUMERIC_COLUMNS, scaled_cols):
        sample_before = transformed[col].iloc[0]
        sample_after = transformed[scaled_col].iloc[0]
        print(
            f"  - {col:18s} -> {scaled_col:25s} | "
            f"range [{transformed[col].min():.2f}, {transformed[col].max():.2f}] -> [0.00, 1.00] | "
            f"row[0]: {sample_before} -> {sample_after:.6f}"
        )

    # B1. Ordinal Encoding for inherently ordered categorical columns.
    transformed["num-of-doors_encoded"] = (
        transformed["num-of-doors"].map(DOOR_COUNT_MAP).astype(int)
    )
    transformed["num-of-cylinders_encoded"] = (
        transformed["num-of-cylinders"].map(CYLINDER_COUNT_MAP).astype(int)
    )
    transformed["horsepower-binned_encoded"] = (
        transformed["horsepower-binned"].map(HORSEPOWER_BIN_MAP).astype(int)
    )

    print("\nB1. Ordinal Encoding:")
    print(
        f"  - num-of-doors      -> num-of-doors_encoded      (map: {DOOR_COUNT_MAP})"
    )
    print(
        f"  - num-of-cylinders  -> num-of-cylinders_encoded  (map: {CYLINDER_COUNT_MAP})"
    )
    print(
        f"  - horsepower-binned -> horsepower-binned_encoded (map: {HORSEPOWER_BIN_MAP})"
    )

    # B2. Frequency Encoding for high-cardinality nominal column 'make'.
    make_freq = transformed["make"].value_counts(normalize=True)
    transformed["make_freq"] = transformed["make"].map(make_freq)
    print(
        f"\nB2. Frequency Encoding:\n  - make -> make_freq "
        f"(sample row[0]: '{transformed['make'].iloc[0]}' -> {transformed['make_freq'].iloc[0]:.6f})"
    )

    # B3. One-Hot Encoding for nominal low-cardinality categorical columns.
    dummies = pd.get_dummies(transformed[ONE_HOT_COLUMNS], prefix=ONE_HOT_COLUMNS, dtype=int)
    transformed = pd.concat([transformed, dummies], axis=1)
    print(
        f"\nB3. One-Hot Encoding ({len(ONE_HOT_COLUMNS)} columns -> {dummies.shape[1]} binary columns):"
    )
    print(f"  - Source columns: {ONE_HOT_COLUMNS}")
    print(f"  - Sample new columns: {list(dummies.columns[:6])} ...")

    # C. Text Cleaning summary (applied to categorical text columns and dates).
    print(
        "\nC. Text Normalization Summary:\n"
        "  - Trimmed trailing/leading spaces & unified casing across categorical text columns\n"
        "  - Standardized transaction_date into ISO format YYYY-MM-DD"
    )
    print(
        f"\nFinal transformed shape: {transformed.shape[0]} rows x {transformed.shape[1]} columns"
    )
    return transformed


def save_dataset(df: pd.DataFrame, output_path: Path = PROCESSED_DATA_PATH) -> None:
    """Save the processed dataset to disk, creating target directories if missing."""
    _print_section("5. LOAD (SAVE PROCESSED DATASET)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed dataset to: {output_path}")
    print(f"Output size               : {output_path.stat().st_size} bytes")


def main() -> None:
    """Run the end-to-end ETL pipeline: Extract -> Inspect -> Clean -> Transform -> Load."""
    raw_df = extract_data(RAW_DATA_PATH)
    inspect_dataset(raw_df)
    cleaned_df = clean_dataset(raw_df)
    transformed_df = transform_dataset(cleaned_df)
    save_dataset(transformed_df, PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()
