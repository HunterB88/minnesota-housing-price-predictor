import requests
import pandas as pd
from pathlib import Path

# API endpoints
BASE_URL = "https://services.arcgis.com/afSMGVsC7QlRK1kZ/ArcGIS/rest/services/"

SALES_API = (
    BASE_URL
    + "Property_Sales_2021_to_2025/FeatureServer/0/query"
)

ASSESSOR_API = (
    BASE_URL
    + "Assessors_Parcel_Data_2020/FeatureServer/0/query"
)

# Configuration
MAX_SALES = 5000
PAGE_SIZE = 1000
BATCH_SIZE = 25

DATA_PATH = Path("data/minneapolis_housing.csv")


def fetch_api(url, params):
    """Send an API request and return its JSON response."""

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    return data


def download_sales(limit=MAX_SALES):
    """Download Minneapolis residential property sales."""

    records = []

    for offset in range(0, limit, PAGE_SIZE):

        params = {
            "where": (
                "PROPTYPE_DESC = 'RESIDENTIAL 1 UNIT' "
                "AND ADJ_SALE_PRICE > 0"
            ),
            "outFields": (
                "PIN,ADJ_SALE_PRICE,SALE_DATE,"
                "PROPTYPE_DESC,NBHD_DESC"
            ),
            "returnGeometry": "false",
            "resultOffset": offset,
            "resultRecordCount": min(PAGE_SIZE, limit - offset),
            "orderByFields": "ObjectId ASC",
            "f": "json"
        }

        data = fetch_api(SALES_API, params)

        batch = [
            record["attributes"]
            for record in data["features"]
        ]

        records.extend(batch)

        print(f"Downloaded {len(records)} sales...")

        if len(batch) < min(PAGE_SIZE, limit - offset):
            break

    if not records:
        raise ValueError("No residential sales were downloaded.")

    sales = pd.DataFrame(records)

    # Standardize property IDs
    sales["APN"] = (
        sales["PIN"]
        .astype(str)
        .str.removeprefix("p")
    )

    return sales


def download_assessor_data(property_ids):
    """Download property characteristics for matching IDs."""

    properties = []

    for i in range(0, len(property_ids), BATCH_SIZE):

        batch = property_ids[i:i + BATCH_SIZE]

        ids = ",".join(f"'{pid}'" for pid in batch)

        params = {
            "where": f"APN IN ({ids})",
            "outFields": (
                "APN,ABOVEGROUNDAREA,BEDROOMS,"
                "BATHROOMS,YEARBUILT,NEIGHBORHOOD"
            ),
            "returnGeometry": "false",
            "f": "json"
        }

        data = fetch_api(ASSESSOR_API, params)

        properties.extend([
            record["attributes"]
            for record in data["features"]
        ])

        print(
            f"\rDownloaded {len(properties)} assessor records...",
            end="",
            flush=True
        )

    print()

    if not properties:
        raise ValueError("No matching assessor records were found.")

    assessor = pd.DataFrame(properties)

    if assessor["APN"].duplicated().any():
        raise ValueError("Duplicate assessor property IDs detected.")

    return assessor


def clean_housing_data(housing):
    """Remove invalid records and duplicate property sales."""

    # Remove records missing essential information
    required_columns = [
        "APN",
        "ADJ_SALE_PRICE",
        "SALE_DATE",
        "ABOVEGROUNDAREA",
        "BEDROOMS",
        "BATHROOMS",
        "YEARBUILT",
        "NEIGHBORHOOD"
    ]

    housing = housing.dropna(subset=required_columns).copy()

    # Remove invalid numerical values
    housing = housing[
        (housing["ADJ_SALE_PRICE"] > 0)
        & (housing["ABOVEGROUNDAREA"] > 0)
        & (housing["BEDROOMS"] > 0)
        & (housing["BATHROOMS"] > 0)
        & (housing["YEARBUILT"] > 1800)
    ].copy()

    # Convert sale dates
    housing["SALE_DATE"] = pd.to_datetime(
        housing["SALE_DATE"],
        unit="ms",
        utc=True
    )

    # Keep the latest recorded sale per property
    housing = housing.sort_values("SALE_DATE")

    housing = housing.drop_duplicates(
        subset=["APN"],
        keep="last"
    )

    return housing.reset_index(drop=True)


def build_dataset():
    """Download, merge, clean, and save the housing dataset."""

    print("Starting Minneapolis housing data pipeline...\n")

    # Step 1: Download sales
    sales = download_sales()

    # Step 2: Download matching assessor records
    property_ids = sales["APN"].dropna().unique().tolist()

    assessor = download_assessor_data(property_ids)

    # Step 3: Merge datasets
    housing = pd.merge(
        sales,
        assessor,
        on="APN",
        how="inner",
        validate="many_to_one"
    )

    print(f"\nMatched {len(housing)} sales.")

    # Step 4: Clean data
    housing = clean_housing_data(housing)

    print(f"Valid unique homes: {len(housing)}")

    if housing.empty:
        raise ValueError("No usable housing records remain.")

    # Step 5: Save dataset
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    housing.to_csv(DATA_PATH, index=False)

    print(f"\nDataset saved to {DATA_PATH}")

    # Summary statistics
    print("\nDataset summary:")
    print(housing[[
        "ADJ_SALE_PRICE",
        "ABOVEGROUNDAREA",
        "BEDROOMS",
        "BATHROOMS",
        "YEARBUILT"
    ]].describe())


if __name__ == "__main__":
    build_dataset()