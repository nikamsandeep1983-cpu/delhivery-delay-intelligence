"""Shared constants and cleaning / feature functions for the Delay Intelligence capstone."""

import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

FILL_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]


def clean_shipments(raw, medians=None):
    """Return (clean_df, medians). Steps in the brief's order:
    1 drop duplicates, 2 service_type strip + title, 3 payment_mode NaN -> "Unknown",
    4 declared_value -> declared_value_inr (drop text column), 5 dispatch_time -> datetime,
    6 parcel_weight_kg > 100 -> NaN, 7 median-fill distance_km, parcel_weight_kg, declared_value_inr.
    """
    # 1. drop exact duplicate rows (work on a copy so the caller's table is untouched)
    df = raw.drop_duplicates().copy()

    # 2. service_type: strip spaces, title case -> Express / Standard / Economy
    df["service_type"] = df["service_type"].str.strip().str.title()

    # 3. payment_mode: missing -> "Unknown"
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. declared_value: remove the rupee sign and commas, convert to a number
    text = df["declared_value"].astype(str).str.replace("\u20b9", "", regex=False).str.replace(",", "", regex=False)
    df["declared_value_inr"] = pd.to_numeric(text, errors="coerce")
    df = df.drop(columns=["declared_value"])

    # 5. dispatch_time: text -> datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # 6. scanner-fault sentinel: any weight above 100 kg is treated as missing
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # 7. fill gaps with medians (computed here only when none are passed in)
    if medians is None:
        medians = {c: float(df[c].median()) for c in FILL_COLUMNS}
    for c in FILL_COLUMNS:
        df[c] = df[c].fillna(medians[c])

    return df, medians


def build_features(clean, hubs):
    """Left-merge hubs on hub_id, add promised_hours, is_cod, is_weekend,
    dispatch_hour, log_declared_value. Never use delivery_hours."""
    df = clean.merge(hubs, on="hub_id", how="left")
    assert len(df) == len(clean), "hub merge changed the row count"

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.day_of_week >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df