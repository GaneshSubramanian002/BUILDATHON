"""Shared input schema for feature extraction, training, and prediction."""

FEATURE_COLUMNS = [
    "temperature_A_mean",
    "temperature_B_mean",
    "current_U_rms",
    "vibration_A_x_rms",
    "vibration_A_y_rms",
    "vibration_B_x_rms",
    "vibration_B_y_rms",
    "load",
]

LABEL_COLUMNS = ["fault_type", "severity_value", "severity_rank"]

METADATA_COLUMNS = [
    "equipment_id",
    "recording_id",
    "window_id",
    "timestamp",
]

VALID_LOADS = {0.0, 2.0, 4.0}

