# Predictive Maintenance Buildathon

This repository contains the coordinated four-person project for detecting abnormal operation in rotating equipment.

## Team integration

- **Person 1:** `person1/` generates independent vibration, current, temperature, load, and RPM test readings. Its labels remain in the offline ground-truth file.
- **Person 2:** `person2/` contains the frozen v2.2 model, feature functions, input contract, examples, and integration instructions.
- **Person 1 + 2 live bridge:** `person1_2/` generates a reading every second, extracts the v2.2 features, calls the model, and provides `start()`/`stop()` controls for the backend.
- **Person 3:** connect the backend to `person2.src.prediction.predict()` and return the documented prediction fields. Convert `ValueError` to HTTP 400 and missing-model errors to HTTP 503.
- **Person 4:** connect the dashboard to the backend response. Display raw monitoring values, anomaly state, risk, condition, recommendation, and history.

## Person 2 model input

The model expects one-second, four-channel vibration data converted into 144 numeric features: 72 fixed-frequency features and 72 RPM-normalized order features. It also requires `load` (`0`, `2`, or `4` Nm) and positive `speed_rpm`. Temperature and current can be displayed as monitoring data but are not model inputs.

The authoritative contract is [`person2/handoff/input_contract.json`](person2/handoff/input_contract.json). Start integration with [`person2/handoff/BACKEND_INTEGRATION.md`](person2/handoff/BACKEND_INTEGRATION.md) and [`person2/handoff/PERSON_1_V2.md`](person2/handoff/PERSON_1_V2.md).

## Local verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s person2/tests -v
.\.venv\Scripts\python.exe -m unittest discover -s person1/tests -v
```

Raw public datasets, processed tables, generated test data, and Python caches are intentionally excluded from Git. The deployable model is `person2/models/final_pipeline.joblib`.
