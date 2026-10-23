# Person 2 — Predictive Maintenance ML

**Current handoff: v2.2.0.** Start with
`handoff/PERSON_1_V2.md`. The authoritative contract is
`handoff/input_contract.json` and the deployable artifact is
`models/final_pipeline.joblib`. Earlier experimental scripts and mock payloads
are retained as development history and do not define the frozen contract.
Read `reports/MODEL_CARD.md` for the evaluation and remaining detection limits.

Reproduce final training and exports:

```powershell
.\.venv\Scripts\python.exe -m person2.src.audit_raw_signals
.\.venv\Scripts\python.exe -m person2.src.import_cross_speed_dataset <extracted-dataset-folder>
.\.venv\Scripts\python.exe -m person2.src.build_order_dataset
.\.venv\Scripts\python.exe -m person2.src.evaluate_gate_ensemble
.\.venv\Scripts\python.exe -m person2.src.finalize_v4
.\.venv\Scripts\python.exe -m unittest discover -s person2/tests -v
```

Only rerun training intentionally: Person 1's test must use a frozen artifact.
Use `requirements-lock.txt` for compatible package versions. Inference never
trains or modifies a model.

Quick local demo:

```powershell
.\.venv\Scripts\python.exe -m person2.src.demo --fault normal
.\.venv\Scripts\python.exe -m person2.src.demo --fault inner_race_fault --rank 3
```

These examples are from training data and are for integration checks only.
To build the shareable ZIP, run `python -m person2.src.package_handoff`.

V2.2 uses 72 fixed-Hz and 72 RPM-normalized order features. Its ensemble gate
uses low-weight real cross-machine data at 600/900 RPM. Temperature and current
remain optional monitoring fields; they are not model inputs. V1 models and
metrics are preserved as `models/final_pipeline_v1.joblib` and
`models/final_metrics_v1.json`; the previous v2 model is also archived. The remainder below describes the historical
v1 extraction experiments, not the active input contract.

This component trains and evaluates fault-detection models using the public
rotating-machine dataset containing temperature, three-phase motor current,
four vibration channels, load, fault type, and severity.

## Model input fields

- `temperature_A_mean` and `temperature_B_mean` in °C
- `current_U_rms` in A
- `vibration_A_x_rms`, `vibration_A_y_rms`, `vibration_B_x_rms`, and
  `vibration_B_y_rms` on the source numeric scale (physical conversion pending)
- `load` in Nm (`0.0`, `2.0`, or `4.0`)

`fault_type` and `severity` are evaluation labels and must never be passed to
the prediction model as input features.

## Data layout

Extract the downloads outside the repository root and place their contents in:

```text
data/raw/current_temp/
data/raw/vibration/
```

Generated window-level feature tables belong in `data/processed/`. Person 1's
independently generated external test data belongs in `data/test/`.

Only phase U is used because all published outer-race fault recordings have
empty V- and W-phase channels. This avoids turning missing-data structure into
an accidental fault label.

Do not commit raw or processed data to Git because the files are large.

## Environment

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r person2\requirements.txt
```

## Planned workflow

1. Match vibration and current/temperature recordings by filename metadata.
2. Split complete recordings into training, validation, and test groups.
3. Extract time-window features without mixing windows from one recording
   across splits.
4. Train a Random Forest fault classifier as the main model.
5. Optionally compare it with an Isolation Forest anomaly detector.
6. Evaluate on held-out public recordings and Person 1's external simulator
   dataset separately.

Run feature extraction from the repository root:

```powershell
.\.venv\Scripts\python.exe -m person2.src.extract_features
```

## Expanded model and integration

```powershell
.\.venv\Scripts\python.exe -m person2.src.improve_model
```

This compares three fixed feature families using the same recording-grouped
folds. Features include peaks, standard deviation, Pearson kurtosis, crest
factor, spectral centroid, and fractional energy in four frequency bands.
It saves `models/improved_classifier.joblib`, `models/improved_metrics.json`,
and an experimental `models/input_contract.json`. The selected feature
contract supersedes the original eight-field mock payload when expanded
features are selected. Do not fabricate missing signal features from RMS.

Backend usage (run from repository root):

```python
from person2.src.prediction import predict
result = predict(sensor_data)
```

Use `data/test/model_input_examples.json` to check integration. Its examples
come from training recordings and cannot be used to claim test accuracy.
Person 1 can use `data/processed/simulator_ranges.csv` and the joint reference
rows to design a simulator. Marginal ranges alone do not preserve realistic
cross-feature relationships. Synthetic tests are reported separately.

### Interpretation and limits

- Grouped cross-validation is development evidence, not an untouched final
  test: earlier evaluation already informed model design. Only three normal
  recordings exist. Independent normal runs are needed for stronger claims.
- The condition model uses a demo policy: normal is NORMAL; smallest defect
  rank is WARNING; larger ranks are CRITICAL. Physical defect size is not a
  verified operational urgency label. Rotor unbalance has five ranks, not three.
- Health scores 90/60/25 are fixed display indices for those conditions. They
  do not measure remaining life or calibrated failure probability. Classifier
  vote confidence is also not a calibrated failure probability.
- MAT and TDMS acquisition use different systems. Windows are paired by
  recording and elapsed sample count under a nominal 25.6 kHz assumption;
  precise cross-device synchronization has not been established.
- Features retain source numeric scale. Vibration unit scaling and physical
  channel location mapping must be checked against acquisition documentation
  before physical sensor deployment. Spectral frequencies use 25.6 kHz.
- BPFO phase V/W are empty, so the model only uses phase U across all faults.
- Raw data are unchanged. No real failure-time prediction is established.

