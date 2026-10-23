# Person 1: active version 2.2 instructions

Version 2 replaces the 59-field v1 contract and improves development performance.
Use only `input_contract.json` for the authoritative fields. V1 instructions
remain for history and must not define your simulator.

## Your task

Build an independent simulator and labeled test CSV. The model requires 144
vibration spectrum features, plus `load` (0, 2 or 4 Nm) and positive
`speed_rpm` as validated context. The features contain 72 fixed-Hz fields and
72 RPM-normalized order fields. Optional metadata: `equipment_id`, `timestamp`.
`load` is not a trained feature. Temperature/current are optional dashboard
monitoring channels and do not affect v2.2 ML decisions.

Use `integration_examples.json` for 45 complete example payloads. Source labels
are outside `sensor_data`; never send them to `predict()`. Use
`reference_features.csv` and `simulator_ranges.csv` to design your generator.
Their rows are public training references, not unseen test data.

## Preferred generation

Create four one-second vibration waveforms at 25,600 Hz. Use the same feature
function as training:

```python
from person2.src.audit_raw_signals import targeted
from person2.src.order_features import order_features

payload = {'equipment_id': 'MOTOR_01', 'load': 2.0, 'speed_rpm': measured_rpm}
for channel_index, waveform in enumerate(four_waveforms, start=1):
    payload.update(targeted(waveform, 25600, f'vibration_{channel_index}'))
    orders = order_features(waveform, 25600, measured_rpm, f'vibration_{channel_index}')
    payload.update({f'order__{name}': value for name, value in orders.items()})
```

Both functions demean and apply a Hann window. They compute fractional spectral
energies, fractions around approximate shaft harmonics, spectral entropy and
dominant frequency/order. Features are amplitude-normalized. Source raw vibration
was stored in MKS; metadata converts to g with factor 0.101971621297793 and zero
offset. Positive constant unit conversion does not change these features.
Do not invent 72 unrelated random numbers. Energy fractions over the twelve
non-overlapping bands sum to 1 for a nonconstant signal. Harmonic fractions
overlap those bands: do not sum them together with band fractions. Dominant
frequency is 0..12800 Hz. Spectral entropy uses natural logarithms.

Normal, unbalance and bearing faults differ in spectral patterns; simply
raising amplitude cannot simulate them for this amplitude-normalized model.
Use reference spectral distributions to design coherent signals, or a
documented joint feature generator. Report that this remains source-informed
synthetic validation. Do not copy complete training rows as your test.

## Labels and deliverables

| fault_type | severity_rank |
|---|---|
| normal | 0 |
| inner_race_fault | 1, 2, 3 |
| outer_race_fault | 1, 2, 3 |
| misalignment | 1, 2, 3 |
| rotor_unbalance | 1, 2, 3, 4, 5 |

Build `sensor_simulator.py`, `generated_test_data.csv` and a README documenting
seed, assumptions and generator rules. CSV contains 144 features, load,
speed_rpm, equipment_id, timestamp, sequence_id, fault_type and severity_rank.
Aim for at least 2,500 rows across five classes, loads, severities and multiple
sequences. Labels stay offline. Start with stable scenarios. Mark transition
intervals separately because their fault/condition labels are not measured
ground truth. Send one valid payload for integration before finishing the CSV.

Keep the final test independent of model training. Do not tune until every
prediction is correct: errors are legitimate evaluation results.

```powershell
python -m person2.src.evaluate_model person1/generated_test_data.csv
```

Person 3 uses `from person2.src.prediction import predict`. No backend rewrite
is needed if it forwards the JSON and stores the output fields. Person 4 can
still display temperature and current as monitoring values, clearly labeled.

## Limits

V2.2 grouped development healthy recall is 77.6%, fault recall 94.8%, and fault
macro F1 86.9%. These are not independent final-test scores. Healthy runs still
produce false alerts, and some faults are missed. Condition labels and fixed
health values remain prototype assumptions. Additional independent healthy
recordings are needed before claiming reliability on real machines.
