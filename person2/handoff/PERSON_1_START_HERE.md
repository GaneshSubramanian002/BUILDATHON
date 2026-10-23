# Person 1: next task

**Archived v1 instructions. Use PERSON_1_V2.md and input_contract.json instead.**

Person 2's v1 prototype is frozen. Build your independent simulator and external
test CSV using this handoff. No additional training or extraction is required
before you start. Detection has substantial known limitations; your simulator
tests compatibility, not real-world accuracy.

## Files to use

- `input_contract.json`: authoritative list of 59 required numeric fields,
  units, load values, window definition and labels. It supersedes earlier
  messages asking for 8 or 10 fields.
- `integration_examples.json`: 45 real feature rows with their source labels,
  covering each load/fault/severity combination. Use `sensor_data` as a live
  payload. The source labels outside it are never model inputs.
- `reference_features.csv`: public-data feature reference, not a test set.
- `simulator_ranges.csv`: statistics grouped by fault, load and severity.
- `environment.json`: versions used by Person 2.

## What you build

1. `sensor_simulator.py`: select equipment, load, fault and severity; emit one
   feature reading per second. Keep backend URL configurable. Person 3 owns
   the server route. Log failed requests and permit clean shutdown.
2. `generated_test_data.csv`: 59 sensor fields plus `equipment_id`, `timestamp`,
   `sequence_id`, `fault_type`, `severity_rank`. Create at least 10 distinct
   sequences of 50 rows per fault class (minimum 2,500 rows), covering loads
   0/2/4 Nm and the available severities. These counts are simulator targets,
   not requirements for fitting the public-data model.
3. A README explaining generation rules, seed and assumptions.

## Exact fault and severity labels

| fault_type | severity_rank |
|---|---|
| normal | 0 |
| inner_race_fault | 1, 2, 3 |
| outer_race_fault | 1, 2, 3 |
| misalignment | 1, 2, 3 |
| rotor_unbalance | 1, 2, 3, 4, 5 |

Ranks describe relative defect sizes within a fault class, not equivalent
physical sizes across classes. Prototype condition policy: normal is NORMAL,
rank 1 is WARNING, ranks 2 and above are CRITICAL. These are project rules,
not certified maintenance urgency labels.

## Generate coherent data

Do not draw each of the 59 fields independently. Prefer simulating raw
one-second waveforms and calling `signal_features()` from
`person2/src/extract_features.py`, or implement a documented joint feature
generator informed by the reference rows. Derive temperature difference from
the two temperature means. RMS, peak, crest, standard deviation and spectral
bands must describe the same signal. Spectral energy fractions sum to one
(zero is allowed for an entirely constant signal). Spectral centroid is 0 to
12,800 Hz. Current and vibration summaries are nonnegative.

A convex blend of nearby, same-load reference windows preserves many feature
relationships better than independent draws, but is still reference-based
synthetic augmentation. Label it honestly and do not copy complete rows as
the final test. Do not invent fault labels solely because a value was increased:
changing amplitude does not create an inner- or outer-race spectral signature.

Use different seeds and trajectories, small noise, and realistic correlations.
You can first build stable scenarios, then add transitions. Transition intervals
do not have experimentally verified ground truth; flag them separately rather
than treating them as guaranteed normal/fault labels.

The reference vibration features retain the source file numeric scale; do not
use earlier example values such as 0.21 or convert units yourself. Physical
conversion to g and exact housing/direction mapping need further source
verification. There is no RPM field, and only U-phase current is used because
the dataset's BPFO files have empty V/W phases.

## Test and submit

Send one schema-valid sample payload first, then finish the generator and CSV.
Keep the final test CSV private while constructing it; the model is now frozen.
Live JSON contains sensor fields plus metadata only. Ground-truth labels stay
in the offline CSV. Do not tune the generator until every prediction agrees:
failures are legitimate test results, including healthy readings called faulty.

From the shared repository root, evaluate the completed CSV with:

```powershell
.\.venv\Scripts\python.exe -m person2.src.evaluate_model person1\generated_test_data.csv
```

This writes JSON metrics and row-level predictions under `person2/reports/`.
Person 3 imports `from person2.src.prediction import predict` and passes one
sensor dictionary. Person 4 displays returned values without recalculating them.
