# Person 3 / Person 4 integration

Active version: 2.2.0. Use `PERSON_1_V2.md` and `input_contract.json`.
ML inputs are 144 vibration spectrum features; load and speed_rpm are validated context.
Temperature/current may be stored and displayed but are not v2 model features.

Keep the `person2` directory and run Python from its parent (repository root).
Install `person2/requirements.txt` into your environment. The model was saved
with the versions in `environment.json`; scikit-learn model files require a
compatible environment. Load only this local trusted artifact.

```python
import json
from pathlib import Path
from person2.src.prediction import predict

examples = json.loads(Path('person2/handoff/integration_examples.json').read_text())
result = predict(examples[0]['sensor_data'])
print(result)
```

Use `input_contract.json` as the definitive field list. Required features are
numeric JSON values. `ValueError` means invalid input and should become HTTP
400. Missing model errors should become HTTP 503. Models are cached after the
first call. Importing the module does not retrain anything.

Output fields:

| Field | Meaning |
|---|---|
| schema_version | 2.2.0 |
| anomaly | boolean normal/fault decision |
| anomaly_score | uncalibrated binary classifier vote/probability score, 0 to 1 |
| possible_fault | normal or one of the four fault identifiers |
| confidence | uncalibrated gate/fault confidence, 0 to 1 |
| condition | NORMAL, WARNING or CRITICAL |
| risk | LOW, MEDIUM or CRITICAL, respectively |
| health_score | fixed demo index 90, 60 or 25, respectively |
| health_score_basis | explicit explanation of the prototype index |
| recommendation | maintenance suggestion string |

Dashboard labels should say 'Prototype health index' and 'Model confidence'.
Neither is real failure probability or remaining useful life. Show raw sensor
values, prediction history, and alerts for WARNING/CRITICAL. Person 3 owns
alert storage and deduplication; Person 2 returns classification only.

See `models/final_metrics.json` for development evaluation and limitations in
the report. The model cannot be described as reliable for unseen healthy runs.
