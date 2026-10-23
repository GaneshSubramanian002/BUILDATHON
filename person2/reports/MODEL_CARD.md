# Active Person 2 prototype v2.2.0

The full recording-held-out development audit tested four sensor feature sets
and four algorithms (16 combinations). All still had zero held-out healthy
recall. A second audit extracted narrower spectral bands, harmonics around
approximate 50.17 Hz shaft frequency, entropy and dominant frequencies.
Standardized balanced logistic regression on these normalized vibration
features generalizes better than the tree classifiers tested.

V2.1 uses a logistic binary gate followed by Extra Trees fault and condition
classifiers. 72 vibration-only spectral features avoid uncertain cross-device
timing. Load is validated context. Temperature/current are optional monitoring
values and do not influence ML decisions. The gate uses KAIST data plus 1%
weighted 600-RPM auxiliary data from DOI 10.17632/y8mv28hjwb.1. The fault and
condition models remain KAIST-only because the auxiliary data lacks bearing
faults and severity labels. Previous artifacts remain archived separately.

| Same grouped development evaluation | V1 | V2 | V2.1 | V2.2 |
|---|---:|---:|---:|---:|
| Healthy recall | 0.0% | 68.1% | 77.4% | 77.6% |
| Fault recall | 98.4% | 94.5% | 94.7% | 94.8% |
| Binary accuracy | 86.6% | 91.4% | 92.6% | 92.8% |
| Binary macro F1 | 46.4% | 80.3% | 83.7% | 83.9% |
| End-to-end fault macro F1 | 69.5% | 85.3% | 86.8% | 86.9% |
| Condition-policy macro F1 | 44.8% | 75.0% | 77.5% | 77.7% |

These folds were reused for feature/model selection. They are development
scores, not an unbiased final estimate or independent industrial validation.
Healthy false-positive rate is still 22.4%; missed-fault rate is 5.2% on these
windows. The gate threshold is 0.50. Ensemble settings were selected
using the same folds, so independent acquisition sessions are still needed.

V2.2 ensembles the v2.1 fixed-Hz gate (75% of its score) with an RPM-normalized
order gate (25%). A replacement was accepted only if both healthy and fault
recall improved on the identical grouped folds. On the 1200-RPM subset excluded
from selection, accuracy improved from 82.1% to 85.6%, normal recall from 63.9%
to 71.4%, fault recall from 91.4% to 92.8%, and macro F1 from 78.9% to 83.3%.
This is cross-machine validation, not proof of industrial reliability.

All nine auxiliary archives and all 3,545 extracted samples passed their
published SHA-256 checksums. Direct unweighted pooling failed because the two
machines occupy different spectral domains. KAIST-based scaling, clipping to
five standard deviations and 1% class-balanced auxiliary weight produced the
selected result. On auxiliary speeds excluded from final training, 900-RPM
macro F1 was 64.2% and 1200-RPM macro F1 was 78.9%; this variation confirms
that the model is not universally speed-independent. See
`cross_dataset_audit.json` and `models/final_metrics.json`.

An additional audit generated 900 healthy feature records from the three real
healthy recordings using small speed warps, sensor gain/coupling changes and
35--50 dB measurement noise. Training with 50--300 synthetic samples per
available training recording did not improve held-out real healthy recall, so
these rows are excluded from the active model. See
`healthy_augmentation_audit.json`. Synthetic rows are useful for robustness
checks but do not replace independent healthy acquisitions.

Audit confirmed all 45 MAT files use 25.6 kHz, unit label g, display factor
0.101971621297793, zero offset, and 'Data saved by Test.Lab in MKS'. Apply the
display transform for physical g. V2 spectral fractions, entropy and dominant
frequency are unaffected by constant amplitude conversion. Published channel
order is A-x, A-y, B-x, B-y (Mendeley dataset documentation:
https://data.mendeley.com/datasets/ztmf3m7h5x/6).

MAT and TDMS metadata times differ and lack a verified common acquisition
trigger. V1 simultaneous fusion claims are unsupported. V2 avoids that input
pairing entirely. A separate known-load study with an early healthy baseline
and a chronological gap was also tested: mean healthy recall 45.8%, mean fault
recall 97.4%. Its same-run healthy test is weaker evidence and not compared
directly as an accuracy gain over grouped CV.

Files: `raw_metadata_audit.json`, `accuracy_audit.json`,
`targeted_accuracy_audit.json`, `healthy_augmentation_audit.json`,
`cross_dataset_audit.json`, `known_load_metrics.json`, plus
`models/final_metrics.json`. Current contract: `handoff/input_contract.json`.
Health index and severity policy remain assumptions as documented below.

## Archived v1 details (historical results)

## Intended use

Local hackathon demonstration of telemetry processing, fault identification,
condition labels and maintenance recommendations. The implementation is ready
for Person 1's simulator integration. Reliability on independent healthy motor
recordings is not established and measured performance is poor on that task.

## Data and extraction

45 paired MAT/TDMS recordings from the public KAIST rotating-machine dataset;
4,500 one-second feature rows. Includes normal, inner/outer race fault,
misalignment and unbalance at 0/2/4 Nm. 59 model features include temperature,
U-phase current, four vibration channels, load, shape and spectral summaries.
All V/W current channels are omitted because BPFO files lack their samples.
Five source `Unbalalnce` filename typos are normalized only in matching code.
Source files are unchanged.

Vibration numeric scale and Point1..4 physical mapping need acquisition
documentation verification. Cross-device absolute synchronization has not
been established: feature windows use paired recording identities and nominal
elapsed sample counts. Do not claim precise synchronous sensor fusion.

## Evaluation

Three-fold StratifiedGroupKFold with complete recordings assigned to folds.
No recording appears in both training and evaluation within a fold. Only three
normal recordings exist (one per load). Every normal test recording therefore
requires generalizing to a load for which a normal run was not in training.
These folds informed development/model selection, so they are not an untouched
final test. Rows are correlated within recordings; row count is not the count
of independent experiments. No confidence interval or industrial claim is made.

| Frozen pipeline development metric | Result |
|---|---:|
| Normal recall | 0.0% |
| Abnormal recall | 98.4% |
| Binary macro F1 | 46.4% |
| End-to-end fault macro F1 | 69.5% |
| Condition-policy macro F1 | 44.8% |

The binary accuracy of 86.6% is misleading in this imbalanced dataset and must
not be presented as evidence of healthy-equipment detection. Unseen healthy
records produce false maintenance alerts. More independent healthy data or a
different deployment-specific baseline would be needed for reliable use.

Compared Random Forest, Extra Trees and standardized RBF SVM binary detectors
using identical folds. Random Forest selected by macro F1. A fault-only Extra
Trees model and a fault-only condition-policy Extra Trees model follow the
binary gate. Models for deployment are fitted on all available public rows
after development evaluation. Their training predictions are not test scores.
Raw reports and full fold assignments are in `models/final_metrics.json`.

## Output semantics

- Anomaly score is an uncalibrated binary model score, not failure probability.
- Confidence combines binary and conditional fault-model scores; uncalibrated.
- Ground-truth condition policy is a project assumption: normal=NORMAL;
  smallest defect rank=WARNING; higher defect ranks=CRITICAL. Fault sizes
  are not verified urgency categories. Predicted condition can be incorrect.
- Health score is a fixed prototype display index: 90/60/25 for those states.
  It is not remaining life, percentage of physical health, or predictive RUL.
- Recommendations are static suggestions selected by the predicted fault.
  Backend owns alert persistence; dashboard owns presentation.

## Next evidence

Person 1 generates independent synthetic sequences under the frozen feature
contract. Evaluate once with the frozen model, report failures, and keep those
metrics separate from public-data development CV. Source-informed synthetic
data do not independently establish real-world performance. Do not tune the
simulator to make the model appear accurate.
