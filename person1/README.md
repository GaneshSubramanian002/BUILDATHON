# Person 1 independent sensor simulator

This code generates raw rotating-machine readings without importing or calling
Person 2's feature extraction or ML model. It produces four synchronized
vibration channels at 25.6 kHz, three-phase current, two temperatures, load and
metadata for one-second windows.

Supported states are normal, inner-race fault, outer-race fault, misalignment
and rotor unbalance. The waveforms use documented physical patterns such as
shaft harmonics and bearing-resonance bursts. They are simulation assumptions,
not measured industrial ground truth.

Run a small dataset from the Buildathon directory:

```powershell
.\.venv\Scripts\python.exe -m person1.generate_dataset
```

Outputs are written to `person1/generated/`:

- `waveforms/*.npz`: raw vibration, current and temperature arrays.
- `sensor_readings.jsonl`: JSON-safe readings without fault labels.
- `ground_truth.csv`: labels kept separately for offline evaluation.
- `manifest.json`: seed, configuration and class counts.

Increase the size when needed:

```powershell
.\.venv\Scripts\python.exe -m person1.generate_dataset --sequences-per-class 10 --seconds-per-sequence 50
```

Do not use generated labels as model inputs. Do not train the final model on
these records if they are intended to be the independent Person 1 test set.
