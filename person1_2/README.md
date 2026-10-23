# Person 1 + Person 2 live integration

This package generates one-second sensor readings with Person 1 and immediately classifies each reading with Person 2's frozen v2.2 model. The default hackathon mode samples public-dataset feature distributions by condition, load, and severity so the demonstration exercises the trained model correctly. Use `--raw-simulator` to test the independent handcrafted waveform generator.

Run five readings:

```powershell
python -m person1_2.service --count 5
```

Run continuously with random condition, severity, and load selection until `Ctrl+C`:

```powershell
python -m person1_2.service --interval 1
```

For the hackathon showcase, cycle through all five conditions automatically:

```powershell
python -m person1_2.service --showcase --interval 1
```

The cycle uses normal, rotor unbalance, misalignment, inner-race fault, and outer-race fault, with varying loads. Random mode is the default; use `--mode random` explicitly if preferred. Use `--mode fixed --fault ... --severity ... --load ...` for one fixed scenario.

Person 3 can import `ReadingService`, provide a callback, and control it directly:

```python
from person1_2 import ReadingService
service = ReadingService(callback=lambda item: send_to_frontend(item.public_json()))
service.start()
service.stop()
```

Simulator ground truth stays out of `public_json()` and is never sent to the model.
