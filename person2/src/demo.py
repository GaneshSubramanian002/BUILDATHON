"""Show a frozen-model prediction on a labeled integration example."""
import argparse
import json
from pathlib import Path
from .prediction import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fault', default='normal', choices=['normal', 'inner_race_fault', 'outer_race_fault', 'misalignment', 'rotor_unbalance'])
    parser.add_argument('--rank', type=int, default=None)
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / 'handoff/integration_examples.json'
    examples = json.loads(path.read_text())
    match = next((e for e in examples if e['source_fault'] == args.fault and (args.rank is None or e['source_severity_rank'] == args.rank)), None)
    if match is None:
        parser.error('No example with that fault/severity combination')
    print('Integration example from public training data; not an accuracy test.')
    print(json.dumps(predict(match['sensor_data']), indent=2))


if __name__ == '__main__':
    main()
