"""Package only code, model, reports and simulator reference data; no raw data."""
from pathlib import Path
import hashlib
import json
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    selected = [root / name for name in ['README.md', 'requirements.txt', 'requirements-lock.txt']]
    for folder in ['src', 'tests']:
        selected.extend(sorted((root / folder).glob('*.py')))
    selected.extend(sorted((root / 'handoff').glob('*.*')))
    selected.extend(root / name for name in ['models/final_pipeline.joblib', 'models/final_metrics.json', 'reports/MODEL_CARD.md', 'reports/accuracy_audit.json', 'reports/targeted_accuracy_audit.json', 'reports/healthy_augmentation_audit.json', 'reports/cross_dataset_audit.json', 'reports/order_transfer_audit.json', 'reports/feature_fusion_audit.json', 'reports/gate_ensemble_audit.json', 'reports/raw_metadata_audit.json', 'reports/known_load_metrics.json'])
    manifest = {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in selected}
    manifest_path = root / 'handoff/SHA256_MANIFEST.json'
    # The manifest cannot hash itself.
    manifest.pop('handoff/SHA256_MANIFEST.json', None)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    selected = [p for p in selected if p != manifest_path] + [manifest_path]
    version_parts = json.loads((root / 'handoff/input_contract.json').read_text())['schema_version'].split('.')[:2]
    version = '_'.join(version_parts)
    archive = root.parent / f'person2_handoff_v{version}.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
        for path in selected:
            output.write(path, 'person2/' + path.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as check:
        if check.testzip() is not None:
            raise RuntimeError('Archive checksum verification failed')
        assert not any('/data/raw/' in name or name.endswith(('.tdms', '.mat')) for name in check.namelist())
    print(f'Created {archive} ({archive.stat().st_size / 1024**2:.1f} MiB); {len(selected)} files; archive verified')


if __name__ == '__main__':
    main()
