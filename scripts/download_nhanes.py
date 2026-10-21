"""Download and validate C3's NHANES 2017-March 2020 source files.

Run from the repository root: python scripts/download_nhanes.py
This step does not filter a cohort, create clinical labels or train a model.
"""
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'data' / 'raw' / 'nhanes_2017_2020'
BASE_URL = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/'

# Preserve survey weights/design variables for later analysis, not as predictors.
REQUIRED = {
    'P_DEMO': ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'WTMECPRP', 'SDMVPSU', 'SDMVSTRA'],
    'P_BMX': ['SEQN', 'BMXBMI'],
    'P_BIOPRO': ['SEQN', 'LBXSASSI', 'LBXSATSI'],
    'P_CBC': ['SEQN', 'LBXPLTSI'],
    'P_DIQ': ['SEQN', 'DIQ010'],
    'P_LUX': ['SEQN', 'LUAXSTAT', 'LUANMVGP', 'LUXSMED', 'LUXSIQRM', 'LUXCAPM'],
}


def validate(payload, name):
    if not payload.startswith(b'HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!'):
        raise ValueError(f'{name}: response is not a SAS XPT file (possibly an HTML error page).')
    frame = pd.read_sas(BytesIO(payload), format='xport')
    missing = set(REQUIRED[name]) - set(frame.columns)
    if missing:
        raise ValueError(f'{name}: required columns missing: {sorted(missing)}')
    if frame.empty or frame.SEQN.isna().any() or frame.SEQN.duplicated().any():
        raise ValueError(f'{name}: empty file or missing/duplicate participant IDs.')
    return frame


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    for name, columns in REQUIRED.items():
        path = DESTINATION / f'{name}.xpt'
        url = BASE_URL + path.name
        try:
            if path.exists():
                payload = path.read_bytes()
                origin = 'cached'
            else:
                print(f'Downloading {name}...', flush=True)
                request = Request(url, headers={'User-Agent': 'C3-NHANES-Research-Prototype/1.0'})
                with urlopen(request, timeout=45) as response:
                    payload = response.read()
                origin = 'downloaded'
            frame = validate(payload, name)
            if origin == 'downloaded':
                partial = path.with_suffix('.part')
                partial.write_bytes(payload)
                partial.replace(path)
            records.append({
                'file': path.name, 'url': url,
                'codebook': BASE_URL + name + '.htm',
                'rows': len(frame), 'columns': len(frame.columns),
                'sha256': sha256(payload).hexdigest(),
                'missing_required_values': {c: int(frame[c].isna().sum()) for c in columns},
            })
            print(f'OK {name}: {len(frame):,} participants ({origin})', flush=True)
        except Exception as exc:
            failures.append({'file': path.name, 'error': str(exc)})
            print(f'FAILED {name}: {exc}', file=sys.stderr, flush=True)

    report = {
        'cycle': 'NHANES 2017-March 2020 pre-pandemic',
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'complete' if not failures else 'incomplete',
        'files': records, 'failures': failures,
        'note': 'Source-file counts are not the final eligible cohort size.',
    }
    (DESTINATION / 'download_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if failures:
        print('Some files failed. Check the errors and rerun; valid cached files are reused.')
        return 1
    print(f'\nAll 6 files validated. Saved in: {DESTINATION}')
    print('Next: join by SEQN, define the cohort and separate predictors from scan outcomes.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
