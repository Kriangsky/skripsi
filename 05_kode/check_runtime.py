"""Read-only check that compatibility hard links still point at the canonical section files."""
import json
import os
from pathlib import Path
import sys


def check(root):
    root=Path(root).resolve()
    plan=json.loads((root/'07_dokumen/organisasi_20261104_lengkap/manifest.json').read_text(encoding='utf8'))
    pairs=[(root/item['new'],root/plan['runtime']/item['old']) for item in plan['files']]
    pairs.extend((root/data['new']/item['relative'],root/plan['runtime']/data['old']/item['relative'])
                 for data in plan['data'] for item in data['files'])
    for canonical,legacy in pairs:
        if not canonical.is_file() or not legacy.is_file() or not os.path.samefile(canonical,legacy):
            raise RuntimeError(f'Compatibility links diverged; both versions preserved. Reconcile before running: {canonical}')
    return len(pairs)


if __name__=='__main__':
    check(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent.parent)
