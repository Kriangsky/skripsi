"""One-time historical archive. Git moves, byte receipts and compatibility junctions."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'old hypothesa'


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    candidates = [ROOT / p for p in ['06_notebook/eksperimen', '06_notebook/hasil',
                  '01_hasil/tahap1a', '01_hasil/tahap1b', '03_insight_exp/tahap1b',
                  '03_insight_exp/arsip_tahap1a', '03_insight_exp/arsip_notebook',
                  '02_exp/tahap1b/linear_probes']]
    candidates += [p for p in (ROOT / '02_exp').rglob('*') if p.is_dir() and p.name in ['runs', 'evaluation']]
    sources = [p for p in candidates if not any(parent in candidates for parent in p.parents)]
    protected = [*ROOT.joinpath('04_dataset/train').rglob('*'),
                 *ROOT.joinpath('02_exp/arsip_pipeline/experiment_yolo26_v3').glob('*.csv'),
                 *ROOT.joinpath('02_exp/arsip_pipeline/experiment_yolo26_v3').glob('*.json'),
                 *ROOT.joinpath('02_exp').rglob('manifest.csv'),
                 *ROOT.joinpath('02_exp').rglob('pilot_manifest_private.csv')]
    protected = [p for p in protected if p.is_file() and not any(p.is_relative_to(s) for s in sources)]
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in protected}
    report = ROOT / 'experiments/model_a/reports/archive_protected_before.json'
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
    planned = []
    for source in sources:
        relative = source.relative_to(ROOT).as_posix()
        destination = ARCHIVE / relative
        if source.resolve() != source.absolute() or destination.exists():
            raise ValueError('Plan requires original real directory and absent destination')
        files = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*') if p.is_file()}
        if source.is_relative_to(ROOT / '02_exp') and any(Path(p).name in
                ['split_lock.csv', 'manifest.csv', 'inventory.csv', 'candidates.csv', 'annotations.csv'] for p in files):
            raise ValueError('Refuse to move a dataset definition')
        planned.append(dict(original=relative, destination=destination.relative_to(ROOT).as_posix(),
                            files=files, tracked_files=sum((relative + '/' + p) in tracked for p in files)))
    plan_path = ROOT / 'experiments/model_a/reports/archive_dry_run.json'
    plan = dict(directories=planned, protected_hashes=hashes,
                rollback='Remove each verified junction using rmdir (no /s), then git mv destination back to original; '
                         'untracked directories use Move-Item. No data deleted; archive receipts preserve hashes.')
    if '--dry-run' in sys.argv:
        plan_path.write_text(json.dumps(plan, indent=2), encoding='utf8')
        print(json.dumps(dict(directories=len(planned), files=sum(len(x['files']) for x in planned),
                              tracked=sum(x['tracked_files'] for x in planned), protected=len(hashes),
                              plan_sha256=sha(plan_path))))
        return
    if '--execute-reviewed-plan' not in sys.argv or json.loads(plan_path.read_text(encoding='utf8')) != plan:
        raise ValueError('Run --dry-run first; execution requires identical source/protected bytes and plan')
    ARCHIVE.mkdir(exist_ok=False)
    report.write_text(json.dumps(hashes, indent=2), encoding='utf8')
    receipts = []
    for source in sources:
        source = source.resolve()
        if not source.is_relative_to(ROOT) or source == ROOT or not source.is_dir():
            raise ValueError(f'Invalid archive source: {source}')
        relative = source.relative_to(ROOT).as_posix()
        destination = ARCHIVE / relative
        if destination.exists() or not destination.resolve().is_relative_to(ARCHIVE.resolve()):
            raise ValueError('Archive destination exists or escapes archive')
        files = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*') if p.is_file()}
        destination.parent.mkdir(parents=True, exist_ok=True)
        operation = 'git mv' if any(x.startswith(relative + '/') for x in tracked) else 'filesystem move (untracked)'
        if operation == 'git mv':
            subprocess.run(['git', 'mv', '--', relative, destination.relative_to(ROOT).as_posix()], cwd=ROOT, check=True)
        else:
            shutil.move(str(source), str(destination))
        # Existing launchers and hard-link validation keep resolving their historical paths.
        quote = lambda p: "'" + str(p).replace("'", "''") + "'"
        subprocess.run(['powershell', '-NoProfile', '-Command',
                        'New-Item -ItemType Junction -Path ' + quote(source) + ' -Value ' + quote(destination) + ' | Out-Null'], check=True)
        for name, digest in files.items():
            if sha(destination / name) != digest or sha(source / name) != digest:
                raise ValueError('Archived byte verification failed')
        receipts.append(dict(original=relative, destination=destination.relative_to(ROOT).as_posix(),
                             operation=operation, compatibility='junction at original directory', files=files))
        (ARCHIVE / 'archive_receipts.json').write_text(json.dumps(receipts, indent=2), encoding='utf8')
    copies = []
    for source in (ROOT / '07_dokumen/protokol').iterdir():
        if source.is_file():
            destination = ARCHIVE / 'protocol_snapshot' / source.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, destination)
            copies.append(dict(original=source.relative_to(ROOT).as_posix(),
                               destination=destination.relative_to(ROOT).as_posix(), sha256=sha(source)))
    for name, digest in hashes.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Protected source changed: {name}')
    text = '# Historical hypothesis archive\n\nArchived 2026-11-10 (Asia/Bangkok). No deletion or checkpoint overwrite.\n\n'
    text += 'These are superseded experiments, not Model A evidence. `archive_receipts.json` records every moved file and SHA256.\n\n'
    text += '| Original directory | Archive destination | Operation |\n|---|---|---|\n'
    text += ''.join(f'| `{x["original"]}` | `{x["destination"]}` | {x["operation"]} |\n' for x in receipts)
    text += '\nOriginal directory paths are compatibility junctions, not second runs. Git renames are staged; no commit created.\n\n'
    text += '## Intentionally retained\n\nRaw internal/external datasets; all frozen split/source manifests, candidate caches, selected-frame manifests, reviewer responses, original protocol documents and data-audit/provenance evidence; historical source/tests/launchers and preparation notebooks; environment and compatibility runtime. Moving these would break provenance or historical dependencies.\n\n'
    text += 'Historical experiment containers remain where they contain these protected inputs. Their run/evaluation subdirectories were archived. Official pretrained assets retained in old containers are never read by Model A.\n\n'
    text += 'Protocol files were copied byte-for-byte to `protocol_snapshot/` as archived evidence. Originals remain because historical hash locks and callers reference them.\n'
    text += '\nOther root consultation documents, addendum reports, draft/handoff/planning files and organization manifests remain untouched: mixed historical provenance and methodological decisions; their current scientific authority is not inferred.\n'
    (ARCHIVE / 'ARCHIVE_MANIFEST.md').write_text(text, encoding='utf8')
    (ARCHIVE / 'protocol_snapshot_receipts.json').write_text(json.dumps(copies, indent=2), encoding='utf8')
    print(json.dumps(dict(archived_directories=len(receipts), files=sum(len(x['files']) for x in receipts),
                          protected_files_verified=len(hashes), protocol_copies=len(copies))))


if __name__ == '__main__':
    main()
