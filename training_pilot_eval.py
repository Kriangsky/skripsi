"""Internal-validation evaluation. Retain extra-class predictions on common-six images."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import bisindo_pipeline as b


def box_iou(a, c):
    x1, y1 = max(a[0], c[0]), max(a[1], c[1])
    x2, y2 = min(a[2], c[2]), min(a[3], c[3])
    inter = max(0., x2-x1)*max(0., y2-y1)
    area_a, area_c = (a[2]-a[0])*(a[3]-a[1]), (c[2]-c[0])*(c[3]-c[1])
    return inter/(area_a+area_c-inter) if area_a+area_c > inter else 0.


def match_frame(gt_class, gt_box, predictions, nc, conf=.25, iou=.5):
    """One reviewed GT per video. Same-class IoU matching first; all unmatched predictions are FP.

    If no correct match exists, the highest-IoU spatial wrong-class match represents FP+FN
    in the confusion matrix. Additional unmatched predictions use the background row.
    """
    b.yolo_box(gt_box)
    if not 0 <= gt_class < nc:
        raise ValueError('Ground truth class outside model vocabulary')
    kept = []
    for p in predictions:
        if not isinstance(p['class_id'], int) or not 0 <= p['class_id'] < nc or not np.isfinite(p['confidence']) or not 0 <= p['confidence'] <= 1:
            raise ValueError('Invalid prediction class/confidence')
        b.yolo_box(p['box'])
        if p['confidence'] >= conf:
            kept.append(dict(p, iou=box_iou(gt_box, p['box'])))
    spatial = [i for i, p in enumerate(kept) if p['iou'] >= iou]
    correct = [i for i in spatial if kept[i]['class_id'] == gt_class]
    chosen = max(correct, key=lambda i: (kept[i]['iou'], kept[i]['confidence'])) if correct else None
    tp = int(chosen is not None)
    entries = []
    wrong = None
    if chosen is not None:
        entries.append((gt_class, gt_class))
    elif spatial:
        wrong = max(spatial, key=lambda i: (kept[i]['iou'], kept[i]['confidence']))
        entries.append((gt_class, kept[wrong]['class_id']))
    else:
        entries.append((gt_class, nc))
    entries.extend((nc, p['class_id']) for i, p in enumerate(kept) if i not in [chosen, wrong])
    for i, p in enumerate(kept):
        p['true_positive'] = i == chosen
    representative = chosen if chosen is not None else wrong
    if representative is None and kept:
        representative = max(range(len(kept)), key=lambda i: kept[i]['confidence'])
    return dict(tp=tp, fp=len(kept)-tp, fn=1-tp, correct=bool(tp),
        confusion_entries=entries, predictions=kept,
        representative=kept[representative] if representative is not None else None)


def rates(tp, fp, fn):
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.)


def plot_confusion(matrix, names, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    ax.imshow(matrix, cmap='Blues', vmin=0)
    labels = names + ['background']
    ax.set(xticks=range(len(labels)), xticklabels=labels, yticks=range(len(labels)),
           yticklabels=labels, xlabel='Predicted', ylabel='Ground truth', title=title)
    ax.tick_params(axis='x', rotation=60)
    for y, row in enumerate(matrix):
        for x, count in enumerate(row):
            ax.text(x, y, str(count), ha='center', va='center')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def contact_sheets(work, records):
    folder = Path(work) / 'reports/contact_sheets'
    folder.mkdir(exist_ok=True)
    paths = []
    for split in ['train', 'val']:
        rows = [r for r in records if r['split'] == split]
        for start in range(0, len(rows), 20):
            tiles = []
            for row in rows[start:start+20]:
                image = cv2.imread(str(Path(work) / 'datasets/B/images' / split / f"{row['sample_id']}.jpg"))
                if image is None:
                    raise ValueError('Cannot render audit image')
                h, w = image.shape[:2]
                lo = tuple(round(v) for v in np.array(row['box'][:2])*[w, h])
                hi = tuple(round(v) for v in np.array(row['box'][2:])*[w, h])
                cv2.rectangle(image, lo, hi, (0, 255, 0), 3)
                tile = np.full((250, 320, 3), 245, dtype=np.uint8)
                factor = min(320/w, 200/h)
                thumb = cv2.resize(image, (max(1, round(w*factor)), max(1, round(h*factor))))
                tile[:thumb.shape[0], :thumb.shape[1]] = thumb
                cv2.putText(tile, f"{row['class_id']} {row['class_name']} {split}", (5, 218), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
                cv2.putText(tile, row['sample_id'], (5, 240), cv2.FONT_HERSHEY_SIMPLEX, .36, (0, 0, 0), 1)
                tiles.append(tile)
            while len(tiles) % 4:
                tiles.append(np.full_like(tiles[0], 245))
            path = folder / f'{split}_{start//20+1:02d}.jpg'
            canvas = np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)])
            if not cv2.imwrite(str(path), canvas):
                raise ValueError('Cannot write contact sheet')
            paths.append(path)
    return paths


def evaluate(work):
    from training_pilot import (BASE, CLASSES, INFERENCE, PILOT_CONFIG, csv, dump,
                                local_yolo, read_json, set_inference_mode, validate_release)
    work = Path(work).resolve()
    release = validate_release(work)
    trained = read_json(work / 'runs/trained_models.json')
    if not trained or trained['release_fingerprint'] != release['fingerprint']:
        raise ValueError('Two completed independent runs on the frozen release required')
    models = {r['model']: r for r in trained['models']}
    if set(models) != {'A', 'B'} or models['A']['initial_checkpoint_sha256'] != models['B']['initial_checkpoint_sha256']:
        raise ValueError('Independent A/B runs must use the identical initial COCO checkpoint')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU evaluation fallback')
    YOLO = local_yolo(work)
    out = work / 'evaluation'
    if (out / 'evaluation_started.json').exists() or (out / 'evaluation_complete.json').exists() or (out / 'common6_metrics.csv').exists():
        raise ValueError('Evaluation already exists; refusing overwrite')
    dump(out / 'evaluation_started.json', dict(release_fingerprint=release['fingerprint'], config=INFERENCE))
    manifest = csv(work / 'datasets/manifest.csv')
    common_rows, full_rows, per_class, fixed, diagnostics = [], [], [], [], []
    prediction_records = []
    for name, scope, cfg in [('A', 'common6', 'A/pilot.yaml'), ('B', 'common6', 'B/common6.yaml'), ('B', 'full8', 'B/pilot.yaml')]:
        record = models[name]
        checkpoint = Path(record['path'])
        if b.sha(checkpoint) != record['sha256']:
            raise ValueError('Trained checkpoint changed')
        model = YOLO(str(checkpoint))
        set_inference_mode(model)
        names = BASE if name == 'A' else CLASSES
        if list(model.names.values()) != names or not model.model.end2end:
            raise ValueError('Model vocabulary/inference mode mismatch')
        # classes=None deliberately retains halo/tidur predictions in B/common6.
        metrics = model.val(data=str(work / 'datasets' / cfg), split='val',
            imgsz=640, batch=16, device=0, workers=0, half=False, augment=False, classes=None,
            agnostic_nms=False, conf=INFERENCE['ap_confidence'], iou=INFERENCE['iou'],
            max_det=INFERENCE['max_det'], plots=False, verbose=False,
            project=str(out / 'ultralytics'), name=f'{name}_{scope}', exist_ok=False)
        mp, mr, map50, map5095 = map(float, metrics.box.mean_results())
        rows = manifest[manifest.split.eq('val') & manifest.class_name.isin(BASE if scope == 'common6' else CLASSES)]
        aggregate = dict(model=name, scope=scope, n=len(rows), mAP50=map50, mAP50_95=map5095,
            precision_ultralytics_curve=mp, recall_ultralytics_curve=mr,
            note='AP averaged over GT-supported classes; use fixed-threshold FP/FN for extra-class errors on common6')
        (common_rows if scope == 'common6' else full_rows).append(aggregate)
        for j, cid in enumerate(metrics.box.ap_class_index):
            per_class.append(dict(model=name, scope=scope, class_id=int(cid), class_name=names[int(cid)],
                gt_n=int(rows.class_id.eq(int(cid)).sum()), AP50=float(metrics.box.all_ap[j, 0]),
                AP50_95=float(np.mean(metrics.box.all_ap[j])),
                precision_ultralytics_curve=float(metrics.box.p[j]), recall_ultralytics_curve=float(metrics.box.r[j])))
        matrix = np.zeros((len(names)+1, len(names)+1), dtype=int)
        tp = fp = fn = 0
        per_counts = {i: dict(tp=0, fp=0, fn=0, gt_n=0) for i in range(len(names))}
        for row in rows.itertuples():
            image_path = work / 'datasets' / name / 'images/val' / f'{row.sample_id}.jpg'
            predictions = model.predict(str(image_path), conf=INFERENCE['fixed_confidence'],
                imgsz=640, device=0, half=False, augment=False, classes=None, agnostic_nms=False,
                iou=INFERENCE['iou'], max_det=INFERENCE['max_det'], save=False, verbose=False)[0]
            raw = [dict(class_id=int(cls), confidence=float(confidence), box=box)
                   for box, confidence, cls in zip(predictions.boxes.xyxyn.cpu().tolist(),
                       predictions.boxes.conf.cpu().tolist(), predictions.boxes.cls.cpu().tolist())]
            result = match_frame(int(row.class_id), json.loads(row.box), raw, len(names),
                                 INFERENCE['fixed_confidence'], INFERENCE['matching_iou'])
            tp += result['tp']; fp += result['fp']; fn += result['fn']
            gt = per_counts[int(row.class_id)]
            gt['gt_n'] += 1; gt['tp'] += result['tp']; gt['fn'] += result['fn']
            for prediction in result['predictions']:
                if not prediction['true_positive']:
                    per_counts[prediction['class_id']]['fp'] += 1
            for actual, predicted in result['confusion_entries']:
                matrix[actual, predicted] += 1
            prediction_records.append(dict(model=name, scope=scope, sample_id=row.sample_id,
                gt_class=int(row.class_id), gt_box=json.loads(row.box), **result))
            if row.class_name in ['apa_kabar', 'halo']:
                rep = result['representative']
                diagnostics.append(dict(model=name, scope=scope, sample_id=row.sample_id,
                    ground_truth=row.class_name, correct=result['correct'], false_negative=result['fn'],
                    false_positive_n=result['fp'], predicted_class=names[rep['class_id']] if rep else 'no_detection',
                    confidence=rep['confidence'] if rep else None, iou=rep['iou'] if rep else None,
                    all_predictions=json.dumps(result['predictions']), quality_flags=row.quality_flags,
                    annotation_reviewer=row.reviewer, quality_override_reason=row.quality_override_reason))
                if not result['correct'] or result['fp']:
                    image = predictions.plot()
                    h, w = image.shape[:2]
                    gt_box = json.loads(row.box)
                    cv2.rectangle(image, (round(gt_box[0]*w), round(gt_box[1]*h)),
                        (round(gt_box[2]*w), round(gt_box[3]*h)), (0, 255, 0), 3)
                    cv2.putText(image, 'GT: '+row.class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
                    if not cv2.imwrite(str(out / 'error_examples' / f'{name}_{scope}_{row.sample_id}.jpg'), image):
                        raise ValueError('Cannot save diagnostic visual')
        fixed.append(dict(model=name, scope=scope, class_name='ALL', n=len(rows),
            confidence=INFERENCE['fixed_confidence'], matching_iou=INFERENCE['matching_iou'], **rates(tp, fp, fn)))
        for cid, count in per_counts.items():
            fixed.append(dict(model=name, scope=scope, class_name=names[cid], n=count['gt_n'],
                confidence=INFERENCE['fixed_confidence'], matching_iou=INFERENCE['matching_iou'],
                **rates(count['tp'], count['fp'], count['fn'])))
        label = f'confusion_matrix_{name}' if scope == 'common6' else 'confusion_matrix_B_full8'
        plot_confusion(matrix, names, out / f'{label}.png', f'{name} {scope}: conf=.25, IoU=.50')
        pd.DataFrame(matrix, index=names+['background'], columns=names+['background']).to_csv(out / f'{label}.csv')
        del model
        torch.cuda.empty_cache()
    for filename, rows in [('common6_metrics.csv', common_rows), ('full8_metrics.csv', full_rows),
                           ('per_class_metrics.csv', per_class), ('fixed_threshold_metrics.csv', fixed),
                           ('apa_kabar_halo_diagnostics.csv', diagnostics)]:
        pd.DataFrame(rows).to_csv(out / filename, index=False)
    dump(out / 'fixed_predictions.json', prediction_records)
    dump(out / 'evaluation_complete.json', dict(release_fingerprint=release['fingerprint'],
        config=INFERENCE, models=models, completion_time=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
    dump(out / 'status.json', dict(status='COMPLETED', metrics_available=True))
    lines = ['# Controlled training pilot — preliminary diagnostic', '',
        'Both independent models completed 30 epochs with reviewed annotations and a frozen internal validation release.',
        '640 positive videos: 480 train + 160 val for B; A uses exactly the shared 360 train + 120 val images/labels.',
        'See reports/dataset_quality_report.csv for all approved/rejected/unreviewed counts and exclusion rates.', '',
        '## Matched common-six validation',
        *[f'- Model {r["model"]}: mAP50={r["mAP50"]:.4f}, mAP50-95={r["mAP50_95"]:.4f} on the same 120 images.' for r in common_rows],
        'B extra-class predictions were retained. AP metrics use GT-supported categories; fixed_threshold_metrics.csv counts every unmatched extra-class detection as FP.',
        'Ultralytics curve precision/recall use its selected operating point; fixed-threshold precision/recall/F1 use conf=.25 and IoU=.50.', '',
        '## Full-eight task', *[f'- B: mAP50={r["mAP50"]:.4f}, mAP50-95={r["mAP50_95"]:.4f} on 160 images.' for r in full_rows],
        'Do not compare this eight-class mAP directly with A six-class mAP as identical tasks.', '',
        '## apa_kabar / halo',
        'Diagnostics include every relevant validation image, GT, predicted classes, confidences, IoUs, FN, extra FP, quality flags and error visuals.']
    frame = pd.DataFrame(diagnostics)
    for (model, scope, cls), group in frame.groupby(['model', 'scope', 'ground_truth']):
        lines.append(f'- {model}/{scope}/{cls}: {int(group.correct.sum())}/{len(group)} correct at fixed thresholds; FN={int(group.false_negative.sum())}.')
    state = read_json(work / 'reports/readiness.json')
    lines += ['', f'Positive reviewed videos={state["verified_positive_n"]}; rejected={state["explicitly_rejected_n"]}.',
        'Quality flags and preparation/transition/temporal annotations describe possible associations, not causal explanations. '
        'No controlled temporal comparator was run; neither single-frame limits nor a need for LSTM/TCN are proven.',
        'Passing pilot gates does not certify the remaining full dataset. Review pilot errors and remaining annotations before final release.',
        'One seed and 30 epochs cannot establish model superiority. After pilot inspection, freeze configuration, '
        'run full-data experiments with multiple seeds, then evaluate Kaggle external data once. No external data was used here.']
    (work / 'pilot_conclusion.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
