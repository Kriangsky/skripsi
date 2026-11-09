"""Fixed-class frame metrics on paired videos. No model or external-data access."""
import json

import numpy as np

from bisindo_pipeline import CLASSES

ALL_CLASSES = tuple(CLASSES)
BASE = ALL_CLASSES[:6]
EXTRA = ALL_CLASSES[6:]


def integer_array(values, name):
    raw = np.asarray(values)
    if raw.dtype.kind not in 'iu' or raw.size == 0:
        raise ValueError(f'{name} must contain nonempty integer labels')
    return raw.astype(np.int64, copy=False)


def probabilities(values, n, rows):
    scores = np.asarray(values, dtype=float)
    if (scores.shape != (rows, n) or not np.isfinite(scores).all()
            or (scores < 0).any() or not np.allclose(scores.sum(1), 1, atol=1e-5, rtol=0)):
        raise ValueError('Invalid full probability vector or class mapping')
    return scores


def align_predictions(a, b, labels):
    """Join by sample_id; require matching video, content, frame and GT identities."""
    labels = tuple(labels)
    parts = []
    for frame, n in [(a, 6), (b, 8)]:
        required = {'sample_id', 'video', 'video_sha256', 'image_sha256',
                    'true_class_id', 'true_class', 'probabilities'}
        if not required.issubset(frame.columns):
            raise ValueError(f'Missing prediction identity fields: {required - set(frame.columns)}')
        for field in ['sample_id', 'video', 'video_sha256', 'image_sha256']:
            if frame[field].isna().any() or frame[field].astype(str).str.strip().eq('').any():
                raise ValueError(f'Missing {field}')
            if field != 'image_sha256' and frame[field].duplicated().any():
                raise ValueError(f'Duplicate {field}')
        ids = integer_array(frame.true_class_id.to_numpy(), 'truth')
        if (ids < 0).any() or (ids >= n).any():
            raise ValueError('Invalid truth class')
        if frame.true_class.tolist() != [ALL_CLASSES[k] for k in ids]:
            raise ValueError('Class mapping mismatch')
        full = probabilities([json.loads(x) for x in frame.probabilities], n, len(frame))
        if 'predicted_class_id' in frame and not np.array_equal(frame.predicted_class_id, full.argmax(1)):
            raise ValueError('Stored prediction differs from full argmax')
        if 'predicted_class' in frame and frame.predicted_class.tolist() != [ALL_CLASSES[k] for k in full.argmax(1)]:
            raise ValueError('Stored prediction class mapping mismatch')
        subset = frame[frame.true_class_id.isin(labels)].sort_values('sample_id').reset_index(drop=True)
        if set(subset.true_class_id) != set(labels):
            raise ValueError('Missing required truth classes')
        parts.append(subset)
    a, b = parts
    identity = ['sample_id', 'video', 'video_sha256', 'image_sha256', 'true_class_id', 'true_class']
    if not a[identity].equals(b[identity]):
        raise ValueError('Missing predictions or mismatched paired video/frame/GT')
    return (a.true_class_id.to_numpy(),
            probabilities([json.loads(x) for x in a.probabilities], 6, len(a)),
            probabilities([json.loads(x) for x in b.probabilities], 8, len(b)))


def metrics(truth, predicted, labels):
    y, p = integer_array(truth, 'truth'), integer_array(predicted, 'prediction')
    labels = tuple(labels)
    if (y.ndim != 1 or y.shape != p.shape or labels not in [tuple(range(6)), tuple(range(8))]
            or set(y) != set(labels) or (p < 0).any() or (p >= 8).any()):
        raise ValueError('Invalid shape, mapping or missing required truth classes')
    matrix = np.bincount(y * 8 + p, minlength=64).reshape(8, 8)
    rows = []
    for k in labels:
        tp = int(matrix[k, k]); support = int(matrix[k].sum()); predicted_n = int(matrix[:, k].sum())
        precision = tp / predicted_n if predicted_n else 0.
        recall = tp / support
        f1 = 2 * tp / (support + predicted_n)
        rows.append(dict(class_id=k, class_name=ALL_CLASSES[k], n=support,
                         precision=precision, recall=recall, f1=f1))
    return dict(n=len(y), accuracy=float(np.mean(y == p)), macro_f1=float(np.mean([r['f1'] for r in rows])),
                classes=rows, confusion_matrix=matrix[:len(labels)].tolist())


def interference(truth, scores_a, scores_b):
    y = integer_array(truth, 'truth')
    a = probabilities(scores_a, 6, len(y)).argmax(1)
    full_b = probabilities(scores_b, 8, len(y))
    b, restricted = full_b.argmax(1), full_b[:, :6].argmax(1)
    ma, mb = metrics(y, a, range(6)), metrics(y, b, range(6))
    rows = [dict(class_id=k, class_name=BASE[k], n=ma['classes'][k]['n'],
                 recall_A=ma['classes'][k]['recall'], recall_B=mb['classes'][k]['recall'],
                 delta_recall=mb['classes'][k]['recall'] - ma['classes'][k]['recall']) for k in range(6)]
    intrusion = []
    for k in [None, *range(6)]:
        p = b if k is None else b[y == k]
        intrusion.append(dict(class_name='overall' if k is None else BASE[k], n=len(p),
                              overall=float(np.mean(p >= 6)), halo=float(np.mean(p == 6)), tidur=float(np.mean(p == 7))))
    aa, ab, ar = ma['accuracy'], mb['accuracy'], float(np.mean(restricted == y))
    decomposition = dict(accuracy_A=aa, accuracy_B=ab, accuracy_B_restricted=ar,
                         gap_A_minus_B=aa-ab, restricted_gap=aa-ar, intrusion_gap=ar-ab)
    if not np.isclose(aa-ab, (aa-ar)+(ar-ab), atol=1e-12):
        raise ArithmeticError('Accuracy decomposition identity failed')
    return dict(A=ma, B=mb, classes=rows, intrusion=intrusion, decomposition=decomposition)
