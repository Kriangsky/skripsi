"""Read-only paired video analysis; writes only to a new output directory."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

import classification_pilot as c

RNG_SEED = 20261103


def wilson(k, n, confidence=.95, one_sided=False):
    if not isinstance(n, (int, np.integer)) or n < 1 or not 0 <= k <= n or not 0 < confidence < 1:
        raise ValueError('Invalid binomial count/confidence')
    z = norm.ppf(confidence if one_sided else (1 + confidence) / 2)
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    radius = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return (0. if k==0 else float(max(0, center-radius)),
            1. if k==n else float(min(1, center+radius)))


def markdown_table(frame):
    rows=[list(frame.columns),['---']*len(frame.columns),*frame.itertuples(index=False,name=None)]
    return '\n'.join('| '+' | '.join(str(v).replace('|','\\|').replace('\n',' ') for v in row)+' |' for row in rows)


def mcnemar_exact(a_only, b_only):
    if any(not isinstance(x, (int, np.integer)) or x < 0 for x in [a_only, b_only]):
        raise ValueError('Discordant counts must be nonnegative integers')
    n = a_only + b_only
    return float(binomtest(a_only, n, .5).pvalue) if n else 1.


def load_pairs(work):
    work = Path(work)
    m = pd.read_csv(work/'manifest.csv', keep_default_na=False)
    p = pd.read_csv(work/'evaluation/predictions.csv', keep_default_na=False)
    if m.sample_id.duplicated().any() or m.video.duplicated().any() or m.video_sha256.duplicated().any():
        raise ValueError('Manifest is not one video per identity/content')
    expected = m[m.split.eq('val') & m.class_id.lt(6)].sort_values('sample_id')
    if expected.empty or not expected.source.eq('train').all():
        raise ValueError('Internal common6 validation required')
    parts = []
    for model, outputs in [('A', 6), ('B', 8)]:
        part = p[p.model.eq(model) & p.true_class_id.lt(6)].copy()
        if part.sample_id.duplicated().any() or set(part.sample_id) != set(expected.sample_id):
            raise ValueError('Missing/duplicate paired predictions')
        part = expected[['sample_id','video','video_sha256','class_id','class_name','image_sha256']].merge(
            part, on='sample_id', validate='one_to_one')
        if not part.class_id.eq(part.true_class_id).all() or not part.class_name.eq(part.true_class).all():
            raise ValueError('Ground truth mismatch')
        for row in part.itertuples():
            scores = np.asarray(json.loads(row.probabilities), dtype=float)
            if scores.shape != (outputs,) or not np.isfinite(scores).all() or (scores < 0).any() or not np.isclose(scores.sum(), 1, atol=1e-5):
                raise ValueError('Invalid full probability vector')
            if int(np.argmax(scores)) != row.predicted_class_id or row.predicted_class != c.CLASSES[row.predicted_class_id]:
                raise ValueError('Prediction argmax/mapping mismatch')
            image = work/'datasets'/model/'val'/c.folder_names(outputs)[row.class_id]/f'{row.sample_id}.png'
            if not image.is_file() or c.b.sha(image) != row.image_sha256:
                raise ValueError('Shared validation image missing/changed')
        part = part.sort_values('sample_id')
        parts.append(part)
    a, b = parts
    result = a[['sample_id','video','video_sha256','class_id','class_name']].copy()
    for model, part in [('A', a), ('B', b)]:
        for field in ['predicted_class_id','predicted_class','confidence','quality_flags']:
            result[f'{field}_{model}'] = part[field].to_numpy()
        result[f'correct_{model}'] = part.predicted_class_id.to_numpy() == result.class_id.to_numpy()
    return result.reset_index(drop=True)


def f1_from_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[-2:] != (6, 8):
        raise ValueError('Common6 matrix must retain eight prediction columns')
    tp = np.diagonal(matrix[..., :, :6], axis1=-2, axis2=-1)
    den = matrix.sum(axis=-1) + matrix[..., :, :6].sum(axis=-2)
    return np.divide(2*tp, den, out=np.zeros_like(tp), where=den>0).mean(axis=-1)


def paired_bootstrap(truth, pred_a, pred_b, repetitions=10000, seed=RNG_SEED):
    y, a, b = (np.asarray(x, dtype=int) for x in [truth, pred_a, pred_b])
    if repetitions < 2 or len(y) == 0 or y.shape != a.shape or y.shape != b.shape:
        raise ValueError('Matched nonempty video arrays required')
    if set(y) != set(range(6)) or (a<0).any() or (a>=6).any() or (b<0).any() or (b>=8).any():
        raise ValueError('Invalid fixed class mapping')
    rng = np.random.default_rng(seed)
    weights = np.zeros((repetitions, len(y)), dtype=np.int32)
    for cid in range(6):
        ids = np.flatnonzero(y==cid)
        weights[:,ids] = rng.multinomial(len(ids), np.full(len(ids),1/len(ids)), size=repetitions)
    matrices = []
    for predictions in [a,b]:
        design = np.eye(48)[y*8+predictions]
        matrices.append((weights @ design).reshape(-1,6,8))
    f1_delta = f1_from_matrix(matrices[1]) - f1_from_matrix(matrices[0])
    acc_delta = weights @ ((b==y).astype(float)-(a==y)) / len(y)
    return np.column_stack([f1_delta, acc_delta])


def analyze(work, out, repetitions=10000):
    work, out = Path(work).resolve(), Path(out).resolve()
    pairs = load_pairs(work)
    if out == work or out.is_relative_to(work) or work.is_relative_to(out):
        raise ValueError('Analysis output must be separate from source run')
    out.mkdir(parents=True, exist_ok=False)
    a, b = pairs.correct_A.to_numpy(), pairs.correct_B.to_numpy()
    counts = dict(both_correct=int((a&b).sum()), a_only=int((a&~b).sum()),
                  b_only=int((~a&b).sum()), both_wrong=int((~a&~b).sum()))
    draws = paired_bootstrap(pairs.class_id, pairs.predicted_class_id_A, pairs.predicted_class_id_B, repetitions)
    summaries=[]
    for j, metric in enumerate(['macro_f1','accuracy']):
        point=[]
        for model, n in [('A',6),('B',8)]:
            aggregate, _, _ = c.metrics(pairs.class_id.tolist(),pairs[f'predicted_class_id_{model}'].tolist(),n,6)
            point.append(aggregate[metric])
        summaries.append(dict(metric=metric,A=point[0],B=point[1],delta_B_minus_A=point[1]-point[0],
            ci95_low=float(np.quantile(draws[:,j],.025)),ci95_high=float(np.quantile(draws[:,j],.975)),
            lower_one_sided95=float(np.quantile(draws[:,j],.05)),bootstrap_sd=float(draws[:,j].std(ddof=1))))
    rows=[]
    for model in ['A','B']:
        for name, group in [('common6',pairs), *list(pairs.groupby('class_name'))]:
            k=int(group[f'correct_{model}'].sum());n=len(group);lo,hi=wilson(k,n)
            rows.append(dict(model=model,scope=name,n=n,correct=k,accuracy=k/n,wilson95_low=lo,wilson95_high=hi))
    errors=pairs[~(pairs.correct_A & pairs.correct_B)]
    errors.to_csv(out/'paired_errors.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'accuracy_wilson.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'paired_ci.csv',index=False)
    per_class=[]
    for model,n in [('A',6),('B',8)]:
        _, details, _=c.metrics(pairs.class_id.tolist(),pairs[f'predicted_class_id_{model}'].tolist(),n,6)
        per_class.extend(dict(model=model,**row) for row in details[:6])
    pd.DataFrame(per_class).to_csv(out/'per_class_metrics.csv',index=False)
    record=dict(kind='FACT: post-hoc exploratory analysis of existing internal pilot',n=len(pairs),
        counts=counts,mcnemar_exact_two_sided=mcnemar_exact(counts['a_only'],counts['b_only']),
        correctness_discordance=(counts['a_only']+counts['b_only'])/len(pairs),
        label_disagreement=float(pairs.predicted_class_id_A.ne(pairs.predicted_class_id_B).mean()),
        seed=RNG_SEED,repetitions=repetitions,training_seeds_observed=[42],seed_variance=None,
        source_sha256={f:c.b.sha(work/f) for f in ['manifest.csv','runs/trained_models.json','evaluation/predictions.csv']},
        analysis_source_sha256=c.b.sha(__file__),statistics=summaries)
    (out/'analysis.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    table=markdown_table(pd.DataFrame(summaries))
    (out/'paired_analysis.md').write_text('# FACT — existing pilot paired analysis\n\n'+table+
        f'\n\nPaired counts: {counts}. Exact two-sided McNemar p={record["mcnemar_exact_two_sided"]}. '
        f'Union errors={len(errors)}.\n\nWilson intervals apply to accuracy, not macro-F1. '
        'Paired bootstrap resamples videos within GT strata; extra B outputs retained. '
        'One training seed only; no measured seed variance. These post-hoc internal results are not confirmatory NI evidence. '
        'Bootstrap cannot invent unseen error categories; signer/session dependence may widen uncertainty.\n',encoding='utf8')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,default=10000)
    args=parser.parse_args()
    print(json.dumps(analyze(args.work,args.out,args.repetitions),indent=2))


if __name__=='__main__':
    main()
