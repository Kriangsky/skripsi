"""Paired GT-stratified video bootstrap; seeds are crossed with the SAME videos."""
import numpy as np
from scipy.stats import norm

from interference_metrics import integer_array, metrics

BOOTSTRAP_SEED = 20261103


def interval(draws):
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 1 or len(draws) < 2 or not np.isfinite(draws).all():
        raise ValueError('Finite bootstrap draws required')
    return dict(ci95_low=float(np.quantile(draws, .025)), ci95_high=float(np.quantile(draws, .975)),
                lower_one_sided95=float(np.quantile(draws, .05)),
                upper_one_sided95=float(np.quantile(draws, .95)), bootstrap_se=float(draws.std(ddof=1)))


def bootstrap(truth, pred_a, pred_b, repetitions=10000, seed=BOOTSTRAP_SEED, labels=tuple(range(6))):
    y = integer_array(truth, 'truth')
    a, b = integer_array(pred_a, 'A'), integer_array(pred_b, 'B')
    labels = tuple(labels)
    if (not isinstance(repetitions, int) or repetitions < 2 or y.ndim != 1 or a.ndim != 2
            or a.shape != b.shape or a.shape[1] != len(y) or a.shape[0] < 1):
        raise ValueError('Matched seed-by-video prediction arrays required')
    if labels == tuple(range(6)) and (a >= 6).any():
        raise ValueError('Model A has six outputs')
    point = np.array([metrics(y, pb, labels)['macro_f1'] - metrics(y, pa, labels)['macro_f1']
                      for pa, pb in zip(a, b)])
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y == k) for k in labels]
    # One paired video draw per replicate, shared across ALL training seed pairs.
    indices = np.array([np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in strata])
                        for _ in range(repetitions)])
    replicate_offsets = np.arange(repetitions)[:, None] * 64
    f1, recalls = [], []
    for predictions in [a, b]:
        seed_f1, seed_recall = [], []
        for p in predictions:
            keys = replicate_offsets + y[indices]*8 + p[indices]
            cm = np.bincount(keys.ravel(), minlength=repetitions*64).reshape(repetitions, 8, 8)
            tp = np.diagonal(cm, axis1=1, axis2=2)[:, :len(labels)]
            den = cm.sum(2)[:, :len(labels)] + cm.sum(1)[:, :len(labels)]
            seed_f1.append(np.divide(2*tp, den, out=np.zeros_like(tp, dtype=float), where=den>0).mean(1))
            seed_recall.append(tp / cm.sum(2)[:, :len(labels)])
        f1.append(np.asarray(seed_f1)); recalls.append(np.asarray(seed_recall))
    deltas = f1[1] - f1[0]
    count = len(a)
    seed_weights = rng.multinomial(count, np.full(count, 1/count), size=repetitions) / count
    hierarchical = np.sum(deltas.T * seed_weights, axis=1)
    recall_a, recall_b = [np.einsum('rsk,rs->rk', x.transpose(1, 0, 2), seed_weights) for x in recalls]
    return dict(estimate=float(point.mean()), per_seed=point, n_videos=len(y), n_seeds=count,
                seed_delta_sd=float(point.std(ddof=1)) if count > 1 else None,
                repetitions=repetitions, bootstrap_seed=seed, hierarchical=hierarchical,
                video_only=deltas.mean(0), seed_only=seed_weights @ point,
                recall_A=recall_a, recall_B=recall_b, recall_delta=recall_b-recall_a)


def simultaneous_recall_intervals(draws, alpha=.05):
    """Bonferroni percentile intervals: familywise nominal 95% over six recall changes.

    Bootstrap coverage is approximate; these intervals do not make a largest delta causal.
    """
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != 6 or not np.isfinite(draws).all() or not 0 < alpha < 1:
        raise ValueError('Six finite recall-change distributions required')
    return np.quantile(draws, [alpha/12, 1-alpha/12], axis=0).T.tolist()


def non_inferiority(lower, margin, justification):
    if margin is None:
        return 'NI_MARGIN_UNRESOLVED'
    if (not isinstance(margin,(int,float)) or not np.isfinite(margin) or not 0 < margin < 1
            or not isinstance(justification,str) or not justification.strip()
            or not np.isfinite(lower) or not -1 <= lower <= 1):
        raise ValueError('Valid prespecified substantive margin and finite bound required')
    return 'non-inferiority demonstrated' if lower > -margin else 'non-inferiority not demonstrated'


def design_effect_sensitivity(estimate, draws):
    """Assumptive normal-SE stress test, NOT signer-cluster inference."""
    se = interval(draws)['bootstrap_se']
    return [dict(design_effect=d, effective_n_multiplier=1/d, adjusted_se=se*np.sqrt(d),
                 ci95_low=float(estimate-norm.ppf(.975)*se*np.sqrt(d)),
                 ci95_high=float(estimate+norm.ppf(.975)*se*np.sqrt(d)),
                 lower_one_sided95=float(estimate-norm.ppf(.95)*se*np.sqrt(d))) for d in [1, 1.5, 2]]
