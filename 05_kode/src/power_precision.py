"""Conditional planning simulation, not measured external performance or model training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import beta, binom, norm

from analysis_paired import RNG_SEED, f1_from_matrix, load_pairs, markdown_table


def joint_model(q, accuracy_effect, accuracy_a=.93):
    """P(A prediction,B prediction | GT), retaining B's two extra outputs."""
    a_only=(q-accuracy_effect)/2
    b_only=(q+accuracy_effect)/2
    masses=[accuracy_a-a_only,a_only,b_only,1-accuracy_a-b_only]
    if min(masses)<-1e-12:
        raise ValueError('Infeasible accuracy/discordance')
    joint=np.zeros((6,6,8))
    for t in range(6):
        wrong_a=np.ones(6)/5;wrong_a[t]=0
        wrong_b=np.r_[np.ones(6)*.8/5,[.1,.1]];wrong_b[t]=0
        joint[t,t,t]+=masses[0]
        joint[t,t,:]+=masses[1]*wrong_b
        joint[t,:,t]+=masses[2]*wrong_a
        # Both-wrong: half same old error, half independent, including extra B classes.
        joint[t,:,:]+=masses[3]*.5*np.outer(wrong_a,wrong_b)
        for a in range(6):joint[t,a,a]+=masses[3]*.5*wrong_a[a]
    if not np.allclose(joint.sum(axis=(1,2)),1) or (joint<0).any():
        raise ValueError('Invalid joint distribution')
    return joint


def expected_delta(joint):
    a=np.pad(joint.sum(axis=2),((0,0),(0,2)))/6
    b=joint.sum(axis=1)/6
    return float(f1_from_matrix(b)-f1_from_matrix(a))


def calibrated_joint(q, f1_effect):
    lo,hi=max(-q, q-2*.93),min(q,2*(1-.93)-q)
    objective=lambda d:expected_delta(joint_model(q,d))-f1_effect
    if lo>hi or objective(lo)*objective(hi)>0:
        raise ValueError('Requested macro-F1 effect infeasible for scenario')
    return joint_model(q,brentq(objective,lo,hi,xtol=1e-12))


def f1_gradient(matrix):
    gradient=np.zeros((6,8))
    for k in range(6):
        den=matrix[k].sum()+matrix[:,k].sum()
        if den<=0:raise ValueError('Absent class in planning population')
        gradient[k,k]+=2/den/6
        gradient[k,:]-=2*matrix[k,k]/den**2/6
        gradient[:,k]-=2*matrix[k,k]/den**2/6
    return gradient


def video_variance(joint,n_per_class=50):
    a=np.pad(joint.sum(axis=2),((0,0),(0,2)))/6;b=joint.sum(axis=1)/6
    ga,gb=f1_gradient(a),f1_gradient(b)
    variance=0.
    for t in range(6):
        influence=gb[t][None,:]-ga[t,:6][:,None]
        mean=(joint[t]*influence).sum()
        variance+=float((joint[t]*(influence-mean)**2).sum())/(36*n_per_class)
    return variance


def simulate_h1(joint, seed_sd=0., repetitions=4000, seed=RNG_SEED, n_per_class=50, delta=.02):
    if seed_sd<0 or repetitions<2:raise ValueError('Invalid simulation settings')
    rng=np.random.default_rng(seed)
    ma=np.zeros((repetitions,6,8));mb=np.zeros_like(ma)
    for t in range(6):
        draws=rng.multinomial(n_per_class,joint[t].ravel(),size=repetitions).reshape(-1,6,8)
        ma[:,t,:6]=draws.sum(axis=2);mb[:,t,:]=draws.sum(axis=1)
    # Same video sample across five seeds. Additive paired seed mean is hypothetical, not observed.
    observed=f1_from_matrix(mb)-f1_from_matrix(ma)+rng.normal(0,seed_sd/np.sqrt(5),repetitions)
    se=np.sqrt(video_variance(joint,n_per_class)+seed_sd**2/5)
    passed=observed-norm.ppf(.95)*se > -delta
    power=float(passed.mean())
    return dict(true_macro_f1_delta=expected_delta(joint),paired_seed_sd_assumed=seed_sd,
        n_common6=n_per_class*6,ci95_width_pp=float(2*norm.ppf(.975)*se*100),
        lower95_halfwidth_pp=float(norm.ppf(.95)*se*100),ni_pass_probability=power,
        monte_carlo_se=float(np.sqrt(power*(1-power)/repetitions)),
        method='SIMULATION: paired multinomial videos + Gaussian seed mean; oracle delta-method CI, NOT exact protocol bootstrap power')


def h2_binomial(true_accuracy,n=50,threshold=.80):
    if not 0<=true_accuracy<=1 or not 0<threshold<1 or n<1:raise ValueError('Invalid binomial scenario')
    critical=next((k for k in range(1,n+1) if beta.ppf(.05,k,n-k+1)>threshold),n+1)
    p=float(binom.sf(critical-1,n,true_accuracy))
    return dict(true_accuracy_assumed=true_accuracy,n_per_class=n,required_correct=critical,
        probability_pass_one_class=p,probability_pass_both_if_independent=p*p,
        both_probability_frechet_low=max(0,2*p-1),both_probability_frechet_high=p,
        method='EXACT binomial + one-sided Clopper-Pearson; conditional single classifier/perfectly correlated seed errors; not pooled n=250')


def run(work,out,repetitions=4000):
    work,out=Path(work).resolve(),Path(out).resolve()
    pairs=load_pairs(work)
    if out==work or out.is_relative_to(work) or work.is_relative_to(out):raise ValueError('Separate output required')
    out.mkdir(parents=True,exist_ok=False)
    empirical=np.zeros((6,6,8))
    for r in pairs.itertuples():empirical[r.class_id,r.predicted_class_id_A,r.predicted_class_id_B]+=1
    empirical/=empirical.sum(axis=(1,2),keepdims=True)
    rows=[]
    for sd in [0.,.005,.01,.02]:
        rows.append(dict(scenario='EMPIRICAL_TRANSPORT_ASSUMPTION',correctness_discordance_assumed=float(pairs.correct_A.ne(pairs.correct_B).mean()),
                         **simulate_h1(empirical,sd,repetitions)))
    for q in [.02,.05,.10]:
        for effect in [-.02,-.01,0.,.01]:
            for sd in [0.,.005,.01,.02]:
                try:
                    result=simulate_h1(calibrated_joint(q,effect),sd,repetitions)
                    rows.append(dict(scenario='HYPOTHETICAL',correctness_discordance_assumed=q,**result))
                except ValueError:
                    rows.append(dict(scenario='INFEASIBLE_GENERATOR_COMBINATION',correctness_discordance_assumed=q,
                        true_macro_f1_delta=effect,paired_seed_sd_assumed=sd))
    h1=pd.DataFrame(rows);h2=pd.DataFrame([h2_binomial(p) for p in [.80,.85,.90,.95,.98]])
    h1.to_csv(out/'h1_power_scenarios.csv',index=False);h2.to_csv(out/'h2_power_scenarios.csv',index=False)
    facts=dict(training_seeds_observed=[42],paired_seed_sd_observed=None,n_pilot=len(pairs),
        correctness_discordance_count=int(pairs.correct_A.ne(pairs.correct_B).sum()),
        predicted_label_disagreement_count=int(pairs.predicted_class_id_A.ne(pairs.predicted_class_id_B).sum()),
        repetitions=repetitions,statistics_rng=RNG_SEED,external_images_read=0,model_fits=0,
        source_predictions_sha256=hashlib.sha256((work/'evaluation/predictions.csv').read_bytes()).hexdigest())
    (out/'simulation_provenance.json').write_text(json.dumps(facts,indent=2)+'\n',encoding='utf8')
    selected=h1[(h1.scenario=='HYPOTHETICAL') & h1.true_macro_f1_delta.abs().lt(1e-8) & h1.paired_seed_sd_assumed.isin([0.,.01,.02])]
    report='''# Precision/power planning — FACT / ASSUMPTION / PROPOSAL

## FACT

Only training seed42 exists; its reproduction is not a new seed. Actual paired counts are in paired analysis.
No seed variance has been measured. No external images/scores were read; zero model-training fits.
Correctness discordance is not the same as changed predicted label or macro-F1 influence.

## SIMULATION ASSUMPTIONS

External common6 planned n=300, 50 independent videos/GT class, five paired seeds. Hypothetical A accuracy .93,
correctness discordance .02/.05/.10, calibrated TRUE macro-F1 effects -.02/-.01/0/+.01, paired seed SD
0/.005/.01/.02. B predictions retain two extra outputs; macro-F1 is computed from full confusion, not accuracy.
Wrong-label choices: 20% extra B output for A-only errors, 10% for both-wrong. Both-wrong labels half shared,
half independently drawn. This is a disclosed generator, not estimated external behavior. Impossible combinations
are flagged in CSV, not dropped or assigned invented power.

Same video sample across seeds; independent Gaussian paired-seed mean SD/sqrt(5) adds unobserved seed variability.
95% CI width uses oracle delta-method video variance plus seed variance. NI uses one-sided normal critical 1.645.
This is APPROXIMATE planning power, not calibration of the final hierarchical percentile-bootstrap test.
It assumes independent videos; unknown signer/session clustering and domain shift can reduce precision.
Conditional empirical transport is optimistic: original error patterns only, no unseen discordances.
Five seeds are not 1,500 independent external videos. Report Monte Carlo SE in CSV.

### H1 at true macro-F1 delta zero

'''+markdown_table(selected[['correctness_discordance_assumed','paired_seed_sd_assumed','ci95_width_pp','ni_pass_probability','monte_carlo_se']])+'''

### H2, 50 videos per new class, threshold unchanged at .80

'''+markdown_table(h2[['true_accuracy_assumed','required_correct','probability_pass_one_class','probability_pass_both_if_independent']])+'''

H2 above uses one-sided 95% Clopper-Pearson, a conservative transparent binomial reference for a fixed classifier,
or perfectly correlated errors across seeds. It is NOT exact power for the preregistered five-seed hierarchical CI.
No observed between-seed new-class variance exists. Both-pass product assumes independence between classes;
Frechet bounds in CSV allow unknown dependence. Do not pool seeds into n=250/class or replace primary CI silently.

## PROPOSAL — decision required, no protocol changes enacted

300 common6 videos may support NI when discordance and seed variation are small, but cannot guarantee a narrow CI.
Use the scenario tables, not the optimistic one-discordance pilot alone. Keep external confirmation with a prespecified
inconclusive outcome if desired; larger independently collected external data is an alternative requiring a new protocol.
Internal CV offers more development evidence, but reused/tuned internal folds do NOT become independent confirmation.
If CV is designated the main H1 evidence, explicitly limit the claim to development and report external corroboration
separately. Selection and testing on the same CV require nested separation for stronger inferential interpretation.
Do not widen delta or lower H2 merely to pass: any scientifically justified change needs researcher decision and
Addendum 02 before prospective training. Delta=.02 and H2=.80 remain unchanged. This analysis cannot decide for
the researcher. No claim that current single-seed pilot has confirmed H1 or H2.
'''
    (out/'power_precision_report.md').write_text(report,encoding='utf8')
    return h1,h2


def development_scenarios(repetitions=10000):
    """Exploratory sensitivity only: margins are scenarios, never protocol changes."""
    rows=[]
    for domain,n_per_class in [('external_common6',50),('internal_CV_one_fold',60),('internal_CV_OOF_optimistic',300)]:
        for q in [.02,.05,.10]:
            joint=calibrated_joint(q,0.)
            for sd in [0.,.01,.02]:
                for margin in [.02,.03,.05]:
                    result=simulate_h1(joint,sd,repetitions,n_per_class=n_per_class,delta=margin)
                    rows.append(dict(status='exploratory/development',domain=domain,ni_margin=margin,
                        correctness_discordance_assumed=q,**result))
    return pd.DataFrame(rows)


def development_report(out,repetitions=10000):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    scenarios=development_scenarios(repetitions)
    h2=pd.DataFrame([h2_binomial(p) for p in [.80,.85,.90,.95,.98]])
    scenarios.to_csv(out/'exploratory_power_precision.csv',index=False)
    h2.to_csv(out/'exploratory_h2_binomial.csv',index=False)
    selected=scenarios[(scenarios.correctness_discordance_assumed==.05)&(scenarios.paired_seed_sd_assumed==.01)]
    text='''# exploratory/development — precision and power, not observed external results

FACT: Only seed42 has observed pilot results. Between-seed variance is unmeasured.
No external image or new training fit is used. NI margins .02/.03/.05 are planning scenarios only;
the candidate margin and H2 threshold remain unchanged. RNG20261103; 10,000 draws per scenario by default.

ASSUMPTIONS: true macro-F1 difference zero; A accuracy .93; correctness discordance .02/.05/.10;
paired seed SD0/.01/.02, five seed pairs sharing the same video sample. Joint wrong-label generator
is `calibrated_joint`: B retains extra output columns, with extra-class mass on discordant/both-wrong errors.
Oracle delta-method paired F1 variance plus independent Gaussian seed mean approximates intervals.
This is NOT exact hierarchical-bootstrap power, nor an estimate of unknown external domain shift.
Video independence assumed; signer/session correlation unmeasured. Five seeds are never five times the video n.

CV sizes: 5-fold over 300 internal videos/class implies 60 held-out/class, n360 common6 per fold;
combined OOF n1800. Those data include previously used train/val videos. OOF intervals below are optimistic
planning references: overlapping fitted training sets, configuration selection and signer dependence are
not modelled. They cannot establish independent confirmation or replace nested evaluation.

Illustration at 5% correctness discordance and hypothetical paired seed SD1 percentage point:

'''+markdown_table(selected[['domain','n_common6','ni_margin','ci95_width_pp','ni_pass_probability','monte_carlo_se']])+'''

H2 n50/class, unchanged threshold .80. One-sided exact Clopper–Pearson lower bound must exceed .80.
Reference is a single classifier/perfectly correlated seed errors, not final five-seed hierarchical power.
Both-class product assumes independent class errors; Frechet bounds are in CSV.

'''+markdown_table(h2)+'''

OPTIONS / CONSEQUENCES:

| Option | Consequence |
| --- | --- |
| Keep external n300 and scientifically fixed margin | Accept inconclusive NI outcomes when CI crosses boundary |
| Collect more independent external videos under a future approved protocol | More precision; cost and new collection before freeze |
| Use internal CV for development, retain external corroboration separately | More development data; selection bias and dependence remain |

No margin, H2 threshold or evidence domain is chosen here. Researcher decision and addendum locking these
items are mandatory BEFORE Stage2 CV tuning or Stage3; Stage1b still needs explicit scoped approval.
'''
    (out/'exploratory_power_precision.md').write_text(text,encoding='utf8')
    return scenarios,h2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,default=4000)
    args=parser.parse_args();run(args.work,args.out,args.repetitions)
