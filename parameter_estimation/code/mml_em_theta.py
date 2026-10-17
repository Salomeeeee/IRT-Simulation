import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.stats import norm
import os
from tqdm import tqdm

"""
Recompute model ability estimates for selected MML-EM simulation conditions.

For each benchmark/IRT/sample-size directory listed in ``rerun.txt`` (e.g.,
``arc/2pl/N30``), this script matches every fitted item-parameter file with
its corresponding simulated response matrix. It removes items that were not
successfully fitted by mirt, then estimates each model's ability (theta) and
posterior standard deviation using expected a posteriori (EAP) estimation on
a dense theta grid with a standard normal prior. Missing item responses are
ignored.

We use this numerically stable, log-space EAP implementation to recover theta
estimates that were missing because the original estimation procedure failed
due to numerical overflow or underflow. As a validation, for replications in
which the original theta estimation succeeded, the recomputed theta estimates
were highly correlated (> 0.99) with the original estimates.

The resulting estimates are saved as ``theta_new_repXX.csv`` in the
corresponding condition directory. This script recomputes theta estimates
only; it does not refit the item parameters.
"""

def eap_theta(response_file, itempars_file, sample_size=None, theta_grid=None):
    resp = pd.read_csv(response_file, nrows=sample_size)
    itempars = pd.read_csv(itempars_file)

    model_ids = resp.iloc[:, 0].astype(str)

    # Only use items actually fitted by mirt.
    # Omitted items were merged back with NA parameters.
    itempars = itempars.dropna(subset=["b"]).copy()

    item_ids = itempars["itemid"].astype(str).tolist()
    Y = resp[item_ids].to_numpy(dtype=float)

    a = itempars["a"].to_numpy(dtype=float)
    b = itempars["b"].to_numpy(dtype=float)
    c = itempars["c"].to_numpy(dtype=float)
    d = itempars["d"].to_numpy(dtype=float)

    if theta_grid is None:
        theta_grid = np.linspace(-6, 6, 601)

    # P(Y=1 | theta)
    # shape = (n_theta, n_items)
    eta = a[None, :] * (theta_grid[:, None] - b[None, :])
    P = c[None, :] + (d[None, :] - c[None, :]) * expit(eta)

    eps = 1e-12
    P = np.clip(P, eps, 1 - eps)

    log_p = np.log(P)
    log_1mp = np.log1p(-P)

    # standard normal prior
    log_prior = norm.logpdf(theta_grid)

    theta_est = []
    theta_se = []

    for y in Y:
        observed = ~np.isnan(y)

        if observed.sum() == 0:
            theta_est.append(np.nan)
            theta_se.append(np.nan)
            continue

        yy = y[observed]

        log_lik = (
            yy[None, :] * log_p[:, observed]
            + (1 - yy[None, :]) * log_1mp[:, observed]
        ).sum(axis=1)

        log_post = log_lik + log_prior

        # normalize in log space
        log_Z = logsumexp(log_post)
        weights = np.exp(log_post - log_Z)

        # EAP
        mu = np.sum(weights * theta_grid)

        # posterior SD
        var = np.sum(weights * (theta_grid - mu) ** 2)

        theta_est.append(mu)
        theta_se.append(np.sqrt(var))

    return pd.DataFrame({
        "model_id": model_ids,
        "theta": theta_est,
        "theta_se": theta_se,
    })

res_dir = '/home/jhu/hjiang66/scratch_zxiao25/jh_data/irt_simulation'
out_dir = '/home/jhu/hjiang66/scratch_zxiao25/jh_data/irt_simulation/em_outputs'
'''
for benchmark in os.listdir(out_dir):
    for irt in ['1pl', '2pl', '3pl']:
        for N in [30, 100, 180, 400, 1000]:
            cond_dir = os.path.join(out_dir, f'{benchmark}/{irt}/N{N}')
            if not os.path.exists(cond_dir):
                continue
            theta_files = [file for file in os.listdir(cond_dir) if file.startswith('theta_est')]
            for tfile in theta_files:
                theta = pd.read_csv(os.path.join(cond_dir, tfile))
                if theta.isna().any().any():
                    print(f'{benchmark}/{irt}/N{N} needs re-run.')
                    break
'''
todos = [t.strip() for t in open('rerun.txt', 'r').readlines()]
for todo in todos:
    benchmark, irt, N = todo.replace('N', '').split('/')
    # print(benchmark, irt, N)
    cond_dir = os.path.join(out_dir, todo)
    item_files = [file for file in os.listdir(cond_dir) if file.startswith('itempars_est')]
    for ifile in tqdm(item_files, desc=todo): 
        idx = int(ifile.split('rep')[-1].split('.')[0])
        new_theta = eap_theta(
            response_file=os.path.join(res_dir, f'{benchmark}/{irt}/resp_sim_rep{idx}.csv'),
            itempars_file=os.path.join(cond_dir, ifile),
            sample_size=int(N)
        )
        new_theta.to_csv(os.path.join(cond_dir, f"theta_new_rep{idx:02d}.csv"), index=False)
