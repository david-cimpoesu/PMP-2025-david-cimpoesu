import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

df = pd.read_csv("date_colesterol.csv")
t = df["Ore_Exercitii"].to_numpy()
y_obs = df["Colesterol"].to_numpy()
t2 = t**2

N = len(y_obs)

Ks = [3, 4, 5]
models = {}
idatas = {}

#1
for K in Ks:
    with pm.Model() as model:
        w = pm.Dirichlet("w", a=np.ones(K) * 4.0)

        alpha = pm.Normal(
            "alpha",
            mu=np.linspace(y_obs.min(), y_obs.max(), K),
            sigma=50,
            shape=K,
            transform=pm.distributions.transforms.ordered,
        )
        beta = pm.Normal("beta", mu=0, sigma=20, shape=K)
        gamma = pm.Normal("gamma", mu=0, sigma=10, shape=K)

        sigma = pm.HalfNormal("sigma", sigma=30, shape=K)

        mu = alpha + beta * t[:, None] + gamma * t2[:, None]

        y = pm.NormalMixture(
            "y",
            w=w,
            mu=mu,
            sigma=sigma,
            observed=y_obs,
        )

        idata = pm.sample(
            draws=1000,
            tune=2000,
            target_accept=0.9,
            chains=4,
            cores=1,
            random_seed=123,
            return_inferencedata=True,
        )

        idata = pm.compute_log_likelihood(idata, model=model)

    models[str(K)] = model
    idatas[str(K)] = idata

#1.1
for K in Ks:
    k = str(K)
    print("\nK =", K)
    print(az.summary(idatas[k], var_names=["w", "alpha", "beta", "gamma", "sigma"]))

#2
comp_waic = az.compare(idatas, ic="waic", method="BB-pseudo-BMA", scale="deviance")
comp_loo = az.compare(idatas, ic="loo", method="BB-pseudo-BMA", scale="deviance")

print("\nWAIC comparison:")
print(comp_waic)

print("\nLOO comparison:")
print(comp_loo)

best_waic = comp_waic.index[0]
best_loo = comp_loo.index[0]


print("\nBest by WAIC:", best_waic)
print("Best by LOO:", best_loo)
