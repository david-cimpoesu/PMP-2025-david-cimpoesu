#a
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def run_scenario(y_obs, theta, mu_prior=10.0,
                 draws=2000, tune=2000, seed=2025):
    with pm.Model() as model:
        n = pm.Poisson("n", mu=mu_prior)
        pm.Binomial("y", n=n, p=theta, observed=y_obs)
        pm.Binomial("y_future", n=n, p=theta)

        step = pm.Metropolis(vars=[n])

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=2,
            cores=1,
            random_seed=seed,
            step=step,
            return_inferencedata=True,
            progressbar=False,
        )

        idata = pm.sample_posterior_predictive(
            idata,
            model=model,
            var_names=["y_future"],
            extend_inferencedata=True,
        )

    return model, idata

def main():
    y_values = [0, 5, 10]
    theta_values = [0.2, 0.5]
    results = {}

    for y_obs in y_values:
        for theta in theta_values:
            _, idata = run_scenario(y_obs, theta)
            results[(y_obs, theta)] = idata
            print(az.summary(idata, var_names=["n"], hdi_prob=0.94))

    fig_n, axes_n = plt.subplots(
        len(y_values), len(theta_values),
        figsize=(10, 8), sharex=True, sharey=True,
        constrained_layout=True
    )

    for i, y_obs in enumerate(y_values):
        for j, theta in enumerate(theta_values):
            idata = results[(y_obs, theta)]
            az.plot_posterior(idata, var_names=["n"], ax=axes_n[i, j], hdi_prob=0.94)
            axes_n[i, j].set_title(f"n | Y={y_obs}, θ={theta}")

#c
    fig_p, axes_p = plt.subplots(
        len(y_values), len(theta_values),
        figsize=(10, 8), sharex=True, sharey=True,
        constrained_layout=True
    )

    for i, y_obs in enumerate(y_values):
        for j, theta in enumerate(theta_values):
            idata = results[(y_obs, theta)]
            samples = idata.posterior_predictive["y_future"].values.ravel()
            az.plot_dist(samples, ax=axes_p[i, j])
            axes_p[i, j].set_title(f"Y* | Y={y_obs}, θ={theta}")

    plt.show()


if __name__ == "__main__":
    main()
