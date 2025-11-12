import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Date
    data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])
    x_bar = data.mean()
    s = data.std(ddof=1)
    print(f"Sample mean = {x_bar:.2f}, Sample std = {s:.2f}, n = {len(data)}")

    #a
    with pm.Model() as weak_model:
        mu = pm.Normal("mu", mu=x_bar, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

        trace_weak = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42, chains=4, cores=1)
        summary_weak = az.summary(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)
    print("\nPosterior summaries (Weak Prior):")
    print(summary_weak)

    #b
    mu_samples_weak = az.extract(trace_weak, var_names="mu", combined=True).to_numpy()
    sigma_samples_weak = az.extract(trace_weak, var_names="sigma", combined=True).to_numpy()
    hdi_mu_weak = az.hdi(mu_samples_weak, hdi_prob=0.95)
    hdi_sigma_weak = az.hdi(sigma_samples_weak, hdi_prob=0.95)
    print(f"95% HDI mu (weak): [{hdi_mu_weak[0]:.3f}, {hdi_mu_weak[1]:.3f}]")
    print(f"95% HDI sigma (weak): [{hdi_sigma_weak[0]:.3f}, {hdi_sigma_weak[1]:.3f}]")

    #c
    print("\nFrequentist estimates:")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"SD:   {np.std(data, ddof=1):.2f}")

    #d
    with pm.Model() as strong_model:
        mu = pm.Normal("mu", mu=50, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=10)
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)
        trace_strong = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42, chains=4, cores=1)
        summary_strong = az.summary(trace_strong, var_names=["mu", "sigma"], hdi_prob=0.95)
    print("\nPosterior summaries (Strong Prior):")
    print(summary_strong)

    mu_samples_strong = az.extract(trace_strong, var_names="mu", combined=True).to_numpy()
    sigma_samples_strong = az.extract(trace_strong, var_names="sigma", combined=True).to_numpy()
    hdi_mu_strong = az.hdi(mu_samples_strong, hdi_prob=0.95)
    hdi_sigma_strong = az.hdi(sigma_samples_strong, hdi_prob=0.95)
    print(f"95% HDI mu (strong): [{hdi_mu_strong[0]:.3f}, {hdi_mu_strong[1]:.3f}]")
    print(f"95% HDI sigma (strong): [{hdi_sigma_strong[0]:.3f}, {hdi_sigma_strong[1]:.3f}]")

    az.plot_posterior(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)
    plt.suptitle("Posterior with Weak Prior", fontsize=14)
    plt.show()

    az.plot_posterior(trace_strong, var_names=["mu", "sigma"], hdi_prob=0.95)
    plt.suptitle("Posterior with Strong Prior", fontsize=14)
    plt.show()
