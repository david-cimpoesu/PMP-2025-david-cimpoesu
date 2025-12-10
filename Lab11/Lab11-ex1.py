import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

if __name__ == "__main__":
    data = pd.read_csv("Prices.csv")
    y = data["Price"].values
    x1 = data["Speed"].values
    x2 = np.log(data["HardDrive"].values)

    #a
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=200)
        beta1 = pm.Normal("beta1", mu=0, sigma=50)
        beta2 = pm.Normal("beta2", mu=0, sigma=50)
        sigma = pm.HalfNormal("sigma", sigma=50)

        mu = alpha + beta1 * x1 + beta2 * x2
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(2000, tune=2000, target_accept=0.9,
                          chains=4, cores=4, random_seed=42)

    #b
    print(az.hdi(idata.posterior["beta1"], hdi_prob=0.95))
    print(az.hdi(idata.posterior["beta2"], hdi_prob=0.95))

    #c
    print(az.summary(idata, var_names=["beta1","beta2"]))
    # Verificăm dacă predictorii sunt utili analizând HDI-urile pentru beta1 și beta2.
    # Dacă intervalele nu includ 0, predictorii au efect semnificativ asupra prețului.
    # În acest caz, ambele HDI-uri sunt strict pozitive, deci predictorii sunt utili.

    #d
    x1_new = 33
    x2_new = np.log(540)
    mu_samples = (
        idata.posterior["alpha"] +
        idata.posterior["beta1"] * x1_new +
        idata.posterior["beta2"] * x2_new
    ).values.flatten()
    print(az.hdi(mu_samples, hdi_prob=0.90))

    #e
    sigma_samples = idata.posterior["sigma"].values.flatten()
    y_pred_samples = mu_samples + np.random.normal(
        0, sigma_samples, size=mu_samples.size
    )
    print(az.hdi(y_pred_samples, hdi_prob=0.90))

    # bonus
    premium = (data["Premium"].str.lower() == "yes").astype(int).values

    with pm.Model() as model_bonus:
        alpha = pm.Normal("alpha", mu=0, sigma=200)
        beta1 = pm.Normal("beta1", mu=0, sigma=50)
        beta2 = pm.Normal("beta2", mu=0, sigma=50)
        beta_p = pm.Normal("beta_p", mu=0, sigma=50)
        sigma = pm.HalfNormal("sigma", sigma=50)

        mu = alpha + beta1 * x1 + beta2 * x2 + beta_p * premium
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        idata_bonus = pm.sample(2000, tune=2000, target_accept=0.9,
                                chains=4, cores=4, random_seed=42)

    print(az.summary(idata_bonus, var_names=["beta_p"]))
