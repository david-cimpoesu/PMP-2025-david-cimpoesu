import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

if __name__ == "__main__":

    data = pd.read_csv("date_promovare_examen.csv")

    y = data["Promovare"].values
    x1 = data["Ore_Studiu"].values
    x2 = data["Ore_Somn"].values

    #a
    print(y.mean())

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta1 = pm.Normal("beta1", mu=0, sigma=5)
        beta2 = pm.Normal("beta2", mu=0, sigma=5)

        eta = alpha + beta1 * x1 + beta2 * x2
        p = pm.math.sigmoid(eta)

        pm.Bernoulli("y_obs", p=p, observed=y)

        idata = pm.sample(
            2000,
            tune=2000,
            target_accept=0.99,
            chains=4,
            cores=4,
            random_seed=42
        )

    #b
    alpha_mean = idata.posterior["alpha"].mean().item()
    beta1_mean = idata.posterior["beta1"].mean().item()
    beta2_mean = idata.posterior["beta2"].mean().item()

    print(alpha_mean, beta1_mean, beta2_mean)

    '''
    Granița de decizie medie este determinată de coeficienții modelului logistic estimați Bayesian.
    Intervalele HDI de 95% pentru coeficienți nu includ valoarea zero, ceea ce indică o separare bună între cele două clase.
    Astfel, modelul reușește să distingă eficient între elevii care promovează și cei care nu promovează.
    '''

    #c
    print(az.hdi(idata.posterior[["beta1", "beta2"]], hdi_prob=0.95))
    print(az.summary(idata, var_names=["beta1", "beta2"]))

    '''
    Coeficientul pentru orele de somn este mai mare decât cel pentru orele de studiu.
    De asemenea, intervalul HDI este mai îndepărtat de zero.
    Acest lucru arată că orele de somn au o influență mai mare asupra promovării.
    '''