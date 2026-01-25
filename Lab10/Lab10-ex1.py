import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

publicity = np.array([1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
                      6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])

sales = np.array([5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0,
                  15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0])

if __name__ == "__main__":

    # a)
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)

        mu = alpha + beta * publicity

        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=sales)

        idata = pm.sample(2000, tune=2000, chains=4, cores=4, target_accept=0.9)

        ppc = pm.sample_posterior_predictive(idata)

    # b)
    print("HDI alpha 95%:", az.hdi(idata.posterior["alpha"], hdi_prob=0.95))
    print("HDI beta 95%:", az.hdi(idata.posterior["beta"], hdi_prob=0.95))

    # c)
    x_new = np.array([12.0])

    alpha_s = idata.posterior["alpha"].values.flatten()
    beta_s = idata.posterior["beta"].values.flatten()
    sigma_s = idata.posterior["sigma"].values.flatten()

    mu_new = alpha_s + beta_s * x_new[0]
    y_new = mu_new + np.random.normal(0, sigma_s)

    print("HDI 90% mu_new:", az.hdi(mu_new, hdi_prob=0.90))
    print("HDI 90% y_new:", az.hdi(y_new, hdi_prob=0.90))

    # plot
    x_grid = np.linspace(publicity.min(), publicity.max(), 200)
    alpha_m = alpha_s.mean()
    beta_m = beta_s.mean()
    y_line = alpha_m + beta_m * x_grid

    plt.scatter(publicity, sales)
    plt.plot(x_grid, y_line)
    plt.show()
