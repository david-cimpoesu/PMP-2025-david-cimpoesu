import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

az.style.use("arviz-darkgrid")
def main():

    #1
    df = pd.read_csv("bike_daily.csv")

    print(df.head())
    print(df.describe())

    sns.pairplot(df[["rentals", "temp_c", "humidity", "wind_kph"]])
    plt.show()

    sns.scatterplot(x="temp_c", y="rentals", data=df)
    plt.show()

    #2
    #a
    df = pd.get_dummies(df, columns=["season"], drop_first=True)

    y = df["rentals"].values

    X_cols = ["temp_c", "humidity", "wind_kph", "is_holiday"] + \
             [c for c in df.columns if c.startswith("season_")]

    X = df[X_cols].astype(float).values

    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0)
    X_s = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std  = y.std()
    y_s = (y - y_mean) / y_std

    temp_idx = X_cols.index("temp_c")

    temp_s = X_s[:, temp_idx]
    temp2_s = temp_s**2

    #b
    with pm.Model() as model_lin:
        alpha = pm.Normal("alpha", 0, 1)
        beta = pm.Normal("beta", 0, 1, shape=X_s.shape[1])
        sigma = pm.HalfNormal("sigma", 1)

        mu = alpha + pm.math.dot(X_s, beta)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_s)
        idata_lin = pm.sample(2000, tune=2000, target_accept=0.9, return_inferencedata=True)
        pm.compute_log_likelihood(idata_lin)

    #c
    X_poly = np.column_stack([X_s, temp2_s])

    with pm.Model() as model_poly:

        alpha = pm.Normal("alpha", 0, 1)
        beta  = pm.Normal("beta", 0, 1, shape=X_poly.shape[1])
        sigma = pm.HalfNormal("sigma", 1)

        mu = alpha + pm.math.dot(X_poly, beta)

        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_s)

        idata_poly = pm.sample(2000, tune=2000, target_accept=0.9, return_inferencedata=True)

        pm.compute_log_likelihood(idata_poly)

    #3
    #am rulat sample

    print("MODEL LINIAR:")
    print(az.summary(idata_lin, var_names=["beta"]))

    print("MODEL POLINOMIAL:")
    print(az.summary(idata_poly, var_names=["beta"]))

    #Temperatura este factorul care influenteaza cel mai mult numarul de inchirieri, mai puternic decât celelalte variabile, in ambele modele.

    #4
    #a

    cmp_waic = az.compare({"lin": idata_lin, "poly": idata_poly}, ic="waic")
    cmp_loo  = az.compare({"lin": idata_lin, "poly": idata_poly}, ic="loo")

    print("WAIC:")
    print(cmp_waic)

    print("LOO:")
    print(cmp_loo)

    az.plot_compare(cmp_waic)
    plt.show()

    #Modelul cu termenul patratic la temperatura este mai bun decat cel liniar,
    #deoarece descrie datele mai bine si ar face predictii mai bune pe date noi, lucru confirmat de valorile WAIC/LOO.

    #b

    pm.sample_posterior_predictive(idata_poly, model=model_poly, extend_inferencedata=True)

    az.plot_ppc(idata_poly, num_pp_samples=100)
    plt.show()

    idx = np.argsort(temp_s)

    alpha_m = idata_poly.posterior["alpha"].mean(("chain","draw")).values
    beta_m  = idata_poly.posterior["beta"].mean(("chain","draw")).values

    mu_hat = alpha_m + np.dot(beta_m, X_poly.T)

    plt.scatter(temp_s, y_s)
    plt.plot(temp_s[idx], mu_hat[idx], color="red")
    plt.show()

    #5

    Q = np.percentile(y, 75)
    is_high = (y >= Q).astype(int)

    #6
    with pm.Model() as model_log:
        alpha = pm.Normal("alpha", 0, 1)
        beta = pm.Normal("beta", 0, 1, shape=X_s.shape[1])

        eta = alpha + pm.math.dot(X_s, beta)
        p = pm.math.sigmoid(eta)

        pm.Bernoulli("y_obs", p=p, observed=is_high)

        idata_log = pm.sample(2000, tune=2000, target_accept=0.95, return_inferencedata=True)

    #7
    print(az.summary(idata_log, var_names=["beta"]))
    print(az.hdi(idata_log.posterior["beta"], hdi_prob=0.95))

    # variabila cu |mean(beta)| maxim:
    beta_means = idata_log.posterior["beta"].mean(("chain", "draw")).values
    best_idx = np.argmax(np.abs(beta_means))

    print("Cea mai influenta variabila:", X_cols[best_idx])
    #wind_kph este mai influenta


if __name__ == "__main__":
    main()