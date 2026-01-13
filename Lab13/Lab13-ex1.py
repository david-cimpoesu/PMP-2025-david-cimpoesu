import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

az.style.use("arviz-darkgrid")

data = np.loadtxt("date.csv")
x_1 = data[:, 0]
y_1 = data[:, 1]

def make_design_and_standardize(x, y, order):
    x_p = np.vstack([x**i for i in range(1, order + 1)])
    x_s = (x_p - x_p.mean(axis=1, keepdims=True)) / x_p.std(axis=1, keepdims=True)
    y_s = (y - y.mean()) / y.std()
    return x_s, y_s

def fit_poly_model(x_s, y_s, beta_sigma, order, draws=1000, tune=1000):
    with pm.Model() as model_p:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta  = pm.Normal("beta", mu=0, sigma=beta_sigma, shape=order)
        eps   = pm.HalfNormal("eps", sigma=5)
        mu    = alpha + pm.math.dot(beta, x_s)
        y_pred = pm.Normal("y_pred", mu=mu, sigma=eps, observed=y_s)
        idata = pm.sample(draws=draws, tune=tune, target_accept=0.9, cores=1, chains=1, return_inferencedata=True)
        pm.compute_log_likelihood(idata, model=model_p)
    return model_p, idata

def plot_fit(x_s, y_s, idata, title=""):
    alpha_post = idata.posterior["alpha"].mean(("chain", "draw")).values
    beta_post  = idata.posterior["beta"].mean(("chain", "draw")).values
    idx = np.argsort(x_s[0])
    y_hat = alpha_post + np.dot(beta_post, x_s)
    plt.figure(figsize=(7,4))
    plt.scatter(x_s[0], y_s, marker=".")
    plt.plot(x_s[0][idx], y_hat[idx])
    plt.title(title)
    plt.show()

# 1.1

order = 5
x_s, y_s = make_design_and_standardize(x_1, y_1, order)
model_p, idata_p = fit_poly_model(x_s, y_s, beta_sigma=10, order=order)
plot_fit(x_s, y_s, idata_p, "Order 5, sd=10")

# 1.1 b

model_p_100, idata_p_100 = fit_poly_model(x_s, y_s, beta_sigma=100, order=order)
plot_fit(x_s, y_s, idata_p_100, "Order 5, sd=100")

beta_sigma_vec = np.array([10, 0.1, 0.1, 0.1, 0.1])
model_p_vec, idata_p_vec = fit_poly_model(x_s, y_s, beta_sigma=beta_sigma_vec, order=order)
plot_fit(x_s, y_s, idata_p_vec, "Order 5, sd vector")

# 1.2

rng = np.random.default_rng(0)

def generate_dummy(n=500, noise=1.0):
    x = rng.uniform(-2, 2, size=n)
    y = 1.0 + 2.0*x - 1.5*(x**2) + 0.7*(x**3) + rng.normal(0, noise, size=n)
    return x, y

x_500, y_500 = generate_dummy(n=500, noise=1.0)
order = 5
x_s_500, y_s_500 = make_design_and_standardize(x_500, y_500, order)

_, idata_500_sd10  = fit_poly_model(x_s_500, y_s_500, beta_sigma=10,  order=order)
_, idata_500_sd100 = fit_poly_model(x_s_500, y_s_500, beta_sigma=100, order=order)

plot_fit(x_s_500, y_s_500, idata_500_sd10,  "n=500 sd=10")
plot_fit(x_s_500, y_s_500, idata_500_sd100, "n=500 sd=100")

# 1.3

def fit_for_order(order, beta_sigma=10):
    x_s, y_s = make_design_and_standardize(x_1, y_1, order)
    model, idata = fit_poly_model(x_s, y_s, beta_sigma=beta_sigma, order=order)
    return idata

idata_1 = fit_for_order(1)
idata_2 = fit_for_order(2)
idata_3 = fit_for_order(3)

waic_cmp = az.compare(
    {"linear": idata_1, "quadratic": idata_2, "cubic": idata_3},
    ic="waic",
    scale="deviance",
    method="BB-pseudo-BMA"
)

loo_cmp = az.compare(
    {"linear": idata_1, "quadratic": idata_2, "cubic": idata_3},
    ic="loo",
    scale="deviance",
    method="BB-pseudo-BMA"
)

print("WAIC")
print(waic_cmp)
print("LOO")
print(loo_cmp)

az.plot_compare(waic_cmp)
plt.title("WAIC")
plt.show()

az.plot_compare(loo_cmp)
plt.title("LOO")
plt.show()
