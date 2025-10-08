import numpy as np
import matplotlib.pyplot as plt

# --Ex 2.1--
X1 = np.random.poisson(1, 1000)
X2 = np.random.poisson(2, 1000)
X3 = np.random.poisson(5, 1000)
X4 = np.random.poisson(10, 1000)

# --Ex 2.2--
# --a--
lambdas = np.random.choice([1, 2, 5, 10], 1000)
X_random = np.array([np.random.poisson(lam) for lam in lambdas])

plt.figure(figsize=(10, 8))

plt.subplot(3, 2, 1)
plt.hist(X1, bins=10)
plt.title("Poisson(1)")

plt.subplot(3, 2, 2)
plt.hist(X2, bins=10)
plt.title("Poisson(2)")

plt.subplot(3, 2, 3)
plt.hist(X3, bins=10)
plt.title("Poisson(5)")

plt.subplot(3, 2, 4)
plt.hist(X4, bins=10)
plt.title("Poisson(10)")

plt.subplot(3, 2, 5)
plt.hist(X_random, bins=10)
plt.title("Randomized Poisson")

plt.tight_layout()
plt.show()

# --b--
'''
Distribuțiile fixe sunt concentrate în jurul propriei medii (un singur varf),
iar cea randomizată are mai multe varfuri. 
Asta arată că incertitudinea parametrului lambda duce la o variabilitate
mai mare a procesului și o distribuție totală mai "răspândită"
'''
