from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
import itertools, math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

H, W = 5, 5
V = [f"X{r}_{c}" for r in range(H) for c in range(W)]
E = []
for r in range(H):
    for c in range(W):
        if r+1 < H: E.append((f"X{r}_{c}", f"X{r+1}_{c}"))
        if c+1 < W: E.append((f"X{r}_{c}", f"X{r}_{c+1}"))

mn = MarkovNetwork()
mn.add_nodes_from(V)
mn.add_edges_from(E)

#a
G = nx.Graph()
G.add_nodes_from(mn.nodes())
G.add_edges_from(mn.edges())
pos = {f"X{r}_{c}": (c, -r) for r in range(H) for c in range(W)}
nx.draw(G, pos, node_size=200, alpha=0.75, with_labels=False)
plt.show()

#b
rng = np.random.default_rng()
lam = 2.5
x_true = rng.choice([-1, 1], size=(H, W))
y = x_true.copy()
k = max(1, int(0.1 * H * W))
idx = rng.choice(H*W, size=k, replace=False)
for t in idx:
    r, c = divmod(t, W)
    y[r, c] *= -1

unary_factors = []
for r in range(H):
    for c in range(W):
        var = f"X{r}_{c}"
        yi = int(y[r, c])
        vals = [math.exp(-lam*((-1)-yi)**2), math.exp(-lam*((+1)-yi)**2)]
        unary_factors.append(DiscreteFactor([var], [2], vals))

pair_vals = [math.exp(-((xi - xj) ** 2)) for xi, xj in itertools.product([-1, 1], repeat=2)]
pair_factors = [DiscreteFactor([u, v], [2, 2], pair_vals) for u, v in E]

mn.add_factors(*(unary_factors + pair_factors))
mn.check_model()

bp = BeliefPropagation(mn)
map_cfg = bp.map_query(variables=V)
x_map = np.array([(-1 if map_cfg[f"X{r}_{c}"]==0 else 1) for r in range(H) for c in range(W)]).reshape(H, W)

fig, axs = plt.subplots(1, 3, figsize=(8, 3))
axs[0].imshow(x_true, cmap="gray", vmin=-1, vmax=1); axs[0].set_title("Original"); axs[0].axis("off")
axs[1].imshow(y, cmap="gray", vmin=-1, vmax=1); axs[1].set_title("Noisy"); axs[1].axis("off")
axs[2].imshow(x_map, cmap="gray", vmin=-1, vmax=1); axs[2].set_title("Denoised (MAP)"); axs[2].axis("off")
plt.tight_layout(); plt.show()
