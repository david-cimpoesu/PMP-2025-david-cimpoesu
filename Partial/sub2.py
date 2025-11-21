import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from hmmlearn.hmm import CategoricalHMM

HMM_KIND = "categorical"

states = ["W", "R", "S"]
symbols = ["L", "M", "H"]

pi = np.array([0.4,0.3,0.3], dtype=float)

A = np.array([
    [0.6,0.3,0.1],
    [0.2,0.7,0.1],
    [0.3,0.2,0.5]
], dtype=float)

B = np.array([
    [0.1,0.7,0.2],
    [0.05,0.25,0.7],
    [0.8,0.15,0.05]
], dtype=float)

model = CategoricalHMM(n_components=3, init_params="")
model.startprob_ = pi
model.transmat_ = A
model.emissionprob_ = B

G = nx.DiGraph()
for s in states:
    G.add_node(s)
for i,si in enumerate(states):
    for j,sj in enumerate(states):
        if A[i,j]>0:
            G.add_edge(si,sj,label=f"{A[i,j]:.2f}")

pos = nx.circular_layout(G)
plt.figure(figsize=(6,6))
nx.draw(G,pos,with_labels=True,node_size=1800,font_size=10)
nx.draw_networkx_edge_labels(G,pos,edge_labels=nx.get_edge_attributes(G,'label'))
plt.show()

#b
sym2idx = {s:i for i,s in enumerate(symbols)}
obs = ["M","H","L"]
obs_idx = np.array([sym2idx[x] for x in obs], dtype=int).reshape(-1,1)

logp = model.score(obs_idx)
print("Forward P =", float(np.exp(logp)))

#c
logp_v, states_hat = model.decode(obs_idx, algorithm="viterbi")
print("Viterbi states =", states_hat.tolist())
print("Viterbi P =", float(np.exp(logp_v)))

#d
T = 10000
seqs = []
for _ in range(T):
    X, Z = model.sample(3)
    seqs.append(tuple(X.flatten().tolist()))
emp = seqs.count(tuple(obs_idx.flatten().tolist())) / T
print("Empirical P =", emp)