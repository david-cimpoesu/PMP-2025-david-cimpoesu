#a
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from hmmlearn.hmm import CategoricalHMM
HMM_KIND = "categorical"
states = ["Hard","Medium","Easy"]
symbols = ["FB","B","S","NS"]
pi = np.array([1/3,1/3,1/3],dtype=float)
A = np.array([[0.0,0.5,0.5],
              [0.5,0.25,0.25],
              [0.5,0.25,0.25]],dtype=float)
B = np.array([[0.10,0.20,0.40,0.30],
              [0.15,0.25,0.50,0.10],
              [0.20,0.30,0.40,0.10]],dtype=float)

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
obs = ["FB","FB","S","B","B","S","B","B","NS","B","B"]
obs_idx = np.array([sym2idx[x] for x in obs],dtype=int).reshape(-1,1)
logp = model.score(obs_idx)
p_seq = float(np.exp(logp))
print(f"P = {p_seq:.6e}")

#c
logp_v, states_hat = model.decode(obs_idx, algorithm="viterbi")
p_vit = float(np.exp(logp_v))
print("Viterbi states (0=Hard,1=Medium,2=Easy):", states_hat.tolist())
print(f"P(Viterbi path) = {p_vit:.6e}")
plt.figure()
plt.plot(states_hat,"-o",label="Viterbi")
plt.yticks(ticks=range(len(states)),labels=states)
plt.xlabel("Time step")
plt.ylabel("State")
plt.legend()
plt.show()
