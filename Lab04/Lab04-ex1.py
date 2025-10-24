from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
import itertools, math
import networkx as nx
import matplotlib.pyplot as plt

V = ["A1","A2","A3","A4","A5"]
E = [("A1","A2"),("A1","A3"),("A2","A4"),("A2","A5"),("A3","A4"),("A4","A5")]

mn = MarkovNetwork()
mn.add_nodes_from(V)
mn.add_edges_from(E)

#a
G = nx.Graph()
G.add_nodes_from(mn.nodes())
G.add_edges_from(mn.edges())
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1200, alpha=0.75)
plt.show()
cliques_a = [tuple(c) for c in nx.find_cliques(G)]
print("Maximal cliques:", cliques_a)

#b
cliques = [("A1","A2"), ("A1","A3"), ("A3","A4"), ("A2","A4","A5")]

def pot(assign, clique):
    s = sum(int(v[1:]) * val for v, val in zip(clique, assign))
    return math.exp(s)

factors = []
for clique in cliques:
    vals = [pot(assg, clique) for assg in itertools.product([-1,1], repeat=len(clique))]
    factors.append(DiscreteFactor(clique, [2]*len(clique), vals))

mn.add_factors(*factors)
mn.check_model()

bp = BeliefPropagation(mn)
map_cfg = bp.map_query(variables=V)
print({v: (-1 if s==0 else 1) for v, s in map_cfg.items()})

assigns = list(itertools.product([-1,1], repeat=5))
score = lambda a1,a2,a3,a4,a5: math.exp(2*a1 + 4*a2 + 6*a3 + 8*a4 + 5*a5)
Z = sum(score(*a) for a in assigns)
print("Z =", Z)

map_vals = [(-1 if map_cfg[v]==0 else 1) for v in V]
p_map = score(*map_vals) / Z
print("P(MAP) =", p_map)
