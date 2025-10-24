from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import itertools

model = DiscreteBayesianNetwork([('S', 'O'), ('S', 'L'), ('S', 'M'), ('L', 'M')])

cpd_S = TabularCPD('S', 2, [[0.6], [0.4]], state_names={'S': [0, 1]})

cpd_O = TabularCPD(
    'O', 2,
    [[0.9, 0.3],   # P(O=0|S=0), P(O=0|S=1)
     [0.1, 0.7]],  # P(O=1|S=0), P(O=1|S=1)
    evidence=['S'], evidence_card=[2],
    state_names={'O': [0, 1], 'S': [0, 1]}
)

cpd_L = TabularCPD(
    'L', 2,
    [[0.7, 0.2],
     [0.3, 0.8]],
    evidence=['S'], evidence_card=[2],
    state_names={'L': [0, 1], 'S': [0, 1]}
)

# Coloane în ordinea (S,L): (0,0) (0,1) (1,0) (1,1)
cpd_M = TabularCPD(
    'M', 2,
    [[0.8, 0.4, 0.5, 0.1],
     [0.2, 0.6, 0.5, 0.9]],
    evidence=['S', 'L'], evidence_card=[2, 2],
    state_names={'M': [0, 1], 'S': [0, 1], 'L': [0, 1]}
)

model.add_cpds(cpd_S, cpd_O, cpd_L, cpd_M)
model.check_model()

print("Independencies:")
print(model.get_independencies())
print()


infer = VariableElimination(model)

print("O  L  M  |  P(S=1|O,L,M) | label")
print("-------------------------------")
for O, L, M in itertools.product([0, 1], [0, 1], [0, 1]):
    p_spam = float(infer.query(['S'], evidence={'O': O, 'L': L, 'M': M}).values[1])
    label = "SPAM" if p_spam >= 0.5 else "NON-SPAM"
    print(f"{O}  {L}  {M}  |     {p_spam:0.3f}      | {label}")

import matplotlib.pyplot as plt
import networkx as nx

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold',
        node_color='skyblue', arrows=True, arrowstyle='-|>', arrowsize=28, width=1.5)
plt.savefig("bn_structure.png", dpi=200); plt.close()

