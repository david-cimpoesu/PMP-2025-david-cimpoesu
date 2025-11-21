from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import itertools


model = DiscreteBayesianNetwork([
    ("O", "H"),
    ("O", "W"),
    ("H", "R"),
    ("W", "R"),
    ("H", "E"),
    ("R", "C")
])


cpd_O = TabularCPD('O', 2, [[0.6], [0.4]], state_names={'O': ['cold', 'mild']})


cpd_H = TabularCPD('H', 2, [[0.9,0.2],[0.1,0.8]],
                   evidence=['O'], evidence_card=[2],
                   state_names={'H': ['yes', 'no'], 'O': ['cold', 'mild']})

cpd_W = TabularCPD('W', 2, [[0.1,0.6],[0.9,0.4]],
                   evidence=['O'], evidence_card=[2],
                   state_names={'W': ['yes', 'no'], 'O': ['cold', 'mild']})

cps_R = TabularCPD('R',2, [[0.6,0.9,0.3,0.5],[0.4,0.1,0.7,0.5]],
                   evidence=['H','W'], evidence_card = [2, 2],
                   state_names={'R': ['warm', 'cool'], 'H': ['yes', 'no'], 'W': ['yes', 'no']})

cpd_E = TabularCPD('E', 2, [[0.8,0.2],[0.2,0.8]],
                   evidence=['H'], evidence_card=[2],
                   state_names={'E': ['high', 'low'], 'H': ['yes', 'no']})

cpd_C = TabularCPD('C', 2, [[0.85,0.4],[0.15,0.6]],
                   evidence=['R'], evidence_card=[2],
                   state_names={'C': ['comfortable', 'uncomfortable'], 'R': ['warm', 'cool']})

model.add_cpds(cpd_O, cpd_H, cpd_W, cps_R, cpd_E, cpd_C)

model.check_model()
infer = VariableElimination(model)

print(infer.query(['H'], evidence={'C':'comfortable'}))
import matplotlib.pyplot as plt
import networkx as nx

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold',
        node_color='skyblue', arrows=True, arrowstyle='-|>', arrowsize=28, width=1.5)
plt.show()



