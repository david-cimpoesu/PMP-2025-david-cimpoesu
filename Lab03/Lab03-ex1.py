from pgmpy.models import BayesianModel
#a
model = BayesianModel([('S','O'), ('S','L'), ('S','M'), ('L','M')])
print(model.get_independencies())

#b
