import pandas
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold, ShuffleSplit

wine_path = "./winequalityN.csv"
wine = pandas.read_csv(wine_path)
wine = wine.dropna()

input_wine = wine.drop(columns = "quality")
output_wine = wine["quality"]
input_wine = pandas.get_dummies(input_wine)

decision_wine = DecisionTreeClassifier("entropy")
decision_wine.fit(input_wine, output_wine)
decision_wine.score(input_wine, output_wine)
export_graphviz(decision_wine)

lr_wine = LogisticRegression()
lr_wine.fit(input_wine, output_wine)
lr_wine.score(input_wine, output_wine)
lr_wine.coef_

mlp_wine = MLPClassifier((10, ), activation = "logistic")
mlp_wine.fit(input_wine, output_wine)
mlp_wine.score(input_wine, output_wine)

kfold = KFold(n_splits = 5)
kfold.split(input_wine, output_wine)
cross_val_score(DecisionTreeClassifier(), input_wine, output_wine, cv = kfold)
cross_val_score(LogisticRegression(), input_wine, output_wine, cv = kfold)
cross_val_score(MLPClassifier(), input_wine, output_wine, cv = kfold)