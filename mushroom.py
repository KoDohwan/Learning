import pandas
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold, ShuffleSplit

mushroom_path = "./mushroom.csv"
mushroom = pandas.read_csv(mushroom_path)

input_mushroom = mushroom.drop(columns = "class")
output_mushroom = mushroom["class"]
input_mushroom = pandas.get_dummies(input_mushroom)


decision_mushroom = DecisionTreeClassifier("entropy")
decision_mushroom.fit(input_mushroom, output_mushroom)
decision_mushroom.score(input_mushroom, output_mushroom)
export_graphviz(decision_mushroom)

lr_mushroom = LogisticRegression()
lr_mushroom.fit(input_mushroom, output_mushroom)
lr_mushroom.score(input_mushroom, output_mushroom)
lr_mushroom.coef_

mlp_mushroom = MLPClassifier((10, ), activation = "logistic")
mlp_mushroom.fit(input_mushroom, output_mushroom)
mlp_mushroom.score(input_mushroom, output_mushroom)

kfold = KFold(n_splits = 5)
kfold.split(input_mushroom, output_mushroom)
cross_val_score(DecisionTreeClassifier(), input_mushroom, output_mushroom, cv = kfold)
cross_val_score(LogisticRegression(), input_mushroom, output_mushroom, cv = kfold)
cross_val_score(MLPClassifier(), input_mushroom, output_mushroom, cv = kfold)