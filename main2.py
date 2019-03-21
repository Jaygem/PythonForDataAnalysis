import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from pandas.compat import StringIO
import numpy as np
from random import randint


listCol = "regionCentroidCol regionCentroidRow regionPixelCount shortLineDensity5 shortLineDensity2 vedgeMean vedgeSd hedgeMean hedgeSd intensityMean rawRedMean rawBlueMean rawGreenMean exRedMean exBlueMean exGreenMean valueMean saturationMean hueMean class".split(" ")

with open("segment.dat", "r") as f:
    text = f.readlines()
text1 = '\n'.join([str(idx)+" " + i for idx, i in enumerate(text)])
data = pd.read_csv(StringIO(text1),
                   sep="\s+")

data = data.set_index("0")
classes = ["brickface",
           "sky",
           "foliage",
           "cement",
           "window",
           "path",
           "grass"]

# Etape 1 : observer la diversité de notre dataset avec le nombre d'élements par classes

print(data["class"].value_counts())


# Etape 2 : Vérifier si il y a un cohérence des valeurs entre les quantités de couleurs et les classes segmentées

a = data.groupby(["class"])["rawRedMean",
                            "rawGreenMean",
                             "rawBlueMean"].median()
a.index = classes
ax = a.plot.bar(rot=0)
fig = ax.get_figure().savefig("colorMeans.png")


# Même calcul pour les couleurs en excès

a = data.groupby(["class"])["exRedMean", "exGreenMean", "exBlueMean"].median()
a.index = classes
ax = a.plot.bar(rot=0)
ax.get_figure().savefig("colorExcess.png")

# Préparation de la data, on sépare le dataset en test et entraînement
_data = data.copy()
labels = _data["class"]
_data['is_train'] = np.random.uniform(0, 1, len(data)) <= .75
y_train, y_test =   _data.query("is_train == True")["class"], _data.query("is_train == False")["class"]
X_train, X_test =   _data.query("is_train == True").drop(["class", "is_train"], axis=1), _data.query("is_train == False").drop(["class", "is_train"], axis=1)
X_train.head()



# Etape 3 : Clustering

n_samples = 1500
random_state = 170

y_pred = KMeans(n_clusters=7,
                random_state=random_state).fit_predict(data)
print(metrics.adjusted_rand_score(labels,y_pred))
plt.subplot(221)
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=y_pred)
plt.title("Clustering color areas based on their data")
plt.savefig("Clustering.png")

# Etape 4 : Logistic regression avec 3 solvers différents
results ={}
model = LogisticRegression(random_state=0, solver='sag',
                           multi_class='multinomial').fit(X_train, y_train)
results["sag"] = model.score(X_test, y_test)
model = LogisticRegression(random_state=0, solver='lbfgs',
                           multi_class='multinomial').fit(X_train, y_train)
results["lbfgs"] = model.score(X_test, y_test)
model = LogisticRegression(random_state=0, solver='newton-cg',
                           multi_class='multinomial').fit(X_train, y_train)
results["newton-cg"] = model.score(X_test, y_test)
results


# Etape 5 : Random Forest classique


Forest = RandomForestClassifier(n_estimators=100, max_depth=4,
                                random_state=0)
Forest.fit(X_train, y_train)
feature_importances = pd.DataFrame(Forest.feature_importances_,
                                   index=X_train.columns,
                                   columns=['importance'])
feature_importances = feature_importances.sort_values('importance', ascending=False)

print(feature_importances)
print("Overall accuracy for the test set : ", Forest.score(X_test, y_test))

# Etape 6 : Grid Search sur un RandomForest

param_grid = { 
    'n_estimators': [50, 600],
    'max_features': ['sqrt', 'log2']
}
ForestToSearch = RandomForestClassifier(
                    n_estimators=100, max_depth=4,
                    random_state=0)
gridForest = GridSearchCV(ForestToSearch,param_grid,cv=5)
gridForest.fit(X_train, y_train)

print(gridForest.best_params_)
print("Overall accuracy for the test set : ",
     gridForest.score(X_test, y_test))

# Etape 7 : Randomized Search sur un RandomForest



ForestToSearch = RandomForestClassifier(
                    n_estimators=100, max_depth=4,
                    random_state=0)
param_dist = {"max_depth": [3, None],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
searchForest = RandomizedSearchCV(ForestToSearch,param_dist,n_iter=8)
searchForest.fit(X_train, y_train)
print(searchForest.best_params_)
print("Overall accuracy for the test set : ",
     gridForest.score(X_test, y_test))