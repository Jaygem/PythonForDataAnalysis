# %% loading
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import flask
from flask import Flask

import sklearn
import pandas as pd
from pandas.compat import StringIO
import numpy as np


listCol = "regionCentroidCol regionCentroidRow regionPixelCount shortLineDensity5 shortLineDensity2 vedgeMean vedgeSd hedgeMean hedgeSd intensityMean rawRedMean rawBlueMean rawGreenMean exRedMean exBlueMean exGreenMean valueMean saturationMean hueMean class".split(
    " ")

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

# %% Etape 1 : observer la diversité de notre dataset

print(data["class"].value_counts())

# Le dataset contient autant de data pour chaque classes ce qui est très intéressant pour notre étude

# %% Etape 2 : Vérifier si il y a un cohérence des valeurs entre les quantités de couleurs et les éléments

a = data.groupby(["class"])["rawRedMean",
                            "rawGreenMean", "rawBlueMean"].median()
a.index = classes
ax = a.plot.bar(rot=0)
# Ce qui est bizarre : Il y a plus de vert que d'autres couleurs...
# En y réfléchissant un peu plus, les images prises par des apareils photos munis d'un CCD ou d'un
# CMOS utilisent un filtre de BAYER. La quantité de pixels vert est supérieur aux autres couleurs.
# Ce qui expliquerait la présence de vert dans le ciel.

# %% Les couleurs en excès

a = data.groupby(["class"])["exRedMean", "exGreenMean", "exBlueMean"].median()
a.index = classes
ax = a.plot.bar(rot=0)

# %% Data preparation
data['is_train'] = np.random.uniform(0, 1, len(data)) <= .75
y_train, y_test = data.query("is_train == True")[
    "class"], data.query("is_train == False")["class"]
X_train, X_test = data.query("is_train == True").drop(["class", "is_train"], axis=1), data.query(
    "is_train == False").drop(["class", "is_train"], axis=1)
X_train.head()
#data = data.drop(labels="class",axis=1)


# %% Etape 3 : Clustering


n_samples = 1500
random_state = 100
maxScore = 0
bestC = 0
for i in range(5, 15):
    y_pred = KMeans(
        n_clusters=i, random_state=random_state).fit_predict(X_train)
    score = metrics.adjusted_rand_score(y_train, y_pred)
    maxScore, bestC = (score, i) if score > maxScore else (maxScore, bestC)
    print("best = ", bestC)
    print("best score = ", maxScore)
    print(str(i)+" gave us a score of " + str(score))
y_pred = KMeans(n_clusters=bestC,
                random_state=random_state).fit_predict(X_train)
plt.subplot(221)
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_pred)
plt.title("Clustering color areas based on their data")

# %% Etape 4 : Logistic regression


model = LogisticRegression(random_state=0, solver='lbfgs',
                           multi_class='multinomial').fit(X_train, y_train)
model.score(X_test, y_test)


# %% Etape 5 : Random Forest


Forest = RandomForestClassifier(n_estimators=100, max_depth=4,
                                random_state=0)
Forest.fit(X_train, y_train)
feature_importances = pd.DataFrame(Forest.feature_importances_,
                                   index=X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

print(feature_importances)
print("Overall accuracy for the test set : ", Forest.score(X_test, y_test))

#%% Etape 6 : Grid Search

from sklearn.model_selection import GridSearchCV

param_grid = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}
ForestToSearch = RandomForestClassifier(
                    n_estimators=100, max_depth=4,
                    random_state=0)
gridForest = GridSearchCV(ForestToSearch,param_grid,cv=5)
gridForest.fit(X_train, y_train)

print(gridForest.best_params_)
print("Overall accuracy for the test set : ", gridForest.score(X_test, y_test))

#%% Etape 7 : Randomized Search

from sklearn.model_selection import RandomizedSearchCV

ForestToSearch = RandomForestClassifier(
                    n_estimators=100, max_depth=4,
                    random_state=0)
searchForest = GridSearchCV(ForestToSearch,param_grid,cv=5)
searchForest.fit(X_train, y_train)

