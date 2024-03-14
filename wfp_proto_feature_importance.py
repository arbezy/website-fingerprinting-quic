print("hello")

# TODO: Copy across wfp_tproto_comparison


# Feature importance code... for diabetes dataset

from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.ensemble import  RandomForestClassifier
import pandas as pd

diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

clf=RandomForestClassifier(n_estimators =10, random_state = 42, class_weight="balanced")
output = cross_validate(clf, X, y, cv=2, scoring = 'accuracy', return_estimator =True)

for idx,estimator in enumerate(output['estimator']):
    print("Features sorted by their score for estimator {}:".format(idx))
    feature_importances = pd.DataFrame(estimator.feature_importances_,
                                       index = diabetes.feature_names,
                                        columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances)
    
# end.