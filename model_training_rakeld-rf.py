import sys
sys.path.append("../..")
import warnings
warnings.filterwarnings("ignore")

from chemocommons import *
import pandas as pd
import numpy as np

from skmultilearn.cluster import NetworkXLabelGraphClusterer # clusterer
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder # as it writes
from skmultilearn.ensemble import LabelSpacePartitioningClassifier # so?
from skmultilearn.ensemble import RakelD

from skmultilearn.adapt import MLkNN, MLTSVM
from skmultilearn.problem_transform import LabelPowerset # sorry, we only used LP
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier # Okay?
from sklearn.model_selection import LeaveOneOut, RepeatedKFold #, KFold # jackknife, "socalled"
from sklearn.metrics import jaccard_similarity_score, f1_score # for some calculation
from sklearn.utils.multiclass import unique_labels
from lightgbm import LGBMClassifier

loocv = LeaveOneOut() # jackknife

label_names = ["ABCG2", "MDR1", "MRP1", "MRP2", "MRP3", "MRP4", "NTCP2", "S15A1", 
               "S22A1", "SO1A2", "SO1B1", "SO1B3", "SO2B1"]

Y = pd.read_csv("label_matrix.txt", sep="\t", names=label_names)
Y[Y==-1]=0

ft_FP = pd.read_csv("query_smiles_feature_similarity_four_average.csv", names=label_names)
ft_FP.rename(mapper= lambda x: x + "_FP", axis=1, inplace=True)
ft_OT = pd.read_csv("feature_similarity_chebi_ontology_DiShIn_2.csv", names=label_names)
ft_OT.rename(mapper= lambda x: x + "_OT", axis=1, inplace=True)

X = np.concatenate((ft_FP, ft_OT), axis=1)

scoring_funcs = {"hamming loss": hamming_func, 
                 "aiming": aiming_func, 
                 "coverage": coverage_func, 
                 "accuracy": accuracy_func, 
                 "absolute true": absolute_true_func, 
                 } # Keep recorded

parameters =  {'labelset_size': [2, 3, 4, 5, 6, 7, 8, 9, 10]}

rakeld = GridSearchCV(RakelD(base_classifier=RandomForestClassifier(),
    baseclassifier_require_dense=[True, True],), param_grid=parameters, n_jobs=-1, cv=loocv, 
                    scoring=scoring_funcs, verbose=3, refit="absolute true")

rakeld.fit(X, Y.values)
print(rakeld.best_score_)




mytuple = (
    rakeld,
)

to_save = dump(mytuple, filename="rakeld-rf.joblib")

