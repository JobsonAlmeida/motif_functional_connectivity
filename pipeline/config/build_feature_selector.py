from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector

def build_feature_selector(feature_selector, k_features):

    if feature_selector == "none":

        selector = None

        selector_config = {
            "type": "None"
        }

    elif feature_selector == "f_classif":

        selector = SelectKBest(
            score_func=f_classif,
            k=k_features
        )

        selector_config = {
            "type": "SelectKBest",
            "score_func": "f_classif",
            "k": k_features
        }

    elif feature_selector == "mutual_info":

        selector = SelectKBest(
            score_func=mutual_info_classif,
            k=k_features
        )

        selector_config = {
            "type": "SelectKBest",
            "score_func": "mutual_info_classif",
            "k": k_features
        }

    elif feature_selector == "sequential_forward_svm":

        selector_estimator = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight=None,
            tol=1e-3,
            max_iter=-1
        )

        selector = SequentialFeatureSelector(
            estimator=selector_estimator,
            n_features_to_select=k_features,
            direction="forward",
            scoring="accuracy",
            cv=5,
            n_jobs=-1
        )

        selector_config = {
            "type": "SequentialFeatureSelector",
            "n_features_to_select": k_features,
            "direction": "forward",
            "scoring": "accuracy",
            "cv": 5,
            "n_jobs": -1,
            "estimator": {
                "type": "SVC",
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "class_weight": None,
                "tol": 1e-3,
                "max_iter": -1
            }
        }

    else:
        raise ValueError(
            f"Feature selector '{feature_selector}' not supported."
        )

    return selector, selector_config
