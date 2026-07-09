"""Additional sklearn classifiers for the unified flattened-window contract."""

from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC

from src.models.classical._sklearn_wrapper import SklearnWindowModel


class LogisticRegressionModel(SklearnWindowModel):
    def __init__(self, task: str = "classification", **params):
        super().__init__(LogisticRegression(**params), task=task)


class RidgeClassifierModel(SklearnWindowModel):
    def __init__(self, task: str = "classification", **params):
        super().__init__(RidgeClassifier(**params), task=task)


class LinearSVCModel(SklearnWindowModel):
    def __init__(self, task: str = "classification", **params):
        super().__init__(LinearSVC(**params), task=task)


class SVCModel(SklearnWindowModel):
    def __init__(self, task: str = "classification", **params):
        super().__init__(SVC(**params), task=task)


class KNNClassifierModel(SklearnWindowModel):
    def __init__(self, task: str = "classification", **params):
        super().__init__(KNeighborsClassifier(**params), task=task)


class ExtraTreesModel(SklearnWindowModel):
    def __init__(self, task: str = "classification", **params):
        super().__init__(ExtraTreesClassifier(**params), task=task)


class GradientBoostingModel(SklearnWindowModel):
    def __init__(self, task: str = "classification", **params):
        super().__init__(GradientBoostingClassifier(**params), task=task)


class HistGradientBoostingModel(SklearnWindowModel):
    def __init__(self, task: str = "classification", **params):
        super().__init__(HistGradientBoostingClassifier(**params), task=task)


class AdaBoostModel(SklearnWindowModel):
    def __init__(self, task: str = "classification", **params):
        super().__init__(AdaBoostClassifier(**params), task=task)


class MLPClassifierModel(SklearnWindowModel):
    def __init__(self, task: str = "classification", **params):
        super().__init__(MLPClassifier(**params), task=task)


class GaussianNBModel(SklearnWindowModel):
    def __init__(self, task: str = "classification", **params):
        super().__init__(GaussianNB(**params), task=task)


class LDAModel(SklearnWindowModel):
    def __init__(self, task: str = "classification", **params):
        super().__init__(LinearDiscriminantAnalysis(**params), task=task)


class QDAModel(SklearnWindowModel):
    def __init__(self, task: str = "classification", **params):
        super().__init__(QuadraticDiscriminantAnalysis(**params), task=task)


class DummyMostFrequentModel(SklearnWindowModel):
    def __init__(self, task: str = "classification", **params):
        params.setdefault("strategy", "most_frequent")
        super().__init__(DummyClassifier(**params), task=task)
