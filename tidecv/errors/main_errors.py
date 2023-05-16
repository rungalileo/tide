from typing import Dict

from .error import BestGTMatch, Error
from dataclasses import dataclass


@dataclass
class ClassError(Error):
    pred: Dict[str, int]
    gt: Dict[str, int]
    short_name: str = "Cls"
    description: str = (
        "Error caused when a prediction would have been marked positive "
        "if it had the correct class."
    )

    def __post_init__(self):
        self.match = BestGTMatch(self.pred, self.gt) if not self.gt["used"] else None

    # def __init__(self, pred: dict, gt: dict):
    #     self.pred = pred
    #     self.gt = gt
    #
    #     self.match = BestGTMatch(pred, gt) if not self.gt["used"] else None

    def fix(self):
        if self.match is None:
            return None
        return self.gt["class"], self.match.fix()


@dataclass
class BoxError(Error):
    pred: Dict[str, int]
    gt: Dict[str, int]
    description: str = (
        "Error caused when a prediction would have been marked "
        "positive if it was localized better."
    )
    short_name: str = "Loc"

    def __post_init__(self):
        self.match = BestGTMatch(self.pred, self.gt) if not self.gt["used"] else None

    # def __init__(self, pred: dict, gt: dict):
    #     self.pred = pred
    #     self.gt = gt
    #
    #     self.match = BestGTMatch(pred, gt) if not self.gt["used"] else None

    def fix(self):
        if self.match is None:
            return None
        return self.pred["class"], self.match.fix()


@dataclass
class DuplicateError(Error):
    pred: Dict[str, int]
    gt: Dict[str, int]
    suppressor: Dict
    description: str = (
        "Error caused when a prediction would have been marked positive "
        "if the GT wasn't already in use by another detection."
    )
    short_name: str = "Dupe"

    # def __init__(self, pred: dict, gt: dict, suppressor: dict):
    #     self.pred = pred
    #     self.gt = gt
    #     self.suppressor = suppressor

    def fix(self):
        return None


@dataclass
class BackgroundError(Error):
    pred: Dict[str, int]
    description: str = (
        "Error caused when this detection should have been classified as "
        "background (IoU < 0.1)."
    )
    short_name: str = "Bkg"

    # def __init__(self, pred: dict):
    #     self.pred = pred

    def fix(self):
        return None


@dataclass
class ClassBoxError(Error):
    pred: Dict[str, int]
    gt: Dict[str, int]
    description: str = (
        "Error caused when a prediction would have been marked positive "
        "if it had the correct class and was localized better."
    )
    short_name: str = "ClsLoc"

    # def __init__(self, pred: dict, gt: dict):
    #     self.pred = pred
    #     self.gt = gt

    def fix(self):
        return None


@dataclass
class MissedError(Error):
    gt: Dict[str, int]
    description: str = (
        "Represents GT missed by the model. Doesn't include "
        "GT corrected elsewhere in the model."
    )
    short_name: str = "Miss"

    # def __init__(self, gt: dict):
    #     self.gt = gt

    def fix(self):
        return self.gt["class"], -1


# These are special errors so no inheritence


@dataclass
class FalsePositiveError:
    description = (
        "Represents the potential AP gained by having perfect precision"
        " (e.g., by scoring all false positives as conf=0) without affecting recall."
    )
    short_name = "FalsePos"

    @staticmethod
    def fix(score: float, correct: bool, info: dict) -> tuple:
        if correct:
            return 1, True, info
        else:
            return 0, False, info


@dataclass
class FalseNegativeError:
    description: str = (
        "Represents the potentially AP gained by having perfect recall"
        " without affecting precision."
    )
    short_name: str = "FalseNeg"
