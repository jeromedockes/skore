from typing import Any

import skrub
from sklearn.base import BaseEstimator


def eval_X_y(data_op: skrub.DataOp, env: dict) -> dict:
    """
    Return a dict in which X and y have been materialized.

    The result is similar to .skb.train_test_split outputs.
    It ensures we can retrieve the ground truth to compute metrics, and that X
    and y are materialized so that results will be consistent even if there is
    randomness in the production of X and y (e.g. they are unsorted database
    query results).
    """
    return data_op.skb.train_test_split(
        env, split_func=lambda X, y: (X, None, y, None)
    )["train"]


def is_skrub_learner(obj: Any) -> bool:
    """Detect if obj is a skrub learner (SkrubLearner, ParamSearch, OptunaSearch)."""
    return hasattr(obj, "__skrub_to_Xy_pipeline__")


def to_learner(estimator: BaseEstimator):
    return (
        skrub.X()
        .skb.apply(estimator, y=skrub.y())
        .skb.set_name("estimator")
        .skb.make_learner()
    )


def to_estimator(learner: skrub.SkrubLearner):
    return learner.find_fitted_estimator("estimator")
