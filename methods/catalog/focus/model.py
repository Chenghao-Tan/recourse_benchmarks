"""

code adapted from:
https://github.com/a-lucic/focus
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.tree import DecisionTreeClassifier

from methods.api import RecourseMethod
from methods.catalog.focus.distances import distance_func
from methods.processing import check_counterfactuals, merge_default_parameters
from models.api import MLModel
from models.catalog import trees


def _filter_hinge_loss(n_class, mask_vector, features, sigma, temperature, model_fn):
    """
    This functions as the prediction loss

    Parameters
    ----------
    n_class : number of classes
    mask_vector : 0 if prediction is flipped, 1 otherwise
    features : current (perturbed) input features
    sigma: float
    temperature: float
    model_fn: model function

    Returns
    -------
    hinge loss
    """
    n_input = features.shape[0]

    if not np.any(mask_vector):
        return torch.zeros(
            (n_input, n_class), device=features.device, dtype=features.dtype
        )

    indices = np.where(mask_vector)[0]
    filtered_input = features[indices]

    if not isinstance(sigma, (float, int)):
        sigma = sigma[indices]
    if not isinstance(temperature, (float, int)):
        temperature = temperature[indices]

    filtered_loss = model_fn(filtered_input, sigma, temperature)

    zero_loss = torch.zeros(
        (n_input, n_class), device=features.device, dtype=features.dtype
    )
    zero_loss[indices] = filtered_loss
    return zero_loss


class FOCUS(RecourseMethod):
    """
    Implementation of Focus [1]_.

    Parameters
    ----------
    mlmodel : model.MLModel
        Black-Box-Model
    checked_hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "optimizer": {"adam", "gd"}
            Determines the optimizer.
        * "n_class": int
            Number of classes.
        * "n_iter": int
            Number of iterations to run for.
        * "sigma": float
            Parameter in sig(z) = (1 + exp(sigma * z)^-1, controls degree of approximation.
        * "temperature": float
            Parameter in the softmax operation, also controls degreee of approximation.
        * "distance_weight": float
            Determines the weight of the counterfactual distance in the loss.
        * "distance_func": {"l1", "l2"}
            Norm to be used.

    .. [1] Lucic, A., Oosterhuis, H., Haned, H., & de Rijke, M. (2018). FOCUS: Flexible optimizable counterfactual
            explanations for tree ensembles. arXiv preprint arXiv:1910.12199.
    """

    _DEFAULT_HYPERPARAMS = {
        "optimizer": "adam",
        "lr": 0.001,
        "n_class": 2,
        "n_iter": 1000,
        "sigma": 1.0,
        "temperature": 1.0,
        "distance_weight": 0.01,
        "distance_func": "l1",
    }

    def __init__(self, mlmodel: MLModel, hyperparams: Optional[Dict] = None) -> None:
        supported_backends = ["sklearn", "xgboost"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super().__init__(mlmodel)
        self.model = mlmodel

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        self.optimizer_name = checked_hyperparams["optimizer"]
        self.lr = checked_hyperparams["lr"]
        self.n_class = checked_hyperparams["n_class"]
        self.n_iter = checked_hyperparams["n_iter"]
        self.sigma_val = checked_hyperparams["sigma"]
        self.temp_val = checked_hyperparams["temperature"]
        self.distance_weight_val = checked_hyperparams["distance_weight"]
        self.distance_function = checked_hyperparams["distance_func"]

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        original_input = self.model.get_ordered_features(factuals)
        original_input = original_input.to_numpy()
        ground_truth = self.model.predict(original_input).reshape(-1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        perturbed = torch.tensor(
            original_input, dtype=torch.float32, device=device, requires_grad=True
        )
        original_tensor = torch.tensor(
            original_input, dtype=torch.float32, device=device
        )

        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam([perturbed], lr=self.lr)
        elif self.optimizer_name == "gd":
            optimizer = torch.optim.SGD([perturbed], lr=self.lr)
        else:
            raise ValueError("optimizer not supported")

        if hasattr(self.model.raw_model, "classes_"):
            class_names = list(self.model.raw_model.classes_)
        else:
            class_names = list(range(self.n_class))

        class_index = np.zeros(len(original_input), dtype=np.int64)
        for i, class_name in enumerate(class_names):
            mask = np.equal(ground_truth, class_name)
            class_index[mask] = i
        class_index_t = torch.tensor(class_index, device=device, dtype=torch.long)

        indicator = np.ones(len(factuals))
        sigma = torch.full(
            (len(factuals),), self.sigma_val, device=device, dtype=torch.float32
        )
        temperature = torch.full(
            (len(factuals),), self.temp_val, device=device, dtype=torch.float32
        )
        distance_weight = torch.full(
            (len(factuals),),
            self.distance_weight_val,
            device=device,
            dtype=torch.float32,
        )

        best_distance = np.full(len(factuals), 1000.0)
        best_perturb = np.zeros_like(original_input)

        for _ in range(self.n_iter):
            indicator_t = torch.tensor(indicator, device=device, dtype=torch.float32)
            optimizer.zero_grad()

            p_model = _filter_hinge_loss(
                self.n_class,
                indicator,
                perturbed,
                sigma,
                temperature,
                self._prob_from_input,
            )
            approx_prob = p_model[
                torch.arange(len(factuals), device=device), class_index_t
            ]

            eps = 1.0e-10
            distance = distance_func(
                self.distance_function, perturbed, original_tensor, eps
            )

            prediction_loss = indicator_t * approx_prob
            distance_loss = distance_weight * distance
            total_loss = torch.mean(prediction_loss + distance_loss)

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                perturbed.clamp_(0.0, 1.0)

            true_distance = (
                distance_func(self.distance_function, perturbed, original_tensor, 0)
                .detach()
                .cpu()
                .numpy()
            )
            current_predict = self.model.predict(
                perturbed.detach().cpu().numpy()
            ).reshape(-1)
            indicator = np.equal(ground_truth, current_predict).astype(np.float64)

            mask_flipped = np.not_equal(ground_truth, current_predict)
            mask_smaller_dist = np.less(true_distance, best_distance)

            temp_dist = best_distance.copy()
            temp_dist[mask_flipped] = true_distance[mask_flipped]
            best_distance[mask_smaller_dist] = temp_dist[mask_smaller_dist]

            temp_perturb = best_perturb.copy()
            temp_perturb[mask_flipped] = perturbed.detach().cpu().numpy()[mask_flipped]
            best_perturb[mask_smaller_dist] = temp_perturb[mask_smaller_dist]

        df_cfs = pd.DataFrame(best_perturb, columns=self.model.data.continuous)
        df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs

    def _prob_from_input(self, perturbed, sigma, temperature):
        feat_columns = self.model.data.continuous
        if not isinstance(self.model.raw_model, DecisionTreeClassifier):
            return trees.get_prob_classification_forest(
                self.model,
                feat_columns,
                perturbed,
                sigma=sigma,
                temperature=temperature,
            )
        return trees.get_prob_classification_tree(
            self.model.raw_model, feat_columns, perturbed, sigma=sigma
        )
