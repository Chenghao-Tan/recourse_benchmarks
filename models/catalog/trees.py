import numpy as np
import torch
import xgboost
import xgboost.core
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from models.catalog.parse_xgboost import parse_booster


def _to_tensor(x, like=None):
    if torch.is_tensor(x):
        return x
    if like is not None:
        return torch.tensor(x, device=like.device, dtype=like.dtype)
    return torch.tensor(x, dtype=torch.float32)


def _split_approx(node, feat_input, feat_index, threshold, sigma):
    """
    Approximate the decision tree split using a sigmoid function.
    """

    def _approx_activation_by_index(feat_input, feat_index, threshold, sigma):
        x = feat_input[:, feat_index] - threshold
        activation = torch.sigmoid(x * sigma)
        return 1.0 - activation, activation

    if node is None:
        node = 1.0

    l_n, r_n = _approx_activation_by_index(feat_input, feat_index, threshold, sigma)
    return node * l_n, node * r_n


def _parse_class_tree(tree, feat_columns, feat_input, split_function):
    if isinstance(tree, xgboost.core.Booster):  # XGBoost
        children_left, children_right, threshold, feature, scores = parse_booster(tree)

        # feature needs to be list of ints, not string
        # -2 is the indicator of a leaf node
        feature = [tree.feature_names.index(f) if not f == "-2" else f for f in feature]

        # TODO make n_classes not hardcoded?
        n_classes = 2
        n_nodes = len(scores)
        values = None
    else:  # Sklearn
        # Code is adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        n_classes = len(tree.classes_)
        n_nodes = tree.tree_.node_count
        values = tree.tree_.value
        scores = None

    nodes = [None] * n_nodes
    leaf_nodes = [[] for _ in range(n_classes)]
    for i in range(n_nodes):
        cur_node = nodes[i]
        if children_left[i] != children_right[i]:  # split node
            l_n, r_n = split_function(cur_node, feat_input, feature[i], threshold[i])
            nodes[children_left[i]] = l_n
            nodes[children_right[i]] = r_n
        else:  # leaf node
            if isinstance(tree, xgboost.core.Booster):
                # TODO score to class depends on the objective function of the XGBoost model
                max_class = int(scores[i] > 0.5)
            else:
                max_class = int(np.argmax(values[i]))
            leaf_nodes[max_class].append(cur_node)

    return leaf_nodes, n_nodes, n_classes


def _sum_tensors(tensors, like):
    if len(tensors) == 0:
        return torch.zeros(like.shape[0], device=like.device, dtype=like.dtype)
    return torch.stack(tensors, dim=0).sum(dim=0)


def get_prob_classification_tree(tree, feat_columns, feat_input, sigma):
    """
    class probability for input
    """
    feat_input_t = _to_tensor(feat_input)

    def split_function(node, feat_input, feat_index, threshold):
        sigma_t = _to_tensor(sigma, like=feat_input_t)
        return _split_approx(node, feat_input, feat_index, threshold, sigma_t)

    # leaf nodes has a value for each feature
    leaf_nodes, n_nodes, n_classes = _parse_class_tree(
        tree, feat_columns, feat_input_t, split_function
    )
    if n_nodes > 1:  # tree has multiple nodes
        empty_idx = next((i for i, v in enumerate(leaf_nodes) if len(v) == 0), None)
        if empty_idx is not None:  # if no leaf predicts this class
            # TODO only works for 2 classes
            out_l = [None] * 2
            other = 1 - empty_idx
            out_l[other] = _sum_tensors(leaf_nodes[other], feat_input_t[:, 0])
            out_l[empty_idx] = 1.0 - out_l[other]
        else:
            out_l = [
                _sum_tensors(leaf_nodes[c_i], feat_input_t[:, 0])
                for c_i in range(n_classes)
            ]
        stacked = torch.stack(out_l, dim=-1)

    else:  # sometimes tree only has one node
        if isinstance(tree, xgboost.core.Booster):
            dm = xgboost.DMatrix(
                feat_input_t[0:1].detach().cpu().numpy(), feature_names=feat_columns
            )
            pred = tree.predict(dm)
            only_class = float(pred[0] > 0.5)
        else:
            pred = tree.predict(feat_input_t[0:1].detach().cpu().numpy())
            only_class = float(pred[0])

        correct_class = torch.ones(
            feat_input_t.shape[0], device=feat_input_t.device, dtype=feat_input_t.dtype
        )
        incorrect_class = torch.zeros_like(correct_class)
        if only_class == 1.0:
            class_0 = incorrect_class
            class_1 = correct_class
        elif only_class == 0.0:
            class_0 = correct_class
            class_1 = incorrect_class
        else:
            raise ValueError
        stacked = torch.stack([class_0, class_1], dim=1)
    return stacked


def get_prob_classification_forest(
    model, feat_columns, feat_input, number_trees=100, sigma=10.0, temperature=1.0
):
    feat_input_t = _to_tensor(feat_input)

    def tree_parser(tree):
        """parse and individual tree"""
        return get_prob_classification_tree(tree, feat_columns, feat_input_t, sigma)

    if model.backend == "sklearn":
        tree_l = [
            tree_parser(estimator) for estimator in model.tree_iterator.estimators_
        ][:number_trees]
    elif model.backend == "xgboost":
        tree_l = [tree_parser(estimator) for estimator in model.tree_iterator][
            :number_trees
        ]
    else:
        raise Exception("model not supported")

    if isinstance(model.tree_iterator, AdaBoostClassifier):
        weights = model.tree_iterator.estimator_weights_
    elif isinstance(model.tree_iterator, RandomForestClassifier) or isinstance(
        model.tree_iterator[0], xgboost.core.Booster
    ):
        weights = np.full(
            len(model.tree_iterator),
            1 / len(model.tree_iterator),
        )
    else:
        raise Exception("model not supported")

    logits = None
    for w, tree in zip(weights, tree_l):
        w_t = torch.tensor(w, device=tree.device, dtype=tree.dtype)
        logits = tree * w_t if logits is None else logits + tree * w_t

    if isinstance(temperature, (float, int)):
        expits = torch.exp(temperature * logits)
    else:
        temp_t = _to_tensor(temperature, like=logits).view(-1, 1)
        expits = torch.exp(temp_t * logits)

    softmax = expits / torch.sum(expits, dim=1, keepdim=True)

    return softmax
