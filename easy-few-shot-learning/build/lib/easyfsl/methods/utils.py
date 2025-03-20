from collections import OrderedDict

import torch
from torch import Tensor
from sklearn.svm import LinearSVC
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
'''
def compute_prototypes(support_features: Tensor, support_labels: Tensor) -> Tensor:
    """
    Compute class prototypes from support features and labels
    Args:
        support_features: for each instance in the support set, its feature vector
        support_labels: for each instance in the support set, its label

    Returns:
        for each label of the support set, the average feature vector of instances with this label
    """

    n_way = len(torch.unique(support_labels))
    # Prototype i is the mean of all instances of features corresponding to labels == i
    support_labels = support_labels * torch.std(support_labels, dim=0)
    return torch.cat(
        [
            support_features[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ]
    )
'''
def compute_prototypes(support_features: Tensor, support_labels: Tensor) -> Tensor:
    """
    Compute class prototypes from support features and labels, with a mask based on the maximum pairwise inter-class variance.
    Args:
        support_features: for each instance in the support set, its feature vector (shape: [n_shot * n_way, feature_dim])
        support_labels: for each instance in the support set, its label (shape: [n_shot * n_way])

    Returns:
        for each label of the support set, the average feature vector of instances with this label (shape: [n_way, feature_dim])
    """
    n_way = len(torch.unique(support_labels))
    feature_dim = support_features.shape[1]
    #svc = LinearSVC()
    #svc.fit(support_features.numpy(), support_labels.numpy())
    svc = mutual_info_classif(support_features.cpu().numpy(), support_labels.cpu().numpy(), n_jobs=-1)
    #svc.fit(support_features.numpy(), support_labels.numpy())
    # Compute prototypes (mean feature vectors for each class)
    prototypes = torch.cat(
        [
            support_features[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ]
    )

    # Compute intra-class variance (average variance within each class)
    intra_variances = []
    for label in range(n_way):
        class_features = support_features[support_labels == label]
        intra_var = torch.var(class_features, dim=0).mean()  # Mean variance across features
        intra_variances.append(intra_var)
    intra_variance = torch.mean(torch.stack(intra_variances))  # Average across classes

    # Compute maximum pairwise inter-class variance (using raw features)
    max_inter_var = 0.0
    for i in range(n_way):
        for j in range(i + 1, n_way):
            # Get all features from classes i and j
            features_i = support_features[support_labels == i]
            features_j = support_features[support_labels == j]
            combined_features = torch.cat([features_i, features_j], dim=0)
            # Compute variance of the combined features
            inter_var = torch.var(combined_features, dim=0).mean()  # Mean variance across features
            if inter_var > max_inter_var:
                max_inter_var = inter_var

    # Compute mask as sigmoid(max_inter_var / intra_variance)
    mask = torch.sigmoid(max_inter_var / intra_variance / torch.mean(support_features, dim=0)) # the 0.1 is a hyperparameter called the temperature
    #mask = torch.sigmoid(max_inter_var / intra_variance) # the 0.1 is a hyperparameter called the temperature
    #mask = (max_inter_var / intra_variance) > 1
    #mask = torch.abs(torch.from_numpy(svc.coef_).to(torch.float32))
    #mask = torch.sigmoid(10 * torch.mean(mask, dim=0))
    #threshold = torch.min(torch.topk(torch.tensor(svc), 512).values).item()
    mask = (torch.from_numpy(svc).to(torch.float32) > 0.2)
    mask = mask ** 0.5
    
    #svc = LinearSVC() # try to use gridsearchcv
    #svc.fit((support_features*mask).numpy(), support_labels.numpy())
    
    #print(mask)
    #mask = torch.sigmoid(max_inter_var / intra_variance)
    # it controls the shape of that activation function

    # Apply the mask to the prototypes
    
    #mask = torch.ones_like(mask) # UNCOMMENT this line to evaluate the baseline
    
    #mask = torch.sigmoid(1 / torch.mean(support_features, dim=0))
    masked_prototypes = prototypes.cuda() * mask.cuda()
    return masked_prototypes, mask, svc

def entropy(logits: Tensor) -> Tensor:
    """
    Compute entropy of prediction.
    WARNING: takes logit as input, not probability.
    Args:
        logits: shape (n_images, n_way)
    Returns:
        Tensor: shape(), Mean entropy.
    """
    probabilities = logits.softmax(dim=1)
    return (-(probabilities * (probabilities + 1e-12).log()).sum(dim=1)).mean()


def k_nearest_neighbours(features: Tensor, k: int, p_norm: int = 2) -> Tensor:
    """
    Compute k nearest neighbours of each feature vector, not included itself.
    Args:
        features: input features of shape (n_features, feature_dimension)
        k: number of nearest neighbours to retain
        p_norm: use l_p distance. Defaults: 2.

    Returns:
        Tensor: shape (n_features, k), indices of k nearest neighbours of each feature vector.
    """
    distances = torch.cdist(features, features, p_norm)

    return distances.topk(k, largest=False).indices[:, 1:]


def power_transform(features: Tensor, power_factor: float) -> Tensor:
    """
    Apply power transform to features.
    Args:
        features: input features of shape (n_features, feature_dimension)
        power_factor: power to apply to features

    Returns:
        Tensor: shape (n_features, feature_dimension), power transformed features.
    """
    return (features.relu() + 1e-6).pow(power_factor)


def strip_prefix(state_dict: OrderedDict, prefix: str):
    """
    Strip a prefix from the keys of a state_dict. Can be used to address compatibility issues from
    a loaded state_dict to a model with slightly different parameter names.
    Example usage:
        state_dict = torch.load("model.pth")
        # state_dict contains keys like "module.encoder.0.weight" but the model expects keys like "encoder.0.weight"
        state_dict = strip_prefix(state_dict, "module.")
        model.load_state_dict(state_dict)
    Args:
        state_dict: pytorch state_dict, as returned by model.state_dict() or loaded via torch.load()
            Keys are the names of the parameters and values are the parameter tensors.
        prefix: prefix to strip from the keys of the state_dict. Usually ends with a dot.

    Returns:
        copy of the state_dict with the prefix stripped from the keys
    """
    return OrderedDict(
        [
            (k[len(prefix) :] if k.startswith(prefix) else k, v)
            for k, v in state_dict.items()
        ]
    )
