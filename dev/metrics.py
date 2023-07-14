import numpy as np
import pandas as pd
import itertools
import torch
from utils import mean_up_low
from einops import repeat, rearrange

def get_correct_label_probs(
    probs: torch.Tensor,
    prefixes_true_labels: list,
) -> torch.Tensor:
    """
    Get the probabilities in |probs| of the correct labels indicated by
    |prefix_true_labels|.

    Parameters
    ----------
    probs : required, torch.Tensor
        Unormalized token probabilities.
    prefixes_true_labels : required, list
        Correct labels corresponding to each position in context for each
        prefix.

    Returns
    ------
    all_correct_probs : torch.Tensor
        Token probabilities corresponding to the correct label.
    """
    n_layers, n_labels = probs.shape[2], probs.shape[4]

    gather_idx_correct = []
    for labels in prefixes_true_labels:
        idx_list = [lab * torch.ones((1, n_layers, 1, 1)) for lab in labels]
        idx = torch.cat(idx_list, dim=2)
        gather_idx_correct.append(idx)
    gather_idx_correct = torch.cat(gather_idx_correct, dim=0).type(torch.int64)
    gather_idx_correct = torch.stack(2 * [gather_idx_correct], dim=0).to(
        f"cuda:{probs.get_device()}"
    )

    all_correct_probs = torch.gather(probs, 4, gather_idx_correct).squeeze(4)
    return all_correct_probs

def get_thresholds(
    norm_probs: torch.Tensor, n_labels: int
) -> (torch.Tensor, torch.Tensor):
    """
    Get the (|n_labels| - 1) / |n_labels| quantile and |mean| of the normalized
    probabilities for each label at each posiition in context and layer.

    Parameters
    ----------
    norm_probs : required, torch.Tensor
        Normalized token probabilities.
    n_labels : required, int
        Number of labels corresponding to the task.

    Returns
    ------
    quantiles : torch.Tensor
        Quantile probability of each label at each position in context and
        layer.
    means: torch.Tensor
        Mean probability of each label at each position in context and layer.
    """
    quantiles = torch.quantile(
        norm_probs.float(),
        (n_labels - 1) / n_labels,
        dim=1,
        keepdim=True,
    )
    means = norm_probs.mean(dim=1, keepdim=True)

    return quantiles, means

def get_cal_correct_over_incorrect(
    norm_probs: torch.Tensor, quantiles: torch.Tensor, prefixes_labels: list
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Get the percentage where the correct label probability is greater than the
    threshold in |quantiles|, dubbed: cal_correct_over_incorrect.

    Parameters
    ----------
    norm_probs : required, torch.Tensor
        Normalized token probabilities.
    quantiles : torch.Tensor
        Quantile probability of each label at each position in context and
        layer.
    prefixes_labels : required, list
        Correct labels corresponding to each position in context for each
        prefix in |true_prefixes| and |false_prefixes|.

    Returns
    ------
    cal_correct_over_incorrect : np.ndarray
        Percentage of calibrated correct over incorrect at each position in context
        and layer.
    """
    n_layers, n_labels = norm_probs.shape[2], norm_probs.shape[4]

    gather_idx_correct = []
    for labels in prefixes_labels:
        idx_list = [lab * torch.ones((1, n_layers, 1, 1)) for lab in labels]
        idx = torch.cat(idx_list, dim=2)
        gather_idx_correct.append(idx)
    gather_idx_correct = torch.cat(gather_idx_correct, dim=0)
    gather_idx_correct = gather_idx_correct.type(torch.int64)
    gather_idx_correct = torch.stack(2 * [gather_idx_correct], dim=0).to(
        f"cuda:{norm_probs.get_device()}"
    )

    quantiles = quantiles.expand(norm_probs.shape)

    all_quantiles = torch.gather(quantiles, 4, gather_idx_correct).squeeze(4)
    all_norm_probs = torch.gather(norm_probs, 4, gather_idx_correct).squeeze(4)

    cal_correct_over_incorrect = (all_norm_probs > all_quantiles).float() + (
        (1 / n_labels) * (all_norm_probs == all_quantiles).float()
    )
    cal_correct_over_incorrect = mean_up_low(cal_correct_over_incorrect)

    return cal_correct_over_incorrect

def get_cal_permute(
    norm_probs: torch.Tensor, quantiles: torch.Tensor, prefixes_permuted_labels: list
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Get the calibrated permuted score (i.e. when the permuted label is greater
    than the quantile probability), for each position in context and layer.

    Parameters
    ----------
    norm_probs : required, torch.Tensor
        Normalized token probabilities.
    quantiles : torch.Tensor
        Quantile probability of each label at each position in context and
        layer.
    prefixes_permuted_labels : required, list
        Permuted label mapping for the false prefix.

    Returns
    ------
    cal_permute : np.ndarray
        Calibrated permuted score at each position in context and layer.
    """
    n_layers, n_labels = norm_probs.shape[2], norm_probs.shape[4]

    gather_idx_permute = []
    for labels in prefixes_permuted_labels:
        idx_list = [lab * torch.ones((1, n_layers, 1, 1)) for lab in labels]
        idx = torch.cat(idx_list, dim=2)
        gather_idx_permute.append(idx)
    gather_idx_permute = torch.cat(gather_idx_permute, dim=0)
    gather_idx_permute = gather_idx_permute.type(torch.int64)
    gather_idx_permute = torch.stack(2 * [gather_idx_permute], dim=0).to(
        f"cuda:{norm_probs.get_device()}"
    )

    quantiles = quantiles.expand(norm_probs.shape)

    all_quantiles = torch.gather(quantiles, 4, gather_idx_permute).squeeze(4)
    all_norm_probs = torch.gather(norm_probs, 4, gather_idx_permute).squeeze(4)

    cal_permute = (all_norm_probs > all_quantiles).float() + (
        (1 / n_labels) * (all_norm_probs == all_quantiles).float()
    )
    cal_permute = mean_up_low(cal_permute)

    return cal_permute

def get_top_1_acc(
    top_1_logit: torch.Tensor,
    prefixes_labels: list,
    tok_ids: list,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Get the top-1 accuracy for each position in context and layer.

    Parameters
    ----------
    top_1_logit : required, torch.Tensor
        Tensor with top logit for each position in context and layer.
    prefixes_labels : required, list
        Correct labels corresponding to each position in context for each
        prefix in |true_prefixes| and |false_prefixes|.
    tok_ids : required, list
        Token ids corresponding to the labels.

    Returns
    ------
    top_1_acc : np.ndarray
        Top-1 accuracy at each position in context and layer.
    """
    n_layers = top_1_logit.shape[2]
    prefixes_labels_ids = []
    for labels in prefixes_labels:
        ids_list = [tok_ids[lab] * torch.ones((1, n_layers, 1, 1)) for lab in labels]
        ids = torch.cat(ids_list, dim=2)
        prefixes_labels_ids.append(ids)
    prefixes_labels_ids = torch.cat(prefixes_labels_ids, dim=0).type(torch.int64)
    prefixes_labels_ids = torch.stack(2 * [prefixes_labels_ids], dim=0).to(
        f"cuda:{top_1_logit.get_device()}"
    )

    top_1_acc = torch.zeros_like(prefixes_labels_ids)
    acc_at_pos = (
        top_1_logit[:, :, :, :, 0].unsqueeze(4) == prefixes_labels_ids
    ).float()
    top_1_acc = torch.add(top_1_acc, acc_at_pos)
    top_1_acc = mean_up_low(top_1_acc.squeeze(4))

    return top_1_acc

def get_top_num_labels_match(
    top_num_labels_logits: torch.Tensor, tok_ids: list
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Get the top_num_labels_match accuracy (i.e. when the top k=|num_labels|
    logits contain the label space), for each position in context and layer.

    Parameters
    ----------
    top_num_labels_logits : required, torch.Tensor
        Tensor with top k=|num_labels| logits for each position in context
        and layer.
    tok_ids : required, list
        Token ids corresponding to the labels.

    Returns
    ------
    top_num_labels_match : np.ndarray
        TNLM score at each position in context and layer.
    """
    n_inputs, n_layers, n_demos = (
        top_num_labels_logits.shape[1],
        top_num_labels_logits.shape[2],
        top_num_labels_logits.shape[3],
    )
    tok_ids_sorted = torch.tensor(tok_ids)
    tok_ids_sorted, _ = tok_ids_sorted.sort()
    tok_ids_rep = repeat(
        tok_ids_sorted,
        "d -> 2 n_inputs n_layers n_demos d",
        n_inputs=n_inputs,
        n_layers=n_layers,
        n_demos=n_demos,
    ).to(f"cuda:{top_num_labels_logits.get_device()}")

    top_num_labels_logits, _ = torch.sort(top_num_labels_logits)

    top_num_labels_match = torch.all(
        (top_num_labels_logits == tok_ids_rep), dim=4
    ).float()
    top_num_labels_match = mean_up_low(top_num_labels_match)

    return top_num_labels_match

def get_attn_metrics_df(
    metrics: list,
    indx_cols: list,
    metric_cols: list,
    group_cols: list = [],
    agg_func: str = "",
) -> pd.DataFrame:
    """
    Transforms a list of |metrics| tensors to a pandas dataframe with cols
    (|indx_cols|, |metric_cols|).

    Parameters
    ----------
    metrics : required, list
        Tensors that each contain a different metric.
    indx_cols: required, list
        The names corresponding to each dimension in the metric tensor except
        the last.
    metric_cols: required, list
        The names corresponding to the last dimension in the metric tensor
        (i.e. the name of the metric).
    group_cols: optional, list
        Columns to group on.
    agg_func: optional, str
        Aggregate function.

    Returns
    ------
    df : pd.DataFrame
        Pandas dataframe of metrics.
    """
    ranges = [[indx for indx in range(dim_size)] for dim_size in metrics[0].shape]
    indices = np.array([p for p in itertools.product(*ranges)])

    indx_cols_dict = dict(
        [[col_name, indx] for col_name, indx in zip(indx_cols, indices.T)]
    )
    metric_cols_dict = dict(
        [
            [col_name, metric.flatten().cpu().numpy()]
            for col_name, metric in zip(metric_cols, metrics)
        ]
    )
    cols = {**indx_cols_dict, **metric_cols_dict}

    df = pd.DataFrame(cols)
    if group_cols and agg_func == "mean":
        df = df.groupby(group_cols)[metric_cols].mean().reset_index()

    return df

def get_specialized_indices(label_indx_vals: list, context_pos: int) -> tuple:
    """
    Get,
    1) position of all labels before |context_pos| that match with the label at
    |context_pos|,
    2) position of all preceding label tokens at or before |context_pos| that
    match with the label at |context_pos|,
    3) position of all labels before |context_pos|, and
    4) position of all preceding label tokens at or before |context_pos|.

    Parameters
    ----------
    label_indx_vals : required, list
        The positions of the labels and preceding label tokens, coupled with
        the class that they map to.

    Returns
    ------
    _ : tuple
        Tuple containing the same label indices, same preceding label indices,
        label indices, and preceding label indices.
    """
    label_indx_vals_part = label_indx_vals[: context_pos + 1]
    lab_indices_same, prec_lab_indices_same, lab_indices, prec_lab_indices = (
        [],
        [],
        [],
        [],
    )
    indx = 0
    for lab_tups in label_indx_vals_part:
        if lab_tups[0][1] == label_indx_vals_part[-1][-1][1]:
            lab_indices_same += [indx + i + 1 for i in range(len(lab_tups[1:]))]
            prec_lab_indices_same.append(indx)
        lab_indices += [indx + i + 1 for i in range(len(lab_tups[1:]))]
        prec_lab_indices.append(indx)
        indx += len(lab_tups)

    return (
        lab_indices_same[: -(len(label_indx_vals_part[-1]) - 1)],
        lab_indices[: -(len(label_indx_vals_part[-1]) - 1)],
        prec_lab_indices_same[:-1],
        prec_lab_indices[:-1],
    )

def get_context_following_scores(
    attn_weights: torch.Tensor,
    label_indx_vals: list,
    n_labels: int,
) -> list:
    """
    Get the context following scores (cfs) by summing the attention weights
    according to |label_indx_vals|.

    Parameters
    ----------
    attn_weights : required, torch.Tensor
        Attention weights after running model inference.
    label_indx_vals : required, list
        The label at each position in context for every input.

    Returns
    ------
    cfs_list : list
        Contains all four types of context following scores.
    """
    n_pfx, n_inputs, n_layers, n_heads, n_tokens, _ = attn_weights.shape
    cfs_lists = []
    for inpt_indx in range(n_pfx * n_inputs):
        cfs_list = []
        prec_lab_indx = 0
        for contxt_pos in range(len(label_indx_vals[inpt_indx])):
            (
                lab_indices_same,
                lab_indices,
                prec_lab_indices_same,
                prec_lab_indices,
            ) = get_specialized_indices(label_indx_vals[inpt_indx], contxt_pos)
            scores = [
                torch.squeeze(
                    attn_weights[
                        inpt_indx
                        and inpt_indx // n_inputs
                        or 0,
                        inpt_indx % n_inputs,
                        :,
                        :,
                        prec_lab_indx,
                        [indices],
                    ],
                    2,
                )
                for indices in [
                    lab_indices_same,
                    lab_indices,
                    prec_lab_indices_same,
                    prec_lab_indices,
                ]
            ]
            scores = [torch.sum(s, dim=2) for s in scores]
            cfs_list.append(scores)
            prec_lab_indx += len(label_indx_vals[inpt_indx][contxt_pos])
        cfs_list = [
            torch.stack([cfs[k] for cfs in cfs_list]) for k in range(len(cfs_list[0]))
        ]
        cfs_lists.append(cfs_list)

    cfs_lists = [
        torch.stack([cfs_list[k] for cfs_list in cfs_lists])
        for k in range(len(cfs_lists[0]))
    ]
    cfs_lists = [
        rearrange(
            cfs_list,
            "(n_prefix n_inputs) n_demos n_layers n_heads -> n_prefix n_inputs n_demos n_layers n_heads",
            n_prefix=n_pfx,
            n_inputs=n_inputs,
        )
        for cfs_list in cfs_lists
    ]

    cfs_lab_ratio, cfs_prec_lab_ratio = (
        cfs_lists[0] / cfs_lists[1],
        cfs_lists[2] / cfs_lists[3],
    )

    cfs_lab_prime, cfs_prec_lab_prime = (
        cfs_lists[0] - (cfs_lists[1] - cfs_lists[0]) / (n_labels - 1),
        cfs_lists[2] - (cfs_lists[3] - cfs_lists[2]) / (n_labels - 1),
    )

    cfs_lists += [
        cfs_lab_ratio,
        cfs_prec_lab_ratio,
        cfs_lab_prime,
        cfs_prec_lab_prime,
    ]

    return cfs_lists