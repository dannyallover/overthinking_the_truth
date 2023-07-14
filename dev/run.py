import sys
import time
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

sys.path.append("../dev")
from model import *
from prefixes import *
from metrics import *

def run_logit_lens(
    model: AutoModelForCausalLM,
    prefixes: Prefixes,
    abl_params: dict = None,
) -> dict:
    """
    Generate the layerwise metrics for the |data| and |model|, and intervene on
    the model as prescribed by |abl_params|.

    Parameters
    ----------
    model : required, AutoModelForCausalLM
        Model to run inference on.
    prefixes : required, Prefixes
        The built prefixes.
    abl_params : optional, dict
        The ablation parameters (e.g. hook, heads).

    Returns
    ------
    metrics : dict
        The layerwise metrics: label_space_probs, top_num_labels_match,
        top_1_acc, correct_over_incorrect, cal_correct_over_incorrect, and
        cal_permute.
    """
    start = time.time()

    print(
        "*Run inference and get top 1 intermediate logit, top number of labels intermediate logits, probabilities of labels, and normalized probabilities of labels.*",
        flush=True,
    )
    (
        top_1_logit,
        top_num_labels_logits,
        probs,
        norm_probs,
    ) = get_label_probs_and_top_logits(model, prefixes, abl_params)

    print("*Compute top_1_acc metric.*", flush=True)
    top_1_acc = get_top_1_acc(
        top_1_logit, prefixes.true_prefixes_labels, prefixes.lab_first_token_ids
    )

    print("*Compute top_num_labels_match metric.*", flush=True)
    top_num_labels_match = get_top_num_labels_match(
        top_num_labels_logits, prefixes.lab_first_token_ids
    )

    print("*Compute label_space metric.*", flush=True)
    label_space_probs = mean_up_low(torch.sum(probs, -1))

    print("*Get probability of correct labels.*", flush=True)
    correct_label_probs = get_correct_label_probs(probs, prefixes.true_prefixes_labels)

    print("*Compute correct_over_incorrect metric.*", flush=True)
    n_labels = norm_probs.shape[-1]
    correct_over_incorrect = (
        correct_label_probs >= torch.max(probs, -1).values
    ).float()
    correct_over_incorrect = mean_up_low(correct_over_incorrect)
    del correct_label_probs

    print("*Get quantile probabilities of labels for calibration.*", flush=True)
    n_labels = norm_probs.shape[-1]
    quantiles, means = get_thresholds(norm_probs, n_labels)

    print("*Compute cal_correct_over_incorrect metric.*", flush=True)
    cal_correct_over_incorrect = get_cal_correct_over_incorrect(
        norm_probs, quantiles, prefixes.true_prefixes_labels
    )

    print("*Compute cal_permute metric.*", flush=True)
    cal_permute = get_cal_permute(norm_probs, quantiles, prefixes.false_prefixes_labels)
    del norm_probs
    del probs
    del quantiles
    del means

    metrics = {
        "label_space_probs": label_space_probs,
        "top_num_labels_match": top_num_labels_match,
        "top_1_acc": top_1_acc,
        "correct_over_incorrect": correct_over_incorrect,
        "cal_correct_over_incorrect": cal_correct_over_incorrect,
        "cal_permute": cal_permute,
    }

    end = time.time()
    print(f"Total time to run: {end - start}.", flush=True)

    return metrics

def run_attn(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefixes: Prefixes,
) -> pd.DataFrame:
    """
    Generate the attention metrics (see below).

    Parameters
    ----------
    model : required, AutoModelForCausalLM
        Model to run inference on.
    tokenizer : required, AutoTokenizer
        Tokenizer corresponding to |model|.
    prefixes : required, Prefixes
        The built prefixes.

    Returns
    ------
    attn_metrics_df : pd.DataFrame
        The attention metrics (e.g. PM score).
    """
    start = time.time()

    print("*Run inference and get attention weights.*", flush=True)
    attn_weights = get_attn_weights(
        model,
        prefixes,
    )

    print("*Compute context following scores.*", flush=True)
    labels = prefixes.true_prefixes_labels + prefixes.false_prefixes_labels
    tok_label_indx = (
        prefixes.true_prefixes_tok_label_indx + prefixes.false_prefixes_tok_label_indx
    )
    label_indx_vals = []
    for indices, labs in zip(tok_label_indx, labels):
        inpt = []
        for indxx, lab in zip(indices, labs):
            cntxt = []
            for indx in indxx:
                cntxt.append((indx, lab))
            inpt.append(cntxt)
        label_indx_vals.append(inpt)
    n_labels = prefixes.num_labels
    context_following_scores = get_context_following_scores(
        attn_weights, label_indx_vals, n_labels
    )

    print("*Average metrics over inputs and massage data into df.*", flush=True)
    indx = ["prefix_type", "n_inputs", "demo_indx", "layer_indx", "head_indx"]
    metrics = [
        "cfs_lab_same",
        "cfs_lab",
        "cfs_prec_lab_same",
        "cfs_prec_lab",
        "cfs_lab_ratio",
        "cfs_prec_lab_ratio",
        "cfs_lab_prime",
        "cfs_prec_lab_prime",
    ]
    group_col_names = ["demo_indx", "prefix_type", "layer_indx", "head_indx"]
    attn_metrics_df = get_attn_metrics_df(
        context_following_scores, indx, metrics, group_col_names, "mean"
    ).fillna(0)

    end = time.time()
    print("total time to run: ", end - start, flush=True)

    del attn_weights
    del context_following_scores
    torch.cuda.empty_cache()

    return attn_metrics_df