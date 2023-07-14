import sys
import pickle
import os
import argparse

sys.path.append("../../dev")
from utils import *
from hooks import *
from model import *
from prefixes import *
from metrics import *
from run import *
from visualize import *

sys.path.append("../../model")
from model_params import *

sys.path.append("../../data")
from dataset_params import *
from prompt_params import *
from demo_params import *

ALL_ABLATION_TYPES = ["heads", "attention", "mlp"]

def run_job(
    models: list,
    datasets: list,
    settings: list,
    abl_types: list,
    num_inputs: int,
    num_demos: int,
):
    print("Running job with the following arguments:", flush=True)
    print("Models:", models, flush=True)
    print("Datasets:", datasets, flush=True)
    print("Settings:", settings, flush=True)
    print("Number of Inputs:", num_inputs, flush=True)
    print("Number of Demonstrations:", num_demos, flush=True)

    for model_name in models:
        model_params = MODEL_PARAMS[model_name]
        model, tokenizer = get_model_and_tokenizer(model_params["file_path"], 0)
        for setting in settings:
            demo_params = DEMO_PARAMS[setting]

            if num_demos == "max":
                n_demos = model_params["max_demos"]
            else:
                n_demos = int(num_demos)

            metrics = {}
            for dataset_name in datasets:
                dataset_params = DATASET_PARAMS[dataset_name]
                prompt_params = PROMPT_PARAMS[dataset_name]

                prefixes = Prefixes(
                    get_dataset(dataset_params),
                    prompt_params[0],
                    demo_params,
                    model_params,
                    tokenizer,
                    num_inputs,
                    n_demos,
                )

                for abl_type in abl_types:
                    if abl_type == "heads":
                        metrics["heads_ours"] = run_head_ablation(
                            model, model_params, prefixes, "heads_ours"
                        )
                        metrics["heads_null"] = run_head_ablation(
                            model, model_params, prefixes, "heads_null"
                        )
                    elif abl_type == "attention":
                        metrics["attention"] = run_attn_or_mlp_ablation(
                            model, model_params, prefixes, "attention"
                        )
                    elif abl_type == "mlp":
                        metrics["mlp"] = run_attn_or_mlp_ablation(
                            model, model_params, prefixes, "mlp"
                        )

                np.save(
                    f"../../results/ablations/{model_name}/{setting}/{dataset_name}.npy",
                    metrics,
                    allow_pickle=True,
                )
                torch.cuda.empty_cache()
    return

def get_random_heads(model_params: dict, num_heads_select: int, num_inputs: int):
    num_heads = model_params["num_heads"]
    num_layers = model_params["num_layers"]
    layer_heads = []
    for i in range(num_inputs):
        samp = {}
        count = 0
        while count < num_heads_select:
            layer, head = (
                random.randint(0, num_heads),
                random.randint(0, num_layers),
            )
            if layer not in samp:
                samp[layer] = []
            if head not in samp[layer]:
                samp[layer].append(head)
                count += 1
        layer_heads.append(samp)
    return layer_heads

def get_our_heads(model_params: dict, num_inputs: int):
    model_name = model_params["model_name"]
    with open(
        f"../../results/attention/{model_name}/unnatural_false_induction_heads.pkl",
        "rb",
    ) as file:
        layer_heads = pickle.load(file)
    layer_heads = [layer_heads] * num_inputs
    return layer_heads

def run_head_ablation(
    model: AutoModelForCausalLM, model_params: dict, prefixes: Prefixes, abl_type: str
):
    num_inputs = len(prefixes.true_prefixes) * 2
    if abl_type == "heads_ours":
        layer_heads = get_our_heads(model_params, num_inputs)
    elif abl_type == "heads_null":
        layer_heads = get_random_heads(model_params, 5, num_inputs)

    abl_params = {
        "type": "heads",
        "layer_heads": layer_heads,
        "hook": zero_ablate_heads,
    }
    metrics = run_logit_lens(model, prefixes, abl_params)
    return metrics

def run_attn_or_mlp_ablation(
    model: AutoModelForCausalLM, model_params: dict, prefixes: Prefixes, abl_type: str
):
    num_heads = model_params["num_heads"]
    num_layers = model_params["num_layers"]
    model_name = model_params["model_name"]
    with open(f"../../results/critical_layers/critical_layers.pkl", "rb") as file:
        critical_layers = pickle.load(file)
    critical_layer = critical_layers[model_params["model_name"]]
    layers = [i for i in range(critical_layer, model_params["num_layers"])]

    abl_params = {
        "type": abl_type,
        "layers": layers,
        "hook": zero_ablate,
    }
    metrics = run_logit_lens(model, prefixes, abl_params)
    return metrics

def check_args(
    models: list,
    datasets: list,
    settings: list,
    abl_types: list,
    num_inputs: int,
    num_demos: int,
):
    our_datasets, our_models, our_settings = (
        list(DATASET_PARAMS.keys()),
        list(MODEL_PARAMS.keys()),
        list(DEMO_PARAMS.keys()),
    )

    for model in models:
        if model not in our_models:
            raise Exception("Model specified is not supported.")
    for dataset in datasets:
        if dataset not in our_datasets:
            print(dataset)
            raise Exception("Dataset specified is not supported.")
    for setting in settings:
        if setting not in our_settings:
            raise Exception("Setting specified is not supported.")
    for abl_type in abl_types:
        if abl_type not in ALL_ABLATION_TYPES:
            raise Exception("Ablation type specified is not supported.")

    if (not num_demos.isdigit() and num_demos != "max") or (
        num_demos.isdigit() and (int(num_demos) > 40 or int(num_demos) < 0)
    ):
        raise Exception(
            "Number of demonstrations must be in the range[0, 40] or use keyword 'max'."
        )
    elif num_inputs < 0:
        raise Exception("Number of inputs must be greater than 0.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define command line arguments.
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Models to run (use 'all' to run all models).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Datasets to run (use 'all' to run all datasets).",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default="permuted_incorrect_labels",
        help="Type of settings (e.g. random_incorrect_labels).",
    )
    parser.add_argument(
        "--abl_types", type=str, default="head", help="Type of ablation (e.g. head)."
    )
    parser.add_argument(
        "--num_inputs", type=int, default=1000, help="Number of inputs."
    )
    parser.add_argument(
        "--num_demos", type=str, default="max", help="Number of demonstrations."
    )

    args = parser.parse_args()

    # Check for "all" option and updating arguments accordingly.
    models = (
        list(MODEL_PARAMS.keys()) if args.models == "all" else args.models.split(",")
    )
    datasets = (
        list(DATASET_PARAMS.keys())
        if args.datasets == "all"
        else args.datasets.split(",")
    )
    settings = (
        list(DEMO_PARAMS.keys()) if args.settings == "all" else args.settings.split(",")
    )
    abl_types = (
        ALL_ABLATION_TYPES if args.abl_types == "all" else args.abl_types.split(",")
    )

    check_args(models, datasets, settings, abl_types, args.num_inputs, args.num_demos)

    # Launch the job.
    run_job(models, datasets, settings, abl_types, args.num_inputs, args.num_demos)