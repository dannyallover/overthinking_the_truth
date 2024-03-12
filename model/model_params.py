gpt2 = {
    "max_token_len": 1024,
    "max_demos": 20,
    "num_layers": 12,
    "file_path": "gpt2",
    "model_name": "gpt2",
}

gpt2_medium = {
    "max_token_len": 1024,
    "max_demos": 20,
    "num_layers": 24,
    "file_path": "gpt2-medium",
    "model_name": "gpt2_medium",
}

gpt2_large = {
    "max_token_len": 1024,
    "max_demos": 20,
    "num_layers": 36,
    "file_path": "gpt2-large",
    "model_name": "gpt2_large",
}

gpt2_xl = {
    "max_token_len": 1024,
    "max_demos": 20,
    "num_layers": 48,
    "num_heads": 25,
    "head_dim": 4800,
    "mlp_dim": 6400,
    "file_path": "gpt2-xl",
    "model_name": "gpt2_xl",
}

gpt_j = {
    "max_token_len": 2048,
    "max_demos": 40,
    "num_layers": 28,
    "num_heads": 16,
    "head_dim": 4096,
    "mlp_dim": 16384,
    "file_path": "EleutherAI/gpt-j-6B",
    "model_name": "gpt_j",
}

gpt_neox_20B = {
    "max_token_len": 2048,
    "max_demos": 40,
    "num_layers": 44,
    "num_heads": 65,
    "head_dim": 18432,
    "mlp_dim": 24576,
    "file_path": "EleutherAI/gpt-neox-20b",
    "model_name": "gpt_neox",
}

pythia_410M = {
    "max_token_len": 2048,
    "max_demos": 40,
    "num_layers": 24,
    "file_path": "EleutherAI/pythia-410m",
    "model_name": "pythia_410M",
}

pythia_2p8B = {
    "max_token_len": 2048,
    "max_demos": 40,
    "num_layers": 32,
    "file_path": "EleutherAI/pythia-2.8b",
    "model_name": "pythia_2p8B",
}

pythia_6p9B = {
    "max_token_len": 2048,
    "max_demos": 40,
    "num_layers": 32,
    "file_path": "EleutherAI/pythia-6.9b",
    "model_name": "pythia_6p9B",
}

pythia_12B = {
    "max_token_len": 2048,
    "max_demos": 40,
    "num_layers": 36,
    "file_path": "EleutherAI/pythia-12b",
    "model_name": "pythia_12B",
}

gpt2_instruct = {
    "max_token_len": 1024,
    "max_demos": 20,
    "num_layers": 12,
    "file_path": "vicgalle/gpt2-open-instruct-v1",
    "model_name": "gpt2_instruct",
}

gpt_j_instruct = {
    "max_token_len": 2048,
    "max_demos": 40,
    "num_layers": 28,
    "file_path": "nlpcloud/instruct-gpt-j-fp16",
    "model_name": "gpt_j_instruct",
}

gpt_neox_20B_instruct = {
    "max_token_len": 2048,
    "max_demos": 40,
    "num_layers": 44,
    "file_path": "jordiclive/instruction-tuned-gpt-neox-20b",
    "model_name": "gpt_neox_instruct",
}

pythia_6p9B_instruct = {
    "max_token_len": 2048,
    "max_demos": 40,
    "num_layers": 32,
    "file_path": "allenai/open-instruct-pythia-6.9b-tulu",
    "model_name": "pythia_6p9B_instruct",
}

llama_7B = {
    "max_token_len": 2048,
    "max_demos": 40,
    "num_layers": 32,
    "file_path": "allenai/open-instruct-pythia-6.9b-tulu",
    "model_name": "pythia_6p9B_instruct",
}

llama2_7B = {
    "max_token_len": 2048,
    "max_demos": 40,
    "num_layers": 32,
    "file_path": "NousResearch/Llama-2-7b-hf",
    "model_name": "llama2_7B",
}

MODEL_PARAMS = {
    "gpt2": gpt2,
    "gpt2_medium": gpt2_medium,
    "gpt2_large": gpt2_large,
    "gpt2_xl": gpt2_xl,
    "gpt_j": gpt_j,
    "gpt_neox_20B": gpt_neox_20B,
    "pythia_410M": pythia_410M,
    "pythia_2p8B": pythia_2p8B,
    "pythia_6p9B": pythia_6p9B,
    "pythia_12B": pythia_12B,
    "llama2_7b": llama2_7B,
    "gpt2_instruct": gpt2_instruct,
    "gpt_j_instruct": gpt_j_instruct,
    "gpt_neox_20B_instruct": gpt_neox_20B_instruct,
    "pythia_6p9B_instruct": pythia_6p9B_instruct,
}
