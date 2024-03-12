import torch
from einops import rearrange
from prefixes import Prefixes
from typing import Callable
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

def get_model_and_tokenizer(
    file_path: str,
    device: int = 0,
) -> (AutoModelForCausalLM, AutoTokenizer):
    """
    Get the model and tokenizer corresponding to |model_name|.

    Parameters
    ----------
    file_path : required, str
        File path to the model.
    device : optional, int
        Device to put model on.

    Returns
    ------
    model : AutoModelForCausalLM
        Model corresponding to |file_path|.
    tokenizer : AutoTokenizer
        Tokenizer corresponding to |file_path|.
    """
    if file_path == "EleutherAI/gpt-neox-20b":
        model = AutoModelForCausalLM.from_pretrained(
            file_path, torch_dtype=torch.float16
        ).to(f"cuda:{device}")
    else:
        model = AutoModelForCausalLM.from_pretrained(file_path).to(f"cuda:{device}")
    tokenizer = AutoTokenizer.from_pretrained(file_path)

    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def get_layernorm(
    model: AutoModelForCausalLM,
) -> torch.nn.LayerNorm:
    """
    Get the final layer norm of |model|.

    Parameters
    ----------
    model : required, AutoModelForCausalLM
        Language model.

    Returns
    ------
    _ : torch.nn.LayerNorm
        Final layer norm.
    """
    if hasattr(model, "transformer"):
        return model.transformer.ln_f
    elif hasattr(model, "gpt_neox"):
        return model.gpt_neox.final_layer_norm
    return None

def get_unembed_matrix(
    model: AutoModelForCausalLM,
) -> torch.nn.Linear:
    """
    Get the unembedding matrix of |model|.

    Parameters
    ----------
    model : required, AutoModelForCausalLM
        Language model.

    Returns
    ------
    _ : torch.nn.Linear
        Unembedding matrix.
    """
    if hasattr(model, "transformer"):
        return model.lm_head
    elif hasattr(model, "gpt_neox"):
        return model.embed_out
    return None

def register_head_hooks(
    model: AutoModelForCausalLM,
    layer_heads: dict,
    hook_fn: Callable,
    index: int,
) -> list:
    """
    Register |hook_fn| on |model| for |layer_heads|.

    Parameters
    ----------
    model : required, AutoModelForCausalLM
        Language model.
    layer_heads : required, dict
        Dictionary of layer/heads to ablate.
    hook_fn : required, Callable
        Ablation hook function.
    index : required, int
        Position of the last token.

    Returns
    ------
    hooks : list
        List of registered forward hooks.
    """
    hooks = []
    for l, h in layer_heads.items():
        if hasattr(model, "transformer") and hasattr(
            model.transformer.h[l].attn, "out_proj"
        ):
            hooks += [
                model.transformer.h[l].attn.k_proj.register_forward_hook(
                    hook_fn(h, index)
                ),
                model.transformer.h[l].attn.v_proj.register_forward_hook(
                    hook_fn(h, index)
                ),
                model.transformer.h[l].attn.q_proj.register_forward_hook(
                    hook_fn(h, index)
                ),
            ]
    return hooks

def register_attention_hooks(
    model: AutoModelForCausalLM,
    layers: list,
    hook_fn: Callable,
    index: int,
) -> list:
    """
    Register |hook_fn| on |model| for |layers|.

    Parameters
    ----------
    model : required, AutoModelForCausalLM
        Language model.
    layer_heads : required, dict
        Dictionary of layer/heads to ablate.
    hook_fn : required, Callable
        Ablation hook function.
    index : required, int
        Position of the last token.

    Returns
    ------
    hooks : list
        List of registered forward hooks.
    """
    hooks = []
    for l in layers:
        if hasattr(model, "gpt_neox"):
            hooks.append(
                model.gpt_neox.layers[
                    l
                ].attention.query_key_value.register_forward_hook(hook_fn(index))
            )
        elif hasattr(model.transformer.h[l].attn, "out_proj"):
            hooks.append(
                model.transformer.h[l].attn.out_proj.register_forward_hook(
                    hook_fn(index)
                )
            )
        elif hasattr(model.transformer.h[l].attn, "c_attn"):
            hooks.append(
                model.transformer.h[l].attn.c_attn.register_forward_hook(hook_fn(index))
            )
    return hooks

def register_mlp_hooks(
    model: AutoModelForCausalLM,
    layers: list,
    hook_fn: Callable,
    index: int,
) -> list:
    """
    Register |hook_fn| on |model| for |layers|.

    Parameters
    ----------
    model : required, AutoModelForCausalLM
        Language model.
    layer_heads : required, dict
        Dictionary of layer/heads to ablate.
    hook_fn : required, Callable
        Ablation hook function.
    index : required, int
        Position of the last token.

    Returns
    ------
    hooks : list
        List of registered forward hooks.
    """
    hooks = []
    for l in layers:
        if hasattr(model, "gpt_neox"):
            hooks.append(
                model.gpt_neox.layers[l].mlp.act.register_forward_hook(hook_fn(index))
            )
        elif hasattr(model, "transformer"):
            hooks.append(
                model.transformer.h[l].mlp.act.register_forward_hook(hook_fn(index))
            )
    return hooks

def get_label_probs_and_top_logits(
    model: AutoModelForCausalLM,
    prefixes: Prefixes,
    abl_params: dict = None,
    device: int = 0,
) -> torch.Tensor:
    """
    Run inference on the |model| and get top_1_logit, top_num_labels_logits,
    probs, norm_prob, while intervening on the model as prescribed by |abl_params|.

    Parameters
    ----------
    model : required, AutoModelForCausalLM
        Model to run inference on.
    prefixes : required, Prefixes
        The built prefixes.
    abl_params : optional, dict
        The ablation parameters (e.g. hook, heads).
    device : optional, int
        Device to put input on.

    Returns
    ------
    hidden : torch.Tensor
        Hidden states for the true_prefixes and false_prefixes, of each input,
        at each position in context, and at each layer.
    """
    tokenized_inputs = prefixes.true_prefixes_tok + prefixes.false_prefixes_tok
    tok_prec_label_indx = (
        prefixes.true_prefixes_tok_prec_label_indx
        + prefixes.false_prefixes_tok_prec_label_indx
    )
    lab_first_token_ids = prefixes.lab_first_token_ids
    ln = get_layernorm(model)
    vocab_matrix = get_unembed_matrix(model).weight

    top_1_logit, top_num_labels_logits, probs, norm_probs = [], [], [], []
    with torch.no_grad():
        for i, (t_inp, indices) in enumerate(
            zip(tokenized_inputs, tok_prec_label_indx)
        ):
            if i % 10 == 0:
                print(f"{i}/{len(tokenized_inputs)} inputs processed", flush=True)

            if abl_params:
                if abl_params["type"] == "heads":
                    hooks = register_head_hooks(
                        model,
                        abl_params["layer_heads"][i],
                        abl_params["hook"],
                        indices[-1],
                    )
                elif abl_params["type"] == "attention":
                    hooks = register_attention_hooks(
                        model,
                        abl_params["layers"],
                        abl_params["hook"],
                        indices[-1],
                    )
                elif abl_params["type"] == "mlp":
                    hooks = register_mlp_hooks(
                        model,
                        abl_params["layers"],
                        abl_params["hook"],
                        indices[-1],
                    )

            out = model(**t_inp.to(f"cuda:{device}"), output_hidden_states=True)
            hidden = torch.stack(out.hidden_states, dim=1)
            hidden = hidden[:, :, indices, :]
            hidden = ln(hidden)

            intermediate_logits = torch.einsum("bldh,vh->bldv", hidden, vocab_matrix)
            intermediate_probs = torch.nn.functional.softmax(intermediate_logits, dim=3)

            probs_ = intermediate_probs[:, :, :, lab_first_token_ids]
            norm_probs_ = (probs_ + 1e-14) / (
                probs_.sum(dim=3, keepdim=True) + (1e-14 * probs_.shape[3])
            )
            probs.append(probs_)
            norm_probs.append(norm_probs_)

            top_1_logit_ = intermediate_logits.topk(1, dim=3).indices
            top_num_labels_logits_ = intermediate_logits.topk(
                len(lab_first_token_ids), dim=3
            ).indices
            top_1_logit.append(top_1_logit_)
            top_num_labels_logits.append(top_num_labels_logits_)

            if abl_params:
                for h in hooks:
                    h.remove()

    top_1_logit, top_num_labels_logits, probs, norm_probs = [
        rearrange(
            torch.cat(l),
            "(n_prefix n_inputs) n_layer n_demos lab_space_size -> "
            "n_prefix n_inputs n_layer n_demos lab_space_size",
            n_prefix=2,
            n_inputs=len(l) // 2,
        )
        for l in [top_1_logit, top_num_labels_logits, probs, norm_probs]
    ]

    return (top_1_logit, top_num_labels_logits, probs, norm_probs)

def get_attn_weights(
    model: AutoModelForCausalLM,
    prefixes: Prefixes,
    device: int = 0,
) -> torch.Tensor:
    """
    Run |model| inference and get the attention weights.

    Parameters
    ----------
    model : required, AutoModelForCausalLM
        Model to run inference on.
    prefixes : required, Prefixes
        The built prefixes.
    device : optional, int
        Device to put input on.

    Returns
    ------
    attn : torch.Tensor
        Tensor of dimension (n_pfx, n_inputs, n_layers, n_heads, n_tokens,
        n_tokens) containing the attention weights.
    """
    tokenized_inputs = prefixes.true_prefixes_tok + prefixes.false_prefixes_tok
    tok_label_indx = (
        prefixes.true_prefixes_tok_label_indx + prefixes.false_prefixes_tok_label_indx
    )
    tok_label_indx_flat = [
        [indx for contxt in prfx for indx in contxt] for prfx in tok_label_indx
    ]
    sample_out = model(
        **tokenized_inputs[0].to(f"cuda:{device}"), output_attentions=True
    )
    n_layers, n_heads, n_inputs, token_pad = (
        len(sample_out.attentions),
        sample_out.attentions[0].shape[1],
        len(tokenized_inputs),
        200,
    )

    padded_attn = torch.zeros((n_inputs, n_layers, n_heads, token_pad, token_pad))
    with torch.no_grad():
        for i, (t_inp, indices) in enumerate(
            zip(tokenized_inputs, tok_label_indx_flat)
        ):
            if i % 10 == 0:
                print(f"{i}/{len(tokenized_inputs)} inputs processed", flush=True)

            out = model(**t_inp.to(f"cuda:{device}"), output_attentions=True)
            attn = torch.cat(out.attentions)
            padded_attn[i, :, :, : len(indices), : len(indices)] = attn[:, :, indices][
                :, :, :, indices
            ]

    attn = rearrange(
        padded_attn,
        "(n_prefix n_inputs) n_layer n_heads token_pad_x token_pad_y -> "
        "n_prefix n_inputs n_layer n_heads token_pad_x token_pad_y",
        n_prefix=2,
        n_inputs=n_inputs // 2,
    )

    return attn