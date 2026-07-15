import argparse
import os
import random
import re
import time
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union
import pickle
from xml.parsers.expat import model
import numpy as np
import torch
from tqdm import tqdm

__all__ = [
    "register_hooks",
    "clear_forward_hooks",
    "clear_hooks_variables",
    "hooks_postprocessing",
    "set_seed",
    "setup_hooks",
]


# Dictionary to store hidden states
HIDDEN_STATES = {}

# Flow matching step
FLOW_MATCHING_STATE = {'STEP': 0}

# Grasping flag and intervention log
ROBOT_IS_GRASPING = False
INTERVENTION_LOG = []


# Setters and getters
def reset_flow_matching_state() -> None:
    """Reset flow-matching state at episode boundaries."""
    global FLOW_MATCHING_STATE
    FLOW_MATCHING_STATE["STEP"] = 0

def reset_interventions_log() -> None:
    """Reset intervention log at episode boundaries."""
    global INTERVENTION_LOG
    INTERVENTION_LOG.clear()


def set_seed(seed_value=42):
    # Python random seed
    random.seed(seed_value)

    # NumPy random seed
    np.random.seed(seed_value)

    # PyTorch random seed
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def append_item_to_dict_of_list(key: str, value: Any, dictionary: Dict[str, Any]):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]
    return dictionary


def update_dict_of_list(item: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in item.items():
        if k in data:
            data[k].append(v)
        else:
            data[k] = [v]
    return data


def fmatch(name: str, patterns: List[str], exact_match: bool = False) -> bool:
    if exact_match:
        return name in patterns
    else:
        # Convert patterns with '*' to proper regex expressions (where * means "any sequence of characters")
        regex_patterns = [
            re.compile(re.sub(r"\*", ".*", pattern)) for pattern in patterns
        ]
        return any([regex.search(name) for regex in regex_patterns])


def compute_time_left(start_time, iteration: int, num_iterations: int):
    elapsed_time = time.time() - start_time  # Time spent so far
    avg_time_per_iter = elapsed_time / iteration  # Average time per iteration
    remaining_iters = num_iterations - iteration
    time_left = avg_time_per_iter * remaining_iters  # Estimated time left
    return time_left / 60


def get_start_idx_generated_tokens(tokens: List[torch.Tensor]) -> int:
    if isinstance(tokens, list) and len(tokens) > 1:
        total_len = torch.cat(tokens, dim=1).shape[1]
        idx = tokens[0].shape[1] - total_len
    else:
        # teacher forcing mode
        v = v[0]
        idx = 0
    return idx  # generated tokens start after the prompt, count from last


def save_hidden_states_input(module_name: str = "", **kwargs: Any):
    """Save module input hidden states."""
    global HIDDEN_STATES

    def hook(module, input, output):
        data = input[0] if isinstance(input, tuple) else input
        data = data.detach().cpu()
        # print(module_name, data.shape)
        if module_name in HIDDEN_STATES:
            HIDDEN_STATES[module_name].append(data)
        else:
            HIDDEN_STATES[module_name] = [data]

    return hook


def save_hidden_states_input_mean(module_name: str = "", **kwargs: Any):
    """Save mean over tokens of module input hidden states (shape: [1, hidden_dim] per step)."""
    global HIDDEN_STATES

    def hook(module, input, output):
        data = input[0] if isinstance(input, tuple) else input
        data = data.detach().cpu()
        data = data.mean(dim=1, keepdim=True)  # [batch, n_tokens, d] → [batch, 1, d]
        if module_name in HIDDEN_STATES:
            HIDDEN_STATES[module_name].append(data)
        else:
            HIDDEN_STATES[module_name] = [data]

    return hook


def save_hidden_states_output(module_name: str = "", **kwargs: Any):
    """Save module output hidden states."""
    global HIDDEN_STATES

    def hook(module, input, output):
        data = output[0] if isinstance(output, tuple) else output
        data = data.detach().cpu()
        # print(module_name, data.shape)
        if module_name in HIDDEN_STATES:
            HIDDEN_STATES[module_name].append(data)
        else:
            HIDDEN_STATES[module_name] = [data]

    return hook




def apply_steering_regression_vlm_with_classifier(
    x: torch.Tensor,
    vector: dict,
) -> torch.Tensor:
    global INTERVENTION_LOG
    clf = pickle.loads(vector['classifiers'])

    assert x.shape[1] > 1
    mean_token_ = torch.mean(x, dim=1).float()
    mean_token  = mean_token_.cpu().numpy()
    proba = clf.predict_proba(mean_token)
    pos_class_idx = list(clf.classes_).index(1)
    prob_positive = float(proba[0, pos_class_idx])

    intervened = prob_positive > 0.5
    INTERVENTION_LOG.append({"fm_step": None, "intervened": intervened, "prob_positive": prob_positive})

    if intervened:
        bias       = vector['classifier'][0, 0, 0]
        clf_vector = vector['classifier'][0, :, 1:]
        steer_vec  = vector['steering'][0]
        s_star     = vector['q_target']

        clf_vector_ = clf_vector.to(x.device).to(x.dtype)
        steer_vec_  = steer_vec.to(x.device).to(x.dtype)
        bias_       = torch.tensor(bias, device=x.device, dtype=x.dtype)
        s_star_     = torch.tensor(s_star, device=x.device, dtype=x.dtype)

        w_norm = torch.linalg.norm(clf_vector_)
        s_hat  = (mean_token_.to(x.device).to(x.dtype) * clf_vector_).sum() + bias_
        alpha  = (s_star_ - s_hat) / w_norm

        x[:, :, :] += alpha * steer_vec_

    return x


def apply_steering_diff_means_vlm_with_classifier(
    x: torch.Tensor,
    vector: dict,
    alpha: float = 1.0,
) -> torch.Tensor:
    global INTERVENTION_LOG
    clf = pickle.loads(vector['classifiers'])

    assert x.shape[1] > 1
    mean_token = torch.mean(x, dim=1).float().cpu().numpy()
    proba = clf.predict_proba(mean_token)
    pos_class_idx = list(clf.classes_).index(1)
    prob_positive = float(proba[0, pos_class_idx])

    intervened = prob_positive > 0.5
    INTERVENTION_LOG.append({"fm_step": None, "intervened": intervened, "prob_positive": prob_positive})

    if intervened:
        steer_vec  = vector['steering'][0]
        steer_vec_ = steer_vec.to(x.device).to(x.dtype)
        x[:, :, :] += alpha * steer_vec_

    return x


def apply_steering_regression_flow_matching_with_classifier(
    x: torch.Tensor,
    vector: dict,
    update_fm_step: bool = False,
) -> torch.Tensor:
    global FLOW_MATCHING_STATE, INTERVENTION_LOG
    step = FLOW_MATCHING_STATE["STEP"]

    clf = pickle.loads(vector['classifiers'][step])
    proba = clf.predict_proba(x[:, 0].float().cpu().numpy())
    pos_class_idx = list(clf.classes_).index(1)
    prob_positive = float(proba[0, pos_class_idx])

    intervened = prob_positive > 0.5
    INTERVENTION_LOG.append({"fm_step": step, "intervened": intervened, "prob_positive": prob_positive})

    if intervened:
        bias       = vector['classifier'][step, 0, 0]
        clf_vector = vector['classifier'][step, :, 1:]
        steer_vec  = vector['steering'][step]
        s_star     = vector['q_target']

        clf_vector_ = clf_vector.to(x.device).to(x.dtype)
        steer_vec_  = steer_vec.to(x.device).to(x.dtype)
        bias_       = torch.tensor(bias, device=x.device, dtype=x.dtype)
        s_star_     = torch.tensor(s_star, device=x.device, dtype=x.dtype)

        w_norm = torch.linalg.norm(clf_vector_)
        s_hat  = (x[:, 0] * clf_vector_).sum() + bias_
        alpha  = (s_star_ - s_hat) / w_norm

        x[:, 0] = x[:, 0] + alpha * steer_vec_

    if update_fm_step:
        FLOW_MATCHING_STATE["STEP"] += 1

    return x



def apply_steering_ot_flow_matching_with_classifier_high_to_low(
    x: torch.Tensor,
    vector: dict,
    alpha: float = 1,
    update_fm_step: bool = False,
) -> torch.Tensor:
    """OT steering: high → low (push activation toward low-feature states)."""
    global FLOW_MATCHING_STATE, INTERVENTION_LOG
    step = FLOW_MATCHING_STATE["STEP"]

    h = x[:, 0].float().cpu().numpy().reshape(1, -1)

    model = vector['classifiers'][step]
    prob_fast = model.predict_proba(h)[0][1]

    print(f"[Classifier H2L] step={step} | prob_high={prob_fast:.3f}")

    if prob_fast < 0.5:
        print(f"[Skip H2L] already low (prob_high={prob_fast:.3f})")
        INTERVENTION_LOG.append({"fm_step": step, "intervened": False, "prob_high": float(prob_fast), "reason": "classifier_skip"})
    else:
        # High → steer toward low using coupling Xs→Xt
        coupling = vector['ot_couplings'][step].detach().clone().float().to(x.device).to(x.dtype)
        Xs       = vector['ot_Xs'][step].detach().clone().float().to(x.device).to(x.dtype)  # high states
        Xt       = vector['ot_Xt'][step].detach().clone().float().to(x.device).to(x.dtype)  # low states

        x_h           = x[:, 0].to(Xs.dtype)
        dist          = ((Xs - x_h) ** 2).sum(dim=1)
        target_idx    = dist.argmin()
        x_transported = torch.matmul(coupling[target_idx:target_idx + 1, :], Xt)
        x[:, 0]       = (1 - alpha) * x[:, 0] + alpha * x_transported.to(x.dtype)
        print(f"[Steering H2L] step={step} | prob_high={prob_fast:.3f} | alpha={alpha}")
        INTERVENTION_LOG.append({"fm_step": step, "intervened": True, "prob_high": float(prob_fast), "reason": "steered"})

    if update_fm_step:
        FLOW_MATCHING_STATE["STEP"] += 1

    return x



def apply_steering_ot_flow_matching_with_classifier_high_to_low_pi05(
    x: torch.Tensor,
    vector: dict,
    alpha: float = 1,
    update_fm_step: bool = False,
) -> torch.Tensor:
    """OT steering: high → low (push activation toward low-feature states)."""
    global FLOW_MATCHING_STATE, INTERVENTION_LOG
    step = FLOW_MATCHING_STATE["STEP"]


    model = vector['classifiers'][step]

    chunk_intervention = []
    for j in range(10):
        h = x[:, j].float().cpu().numpy().reshape(1, -1)
        prob_fast = model.predict_proba(h)[0][1]
        chunk_intervention.append(prob_fast>0.5)


    intervention = sum(chunk_intervention) > 5
    if intervention == False:
        INTERVENTION_LOG.append({"fm_step": step, "intervened": False, "prob_high": float(prob_fast), "reason": "classifier_skip"})
    else:
        # High → steer toward low using coupling Xs→Xt
        coupling = vector['ot_couplings'][step].detach().clone().float().to(x.device).to(x.dtype)
        Xs       = vector['ot_Xs'][step].detach().clone().float().to(x.device).to(x.dtype)  # high states
        Xt       = vector['ot_Xt'][step].detach().clone().float().to(x.device).to(x.dtype)  # low states
        for j in range(10):
            x_h = x[:, j].to(Xs.dtype)
            dist          = ((Xs - x_h) ** 2).sum(dim=1)
            target_idx    = dist.argmin()
            x_transported = torch.matmul(coupling[target_idx:target_idx + 1, :], Xt)
            x[:, j]       = (1 - alpha) * x[:, j] + alpha * x_transported.to(x.dtype)

        INTERVENTION_LOG.append({"fm_step": step, "intervened": True, "prob_high": float(prob_fast), "reason": "steered"})

    if update_fm_step:
        FLOW_MATCHING_STATE["STEP"] += 1

    return x


def apply_steering_ot_flow_matching_with_classifier_low_to_high(
    x: torch.Tensor,
    vector: dict,
    alpha: float = 1,
    update_fm_step: bool = False,
) -> torch.Tensor:
    """Reverse OT steering: slow → fast (low-to-high speed)."""
    global FLOW_MATCHING_STATE, INTERVENTION_LOG
    step = FLOW_MATCHING_STATE["STEP"]

    if x.is_cuda:
        torch.cuda.synchronize()
    _clf_t0 = time.time()

    h = x[:, 0].float().cpu().numpy().reshape(1, -1)

    model = vector['classifiers'][step]
    prob_fast = model.predict_proba(h)[0][1]

    _clf_s = time.time() - _clf_t0

    print(f"[Classifier L2H] step={step} | prob_fast={prob_fast:.3f} | clf_s={_clf_s*1000:.3f}ms")

    if prob_fast >= 0.5:
        print(f"[Skip L2H] already fast (prob_fast={prob_fast:.3f})")
        INTERVENTION_LOG.append({"fm_step": step, "intervened": False, "prob_fast": float(prob_fast), "reason": "classifier_skip", "classifier_s": _clf_s})
    else:
        # Slow -> steer to fast using transposed coupling
        if x.is_cuda:
            torch.cuda.synchronize()
        _load_t0 = time.time()

        coupling = vector['ot_couplings'][step].detach().clone().float().to(x.device).to(x.dtype)
        Xs       = vector['ot_Xs'][step].detach().clone().float().to(x.device).to(x.dtype)  # fast states
        Xt       = vector['ot_Xt'][step].detach().clone().float().to(x.device).to(x.dtype)  # slow states

        if x.is_cuda:
            torch.cuda.synchronize()
        _load_s = time.time() - _load_t0

        x_h        = x[:, 0].to(Xt.dtype)

        if x.is_cuda:
            torch.cuda.synchronize()
        _transport_t0 = time.time()

        dist       = ((Xt - x_h) ** 2).sum(dim=1)
        target_idx = dist.argmin()
        col        = coupling[:, target_idx]           # (N_fast,)
        x_transported = ((col / col.sum()).unsqueeze(0) @ Xs)  # (1, hidden_dim)
        x[:, 0]    = (1 - alpha) * x[:, 0] + alpha * x_transported.to(x.dtype)

        if x.is_cuda:
            torch.cuda.synchronize()
        _transport_s = time.time() - _transport_t0

        print(f"[Steering L2H] step={step} | prob_fast={prob_fast:.3f} | alpha={alpha} | "
                f"clf_s={_clf_s*1000:.3f}ms load_s={_load_s*1000:.3f}ms transport_s={_transport_s*1000:.3f}ms")
        INTERVENTION_LOG.append({
            "fm_step": step, "intervened": True, "prob_fast": float(prob_fast), "reason": "steered",
            "classifier_s": _clf_s, "load_vectors_s": _load_s, "transport_s": _transport_s,
        })

    if update_fm_step:
        FLOW_MATCHING_STATE["STEP"] += 1

    return x






def apply_steering_ot_flow_matching_with_classifier_low_to_high_pi05(
    x: torch.Tensor,
    vector: dict,
    alpha: float = 1,
    update_fm_step: bool = False,
) -> torch.Tensor:
    """Reverse OT steering: slow → fast (low-to-high speed)."""
    global FLOW_MATCHING_STATE, INTERVENTION_LOG
    step = FLOW_MATCHING_STATE["STEP"]


    model = vector['classifiers'][step]
    chunk_intervention = []
    for j in range(10):
        h = x[:, j].float().cpu().numpy().reshape(1, -1)
        prob_fast = model.predict_proba(h)[0][1]
        chunk_intervention.append(prob_fast<0.5)


    intervention = sum(chunk_intervention) > 5

    if intervention == False:
        INTERVENTION_LOG.append({"fm_step": step, "intervened": False, "prob_fast": float(prob_fast), "reason": "classifier_skip"})
    else:
        # Slow -> steer to fast using transposed coupling

        coupling = vector['ot_couplings'][step].detach().clone().float().to(x.device).to(x.dtype)
        Xs       = vector['ot_Xs'][step].detach().clone().float().to(x.device).to(x.dtype)  # fast states
        Xt       = vector['ot_Xt'][step].detach().clone().float().to(x.device).to(x.dtype)  # slow states

        for j in range(10):
            x_h = x[:, j].to(Xt.dtype)
            dist          = ((Xt - x_h) ** 2).sum(dim=1)
            target_idx    = dist.argmin()
            col        = coupling[:, target_idx]           # (N_fast,)
            x_transported = ((col / col.sum()).unsqueeze(0) @ Xs)  # (1, hidden_dim)
            x[:, j]    = (1 - alpha) * x[:, j] + alpha * x_transported.to(x.dtype)



        INTERVENTION_LOG.append({
            "fm_step": step, "intervened": True, "prob_fast": float(prob_fast), "reason": "steered",})

    if update_fm_step:
        FLOW_MATCHING_STATE["STEP"] += 1

    return x





def shift_hidden_states(
    vector: Union[torch.Tensor, dict] = None,
    operation: str = "add",
    alpha: float = 1,
    only_generated_tokens: bool = False,
    include_last_prompt_token: bool = False,
    start_prompt_token_idx: int = 0,
    update_fm_step: bool = False,
    **kwargs: Any,
):
    """
    Shift features in the vector's direction.
    """

    if operation == "regressor_vlm_clf":
        def hook(module, input, output):
            if isinstance(output, tuple):
                output_ = apply_steering_regression_vlm_with_classifier(
                    output[0],
                    vector,
                    only_generated_tokens=only_generated_tokens,
                    include_last_prompt_token=include_last_prompt_token,
                    start_prompt_token_idx=start_prompt_token_idx,
                )
                return (output_,) + output[1:]
            else:
                return apply_steering_regression_vlm_with_classifier(
                    output,
                    vector,
                    only_generated_tokens=only_generated_tokens,
                    include_last_prompt_token=include_last_prompt_token,
                    start_prompt_token_idx=start_prompt_token_idx,
                )

    elif operation == "mean_vlm_clf":
        def hook(module, input, output):
            if isinstance(output, tuple):
                output_ = apply_steering_diff_means_vlm_with_classifier(
                    output[0],
                    vector,
                    alpha=alpha,
                    only_generated_tokens=only_generated_tokens,
                    include_last_prompt_token=include_last_prompt_token,
                    start_prompt_token_idx=start_prompt_token_idx,
                )
                return (output_,) + output[1:]
            else:
                return apply_steering_diff_means_vlm_with_classifier(
                    output,
                    vector,
                    alpha=alpha,
                    only_generated_tokens=only_generated_tokens,
                    include_last_prompt_token=include_last_prompt_token,
                    start_prompt_token_idx=start_prompt_token_idx,
                )

    elif operation == "regressor_fm_clf":
        def hook(module, input, output):
            if isinstance(output, tuple):
                output_ = apply_steering_regression_flow_matching_with_classifier(
                    output[0],
                    vector,
                    only_generated_tokens=only_generated_tokens,
                    include_last_prompt_token=include_last_prompt_token,
                    start_prompt_token_idx=start_prompt_token_idx,
                    update_fm_step=update_fm_step,
                )
                return (output_,) + output[1:]
            else:
                return apply_steering_regression_flow_matching_with_classifier(
                    output,
                    vector,
                    only_generated_tokens=only_generated_tokens,
                    include_last_prompt_token=include_last_prompt_token,
                    start_prompt_token_idx=start_prompt_token_idx,
                    update_fm_step=update_fm_step,
                )

    elif "ot_steering_decision_func_high_to_low" in operation:
        if "pi05" in operation:
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output_ = apply_steering_ot_flow_matching_with_classifier_high_to_low_pi05(
                        output[0],
                        vector,
                        alpha=alpha,
                        only_generated_tokens=only_generated_tokens,
                        include_last_prompt_token=include_last_prompt_token,
                        start_prompt_token_idx=start_prompt_token_idx,
                        update_fm_step=update_fm_step,
                    )
                    return (output_,) + output[1:]
                else:
                    return apply_steering_ot_flow_matching_with_classifier_high_to_low_pi05(
                        output,
                        vector,
                        alpha=alpha,
                        only_generated_tokens=only_generated_tokens,
                        include_last_prompt_token=include_last_prompt_token,
                        start_prompt_token_idx=start_prompt_token_idx,
                        update_fm_step=update_fm_step,
                    )
        else:
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output_ = apply_steering_ot_flow_matching_with_classifier_high_to_low(
                        output[0],
                        vector,
                        alpha=alpha,
                        only_generated_tokens=only_generated_tokens,
                        include_last_prompt_token=include_last_prompt_token,
                        start_prompt_token_idx=start_prompt_token_idx,
                        update_fm_step=update_fm_step,
                    )
                    return (output_,) + output[1:]
                else:
                    return apply_steering_ot_flow_matching_with_classifier_high_to_low(
                        output,
                        vector,
                        alpha=alpha,
                        only_generated_tokens=only_generated_tokens,
                        include_last_prompt_token=include_last_prompt_token,
                        start_prompt_token_idx=start_prompt_token_idx,
                        update_fm_step=update_fm_step,
                    )

    elif "ot_steering_decision_func_low_to_high" in operation:
        if "pi05" in operation:
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output_ = apply_steering_ot_flow_matching_with_classifier_low_to_high_pi05(
                        output[0],
                        vector,
                        alpha=alpha,
                        only_generated_tokens=only_generated_tokens,
                        include_last_prompt_token=include_last_prompt_token,
                        start_prompt_token_idx=start_prompt_token_idx,
                        update_fm_step=update_fm_step,
                    )
                    return (output_,) + output[1:]
                else:
                    return apply_steering_ot_flow_matching_with_classifier_low_to_high_pi05(
                        output,
                        vector,
                        alpha=alpha,
                        only_generated_tokens=only_generated_tokens,
                        include_last_prompt_token=include_last_prompt_token,
                        start_prompt_token_idx=start_prompt_token_idx,
                        update_fm_step=update_fm_step,
                    )
        else:
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output_ = apply_steering_ot_flow_matching_with_classifier_low_to_high(
                        output[0],
                        vector,
                        alpha=alpha,
                        only_generated_tokens=only_generated_tokens,
                        include_last_prompt_token=include_last_prompt_token,
                        start_prompt_token_idx=start_prompt_token_idx,
                        update_fm_step=update_fm_step,
                    )
                    return (output_,) + output[1:]
                else:
                    return apply_steering_ot_flow_matching_with_classifier_low_to_high(
                        output,
                        vector,
                        alpha=alpha,
                        only_generated_tokens=only_generated_tokens,
                        include_last_prompt_token=include_last_prompt_token,
                        start_prompt_token_idx=start_prompt_token_idx,
                        update_fm_step=update_fm_step,
                    )

    else:
        raise NotImplementedError(
            f"Only the following steering operations are supported: "
            f"regressor_vlm_clf, mean_vlm_clf, regressor_fm_clf, "
            f"ot_steering_decision_func_high_to_low, ot_steering_decision_func_low_to_high. Got {operation}"
        )

    return hook


def extract_token_of_interest_states(
    tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
    token_of_interest_idx: Union[int, torch.Tensor] = None,
    token_of_interest_start_token: int = 0,
) -> Tuple[torch.Tensor]:

    if token_of_interest_start_token != 0:
        # e.g. consider only te answers
        tokens = tokens[:, token_of_interest_start_token:]
        pred_tokens = pred_tokens[:, token_of_interest_start_token:]

    # Concider only text, no preds tokens for image tokens
    if pred_tokens.shape[1] > tokens.shape[1]:
        pred_tokens = pred_tokens[
            :, -tokens.shape[1] :
        ]  # e.g. in case of language_model.lm_head only the hidden states for generated tokens are saved
    elif pred_tokens.shape[1] < tokens.shape[1]:
        tokens = tokens[:, -pred_tokens.shape[1] :]

    assert (
        token_of_interest_idx is not None
    ), f"Please provide the token_of_interest_idx, got {token_of_interest_idx}"

    # If the token_of_interest splits into different ids, we consider the first one (while skipping eos/bos tokens)
    if not isinstance(token_of_interest_idx, torch.Tensor):
        token_of_interest_idx = torch.tensor([token_of_interest_idx])
    token_of_interest_idx = token_of_interest_idx.to(pred_tokens.device)

    # Step 1: Find where the tokens of interest exist in the batch (B, L)
    token_of_interest_batch_presence = torch.isin(
        pred_tokens, token_of_interest_idx
    )  # (B, L)
    # Step 2: Get the first occurrence index for each sequence
    token_of_interest_batch_first_pos = torch.argmax(
        token_of_interest_batch_presence.long(), dim=1
    )  # (B,)

    # Step 3: Mask for sequences with no token of interest
    no_token_found_mask = ~token_of_interest_batch_presence.any(dim=1)

    # Set the position to -1 if no token of interest is found
    token_of_interest_batch_first_pos[no_token_found_mask] = -1

    # Step 4: Now handle indexing into `v` based on the first position
    # Extract v at the first position for each batch (B,)
    # Select only valid positions in `v`
    v_selected = tokens[
        range(tokens.shape[0]),
        token_of_interest_batch_first_pos.clamp(min=0).to(tokens.device),
    ].unsqueeze(1)
    return v_selected, ~no_token_found_mask


def extract_states_before_special_tokens(
    tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
    end_special_tokens: List[str],
    tokenizer: Callable,
    token_of_interest_start_token: int = 0,
) -> Tuple[torch.Tensor]:
    if token_of_interest_start_token != 0:
        # e.g. consider only te answers
        tokens = tokens[:, token_of_interest_start_token:]
        pred_tokens = pred_tokens[:, token_of_interest_start_token:]

    # Concider only text, no preds tokens for image tokens
    if pred_tokens.shape[1] > tokens.shape[1]:
        pred_tokens = pred_tokens[
            :, -tokens.shape[1] :
        ]  # e.g. in case of language_model.lm_head only the hidden states for generated tokens are saved
    elif pred_tokens.shape[1] < tokens.shape[1]:
        tokens = tokens[:, -pred_tokens.shape[1] :]

    assert end_special_tokens is not None and isinstance(
        end_special_tokens, list
    ), f"Please provide the list of token_of_interest, got {end_special_tokens}"

    # If the token_of_interest splits into different ids, we consider the first one (while skipping eos/bos tokens)
    end_special_tokens_idx = torch.tensor(
        [
            tokenizer.encode(tok, add_special_tokens=False)[0]
            for tok in end_special_tokens
        ]
    ).to(pred_tokens.device)

    # Step 1: Find where the tokens of interest exist in the batch (B, L)
    token_of_interest_batch_presence = torch.isin(
        pred_tokens, end_special_tokens_idx
    )  # (B, L)
    # Step 2: Get the first occurrence index for each sequence
    token_of_interest_batch_first_pos = torch.argmax(
        token_of_interest_batch_presence.long(), dim=1
    )  # (B,)

    # Step 3: Mask for sequences with no token of interest
    no_token_found_mask = ~token_of_interest_batch_presence.any(dim=1)

    # Set the position to -1 if no token of interest is found
    token_of_interest_batch_first_pos[no_token_found_mask] = -1

    # Step 4: Now handle indexing into `v` based on the first position
    # Extract v at the first position for each batch (B,)
    # Select only valid positions in `v`
    v_selected = (
        tokens[
            range(tokens.shape[0]),
            : token_of_interest_batch_first_pos.to(tokens.device),
        ]
        .mean(1)
        .unsqueeze(1)
    )
    return v_selected, no_token_found_mask


def get_hidden_states(
    token_idx: int = None,
    token_start_end_idx: List[List[int]] = None,
    extract_token_of_interest: bool = False,
    token_of_interest_start_token: int = 0,
    extract_before_special_tokens: bool = False,
    save_only_generated_tokens: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    hidden_states = {}
    output = {}
    for k, v in HIDDEN_STATES.items():
        if isinstance(v, list) and len(v) > 1:
            #v = torch.cat(v, dim=1) # Old code changed by JAYNEEL
            v = torch.cat(v, dim=0)
            #print ("Found a LIST", v.shape)
        else:
            v = v[0]
        if token_idx is not None:
            v = v[:, token_idx, :].unsqueeze(1)
        elif token_start_end_idx is not None:
            v = v[:, int(token_start_end_idx[0]) : int(token_start_end_idx[1]), :]
        elif extract_token_of_interest:

            if save_only_generated_tokens:
                start_idx_generated_tokens = -kwargs["model_generated_output"].shape[1]
                token_of_interest_start_token = start_idx_generated_tokens

            v, token_of_interest_mask = extract_token_of_interest_states(
                tokens=v,
                pred_tokens=kwargs["model_output"],
                token_of_interest_idx=kwargs.get("token_of_interest_idx", None),
                token_of_interest_start_token=token_of_interest_start_token,
            )
            output["token_of_interest_mask"] = token_of_interest_mask
            output["image"] = kwargs["image"]
        elif extract_before_special_tokens:

            if save_only_generated_tokens:
                start_idx_generated_tokens = -kwargs["model_generated_output"].shape[1]
                token_of_interest_start_token = start_idx_generated_tokens

            v, token_of_interest_mask = extract_states_before_special_tokens(
                tokens=v,
                pred_tokens=kwargs["model_output"],
                end_special_tokens=kwargs["end_special_tokens"],
                tokenizer=kwargs["tokenizer"],
                token_of_interest_start_token=token_of_interest_start_token,
            )
            output["token_of_interest_mask"] = torch.ones_like(
                token_of_interest_mask
            ).bool()
            output["image"] = kwargs["image"]
        else:
            pass
        hidden_states[k] = v.clone()
    output["hidden_states"] = hidden_states
    return output


def save_hidden_states_to_file(
    data: Dict[str, Any],
    data_keys: List[str] = ["hidden_states"],
    hook_name: str = "",
    args: argparse.Namespace = None,
    logger: Callable = None,
) -> None:
    saved_data = {}

    for data_key in data.keys():
        if data_key in data_keys:
            assert (
                data_key in data
            ), f"{data_key} not found in data, there is only: {data.keys()}"

            saved_data[data_key] = data[data_key]  # List[Any]
    file_name = os.path.join(
        args.save_dir, "features", f"{hook_name}_{args.save_filename}.pth"
    )
    torch.save(saved_data, file_name)
    if logger is not None:
        logger.info(f"Saving data to: {file_name}")


def save_analysis_to_file(
    data: Dict[str, Any],
    analysis_saving_path: str,
    data_keys: List[str] = ["text_grounding"],
    logger: Callable = None,
) -> None:
    saved_data = {}

    for data_key in data_keys:
        assert (
            data_key in data
        ), f"{data_key} not found in data, there is only: {data.keys()}"

        saved_data[data_key] = data[data_key]  # List[Any]
    file_name = f"{analysis_saving_path}.pth"
    torch.save(saved_data, file_name)
    if logger is not None:
        logger.info(f"Saving analysis data to: {file_name}")


def register_hooks(
    hook_index: int,
    model: Callable,
    modules_to_hook: List[str],
    hook_name: str = "save_hidden_states",
    tokenizer: Callable = None,
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Callable:
    hook_function, hook_return_function = None, None

    # Detect input/output save preference and normalize hook_name for matching
    if "_input_hidden_states" in hook_name and hook_name != "save_input_hidden_states_mean":
        save_hidden_states = save_hidden_states_input
        hook_name = hook_name.replace("_input_hidden_states", "_hidden_states")
    elif "_output_hidden_states" in hook_name:
        save_hidden_states = save_hidden_states_output
        hook_name = hook_name.replace("_output_hidden_states", "_hidden_states")
    # else: save_hidden_states stays as the original function


    if "save_hidden_states" == hook_name:
        # Save the hidden states of all tokens in the sequence
        hook_function = save_hidden_states
        hook_return_function = get_hidden_states
    elif "save_hidden_states_given_token_idx" == hook_name:
        # Save the hidden states at given token index
        hook_function = save_hidden_states
        hook_return_function = partial(get_hidden_states, token_idx=args.token_idx[0] if len(args.token_idx)==1 else args.token_idx[hook_index])
    elif "save_hidden_states_given_token_start_end_idx" == hook_name:
        # Save the hidden states of tokens between start and end index
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states, token_start_end_idx=args.token_start_end_idx
        )
    elif "save_hidden_states_for_token_of_interest" == hook_name:
        # Save the hidden states of tokens between start and end index
        token_of_interest = args.token_of_interest

        # Get index in tokenizer vocabulary for token of interest
        # Some tokenizers encode/decode space along with token, so include index of whitespace + token_of_interest
        tokens_of_interest = set(
            [
                token_of_interest,
                token_of_interest.capitalize(),
                token_of_interest.lower(),
            ]
        )
        token_of_interest_idx = args.token_of_interest_idx
        if token_of_interest_idx is None:
            token_of_interest_idx = torch.tensor(
                [
                    tokenizer.encode(tok, add_special_tokens=False)[0]
                    for tok in tokens_of_interest
                ]
            )
            check_token = tokenizer.encode(" " + token_of_interest, add_special_tokens=False)[0]
            if token_of_interest in tokenizer.decode([check_token]): # Check if this check_token is only encoding whitespace
                token_of_interest_idx = torch.tensor(list(token_of_interest_idx) + [check_token])
            

        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            extract_token_of_interest=True,
            token_of_interest_idx=token_of_interest_idx,
            token_of_interest_start_token=args.token_of_interest_start_token,
            save_only_generated_tokens=args.save_only_generated_tokens,
        )
    elif "save_hidden_states_for_token_of_interest_class" == hook_name:
        # Save the hidden states of tokens between start and end index
        token_of_interest = []
        tokens = list(WORDS[args.token_of_interest_class])
        for tok in tqdm(tokens):
            toks = [
                tok,
                tok.capitalize(),
                tok.lower(),
            ]
            token_of_interest.extend(toks)
        tokens_of_interest = list(set(token_of_interest))

        token_of_interest_idx = args.token_of_interest_idx
        if token_of_interest_idx is None:
            token_of_interest_idx = torch.tensor(
                [
                    tokenizer.encode(tok, add_special_tokens=False)[0]
                    for tok in tokens_of_interest
                ]
            )
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            extract_token_of_interest=True,
            token_of_interest_idx=token_of_interest_idx,
            token_of_interest_start_token=args.token_of_interest_start_token,
            save_only_generated_tokens=args.save_only_generated_tokens,
        )
    elif "save_hidden_states_before_special_tokens" == hook_name:
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            extract_before_special_tokens=True,
            end_special_tokens=args.end_special_tokens,
            tokenizer=tokenizer,
            save_only_generated_tokens=args.save_only_generated_tokens,
        )
    elif hook_name == "save_input_hidden_states_mean":
        hook_function = save_hidden_states_input_mean
        hook_return_function = get_hidden_states
    elif "shift_hidden_states" in hook_name:
        # Only 5 steering operations are wired up in shift_hidden_states; map hook_name -> operation.
        if "regressor" in hook_name and "vlm" in hook_name and "clf" in hook_name:
            operation = "regressor_vlm_clf"
        elif "mean" in hook_name and "vlm" in hook_name and "clf" in hook_name:
            operation = "mean_vlm_clf"
        elif "regressor" in hook_name and "fm" in hook_name and "clf" in hook_name:
            operation = "regressor_fm_clf"
        elif "dimas" in hook_name and "high_to_low" in hook_name:
            operation = "ot_steering_decision_func_high_to_low"
        elif "dimas" in hook_name and "low_to_high" in hook_name:
            operation = "ot_steering_decision_func_low_to_high"
        else:
            raise NotImplementedError(
                f"Unsupported shift_hidden_states hook_name: {hook_name}. Supported patterns: "
                f"*regressor*vlm*clf*, *mean*vlm*clf*, *regressor*fm*clf*, "
                f"*dimas*high_to_low*, *dimas*low_to_high*"
            )
        
        if "pi05" in hook_name:
            operation += "_pi05"

        only_generated_tokens = "only_generated" in hook_name
        include_last_prompt_token = "last_prompt_token" in hook_name

        shift_vector_path = args.shift_vector_path[0] if len(args.shift_vector_path)==1 else args.shift_vector_path[hook_index]
        logger.info(f"Loading steering vector from: {shift_vector_path}")
        vector = torch.load(shift_vector_path , weights_only=False)

        hook_function = partial(
            shift_hidden_states,
            vector=vector,
            operation=operation,
            alpha=args.steering_alpha,
            only_generated_tokens=only_generated_tokens,
            include_last_prompt_token=include_last_prompt_token,
            start_prompt_token_idx=args.start_prompt_token_idx_steering,
            update_fm_step=True,
        )
        hook_return_function = partial(get_hidden_states, token_idx=args.token_idx[0] if len(args.token_idx)==1 else args.token_idx[hook_index])
    else:
        warnings.warn(f"{hook_name} is not supported. No hooks attached to model.")
    if hook_function is not None:
        hooked_modules = []
        for name, module in model.named_modules():
            if fmatch(
                name, modules_to_hook, exact_match=args.exact_match_modules_to_hook
            ):
                module.register_forward_hook(hook_function(module_name=name))
                hooked_modules.append(name)
        if logger is not None:
            logger.info(f"Apply {hook_name} to hooked_modules: {hooked_modules}")

    return hook_return_function


def hooks_postprocessing(
    hook_name: str = "save_hidden_states", args: argparse.Namespace = None
) -> Callable:
    hook_postprocessing_function = None
    if "save_hidden_states" in hook_name:

        data_keys = ["hidden_states", "image", "model_predictions"]

        if "token_of_interest" in hook_name:
            data_keys.append("token_of_interest_mask")
        hook_postprocessing_function = partial(
            save_hidden_states_to_file,
            args=args,
            data_keys=data_keys,
            hook_name=hook_name,
        )
    else:
        warnings.warn(f"{hook_name} is not supported. No hooks attached to model.")

    return hook_postprocessing_function


def clear_forward_hooks(model: Callable) -> None:
    for module in model.modules():
        module._forward_hooks.clear()


def clear_hooks_variables():
    global HIDDEN_STATES
    HIDDEN_STATES = {}
    # Also reset flow-matching progression when clearing hook state.
    reset_flow_matching_state()


def setup_hooks(
    model: Callable,
    modules_to_hook: List[str],
    hook_names: str,
    tokenizer: Callable = None,
    logger: Callable = None,
    args: argparse.Namespace = None,
):
    hook_return_functions, hook_postprocessing_functions = [], []
    for i, hook_name in enumerate(hook_names):
        if modules_to_hook is not None and i < len(modules_to_hook):
            modules_to_hook_ = modules_to_hook[i]
            assert isinstance(
                modules_to_hook_, list
            ), f"modules_to_hook_ must be of type list. modules_to_hook_: {modules_to_hook_}"
            hook_return_function = register_hooks(
                hook_index=i,
                model=model,
                modules_to_hook=modules_to_hook_,
                hook_name=hook_name,
                logger=logger,
                args=args,
            )
        else:
            hook_return_function = None
        hook_postprocessing_function = hooks_postprocessing(
            hook_name=hook_name, args=args
        )

        hook_return_functions.append(hook_return_function)
        hook_postprocessing_functions.append(hook_postprocessing_function)

    return hook_return_functions, hook_postprocessing_functions


