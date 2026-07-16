import os
import pickle
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr


# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────


def get_speed(info):
    speed_info = (info['action'][:, :3]**2).sum(dim=1).sqrt()
    return speed_info, info

def get_eef_height_displacement(info):
    eef_height = info['action'][:, 2].abs()
    return eef_height, info


GET_FEATURE_FUNC = None
FEATURE_TAG = None

def set_feature_func_speed():
    global GET_FEATURE_FUNC, FEATURE_TAG
    GET_FEATURE_FUNC = get_speed
    FEATURE_TAG = "speed"

def set_feature_func_eef_height_displacement():
    global GET_FEATURE_FUNC, FEATURE_TAG
    GET_FEATURE_FUNC = get_eef_height_displacement
    FEATURE_TAG = "eef_height_displacement"

def get_feature_distribution(all_data, episode_list, success_only=True, max_ep_per_task=None):
    all_features = []
    success = 0
    total_episodes = 0

    for episode in episode_list:
        feature_info, info = GET_FEATURE_FUNC(all_data[episode])

        episode_indices = info['episode_index']
        unique_episodes = episode_indices.unique()
        if max_ep_per_task is not None:
            unique_episodes = unique_episodes[:max_ep_per_task]

        for ep_idx in unique_episodes:
            total_episodes += 1
            mask = (episode_indices == ep_idx)  # steps of that ep

            ep_success = info['next.success'][mask].sum() > 0
            ep_feature = feature_info[mask][:-1]

            if ep_success:
                success += 1

            if (success_only and ep_success) or not success_only:
                all_features.append(ep_feature.data.numpy())

    feature = np.concatenate(all_features, axis=0) if all_features else None
    return feature, success / total_episodes if total_episodes > 0 else 0


def get_success_labels(all_data, episode_list, success_only=True, max_ep_per_task=None):
    all_success_labels = []

    for episode in episode_list:
        info = all_data[episode]

        episode_indices = info['episode_index']
        unique_episodes = episode_indices.unique()
        if max_ep_per_task is not None:
            unique_episodes = unique_episodes[:max_ep_per_task]

        for ep_idx in unique_episodes:
            mask = (episode_indices == ep_idx)

            ep_success = info['next.success'][mask].sum() > 0

            target_length = mask.sum().item() - 1

            if (success_only and ep_success) or not success_only:
                if target_length > 0:
                    success_val = 1 if ep_success else 0
                    all_success_labels.append(np.full((target_length,), success_val))

    if not all_success_labels:
        return np.array([])

    return np.concatenate(all_success_labels, axis=0)


def get_hidden_repr(episode_data, layer_key=None, fm_time_step=None, max_ep_per_task=None):
    hook_data = episode_data['hook_data']

    if isinstance(hook_data, dict):
        hook_data = [hook_data]
    if max_ep_per_task is not None:
        hook_data = hook_data[:max_ep_per_task]

    all_repr = []

    for ep_hook in hook_data:
        if 'hidden_states' not in ep_hook:
            continue

        hidden_states = ep_hook['hidden_states']

        if layer_key is None:
            layer_key = list(hidden_states[0].keys())[0]

        for i in range(len(hidden_states)):
            hs = hidden_states[i][layer_key]
            # FM case: hs is indexed by denoising step
            # VLM case: hs is the repr directly
            repr_i = hs[fm_time_step] if fm_time_step is not None else hs[0]
            all_repr.append(repr_i)

    if len(all_repr) == 0:
        return None

    return torch.cat(all_repr, dim=0)


def load_episode_data(extraction_dir, episode_list, suffix_episode="", flat=False):

    all_data = {}
    for episode in episode_list:
        if flat:
            path = os.path.join(extraction_dir, 'videos', episode, 'episode_output_data.pt')
        else:
            path = os.path.join(extraction_dir, episode + suffix_episode, 'videos', episode, 'episode_output_data.pt')
        all_data[episode] = torch.load(path)
    return all_data




def vlm_steering_generate_regression_with_classifier(layer_num, extraction_dir, episode_list,
                                                     low_q=0.25, high_q=0.75, output_dir=None, suffix=''):
    global FEATURE_TAG

    layer = f'model.vlm_with_expert.layer_hooks.0.{layer_num}'

    all_data = load_episode_data(extraction_dir, episode_list, suffix_episode="_mean_vlm")
    all_repr = []
    for episode in episode_list:
        all_repr.append(get_hidden_repr(all_data[episode], layer_key=layer))

    repr_all = torch.cat(all_repr).float().cpu().numpy()
    print(f"Extracted representations shape: {repr_all.shape} | Layer: {layer}")

    feature, _ = get_feature_distribution(all_data, episode_list, success_only=True)
    q_low_val  = float(np.quantile(feature, low_q))
    q_high_val = float(np.quantile(feature, high_q))
    print(f"Feature quantiles — q_low ({low_q}): {q_low_val:.4f} | q_high ({high_q}): {q_high_val:.4f}")

    success_labels = get_success_labels(all_data, episode_list, success_only=False)
    repr_all = repr_all[success_labels == 1]

    X_train = repr_all.copy()
    y_train = feature.copy()
    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)

    coef        = reg.coef_
    coef_norm   = np.linalg.norm(coef)
    coef_normed = coef / coef_norm
    hidden_dim  = coef.shape[0]

    classifier_vecs = torch.zeros([1, 1, hidden_dim + 1])
    steering_vecs   = torch.zeros([1, 1, hidden_dim])
    classifier_vecs[0, 0,  0] = reg.intercept_.item()
    classifier_vecs[0, :, 1:] = torch.tensor(coef)
    steering_vecs[0, :, :]    = torch.tensor(coef_normed)

    feature_train = y_train.copy()
    train_idx = list(np.where(feature_train < q_low_val)[0]) + list(np.where(feature_train > q_high_val)[0])
    labels_low  = np.zeros(len(y_train)); labels_low[np.where(feature_train > q_high_val)[0]] = 1
    labels_high = np.zeros(len(y_train)); labels_high[np.where(feature_train < q_low_val)[0]] = 1

    clf_low = SVC(C=0.1, kernel='linear', probability=True)
    clf_low.fit(X_train[train_idx], labels_low[train_idx])
    clf_high = SVC(C=0.1, kernel='linear', probability=True)
    clf_high.fit(X_train[train_idx], labels_high[train_idx])

    for q_target_val, tag, clf in [(q_low_val, f'{FEATURE_TAG}_low', clf_low), (q_high_val, f'{FEATURE_TAG}_high', clf_high)]:
        fname = f"VLM_steering_vecs_regression_{layer_num}_{tag}{suffix}.pt"
        save_path = fname if output_dir is None else os.path.join(output_dir, fname)
        torch.save({
            'classifier':  classifier_vecs,
            'classifiers': pickle.dumps(clf),
            'steering':    steering_vecs,
            'q_target':    q_target_val,
            'low_q':       low_q,
            'high_q':      high_q,
        }, save_path)
        print(f"Saved: {save_path}")


def vlm_steering_generate_diff_means_with_classifier(layer_num, extraction_dir, episode_list,
                                                      low_q=0.25, high_q=0.75, output_dir=None, suffix='',
                                                      svm_C=0.1):
    
    global FEATURE_TAG

    layer = f'model.vlm_with_expert.layer_hooks.0.{layer_num}'

    all_data = load_episode_data(extraction_dir, episode_list, suffix_episode="_mean_vlm")
    all_repr = []
    for episode in episode_list:
        all_repr.append(get_hidden_repr(all_data[episode], layer_key=layer))

    repr_all = torch.cat(all_repr).float().cpu().numpy()
    print(f"Extracted representations shape: {repr_all.shape} | Layer: {layer}")

    feature, _ = get_feature_distribution(all_data, episode_list, success_only=True)
    q_low_val  = float(np.quantile(feature, low_q))
    q_high_val = float(np.quantile(feature, high_q))
    print(f"Feature quantiles — q_low ({low_q}): {q_low_val:.4f} | q_high ({high_q}): {q_high_val:.4f}")

    success_labels = get_success_labels(all_data, episode_list, success_only=False)
    repr_all = repr_all[success_labels == 1]

    low_feature_idx = np.where(feature < q_low_val)[0]
    high_feature_idx = np.where(feature > q_high_val)[0]

    mean_low_feature = repr_all[low_feature_idx].mean(axis=0)
    mean_high_feature = repr_all[high_feature_idx].mean(axis=0)
    diff_to_low_feature = mean_low_feature - mean_high_feature
    diff_to_high_feature = mean_high_feature - mean_low_feature
    midpoint  = (mean_low_feature + mean_high_feature) / 2.0
    bias_to_low_feature = -float(np.dot(diff_to_low_feature, midpoint))
    bias_to_high_feature = -float(np.dot(diff_to_high_feature, midpoint))

    print(f"Layer {layer_num} | Slow: {len(low_feature_idx)} | Fast: {len(high_feature_idx)} | ‖diff‖: {np.linalg.norm(diff_to_low_feature):.4f}")

    hidden_dim = diff_to_low_feature.shape[0]
    steering_vecs_to_low_feature = torch.zeros([1, 1, hidden_dim])
    steering_vecs_to_high_feature = torch.zeros([1, 1, hidden_dim])
    steering_vecs_to_low_feature[0, :, :] = torch.tensor(diff_to_low_feature)
    steering_vecs_to_high_feature[0, :, :] = torch.tensor(diff_to_high_feature)
    classifier_vecs_to_low_feature = torch.zeros([1, 1, hidden_dim + 1])
    classifier_vecs_to_high_feature = torch.zeros([1, 1, hidden_dim + 1])
    classifier_vecs_to_low_feature[0, 0, 0] = bias_to_low_feature
    classifier_vecs_to_low_feature[0, :, 1:] = torch.tensor(diff_to_low_feature)
    classifier_vecs_to_high_feature[0, 0, 0] = bias_to_high_feature
    classifier_vecs_to_high_feature[0, :, 1:] = torch.tensor(diff_to_high_feature)

    X_clf = np.concatenate([repr_all[low_feature_idx], repr_all[high_feature_idx]], axis=0)
    is_fast = np.concatenate([np.zeros(len(low_feature_idx)), np.ones(len(high_feature_idx))])

    clf_low  = SVC(C=svm_C, kernel='linear', probability=True)
    clf_low.fit(X_clf, is_fast)
    clf_high = SVC(C=svm_C, kernel='linear', probability=True)
    clf_high.fit(X_clf, 1 - is_fast)
    print(f"Layer {layer_num} | SVM acc (low): {clf_low.score(X_clf, is_fast):.4f} | (high): {clf_high.score(X_clf, 1 - is_fast):.4f}")

    for tag, clf, clf_vecs, steer_vecs, q_target in [
        (f'{FEATURE_TAG}_low', clf_low,  classifier_vecs_to_low_feature, steering_vecs_to_low_feature,  q_low_val),
        (f'{FEATURE_TAG}_high', clf_high, classifier_vecs_to_high_feature, steering_vecs_to_high_feature, q_high_val),
    ]:
        fname = f"VLM_steering_vecs_diff_means_{layer_num}_{tag}{suffix}.pt"
        save_path = fname if output_dir is None else os.path.join(output_dir, fname)
        torch.save({
            'classifier':  clf_vecs,
            'classifiers': pickle.dumps(clf),
            'steering':    steer_vecs,
            'q_target':    q_target,
            'low_q':       low_q,
            'high_q':      high_q,
        }, save_path)
        print(f"Saved: {save_path}")


def fm_steering_generate_regression_with_classifier(layer_num, extraction_dir, episode_list,
                                                     steps=list(range(10)), num_steps=10,
                                                     low_q=0.25, high_q=0.75, output_dir=None, suffix='',
                                                     svm_C=0.1, n_train_tasks=None):
    
    global FEATURE_TAG

    layer = f'model.vlm_with_expert.layer_hooks.1.{layer_num}'
    classifier_vecs  = torch.zeros([num_steps, 1, 481])
    steering_vecs = torch.zeros([num_steps, 1, 480])
    classifiers_low  = [None] * num_steps
    classifiers_high = [None] * num_steps

    train_episodes = episode_list[:n_train_tasks] if n_train_tasks else episode_list
    all_data = load_episode_data(extraction_dir, train_episodes, flat=True)

    feature, _ = get_feature_distribution(all_data, train_episodes, success_only=True)
    q_low_val  = float(np.quantile(feature, low_q))
    q_high_val = float(np.quantile(feature, high_q))
    print(f"Feature quantiles — q_low ({low_q}): {q_low_val:.4f} | q_high ({high_q}): {q_high_val:.4f}")

    success_labels = get_success_labels(all_data, train_episodes, success_only=False)

    for j, step in enumerate(steps):
        all_repr = []
        for episode in train_episodes:
            all_repr.append(get_hidden_repr(all_data[episode], fm_time_step=step, layer_key=layer))

        repr_all = torch.cat(all_repr).float().cpu().numpy()
        repr_all = repr_all[success_labels == 1]

        X_train = repr_all.copy()
        y_train = feature.copy()
        reg = Ridge(alpha=1.0)
        reg.fit(X_train, y_train)

        coef        = reg.coef_
        coef_norm   = np.linalg.norm(coef)
        coef_normed = coef / coef_norm

        classifier_vecs[j, 0,  0] = reg.intercept_.item()
        classifier_vecs[j, :, 1:] = torch.tensor(coef)
        steering_vecs[j, :, :]    = torch.tensor(coef_normed)

        low_feature_idx  = np.where(y_train < q_low_val)[0]
        high_feature_idx  = np.where(y_train > q_high_val)[0]
        train_idx = np.concatenate([low_feature_idx, high_feature_idx])

        labels_low_feature  = np.zeros(len(y_train)); labels_low_feature[high_feature_idx]  = 1
        labels_high_feature = np.zeros(len(y_train)); labels_high_feature[low_feature_idx] = 1

        clf_low = SVC(C=svm_C, kernel='linear', probability=True)
        clf_low.fit(X_train[train_idx], labels_low_feature[train_idx])
        clf_high = SVC(C=svm_C, kernel='linear', probability=True)
        clf_high.fit(X_train[train_idx], labels_high_feature[train_idx])

        classifiers_low[j]  = pickle.dumps(clf_low)
        classifiers_high[j] = pickle.dumps(clf_high)
        print(f"Step {step} | SVM acc (low/fast=1): {clf_low.score(X_train[train_idx], labels_low_feature[train_idx]):.4f} "
              f"| (high/slow=1): {clf_high.score(X_train[train_idx], labels_high_feature[train_idx]):.4f}")

    for q_target_val, tag, clfs in [(q_high_val, f'{FEATURE_TAG}_high', classifiers_high),
                                    (q_low_val,  f'{FEATURE_TAG}_low',  classifiers_low)]:
        fname = f"FM_steering_vecs_regression_{layer_num}_{tag}{suffix}.pt"
        save_path = fname if output_dir is None else os.path.join(output_dir, fname)
        torch.save({
            'classifier':  classifier_vecs,
            'classifiers': clfs,
            'steering':    steering_vecs,
            'q_target':    q_target_val,
            'low_q':       low_q,
            'high_q':      high_q,
        }, save_path)
        print(f"Saved: {save_path}")


# ─────────────────────────────────────────────
# Optimal Transport helpers
# ─────────────────────────────────────────────

def get_OT_flow_matching(repr, feature, low_thresh, high_thresh, ot_type='lowrank_sinkhorn', plot=False):
    # TODO: Add support to sample random N points from the two distributions for OT. Depends on space the OT data takes to store
    import ot
    high_feature_idx = np.where(feature > high_thresh)[0]
    low_feature_idx = np.where(feature < low_thresh)[0]

    print('Number of Xs : ' , len(low_feature_idx))
    print('Number of Xs : ' , len(high_feature_idx))
    
    Xs = repr[high_feature_idx]
    Xt = repr[low_feature_idx]
    if ot_type == 'linear':
        ot_mapping_linear = ot.da.MappingTransport(
            kernel="linear", mu=1e0, eta=1e-8, bias=True, max_iter=20, verbose=True
        )
        ot_mapping_linear.fit(Xs=Xs, Xt=Xt)
        return ot_mapping_linear
    
    elif ot_type == 'lowrank_sinkhorn':
        val = ot.lowrank.lowrank_sinkhorn(X_s=Xs, X_t=Xt, reg=0.0001, numItermax=5000 , log=True)
        coupling = np.dot(np.dot(val[0], np.diag(1.0/val[2])), val[1].T)
        coupling = coupling.shape[0] * coupling
        return coupling, Xs, Xt
    




# ─────────────────────────────────────────────
# Steering vector training with OT
# ─────────────────────────────────────────────

def fm_steering_generate_OT(layer_num, extraction_dir, episode_list,
                          mode='classifier',  # 'classifier' ou 'regressor'
                          kernel='linear', steps=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], num_steps=10,
                          low_thresh=None, high_thresh=None,
                          low_quantile=0.25, high_quantile=0.75,
                          n_train_tasks=5, n_rollouts_per_task=1,
                          max_ep_per_task=None,
                          output_dir=None, suffix=''):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    couplings, Xs_list, Xt_list = [], [], []
    model_list = []
    for i in range(num_steps):
        couplings.append(-1); Xs_list.append(-1); Xt_list.append(-1)

    layer = f'model.vlm_with_expert.layer_hooks.1.{layer_num}'

    # ── Split train / test ────────────────────────────────────────────────
    split_idx      = n_train_tasks * n_rollouts_per_task
    train_episodes = episode_list[:split_idx]
    test_episodes  = episode_list[split_idx:]
    print(f"Train episodes ({len(train_episodes)}): {train_episodes}")
    print(f"Test  episodes ({len(test_episodes)}):  {test_episodes}")
    if max_ep_per_task is not None:
        print(f"Using first {max_ep_per_task} rollouts per task")

    all_data = load_episode_data(extraction_dir, episode_list, flat=True)

    # ── Speed distribution ────────────────────────────────────────────────
    feature_train, _ = get_feature_distribution(all_data, train_episodes, success_only=True, max_ep_per_task=max_ep_per_task)
    has_test = bool(test_episodes)
    if has_test:
        feature_test, _ = get_feature_distribution(all_data, test_episodes, success_only=True, max_ep_per_task=max_ep_per_task)

    if low_thresh is None:
        low_thresh  = float(np.quantile(feature_train, low_quantile))
    if high_thresh is None:
        high_thresh = float(np.quantile(feature_train, high_quantile))
    print(f"Thresholds -> low={low_thresh:.4f} (q{low_quantile}) | high={high_thresh:.4f} (q{high_quantile})")

    # ── Success labels ────────────────────────────────────────────────────
    success_train = get_success_labels(all_data, train_episodes, success_only=False, max_ep_per_task=max_ep_per_task)
    if has_test:
        success_test = get_success_labels(all_data, test_episodes, success_only=False, max_ep_per_task=max_ep_per_task)

    for j, step in enumerate(steps):
        train_repr = []

        for episode in train_episodes:
            train_repr.append(get_hidden_repr(all_data[episode], fm_time_step=step, layer_key=layer, max_ep_per_task=max_ep_per_task))

        repr_train = torch.cat(train_repr).float().cpu().numpy()
        repr_train = repr_train[success_train == 1]

        if has_test:
            test_repr = []
            for episode in test_episodes:
                test_repr.append(get_hidden_repr(all_data[episode], fm_time_step=step, layer_key=layer, max_ep_per_task=max_ep_per_task))
            repr_test = torch.cat(test_repr).float().cpu().numpy()
            repr_test = repr_test[success_test == 1]
            print(f"Step {step} | train: {len(repr_train)} | test: {len(repr_test)}")
        else:
            print(f"Step {step} | train: {len(repr_train)} | test: (none)")

        # ── Fit model ─────────────────────────────────────────────────────
        if mode == 'classifier':
            train_idx = list(np.where(feature_train < low_thresh)[0]) + \
                        list(np.where(feature_train > high_thresh)[0])
            y_train = (feature_train[train_idx] > high_thresh).astype(int)

            model = SVC(C=0.1, kernel=kernel, probability=True)
            model.fit(repr_train[train_idx], y_train)

            if has_test:
                test_idx = list(np.where(feature_test < low_thresh)[0]) + \
                           list(np.where(feature_test > high_thresh)[0])
                y_test = (feature_test[test_idx] > high_thresh).astype(int)
                acc = (model.predict(repr_test[test_idx]) == y_test).mean() * 100
                prob_fast_test = model.predict_proba(repr_test[test_idx])[:, 1]
                print(f"Step {step} | Classifier accuracy: {acc:.1f}% | mean prob_fast: {prob_fast_test.mean():.3f}")

        else:  # regressor
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('ridge',  Ridge(alpha=1.0))
            ])
            model.fit(repr_train, feature_train)
            if has_test:
                r2 = model.score(repr_test, feature_test)
                pearson_r, _ = pearsonr(model.predict(repr_test), feature_test)
                print(f"Step {step} | R²={r2:.3f} | Pearson={pearson_r:.3f}")

        model_list.append(model)

        # ── OT ────────────────────────────────────────────────────────────
        if has_test:
            repr_all  = np.concatenate([repr_train, repr_test])
            feature_all = np.concatenate([feature_train, feature_test])
        else:
            repr_all  = repr_train
            feature_all = feature_train

        ot_details      = get_OT_flow_matching(repr_all, feature_all,
                                               low_thresh=low_thresh,
                                               high_thresh=high_thresh)
        couplings[j] = ot_details[0]
        Xs_list[j]   = ot_details[1]
        Xt_list[j]   = ot_details[2]

    save_name = f"FM_steering_vecs_OT_{FEATURE_TAG}{suffix}.pt"
    save_path = save_name if output_dir is None else os.path.join(output_dir, save_name)

    torch.save({
        'mode':         mode,
        'kernel':       kernel,
        'layer_num':    layer_num,
        'low_thresh':   low_thresh,
        'high_thresh':  high_thresh,
        'low_quantile': low_quantile,
        'high_quantile':high_quantile,
        'classifiers':       model_list,
        'ot_couplings': [torch.tensor(c).float() if not isinstance(c, int) else c for c in couplings],
        'ot_Xs':        [torch.tensor(x).float() if not isinstance(x, int) else x for x in Xs_list],
        'ot_Xt':        [torch.tensor(t).float() if not isinstance(t, int) else t for t in Xt_list],
    }, save_path)
    print(f"Saved: {save_path}")

