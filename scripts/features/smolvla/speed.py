import argparse


from utils import (vlm_steering_generate_regression_with_classifier,
                                            vlm_steering_generate_diff_means_with_classifier, 
                                            fm_steering_generate_regression_with_classifier, 
                                            fm_steering_generate_OT,
                                            set_feature_func_speed)

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

EPISODE_LIST = [f'libero_object_{i}' for i in range(1)]

def build_parser():
    p = argparse.ArgumentParser(description="Speed probe & steering analysis")

    p.add_argument("--extraction-dir",    required=True)
    p.add_argument("--layer-nums",     nargs="+")
    p.add_argument("--episodes",      nargs="+", default=EPISODE_LIST)
    p.add_argument("--output-dir",    type=str)
    p.add_argument("--suffix",    type=str)

    sub = p.add_subparsers(dest="command", required=True)

    # train-regression-vlm-clf
    s = sub.add_parser("train-regression-vlm-clf", help="Train VLM regression steering vector with sklearn SVM gate")
    s.add_argument("--low-q",  type=float, default=0.25)
    s.add_argument("--high-q", type=float, default=0.75)
    s.add_argument("--svm-C",  type=float, default=0.1)

    # train-diff-means-vlm-clf
    s = sub.add_parser("train-diff-means-vlm-clf", help="Train VLM diff-of-means steering vector with sklearn SVM gate")
    s.add_argument("--low-q",  type=float, default=0.25)
    s.add_argument("--high-q", type=float, default=0.75)
    s.add_argument("--svm-C",  type=float, default=0.1)

    # train-regression-fm-clf
    s = sub.add_parser("train-regression-fm-clf", help="Train FM regression steering vector with sklearn SVM gate")
    s.add_argument("--steps",        nargs="+", type=int, default=list(range(10)))
    s.add_argument("--low-q",        type=float, default=0.25)
    s.add_argument("--high-q",       type=float, default=0.75)
    s.add_argument("--svm-C",        type=float, default=0.1)
    s.add_argument("--n-train-tasks", type=int,  default=None,
                   help="Use only the first N tasks for training (default: all episodes)")

    # train-OT
    s = sub.add_parser("train-OT", help="Train steering vector with Optimal Transport")
    s.add_argument("--steps",              nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    s.add_argument("--low-thresh",         type=float, default=None)
    s.add_argument("--high-thresh",        type=float, default=None)
    s.add_argument("--mode",   choices=["classifier", "regressor"], default="classifier")
    s.add_argument("--kernel", choices=["linear", "rbf"],           default="linear")
    s.add_argument("--low-q",              type=float, default=0.25)
    s.add_argument("--high-q",             type=float, default=0.75)
    s.add_argument("--n-episodes-per-task", type=int,  default=None,
                   help="Use only the first N rollouts per task folder (default: all)")
    s.add_argument("--n-train-tasks",      type=int,   default=4,
                   help="Number of tasks used for training the classifier (rest used for test)")

    s = sub.add_parser("nothing", help="do nothing")

    return p


def main():
    args = build_parser().parse_args()

    set_feature_func_speed()

    extraction_dir = args.extraction_dir
    layer_nums   = args.layer_nums
    layer_num    = layer_nums[0] if len(layer_nums) == 1 else None
    episode_list = args.episodes

    if args.command == "nothing":
        print("No command specified, exiting.")
        return

    if args.command == "train-regression-vlm-clf":
        for layer_num in layer_nums:
            vlm_steering_generate_regression_with_classifier(
                layer_num, extraction_dir, episode_list,
                low_q=args.low_q,
                high_q=args.high_q,
                output_dir=args.output_dir,
                suffix=args.suffix or '',
            )

    elif args.command == "train-diff-means-vlm-clf":
        for layer_num in layer_nums:
            vlm_steering_generate_diff_means_with_classifier(
                layer_num, extraction_dir, episode_list,
                low_q=args.low_q,
                high_q=args.high_q,
                output_dir=args.output_dir,
                suffix=args.suffix or '',
                svm_C=args.svm_C,
            )

    elif args.command == "train-regression-fm-clf":
        for layer_num in layer_nums:
            fm_steering_generate_regression_with_classifier(
                layer_num, extraction_dir, episode_list,
                steps=args.steps,
                num_steps=len(args.steps),
                low_q=args.low_q,
                high_q=args.high_q,
                svm_C=args.svm_C,
                n_train_tasks=args.n_train_tasks,
                output_dir=args.output_dir,
                suffix=args.suffix or '',
            )

    elif args.command == "train-OT":
        fm_steering_generate_OT(layer_num, extraction_dir, episode_list,
                                mode=args.mode,
                                kernel=args.kernel,
                                steps=args.steps,
                                num_steps=len(args.steps),
                                low_thresh=args.low_thresh,
                                high_thresh=args.high_thresh,
                                low_quantile=args.low_q,
                                high_quantile=args.high_q,
                                n_train_tasks=args.n_train_tasks,
                                max_ep_per_task=args.n_episodes_per_task,
                                output_dir=args.output_dir,
                                suffix=args.suffix or '')


if __name__ == "__main__":
    main()