import argparse

from utils import (
    vlm_steering_generate_regression_with_classifier,
    vlm_steering_generate_diff_means_with_classifier,
    fm_steering_generate_regression_with_classifier,
    fm_steering_generate_OT,
    set_feature_func_eef_height_displacement,
)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

EPISODE_LIST = [f"libero_object_{i}" for i in range(1)]


def build_parser():
    p = argparse.ArgumentParser(
        description="Height displacement probe & steering analysis"
    )

    set_feature_func_eef_height_displacement()

    # p.add_argument("--videos-dir",    required=True)
    p.add_argument("--extraction-dir", required=True)
    # p.add_argument("--fld",           required=True)
    p.add_argument("--layer-nums", nargs="+")
    p.add_argument("--fm-step", type=int, default=9)
    p.add_argument("--episodes", nargs="+", default=EPISODE_LIST)
    p.add_argument("--output-dir", type=str)
    p.add_argument("--suffix", type=str)
    p.add_argument(
        "--extraction-suffix",
        type=str,
        default="",
        help="Appended after the modality tag (_fm/_mean_vlm) when looking up extraction "
        "folders, e.g. '_30' to read from the <episode>_fm_30 / <episode>_mean_vlm_30 dirs.",
    )

    sub = p.add_subparsers(dest="command", required=True)

    # train-classifier
    s = sub.add_parser("train-classifier", help="Train SVM steering vector")
    s.add_argument("--steps", nargs="+", type=int, default=[7, 8, 9])
    s.add_argument("--low-q", type=float, default=0.45)
    s.add_argument("--high-q", type=float, default=0.75)

    # train-regression
    s = sub.add_parser(
        "train-regression-fm", help="Train Ridge regression steering vector"
    )
    s.add_argument("--steps", nargs="+", type=int, default=[7, 8, 9])
    s.add_argument("--low-q", type=float, default=0.45)
    s.add_argument("--high-q", type=float, default=0.75)

    # train-vlm-regression
    s = sub.add_parser(
        "train-regression-vlm", help="Train Ridge regression steering vector"
    )
    s.add_argument("--low-q", type=float, default=0.45)
    s.add_argument("--high-q", type=float, default=0.75)

    # train-vlm-diff-means
    s = sub.add_parser(
        "train-diff-means-vlm", help="Train difference of means steering vector"
    )
    s.add_argument("--low-q", type=float, default=0.45)
    s.add_argument("--high-q", type=float, default=0.75)

    # train-OT
    s = sub.add_parser("train-OT", help="Train steering vector with Optimal Transport")
    s.add_argument(
        "--steps", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    s.add_argument("--low-thresh", type=float, default=None)
    s.add_argument("--high-thresh", type=float, default=None)
    s.add_argument("--mode", choices=["classifier", "regressor"], default="classifier")
    s.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    s.add_argument("--low-q", type=float, default=0.45)
    s.add_argument("--high-q", type=float, default=0.75)
    s.add_argument(
        "--n-episodes-per-task",
        type=int,
        default=None,
        help="Use only the first N rollouts per task folder (default: all)",
    )
    s.add_argument(
        "--n-train-tasks",
        type=int,
        default=5,
        help="Number of leading --episodes entries used to fit the OT coupling "
        "(the rest are only used for the unused test-accuracy print). "
        "Pass the full episode count to use all of them.",
    )

    # Apply difference of means
    s = sub.add_parser(
        "train-diff-means-fm", help="Steering vector via difference of means"
    )
    s.add_argument(
        "--steps", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    s.add_argument("--low-q", type=float, default=0.45)
    s.add_argument("--high-q", type=float, default=0.75)

    return p


def main():
    args = build_parser().parse_args()

    # videos_dir   = args.videos_dir
    extraction_dir = args.extraction_dir
    # fld          = args.fld
    layer_nums = args.layer_nums
    layer_num = layer_nums[0] if len(layer_nums) == 1 else None
    episode_list = args.episodes

    # stats = count_success_per_task(extraction_dir, episode_list)

    # for task, s in stats.items():
    #     print(f"{task:30s} | {s['success']:3d}/{s['total']:3d}  ({s['rate']*100:.1f}%)")

    if args.command == "nothing":
        print("No command specified, exiting.")
        return

    elif args.command == "train-regression-fm":
        for layer_num in layer_nums:
            fm_steering_generate_regression_with_classifier(
                layer_num,
                extraction_dir,
                episode_list,
                steps=args.steps,
                num_steps=len(args.steps),
                low_q=args.low_q,
                high_q=args.high_q,
                output_dir=args.output_dir,
                suffix=args.suffix,
            )

    elif args.command == "train-regression-vlm":
        for layer_num in layer_nums:
            vlm_steering_generate_regression_with_classifier(
                layer_num,
                extraction_dir,
                episode_list,
                low_q=args.low_q,
                high_q=args.high_q,
                output_dir=args.output_dir,
                suffix=args.suffix,
            )

    elif args.command == "train-diff-means-vlm":
        for layer_num in layer_nums:
            vlm_steering_generate_diff_means_with_classifier(
                layer_num,
                extraction_dir,
                episode_list,
                low_q=args.low_q,
                high_q=args.high_q,
                output_dir=args.output_dir,
                suffix=args.suffix,
            )

    elif args.command == "train-OT":
        for layer_num in layer_nums:
            fm_steering_generate_OT(
                layer_num,
                extraction_dir,
                episode_list,
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
                suffix=args.suffix or "",
            )


if __name__ == "__main__":
    main()
