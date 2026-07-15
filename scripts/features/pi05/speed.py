import argparse

from scripts.features.pi05.utils import (vlm_steering_generate_regression_with_classifier,
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

    set_feature_func_speed()

    p.add_argument("--extraction-dir",    required=True)
    p.add_argument("--layer-nums",     nargs="+")
    p.add_argument("--episodes",      nargs="+", default=EPISODE_LIST)
    p.add_argument("--output-dir",    type=str)
    p.add_argument("--suffix",    type=str)
    p.add_argument("--extraction-suffix", type=str, default="",
                    help="Appended after the modality tag (_fm/_mean_vlm) when looking up extraction "
                         "folders, e.g. '_30' to read from the <episode>_fm_30 / <episode>_mean_vlm_30 dirs.")

    sub = p.add_subparsers(dest="command", required=True)

    # train-regression
    s = sub.add_parser("train-regression-fm", help="Train Ridge regression steering vector")
    s.add_argument("--steps", nargs="+", type=int, default=[7, 8, 9])
    s.add_argument("--low-q",  type=float, default=0.45)
    s.add_argument("--high-q", type=float, default=0.75)


    # train-vlm-regression
    s = sub.add_parser("train-regression-vlm", help="Train Ridge regression steering vector")
    s.add_argument("--low-q",  type=float, default=0.45)
    s.add_argument("--high-q", type=float, default=0.75)

    # train-vlm-diff-means
    s = sub.add_parser("train-diff-means-vlm", help="Train difference of means steering vector")
    s.add_argument("--low-q",  type=float, default=0.45)
    s.add_argument("--high-q", type=float, default=0.75)


    # train-OT-pegah
    s = sub.add_parser("train-OT-pegah", help="Train steering vector with Optimal Transport")
    s.add_argument("--steps",       nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    s.add_argument("--low-thresh",  type=float, default=None)
    s.add_argument("--high-thresh", type=float, default=None)
    s.add_argument("--low-q",  type=float, default=0.45)
    s.add_argument("--high-q", type=float, default=0.75)
    s.add_argument("--n-train-tasks", type=int, default=5,
                    help="Number of leading --episodes entries used to fit the OT coupling "
                         "(the rest are only used for the unused test-accuracy print). "
                         "Pass the full episode count to use all of them.")


    # Apply difference of means
    s = sub.add_parser("train-diff-means-fm", help="Steering vector via difference of means")
    s.add_argument("--steps",       nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8 , 9])
    s.add_argument("--low-q",  type=float, default=0.45)
    s.add_argument("--high-q", type=float, default=0.75)


    return p


def main():
    args = build_parser().parse_args()

    extraction_dir = args.extraction_dir
    layer_nums   = args.layer_nums
    episode_list = args.episodes

    if args.command == "train-regression-fm":
        for layer_num in layer_nums:
            fm_steering_generate_regression_with_classifier(layer_num, extraction_dir, episode_list,
                                            steps=args.steps,
                                            num_steps=len(args.steps),
                                            low_q=args.low_q,
                                            high_q=args.high_q,
                                            output_dir=args.output_dir,
                                            suffix=args.suffix,
                                            extraction_suffix=args.extraction_suffix)


    elif args.command == "train-regression-vlm":
        for layer_num in layer_nums:
            vlm_steering_generate_regression_with_classifier(layer_num, extraction_dir, episode_list,
                                            low_q=args.low_q,
                                            high_q=args.high_q,
                                            output_dir=args.output_dir,
                                            suffix=args.suffix,
                                            extraction_suffix=args.extraction_suffix)


    elif args.command == "train-diff-means-vlm":
        for layer_num in layer_nums:
            vlm_steering_generate_diff_means_with_classifier(layer_num, extraction_dir, episode_list,
                                            low_q=args.low_q,
                                            high_q=args.high_q,
                                            output_dir=args.output_dir,
                                            suffix=args.suffix,
                                            extraction_suffix=args.extraction_suffix)

        
    elif args.command == "train-OT-pegah":
        for layer_num in layer_nums:
            fm_steering_generate_OT(layer_num, extraction_dir, episode_list,
                                    steps=args.steps,
                                    num_steps=len(args.steps),
                                    low_quantile=args.low_q,
                                    high_quantile=args.high_q,
                                    output_dir=args.output_dir,
                                    suffix=args.suffix,
                                    extraction_suffix=args.extraction_suffix,
                                    n_train_tasks=args.n_train_tasks)

        




if __name__ == "__main__":
    main()