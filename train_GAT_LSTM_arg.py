import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_type", type=str, default="Nonlinear_Analysis")
    parser.add_argument(
        "--train_dataset_dir_list",
        type=str,
        nargs="+",
        default=[
            "./Data/Nonlinear_Analysis/train/ChiChi_DBE",
            "./Data/Nonlinear_Analysis/train/NGAWest2_DBE",
            "./Data/Nonlinear_Analysis/train/ChiChi_MCE",
            "./Data/Nonlinear_Analysis/train/NGAWest2_MCE",
        ],
    )
    parser.add_argument(
        "--eval_dataset_dir_list",
        type=str,
        nargs="+",
        default=[
            "./Data/Nonlinear_Analysis/eval/ChiChi_DBE",
            "./Data/Nonlinear_Analysis/eval/NGAWest2_DBE",
            "./Data/Nonlinear_Analysis/eval/ChiChi_MCE",
            "./Data/Nonlinear_Analysis/eval/NGAWest2_MCE",
        ],
    )
    parser.add_argument("--max_train_dataset_samples", type=int, default=None)
    parser.add_argument("--max_eval_dataset_samples", type=int, default=None)
    parser.add_argument("--max_train_plot_samples", type=int, default=1)
    parser.add_argument("--max_eval_plot_samples", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./Results")
    parser.add_argument("--response_type", type=str, default="Displacement")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--plot_num", type=int, default=1)
    parser.add_argument("--plot", action="store_true")
    # model hyperparamter
    parser.add_argument("--compression_rate", type=int, default=40)
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--edge_dim", type=int, default=6)
    parser.add_argument("--gnn_embed_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--max_story", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--sch_factor", type=float, default=0.75)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--pack_mode", action="store_true")
    args = parser.parse_args(
        args=[
            "--pack_mode",
            # "--plot"
        ]
    )
    return args
