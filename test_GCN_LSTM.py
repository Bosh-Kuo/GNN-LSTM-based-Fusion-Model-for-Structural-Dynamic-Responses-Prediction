import argparse
import json
import os
import numpy as np
import torch
from Models.GCN_LSTM import GCN_LSTM, LSTM, GCN_Encoder
from torch_geometric.loader import DataLoader as GraphDataLoader
from tqdm import tqdm
from Utils.dataset import GraphDataset
from Utils.utils import (
    R_square,
    normalized_MSE,
    peakError,
    plot_error_histogram,
    plot_ground_motion,
    plot_scatter,
    plot_story_response,
    same_seeds,
)

# * Basic Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./Results/GCN_LSTM/Nonlinear_Analysis/Acceleration/2022_09_09__18_27_46",
    )
    parser.add_argument("--MAX_plot_num", type=int, default=4)
    args = parser.parse_args()
    return args


# args
args = parse_args()
args_dict = vars(args)
with open(os.path.join(args.output_dir, "arg.json"), "rt") as f:
    args_dict.update(json.load(f))

# seed
same_seeds(args.seed)

# device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
GPU_name = torch.cuda.get_device_name()
print(f"\n\n** GPU Info **")
print(f"{'=' * 100}")
print(f"My GPU is {GPU_name}")
print(f"{'=' * 100}")

# * Load Data
eval_dataset = GraphDataset(
    folder_path_list=args.eval_dataset_dir_list,
    response_type=args.response_type,
    numOfData_per_folder=None,
)
print(f"\n\n** Load Data **")
print(f"{'=' * 100}")
print(f"Response Type: {args.response_type}")
print(f"Eval dataset: {args.eval_dataset_dir_list}")
print(f"# of effective eval data: {len(eval_dataset)}")
print(f"{'=' * 100}")

# * Data Preprocessing
# get normalized_item_dict
normalized_item_dict = torch.load(
    os.path.join(args.output_dir, "normalized_item_dict.pth")
)
print(f"\n\n** Get Normalization Dictionary **")
print(f"{'=' * 100}")
print(f"\nnormalization dictionary: \n{normalized_item_dict}")
print(f"{'=' * 100}")

# Normalize eval data
# plot 不需要更新 model，不需要算 loss 所以不需要 normalize target
eval_dataset.normalize_source(normalized_item_dict)

# * Prepare GraphDataloader
eval_loader = GraphDataLoader(eval_dataset, batch_size=1, shuffle=False)

# * Model (GAT_LSTM)
# Build GCN Encoder and LSTM
input_dim = eval_dataset[0].x.size(-1)
GCN_Encoder_constructor_args = {
    "input_dim": input_dim,
    "hid_dim": args.hid_dim,
    "gnn_embed_dim": args.gnn_embed_dim,
    "dropout": args.dropout,
}

LSTM_constructor_args = {
    "gnn_embed_dim": args.gnn_embed_dim,
    "hid_dim": args.hid_dim,
    "n_layers": args.n_layers,
    "dropout": args.dropout,
    "pack_mode": args.pack_mode,
    "compression_rate": args.compression_rate,
    "max_story": args.max_story,
}
gcn_encoder = GCN_Encoder(**GCN_Encoder_constructor_args)
lstm = LSTM(**LSTM_constructor_args)
model = GCN_LSTM(gcn_encoder, lstm, device).to(device)
print(f"\n\n** Model Info **")
print(f"{'=' * 100}")
print(model)
print(f"{'=' * 100}")


# * Load best model
model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))

# * create folder
eval_fig_dir = os.path.join(args.output_dir, f"{args.response_type}_inference")
if not os.path.exists(eval_fig_dir):
    os.mkdir(eval_fig_dir)

normMSE_list = []
R_square_list = []
peakError_list = []
time_length_list = []
first_period_list = []
height_list = []
with torch.no_grad():
    for i, Data in enumerate(tqdm(eval_loader)):
        model.eval()
        # y_pred = [1, trg_length, max_story]
        y_pred = model(Data) * normalized_item_dict["y"]
        # y_pred = [trg_length, max_story]
        y_pred = torch.squeeze(y_pred).detach().cpu().numpy()
        # y_ture = [trg_length, max_story]
        y_true = torch.squeeze(Data.y).numpy()
        # ground_motion = [1, src_length]
        ground_motion = Data.ground_motion * normalized_item_dict["ground_motion"]
        # ground_motion = [src_length]
        ground_motion = torch.squeeze(ground_motion).numpy()
        # Data batch size = 1
        story = Data.story[0].item()
        first_period = (Data.x[0, 6] * normalized_item_dict["x"]["period"]).item()
        sample_rate = Data.sample_rate[0].item()
        ground_motion_name = Data.ground_motion_name[0]
        src_seq_len = Data.time_steps[0].item()
        trg_seq_len = int(src_seq_len / 10)

        if i % int(len(eval_dataset) / args.MAX_plot_num) == 0:
            sample_folder = os.path.join(eval_fig_dir, f"sample{i}")
            if not os.path.isdir(sample_folder):
                os.mkdir(sample_folder)
            ## Plot ground motion
            plot_ground_motion(
                ground_motion[:src_seq_len],
                sample_rate,
                ground_motion_name,
                sample_folder,
            )
            ## Plot Response
            plot_story_response(
                y_true[:trg_seq_len, :],
                y_pred[:trg_seq_len, :],
                sample_rate * 10,
                ground_motion_name,
                story,
                args.response_type,
                sample_folder,
            )

        ## Calculate error
        normMSE_list += [
            normalized_MSE(y_true[:trg_seq_len, i], y_pred[:trg_seq_len, i])
            for i in range(story)
        ]
        R_square_list += [
            R_square(y_true[:trg_seq_len, i], y_pred[:trg_seq_len, i])
            for i in range(story)
        ]
        peakError_list += [
            peakError(y_true[:trg_seq_len, i], y_pred[:trg_seq_len, i])
            for i in range(story)
        ]
        time_length_list += [int(src_seq_len * sample_rate) for i in range(story)]
        first_period_list += [first_period for i in range(story)]
        height_list += [story for i in range(story)]

        ## check
        # if sum([R_square(y_true[:trg_seq_len, i], y_pred[:trg_seq_len, i]) for i in range(story)]) / story < 0.8:
        #     print(f"R_square: {sum([R_square(y_true[:trg_seq_len, i], y_pred[:trg_seq_len, i]) for i in range(story)]) / story}, data: {i}, ground motion: {Data.ground_motion_name[0]}")
        if any(
            [
                R_square(y_true[:trg_seq_len, i], y_pred[:trg_seq_len, i]) < 0.5
                for i in range(story)
            ]
        ):
            print(
                f"R_square: {sum([R_square(y_true[:trg_seq_len, i], y_pred[:trg_seq_len, i]) for i in range(story)]) / story}, data: {i}, ground motion: {Data.ground_motion_name[0]}"
            )

# * plot statistics figures
## Histogram
# normMSE
plot_error_histogram(
    error=np.array(normMSE_list),
    response_type=args.response_type,
    error_type="NMSE",
    save_folder=eval_fig_dir,
)
# R_square
plot_error_histogram(
    error=np.array(R_square_list),
    response_type=args.response_type,
    error_type="R_squared",
    save_folder=eval_fig_dir,
)
# peak Error
plot_error_histogram(
    error=np.array(peakError_list),
    response_type=args.response_type,
    error_type="peak_Error(%)",
    save_folder=eval_fig_dir,
)

## Scatter
# X = sequence length
# normMSE
plot_scatter(
    x=time_length_list,
    y=normMSE_list,
    colors=height_list,
    x_label="sequence length (sec)",
    y_label="NMSE",
    color_label="building height(F)",
    response_type=args.response_type,
    save_folder=eval_fig_dir,
)
# R_square
plot_scatter(
    x=time_length_list,
    y=R_square_list,
    colors=height_list,
    x_label="sequence length (sec)",
    y_label="R_squared",
    color_label="building height(F)",
    response_type=args.response_type,
    save_folder=eval_fig_dir,
)
# peak Error
plot_scatter(
    x=time_length_list,
    y=peakError_list,
    colors=height_list,
    x_label="sequence length (sec)",
    y_label="peak_Error(%)",
    color_label="building height(F)",
    response_type=args.response_type,
    save_folder=eval_fig_dir,
)

# X = first period
# normMSE
plot_scatter(
    x=first_period_list,
    y=normMSE_list,
    colors=height_list,
    x_label="first period (sec)",
    y_label="NMSE",
    color_label="building height(F)",
    response_type=args.response_type,
    save_folder=eval_fig_dir,
)
# R_square
plot_scatter(
    x=first_period_list,
    y=R_square_list,
    colors=height_list,
    x_label="first period (sec)",
    y_label="R_squared",
    color_label="building height(F)",
    response_type=args.response_type,
    save_folder=eval_fig_dir,
)
# peak Error
plot_scatter(
    x=first_period_list,
    y=peakError_list,
    colors=height_list,
    x_label="first period (sec)",
    y_label="peak_Error(%)",
    color_label="building height(F)",
    response_type=args.response_type,
    save_folder=eval_fig_dir,
)
