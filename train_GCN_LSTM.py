import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
from datetime import datetime
from torch_geometric.loader import DataLoader as GraphDataLoader
from Utils.dataset import GraphDataset
from Utils.setLogger import setLogger
from Utils.utils import (
    same_seeds,
    plot_ground_motion,
    plot_story_response,
    plot_lossCurve,
)
from Models.GCN_LSTM import GCN_LSTM, LSTM, GCN_Encoder
from train_GCN_LSTM_arg import parse_args

# * Basic Settings
# args
args = parse_args()

# seed
same_seeds(args.seed)

# output dir
date_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
analysis_dir = os.path.join("./Results/GCN_LSTM", args.analysis_type)
if not os.path.exists(analysis_dir):
    os.mkdir(analysis_dir)
response_dir = os.path.join(analysis_dir, args.response_type)
if not os.path.exists(response_dir):
    os.mkdir(response_dir)
output_dir = os.path.join(response_dir, date_str)
os.mkdir(output_dir)
args.output_dir = output_dir

# logger
logger = setLogger(os.path.join(args.output_dir, "log.log"))

# device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
GPU_name = torch.cuda.get_device_name()
logger.info(f"\n\n** GPU Info **")
logger.info(f"{'=' * 100}")
logger.info(f"My GPU is {GPU_name}")
logger.info(f"{'=' * 100}")


# * Load Data
train_dataset = GraphDataset(
    folder_path_list=args.train_dataset_dir_list,
    response_type=args.response_type,
    numOfData_per_folder=args.max_train_dataset_samples,
)
eval_dataset = GraphDataset(
    folder_path_list=args.eval_dataset_dir_list,
    response_type=args.response_type,
    numOfData_per_folder=args.max_eval_dataset_samples,
)
logger.info(f"\n\n** Load Data **")
logger.info(f"{'=' * 100}")
logger.info(f"Response Type: {args.response_type}")
logger.info(f"Train dataset: {args.train_dataset_dir_list}")
logger.info(f"Eval dataset: {args.eval_dataset_dir_list}")
logger.info(f"# of effective train data: {len(train_dataset)}")
logger.info(f"# of effective eval data: {len(eval_dataset)}")
logger.info(f"{'=' * 100}")

if args.plot:
    train_fig_dir = os.path.join(output_dir, f"{args.response_type}_train")
    eval_fig_dir = os.path.join(output_dir, f"{args.response_type}_eval")
    os.mkdir(train_fig_dir)
    os.mkdir(eval_fig_dir)
    plot_train_dataset = GraphDataset(
        folder_path_list=args.train_dataset_dir_list,
        response_type=args.response_type,
        numOfData_per_folder=args.max_train_plot_samples,
    )
    plot_eval_dataset = GraphDataset(
        folder_path_list=args.eval_dataset_dir_list,
        response_type=args.response_type,
        numOfData_per_folder=args.max_eval_plot_samples,
    )

# * Data Preprocessing
# get normalization dictionary
normalized_item_dict = train_dataset.get_normalized_item_dict()
logger.info(f"\n\n** Get Normalization Dictionary **")
logger.info(f"{'=' * 100}")
logger.info(f"\nnormalization dictionary: \n{normalized_item_dict}")
logger.info(f"{'=' * 100}")

# save normalized_item_dict
torch.save(
    normalized_item_dict, os.path.join(args.output_dir, "normalized_item_dict.pth")
)

# Normalize train data
train_dataset.normalize_source(normalized_item_dict)
train_dataset.normalize_target(normalized_item_dict)
# Normalize eval data
eval_dataset.normalize_source(normalized_item_dict)
eval_dataset.normalize_target(normalized_item_dict)
# Normalize plot train/eval source data
if args.plot:
    plot_train_dataset.normalize_source(normalized_item_dict)
    plot_eval_dataset.normalize_source(normalized_item_dict)

# * Prepare GraphDataloader
assert args.batch_size <= len(
    train_dataset
), "batch size must equal or less than # of train_dataset"
assert args.batch_size <= len(
    eval_dataset
), "batch size must equal or less than # of eval_dataset"
train_loader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
eval_loader = GraphDataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
if args.plot:
    plot_train_loader = GraphDataLoader(plot_train_dataset, batch_size=1, shuffle=False)
    plot_eval_loader = GraphDataLoader(plot_eval_dataset, batch_size=1, shuffle=False)

# * Model (GCN_LSTM)

# Build GCN Encoder and LSTM
input_dim = train_dataset[0].x.size(-1)
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
logger.info(f"\n\n** Model Info **")
logger.info(f"{'=' * 100}")
logger.info(model)
logger.info(f"{'=' * 100}")

# * Loss function and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", factor=args.sch_factor, patience=args.patience
)

# * Plot Postprocess
def postprocess_and_plot(y_pred, Data, epoch, fig_folder):
    # y_pred = [trg_length, max_story]
    y_pred = y_pred.detach().cpu().numpy()
    # y_ture = [trg_length, max_story]
    y_true = torch.squeeze(Data.y).numpy()
    # ground_motion = [1, src_length]
    ground_motion = Data.ground_motion * normalized_item_dict["ground_motion"]
    # ground_motion = [src_length]
    ground_motion = torch.squeeze(ground_motion).numpy()

    # Data batch size = 1
    story = Data.story[0].item()
    sample_rate = Data.sample_rate[0].item()
    ground_motion_name = Data.ground_motion_name[0]
    src_seq_len = Data.time_steps[0].item()
    trg_seq_len = int(src_seq_len / 10)
    ## Plot ground motion
    plot_ground_motion(
        ground_motion[:src_seq_len], sample_rate, ground_motion_name, fig_folder
    )
    ## Plot Response
    plot_story_response(
        y_true[:trg_seq_len, :],
        y_pred[:trg_seq_len, :],
        sample_rate * 10,
        ground_motion_name,
        story,
        args.response_type,
        fig_folder,
        epoch,
    )


# * Train
train_loss_record = []
eval_loss_record = []
best_loss = np.inf
logger.info(f"\n\n** Train **")
logger.info(f"{'=' * 100}")
logger.info(f"  Packed Mode = {args.pack_mode}")
logger.info(f"  Compression Rate of Ground Motion = {args.compression_rate}")
logger.info(
    f"  Compression Rate of Response Sequence  = {int(args.compression_rate / 10)}"
)
logger.info(f"  Compressed Seqence Length  = {int(20000/args.compression_rate)}")
logger.info(f"  Num Epochs = {args.epoch}")
logger.info(f"  Num Train Examples = {len(train_dataset)}")
logger.info(f"  Num Eval Examples = {len(eval_dataset)}")
logger.info(f"  Batch Size = {args.batch_size}")
logger.info(f"  Evaluation Interval = {args.eval_interval}")
logger.info(f"  Plot Interval = {int(args.epoch / args.plot_num)}")
logger.info(f"{'=' * 100}")

for epoch in range(args.epoch):
    train_loss = 0
    model.train()
    ## Train
    for i, Data in enumerate(tqdm(train_loader)):
        # clean the gradient
        optimizer.zero_grad()
        # outputs = [batch, original_trg_seq_len, max_story]
        output = model(Data)
        # calculate loss
        y = Data.y.to(device)
        loss = criterion(output, y)
        train_loss += loss.item()
        # calculate gradient and back propagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    train_loss = train_loss / len(train_loader)
    train_loss_record.append([epoch, train_loss])
    logger.info(f"Epoch: {epoch:03d}, Train_Loss: {train_loss:.8f}")

    ## Eval
    if (epoch + 1) % args.eval_interval == 0:
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for i, Data in enumerate(tqdm(eval_loader)):
                output = model(Data)
                # calculate loss
                y = Data.y.to(device)
                loss = criterion(output, y)
                eval_loss += loss.item()
            eval_loss = eval_loss / len(eval_loader)
            eval_loss_record.append([epoch, eval_loss])
            scheduler.step(eval_loss)
            logger.info(f"Epoch: {epoch:03d}, Eval_Loss: {eval_loss:.8f}")

        # Save model if train_loss is better
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
            logger.info(f"Epoch: {epoch:03d}, Save the best checkpoint")

    ## Plot
    if args.plot:
        if (epoch + 1) % int(args.epoch / args.plot_num) == 0:
            logger.info(f"\nEpoch: {epoch:03d}, save response figures\n")
            with torch.no_grad():
                for i, Data in enumerate(tqdm(plot_train_loader)):
                    model.eval()
                    sample_folder = os.path.join(train_fig_dir, f"sample{i}")
                    if not os.path.isdir(sample_folder):
                        os.mkdir(sample_folder)
                    # y_pred = [1, trg_length, max_story]
                    y_pred = model(Data) * normalized_item_dict["y"]
                    # y_pred = [trg_length, max_story]
                    y_pred = torch.squeeze(y_pred)
                    postprocess_and_plot(y_pred, Data, (epoch + 1), sample_folder)

                for i, Data in enumerate(tqdm(plot_eval_loader)):
                    model.eval()
                    sample_folder = os.path.join(eval_fig_dir, f"sample{i}")
                    if not os.path.isdir(sample_folder):
                        os.mkdir(sample_folder)
                    # y_pred = [1, trg_length, max_story]
                    y_pred = model(Data) * normalized_item_dict["y"]
                    # y_pred = [trg_length, max_story]
                    y_pred = torch.squeeze(y_pred)
                    postprocess_and_plot(y_pred, Data, (epoch + 1), sample_folder)

# * Save and Plot Loss Record
torch.save(model.state_dict(), os.path.join(args.output_dir, "last_model.pt"))
train_loss_record = np.array(train_loss_record)
eval_loss_record = np.array(eval_loss_record)
np.save(os.path.join(args.output_dir, "train_loss_record"), train_loss_record)
np.save(os.path.join(args.output_dir, "eval_loss_record"), eval_loss_record)
plot_lossCurve(train_loss_record, eval_loss_record, args.output_dir)

# Save args
with open(os.path.join(args.output_dir, "arg.json"), "wt") as f:
    json.dump(vars(args), f, indent=4)
