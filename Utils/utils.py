import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def peakError(seq_ture, seq_pred):
    """use numpy array
    seq_ture = [seq_len]
    seq_pred = [seq_len]
    """
    error = (
        (np.max(np.absolute(seq_pred)) - np.max(np.absolute(seq_ture)))
        / np.max(np.absolute(seq_ture))
        * 100
    )  # (%)
    return error


def normalized_MSE(seq_ture, seq_pred):
    """use numpy array
    seq_ture = [seq_len]
    seq_pred = [seq_len]
    """
    N = seq_ture.size
    max_val = max(np.max(np.absolute(seq_ture)), np.max(np.absolute(seq_pred)))
    seq_ture, seq_pred = seq_ture / max_val, seq_pred / max_val
    error = (1 / N) * np.sum((seq_ture - seq_pred) ** 2)
    return error


def R_square(seq_ture, seq_pred):
    """use numpy array
    seq_ture = [seq_len]
    seq_pred = [seq_len]
    """
    y_mean = np.mean(seq_ture)
    SS_tot = np.sum((seq_ture - y_mean) ** 2)
    SS_res = np.sum((seq_ture - seq_pred) ** 2)
    return 1 - SS_res / SS_tot


def plot_error_residualDisp(
    residual_response, residual_error, compression_rate_list, save_path
):
    """use numpy array
    residual_response = [# of data]
    residual_error[0] = [# of data](compression rate 0)
    residual_error[1] = [# of data](compression rate 1)
    compression_rate_list[0] = compression rate 0
    compression_rate_list[1] = compression rate 1
    """

    x = residual_response + residual_response
    y = residual_error[0] + residual_error[1]
    colors = [compression_rate_list[0] for i in range(len(residual_response))] + [
        compression_rate_list[1] for i in range(len(residual_response))
    ]
    data = pd.DataFrame(
        {"residual_response": x, "residual_error": y, "compression rate": colors}
    )
    plt.figure(figsize=(10, 6), facecolor="w")
    plt.tick_params(labelsize=15)
    sns.set_style("whitegrid")
    sns.lmplot(
        x="residual_response",
        y="residual_error",
        hue="compression rate",
        data=data,
        height=6,
        aspect=1.5,
        legend_out=False,
    )
    plt.xlabel("Absolute Residual Displacement ($mm$)", fontsize=18)
    plt.ylabel("Absolute Error ($mm$)", fontsize=18)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_lossCurve(train_loss_record, eval_loss_record, save_dir):
    """use numpy array
    train_loss_record[:,0] = [epoch]
    train_loss_record[:,1] = [loss]
    """
    plt.figure(figsize=(8, 6))
    plt.tick_params(labelsize=15)
    plt.plot(train_loss_record[:, 0], train_loss_record[:, 1], label="train")
    plt.plot(eval_loss_record[:, 0], eval_loss_record[:, 1], label="valid")
    plt.yscale("log")
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("MSE Loss", fontsize=18)
    plt.title("Loss curve", fontsize=20)
    plt.legend(loc="upper right", fontsize=18)
    plt.grid()
    plt.savefig(os.path.join(save_dir, "lossCurve.png"))
    plt.close()


def plot_ground_motion(ground_motion, time_step, ground_motion_name, save_folder):
    """use numpy array
    ground_motion = [src_length]
    """
    timeline = (np.arange(ground_motion.shape[0]) + 1) * time_step
    # First plot ground motion shape
    plt.figure(figsize=(30, 12), facecolor="w")
    plt.plot(timeline, ground_motion, color="black")
    plt.tick_params(labelsize=16)
    plt.xlabel("Time(sec)", fontsize=24)
    plt.ylabel("Acceleration ($mm/s^2$)", fontsize=24)
    plt.title(f"{ground_motion_name}", fontsize=28)
    plt.grid()
    plt.savefig(os.path.join(save_folder, "ground_motion.png"))
    plt.close()


def plot_story_response(
    y_true,
    y_pred,
    time_step,
    ground_motion_name,
    story,
    response_type,
    save_folder,
    epoch=None,
):
    """use numpy array
    y_true = [trg_length, max_story(8)]
    y_pred = [trg_length, max_story(8)]
    sample_rate: output time step size
    story: int
    """
    if response_type == "Acceleration":
        unit = "$mm/s^2$"
    elif response_type == "Velocity":
        unit = "$mm/s$"
    elif response_type == "Displacement":
        unit = "$mm$"

    timeline = (np.arange(y_true.shape[0]) + 1) * time_step
    for i in range(story):
        F = i + 1
        story_MSE = normalized_MSE(y_true[:, i], y_pred[:, i])
        story_R_square = R_square(y_true[:, i], y_pred[:, i])
        story_peakError = peakError(y_true[:, i], y_pred[:, i])
        plt.figure(figsize=(30, 12), facecolor="w")
        plt.plot(timeline, y_true[:, i], label="true", color="silver", linewidth=3)
        plt.plot(timeline, y_pred[:, i], label="pred", color="black", linewidth=1)
        plt.tick_params(labelsize=16)
        plt.xlabel("Time(sec)", fontsize=24)
        plt.ylabel(f"{response_type} ({unit})", fontsize=24)
        plt.title(
            f"{ground_motion_name}  \n{F}F {response_type} \n NMSE = {story_MSE:.5f}, $R^2$ = {story_R_square:.3f}, peak Error = {story_peakError:.3f} %",
            fontsize=24,
        )
        plt.legend(loc="upper right", fontsize=18)
        plt.grid()
        if epoch is None:
            plt.savefig(os.path.join(save_folder, f"{F}F_{response_type}.png"))
        else:
            plt.savefig(
                os.path.join(save_folder, f"{F}F_{response_type}_epoch{epoch}.png")
            )
        plt.close()


def plot_story_response_overlay(
    y_true,
    y_pred_list,
    label_list,
    time_step,
    ground_motion_name,
    story,
    response_type,
    save_folder,
    epoch=None,
):
    """use numpy array
    y_true = [trg_length, max_story(8)]
    y_pred = [trg_length, max_story(8)]
    sample_rate: output time step size
    story: int
    """
    if response_type == "Acceleration":
        unit = "$mm/s^2$"
    elif response_type == "Velocity":
        unit = "$mm/s$"
    elif response_type == "Displacement":
        unit = "$mm$"

    timeline = (np.arange(y_true.shape[0]) + 1) * time_step
    for i in range(story):
        F = i + 1
        plt.figure(figsize=(30, 12), facecolor="w")
        plt.plot(
            timeline[::2],
            y_true[::2, i],
            label="true",
            color="black",
            linewidth=3,
            alpha=0.65,
        )
        colors = [
            "blue",
            "red",
            "green",
        ]
        lines = ["--", "-.", "--"]
        for j in range(len(y_pred_list)):
            plt.plot(
                timeline[::2],
                y_pred_list[j][::2, i],
                label=label_list[j],
                color=colors[j],
                linewidth=2,
                alpha=0.7,
                linestyle=lines[j],
            )
        plt.tick_params(labelsize=28)  # 原 16
        plt.xlabel("Time(sec)", fontsize=36)  # 原 24
        plt.ylabel(f"{response_type} ({unit})", fontsize=36)  # 原 24
        plt.title(f"{ground_motion_name}  \n{F}F {response_type}", fontsize=36)  # 原 24
        plt.legend(fontsize=32)  # 原 18
        plt.grid()
        if epoch is None:
            plt.savefig(os.path.join(save_folder, f"{F}F_{response_type}.png"))
        else:
            plt.savefig(
                os.path.join(save_folder, f"{F}F_{response_type}_epoch{epoch}.png")
            )
        plt.close()


def plot_error_histogram(error, response_type, error_type, save_folder):
    """use numpy array
    error = [# of data]
    """
    mean = np.mean(error)
    deviation = np.std(error)
    plt.figure(figsize=(12, 12), facecolor="w")
    plt.hist(error, bins=20, color="slategrey", edgecolor="black")
    plt.tick_params(labelsize=20)
    plt.xlabel(f"{error_type}", fontsize=24)
    plt.ylabel("# of data", fontsize=24)
    plt.title(
        f"{response_type} {error_type} Distribution \n $\mu={mean:.4f}$, \n $\sigma ={deviation:.4f}$",
        fontsize=24,
    )
    plt.savefig(
        os.path.join(save_folder, f"{response_type}_{error_type.replace('(%)','')}.png")
    )
    plt.close()


def plot_error_histogram_overlay(
    error_list, label_list, response_type, error_type, save_folder
):
    """use numpy array
    error = [# of data]
    """
    title_string = ""
    for i in range(len(error_list)):
        mean = np.mean(error_list[i])
        deviation = np.std(error_list[i])
        title_string += (
            f"\n{label_list[i]}: $\mu={mean:.4f}$, $\sigma ={deviation:.4f}$"
        )

    plt.figure(figsize=(12, 12), facecolor="w")
    plt.hist(error_list, bins=20, label=label_list, edgecolor="black")
    plt.tick_params(labelsize=20)
    plt.xlabel(f"{error_type}", fontsize=28)  # 原 24
    plt.ylabel("# of data", fontsize=28)  # 原 24
    plt.legend(fontsize=24)  # 原 16
    plt.title(
        f"{response_type} {error_type} Distribution" + title_string, fontsize=28
    )  # 原 24
    plt.savefig(
        os.path.join(save_folder, f"{response_type}_{error_type.replace('(%)','')}.png")
    )
    plt.close()


def plot_scatter(
    x, y, colors, x_label, y_label, color_label, response_type, save_folder
):
    """use numpy array
    x = [# of data]
    y = [# of data]
    """
    data = pd.DataFrame({x_label: x, y_label: y, color_label: colors})
    if x_label == "sequence length (sec)":
        xlimit = (0, 110)
    elif x_label == "first period (sec)":
        xlimit = (0, 1.6)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.6)
    axe = sns.jointplot(
        data=data, x=x_label, y=y_label, xlim=xlimit, hue=color_label, height=12, s=120
    )
    axe.fig.set_figwidth(14)
    axe.fig.subplots_adjust(top=0.92)
    axe.fig.suptitle(f"{x_label} - {y_label}", fontsize=32)  # 原 24
    plt.savefig(
        os.path.join(
            save_folder, f"{response_type}_{x_label}_{y_label.replace('(%)','')}.png"
        )
    )
    plt.close()


def plot_attention(
    x_coord, y_coord, z_coord, node_color, edge_color, edge_pos_xyz, save_folder
):
    fig = plt.figure(figsize=(15, 15), facecolor="w")
    ax = fig.add_subplot(111, projection="3d", facecolor="w")
    ax.set_axis_off()

    # plot nodes
    p = ax.scatter(
        x_coord,
        y_coord,
        z_coord,
        c=node_color,
        s=250,
        alpha=0.75,
        edgecolors="black",
        cmap="Greys",
    )
    ax.set_box_aspect((np.ptp(x_coord), np.ptp(y_coord), np.ptp(z_coord)))
    # plot edges
    for e, color in zip(edge_pos_xyz, edge_color):
        xx = [e[0], e[3]]
        yy = [e[1], e[4]]
        zz = [e[2], e[5]]
        ax.plot(
            xx,
            yy,
            zz,
            c="grey",
            linewidth=10 * color[0] if 10 * color[0] >= 0.5 else 0.5,
        )

    cbar = fig.colorbar(p, shrink=0.5, aspect=4, pad=0.001)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label("Attention Weights", size=24)
    # fig.suptitle("Visualization of Attention Weights", fontsize=30)
    plt.savefig(os.path.join(save_folder, "attention.png"))
    plt.close()
