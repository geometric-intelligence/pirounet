import logging
import os
import random
import time

import default_config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = default_config.which_device
import matplotlib
import matplotlib.pyplot as plt
import models.utils as utils
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import torch
import wandb
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import juggle_axes

point_labels = [
    "ARIEL",
    "C7",
    "CLAV",
    "LANK",
    "LBHD",
    "LBSH",
    "LBWT",
    "LELB",
    "LFHD",
    "LFRM",
    "LFSH",
    "LFWT",
    "LHEL",
    "LIEL",
    "LIHAND",
    "LIWR",
    "LKNE",
    "LKNI",
    "LMT1",
    "LMT5",
    "LOHAND",
    "LOWR",
    "LSHN",
    "LTHI",
    "LTOE",
    "LUPA",
    # 'LabelingHips',
    "MBWT",
    "MFWT",
    "RANK",
    "RBHD",
    "RBSH",
    "RBWT",
    "RELB",
    "RFHD",
    "RFRM",
    "RFSH",
    "RFWT",
    "RHEL",
    "RIEL",
    "RIHAND",
    "RIWR",
    "RKNE",
    "RKNI",
    "RMT1",
    "RMT5",
    "ROHAND",
    "ROWR",
    "RSHN",
    "RTHI",
    "RTOE",
    "RUPA",
    "STRN",
    # 'SolvingHips',
    "T10",
]

# This array defines the points between which skeletal lines should
# be drawn. Each segment is defined as a line between a group of one
# or more named points -- the line will be drawn at the average position
# of the points in the group
skeleton_lines = [
    #     ( (start group), (end group) ),
    (("LHEL",), ("LTOE",)),  # toe to heel
    (("RHEL",), ("RTOE",)),
    (("LKNE", "LKNI"), ("LHEL",)),  # heel to knee
    (("RKNE", "RKNI"), ("RHEL",)),
    (("LKNE", "LKNI"), ("LFWT", "RFWT", "LBWT", "RBWT")),  # knee to "navel"
    (("RKNE", "RKNI"), ("LFWT", "RFWT", "LBWT", "RBWT")),
    (
        ("LFWT", "RFWT", "LBWT", "RBWT"),
        (
            "STRN",
            "T10",
        ),
    ),  # "navel" to chest
    (
        (
            "STRN",
            "T10",
        ),
        (
            "CLAV",
            "C7",
        ),
    ),  # chest to neck
    (
        (
            "CLAV",
            "C7",
        ),
        (
            "LFSH",
            "LBSH",
        ),
    ),  # neck to shoulders
    (
        (
            "CLAV",
            "C7",
        ),
        (
            "RFSH",
            "RBSH",
        ),
    ),
    (
        (
            "LFSH",
            "LBSH",
        ),
        (
            "LELB",
            "LIEL",
        ),
    ),  # shoulders to elbows
    (
        (
            "RFSH",
            "RBSH",
        ),
        (
            "RELB",
            "RIEL",
        ),
    ),
    (
        (
            "LELB",
            "LIEL",
        ),
        (
            "LOWR",
            "LIWR",
        ),
    ),  # elbows to wrist
    (
        (
            "RELB",
            "RIEL",
        ),
        (
            "ROWR",
            "RIWR",
        ),
    ),
    (("LFHD",), ("LBHD",)),  # draw lines around circumference of the head
    (("LBHD",), ("RBHD",)),
    (("RBHD",), ("RFHD",)),
    (("RFHD",), ("LFHD",)),
    (("LFHD",), ("ARIEL",)),  # connect circumference points to top of head
    (("LBHD",), ("ARIEL",)),
    (("RBHD",), ("ARIEL",)),
    (("RFHD",), ("ARIEL",)),
]

# Normal, connected skeleton:
skeleton_idxs = []
for g1, g2 in skeleton_lines:
    entry = []
    entry.append([point_labels.index(line) for line in g1])
    entry.append([point_labels.index(line) for line in g2])
    skeleton_idxs.append(entry)


def getlinesegments(seq, zcolor=None, cmap=None):
    """Calculate coordinates for the lines."""
    xline = np.zeros((seq.shape[0], len(skeleton_idxs), 3, 2))

    if cmap:
        colors = np.zeros((len(skeleton_idxs), 4))
    for i, (g1, g2) in enumerate(skeleton_idxs):
        xline[:, i, :, 0] = np.mean(seq[:, g1], axis=1)
        xline[:, i, :, 1] = np.mean(seq[:, g2], axis=1)
        if cmap is not None:
            colors[i] = cmap(0.5 * (zcolor[g1].mean() + zcolor[g2].mean()))
    if cmap:
        return xline, colors
    else:
        return xline


def putlines(ax, segments, color=None, lw=2.5, alpha=None):
    """Put line segments on the given axis with given colors."""
    lines = []
    # Main skeleton
    for i in range(len(skeleton_idxs)):
        if isinstance(color, (list, tuple, np.ndarray)):
            c = color[i]
        else:
            c = color
        line = ax.plot(
            np.linspace(segments[i, 0, 0], segments[i, 0, 1], 2),
            np.linspace(segments[i, 1, 0], segments[i, 1, 1], 2),
            np.linspace(segments[i, 2, 0], segments[i, 2, 1], 2),
            color=c,
            alpha=alpha,
            lw=lw,
        )[0]
        lines.append(line)
    return lines


def animatestick(
    seq,
    fname,
    ghost=None,
    ghost_shift=1,
    figsize=(12, 8),
    zcolor=None, #'blue',
    pointer=None,
    ax_lims=(-0.4, 0.4),
    speed=45,
    dot_size=20,
    dot_alpha=0.5,
    lw=2.5,
    cmap="inferno",
    pointer_color="black",
    condition=None,
):
    """Create skeleton animation."""
    # Put data on CPU and convert to numpy array
    if torch.is_tensor(seq):
        seq = seq.cpu().data.numpy()
    if ghost is not None:
        ghost = ghost.cpu().data.numpy()

    if zcolor is None:
        zcolor = np.zeros(seq.shape[1])

    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)

    # The following lines eliminate background lines/axes:
    ax.axis("off")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    if ghost_shift and ghost is not None:
        seq = seq.copy()
        ghost = ghost.copy()
        seq[:, :, 0] -= ghost_shift
        ghost[:, :, 0] += ghost_shift

    cm = matplotlib.cm.get_cmap(cmap)

    pts = ax.scatter(
        seq[0, :, 0],
        seq[0, :, 1],
        seq[0, :, 2],
        c=zcolor,
        s=dot_size,
        cmap=cm,
        alpha=dot_alpha,
    )

    ghost_color = "blue"

    if ghost is not None:
        pts_g = ax.scatter(
            ghost[0, :, 0],
            ghost[0, :, 1],
            ghost[0, :, 2],
            c=ghost_color,
            s=dot_size,
            alpha=dot_alpha,
        )

    if ax_lims:
        ax.set_xlim(*ax_lims)
        ax.set_ylim(*ax_lims)
        ax.set_zlim(0, ax_lims[1] - ax_lims[0])
    plt.close(fig)

    xline, colors = getlinesegments(seq, zcolor, cm)
    lines = putlines(ax, xline[0], colors, lw=lw, alpha=0.9)

    if ghost is not None:
        xline_g = getlinesegments(ghost)
        lines_g = putlines(ax, xline_g[0], ghost_color, lw=lw, alpha=1.0)

    if pointer is not None:
        vR = 0.15
        dX, dY = vR * np.cos(pointer), vR * np.sin(pointer)
        zidx = point_labels.index("CLAV")
        X = seq[:, zidx, 0]
        Y = seq[:, zidx, 1]
        Z = seq[:, zidx, 2]
        quiv = ax.quiver(X[0], Y[0], Z[0], dX[0], dY[0], 0, color=pointer_color)
        ax.quiv = quiv

    if condition is not None:
        title = str("Generated for label " + str(condition))

    def update(t):
        if condition is not None:
            plt.title(title)
        pts._offsets3d = juggle_axes(seq[t, :, 0], seq[t, :, 1], seq[t, :, 2], "z")
        for i, l in enumerate(lines):
            l.set_data(xline[t, i, :2])
            l.set_3d_properties(xline[t, i, 2])

        if ghost is not None:
            pts_g._offsets3d = juggle_axes(
                ghost[t, :, 0], ghost[t, :, 1], ghost[t, :, 2], "z"
            )
            for i, l in enumerate(lines_g):
                l.set_data(xline_g[t, i, :2])
                l.set_3d_properties(xline_g[t, i, 2])

        if pointer is not None:
            ax.quiv.remove()
            ax.quiv = ax.quiver(X[t], Y[t], Z[t], dX[t], dY[t], 0, color=pointer_color)

    anim = animation.FuncAnimation(
        fig, update, len(seq), interval=speed, blit=False, save_count=200
    )

    anim.save(fname, writer="pillow", fps=30)
    logging.info(f"Artifact saved at {fname}.")

    return fname


def reconstruct(
    model,
    epoch,
    input_data,
    input_label,
    purpose,
    config,
    log_to_wandb=False,
    single_epoch=None,
    comic=False
):
    """
    Make and save stick video on seq from input_data dataset.
    No conditions on output.

    Parameters
    ----------
    purpose : str, {"train", "valid"}
    """
    if config is None:
        logging.info("!! Parameter config is not given: Using default_config")
        config = default_config
 
    x = input_data
    y = input_label
    x_recon = model(x, y)  # has shape [batch_size, seq_len, 159]
    _, seq_len, _ = x.shape

    x_formatted = x[0].reshape((seq_len, -1, 3))
    x_recon_formatted = x_recon[0].reshape((seq_len, -1, 3))

    if epoch is not None:
        name = f"recon_epoch_{epoch}_{purpose}_{config.run_name}.gif"
    else:
        name = f"recon_{purpose}_{config.run_name}.gif"
    
    if single_epoch is None:
        filepath = os.path.join(os.path.abspath(os.getcwd()), "animations/" + config.run_name)
        fname = os.path.join(filepath, name)

    if single_epoch is not None:
        fname = os.path.join(str(single_epoch), name)
        plotname = f"comic_{purpose}_{config.run_name}"
        comicname_recon = os.path.join(str(single_epoch), plotname + '_recon.png')
        comicname = os.path.join(str(single_epoch), plotname + '.png')

    fname = animatestick(
        x_recon_formatted,
        fname=fname,
        ghost=x_formatted,
        dot_alpha=0.7,
        ghost_shift=0.2,
    )

    if comic:
        draw_comic(
            x_recon_formatted,
            comicname_recon,
            recon=True
        )
        draw_comic(
            x_formatted,
            comicname,
        )

    if log_to_wandb:
        animation_artifact = wandb.Artifact("animation", type="video")
        animation_artifact.add_file(fname)
        wandb.log_artifact(animation_artifact)
        logging.info("ARTIFACT: logged reconstruction to wandb.")


def generate(
    model,
    y_given=None,
    config=None,
):
    """Generate a dance from a given label by sampling in the latent space.

    If no label y is given, then a random label is chosen.
    """

    if config is None:
        logging.info("!! Parameter config is not given: Using default_config")
        config = default_config

    onehot_encoder = utils.make_onehot_encoder(config.label_dim)
    if y_given is not None:
        y_onehot = onehot_encoder(y_given)
        y_onehot = y_onehot.reshape((1, y_onehot.shape[0]))
        y_onehot = y_onehot.to(config.device)
        y_title = y_given

    else:
        y_rand = random.randint(0, config.label_dim - 1)
        y_onehot = onehot_encoder(y_rand)
        y_onehot = y_onehot.reshape((1, y_onehot.shape[0]))
        y_onehot = y_onehot.to(config.device)
        y_title = y_rand

    z_create = torch.randn(size=(1, config.latent_dim))
    z_create = z_create.to(config.device)

    x_create = model.sample(z_create, y_onehot)

    return x_create, y_title


def generate_one_move(
    model,
    config=None,
):
    """
    Generate a dance from the same body-motion latent variable
    with all possible different labels.

    """

    if config is None:
        logging.info("!! Parameter config is not given: Using default_config")
        config = default_config

    onehot_encoder = utils.make_onehot_encoder(config.label_dim)
    z_create = torch.randn(size=(1, config.latent_dim)).to(config.device)

    all_moves = []
    all_labels = []
    for y in range(config.label_dim):
        y_onehot = onehot_encoder(y)
        
        y_onehot = y_onehot.reshape((1, y_onehot.shape[0]))
        print(y_onehot)
        y_onehot = y_onehot.to(config.device)

        x_create = model.sample(z_create, y_onehot)
        all_moves.append(x_create.cpu().data.numpy())
        all_labels.append(y)

    return all_moves, all_labels

def generate_and_save(
    model,
    epoch=None,
    y_given=None,
    config=None,
    single_epoch=None,
    log_to_wandb=False,
    comic=False,
):
    """Generate a dance from a given label and save the corresponding artifact."""
    if config is None:
        logging.info("!! Parameter config is not given: Using default_config")
        config = default_config

    filepath = os.path.join(os.path.abspath(os.getcwd()), "animations/" + config.run_name)

    x_create, y_title = generate(model, y_given)
    x_create_formatted = x_create[0].reshape((config.seq_len, -1, 3))
    
    if epoch is not None:
        name = f"create_{y_given}_epoch_{epoch}_{config.run_name}.gif"
    else:
        name = f"create_{y_given}_{config.run_name}.gif"
    
    if single_epoch is None:    
        filepath = os.path.join(os.path.abspath(os.getcwd()), "animations/" + config.run_name)
        fname = os.path.join(filepath, name)

    if single_epoch is not None:
        fname = os.path.join(str(single_epoch), name)
        plotname = f"comic_{y_given}_{epoch}_{config.run_name}.png"
        comicname = os.path.join(str(single_epoch), plotname)


    fname = animatestick(
        x_create_formatted,
        fname=fname,
        ghost=None,
        dot_alpha=0.7,
        ghost_shift=0.2,
        condition=y_title,
    )

    if comic:
        draw_comic(
            x_create_formatted,
            comicname,
            recon=True
        )

    if log_to_wandb:
        animation_artifact = wandb.Artifact("animation", type="video")
        animation_artifact.add_file(fname)
        wandb.log_artifact(animation_artifact)
        logging.info("ARTIFACT: logged conditional generation to wandb.")

def draw_comic(frames, comicname, angles=None, figsize=None, window_size=0.8, dot_size=0, lw=0.8, zcolor=None,recon=False):
    
    if torch.is_tensor(frames):
        frames = frames.cpu().data.numpy()
    frames = frames[::4, :, :]

    if recon:
        cmap = 'autumn' #'cool_r'
    if not recon:
        cmap = 'cool_r'
    #cmap = 'cool_r'
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    ax.view_init(30, 0)
    shift_size=0.6
    
    ax.set_xlim(-0.4*window_size,0.4*window_size)
    ax.set_ylim(0, 6)  #(-window_size,2*len(frames)*window_size)
    ax.set_zlim(-0.1,0.5)
    ax.set_box_aspect([1,8,0.8])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
    cm = matplotlib.cm.get_cmap(cmap)
    
    if angles is not None:
        vR = 0.15
        zidx = point_labels.index("CLAV")
        X = frames[:,zidx,0]
        Y = frames[:,zidx,1]
        dX,dY = vR*np.cos(angles), vR*np.sin(angles)
        Z = frames[:,zidx,2]
        #Z = frames[:,2,2]
    n_frame=0
    for iframe,frame in enumerate(frames):
        n_frame += 1
        ax.scatter(frame[:,0],
                       frame[:,1]+0.4+n_frame*shift_size,
                       frame[:,2],
                       #alpha=0.3,
                       c= frame[:, 2], #'black', #zcolor,np.arange(0, 53),#
                       cmap=cm,
                       s=dot_size,
                       depthshade=True)

        zcolor = frame[:, 2] * 2
        
        if angles is not None:
            ax.quiver(X[iframe],iframe*shift_size+Y[iframe],Z[iframe],dX[iframe],dY[iframe],0, color='black')
        
        for i,(g1,g2) in enumerate(skeleton_lines):
            g1_idx = [point_labels.index(l) for l in g1]
            g2_idx = [point_labels.index(l) for l in g2]

            if zcolor is not None:
                color = cm(0.5*(zcolor[g1_idx].mean() + zcolor[g2_idx].mean()))
            else:
                color = None

            x1 = np.mean(frame[g1_idx],axis=0)
            x2 = np.mean(frame[g2_idx],axis=0)
            
            ax.plot(np.linspace(x1[0],x2[0],10),
                    np.linspace(x1[1],x2[1],10)+iframe*shift_size,
                    np.linspace(x1[2],x2[2],10),
                    color=color,
                    lw=lw)
        plt.savefig(comicname)

def generate_and_save_one_move(
    model,
    config,
    path,
):
    """
    Generate label_dim dances using the 
    samebody motion latent variable and different labels.
    Save the corresponding artifacts.
    """

    all_moves, all_labels = generate_one_move(model, config)

    for i, y in enumerate(all_labels):
        x_create_formatted = all_moves[y].reshape((config.seq_len, -1, 3))
        name = f"one_move_{y}.gif"
        fname = os.path.join(path, name)
        plotname = f"comic_{y}.png"
        comicname = os.path.join(str(path), plotname)

        fname = animatestick(
            x_create_formatted,
            fname=fname,
            ghost=None,
            dot_alpha=0.7,
            ghost_shift=0.2,
            condition=y,
        )

        draw_comic(
            x_create_formatted,
            comicname,
            recon=True
        )
