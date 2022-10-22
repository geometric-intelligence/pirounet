"Functions for generating and visualizing dance sequences."

import logging
import os
import random

import default_config
import matplotlib
import matplotlib.pyplot as plt
import models.utils as utils
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import torch
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import juggle_axes
from sklearn.decomposition import PCA
from torch.autograd import Variable

import wandb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = default_config.which_device

# This array stores the physical meaning of all 53 keypoints
# forming the skeleton in each dance pose.
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

# This operation creates a normal, connected skeleton:
skeleton_idxs = []
for g1, g2 in skeleton_lines:
    entry = []
    entry.append([point_labels.index(line) for line in g1])
    entry.append([point_labels.index(line) for line in g2])
    skeleton_idxs.append(entry)


def getlinesegments(seq, zcolor=None, cmap=None):
    """Calculates coordinates for the lines.

    If using a colormap, calculates the color based on the location
    of each segment's middle point.

    Parameters
    ----------
    seq :           array
                    Shape = [seq_len, keypoints (53), 3]
                    Dance sequence.
    zcolor :        string
                    Color of first skeleton's keypoints and segments.
    cmap :          string
                    Matplotlib colormap.

    Returns
    ----------
    xline :         array
                    Shape = [seq_len, n_segments, 3, 2]
                    Returns 3D start and end points for every
                    segment in a skeleton by taking mean of
                    relevant keypoints.

    colors :        array
                    Shape = [n_segments, 4]
                    Stores colors from color map based on
                    location of middle point of segment.

    """
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
    """Puts line segments on the given axis with given colors.

    Parameters
    ----------
    ax :        Plot axis to plot segments on.
    segments :  array
                Shape = [n_segments, 3, 2]
                3D start and end points for each segment in
                skeleton.
    color :     array
                Shape = [n_segments, 4]
                Color associated to each segment.
    lw :        float
                Line width of each segment.
    alpha :     float
                Between 0 and 1. Determines transparency of
                segments. 1 = transparent.

    Returns
    ----------
    lines :     list
                Lines connecting the start and end points of
                each segment.
    """
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
    zcolor=None,
    ax_lims=(-0.4, 0.4),
    speed=45,
    dot_size=20,
    dot_alpha=0.5,
    lw=2.5,
    cmap="inferno",
    condition=None,
):
    """Creates skeleton animation of one sequence.

    Draws up to two connected skeletons moving in
    a unit box, centered at a point on the x,y plane.

    Parameters
    ----------
    seq :           array
                    Shape = [seq_len, keypoints (53), 3]
                    Sequence of seq_len poses with 53 keypoints each
                    to be plotted.
    fname :         string
                    Filepath + name of animation for saving purposes.
    ghost :         bool
                    Shape = [seq_len, keypoints (53), 3]
                    Sequence of second skeleton. None when plotting
                    only one skeleton.
    ghost_shift :   float
                    Lateral shift between two skeletons.
    figsize :       tuple
                    Size of figure.
    zcolor :        string
                    Color of first skeleton's keypoints and segments.
    ax_lims :       tuple
                    Beginning and ending point of x,y axes of the
                    unit cube. z axis limits is a function of these.
    speed :         int
                    Intervals for animation.
    dot_size :      int
                    Size of keypoint scatterpoints for skeleton(s).
    dot_alpha :     float
                    Between 0 and 1. Determines transparency of
                    keypoint scatterpoints. 1 = transparent.
    lw :            float
                    Line width of skeleton segments.
    cmap :          string
                    Matplotlib colormap.
    condition :     int
                    Integer representing categorical label associated
                    to dance sequence, if known.


    Returns
    ----------
    fname :         string
                    Filepath + name of animation for saving purposes.
    """
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

    anim = animation.FuncAnimation(
        fig, update, len(seq), interval=speed, blit=False, save_count=200
    )
    anim.save(fname, writer="pillow", fps=30)
    logging.info(f"Artifact saved at {fname}.")
    return fname


def reconstruct(
    model,
    config,
    epoch,
    input_data,
    input_label,
    purpose,
    log_to_wandb=False,
    results_path=None,
    comic=False,
):
    """
    Make and save stick gif on sequence from input_data dataset.
    No conditions on output.
    Animation features original sequence (blue) and reconstructed
    sequence (black).
    Option to create strip-comic plot of each sequence as well.

    Parameters
    ----------
    model :         serialized object
                    Model to evaluate.
    epoch :         int
                    Epoch to evaluate it at.
    input_data :    array
                    Shape = [n_seqs, seq_len, input_dim]
                    Input sequences to be reconstructed.
    input_label :   array
                    Shape = [n_seqs, label_dim]
                    One hots associated to input sequences.
    purpose :       string
                    {"train", "valid", "test"}
    config :        dict
                    Configuration for run.
    log_to_wandb :  bool
                    If True, logs the animation file to wandb.
    results_path :  string
                    Filepath to folder for saving an artifact
                    not during training (i.e. only for 1 epoch).
                    Set to None for artifact generation during
                    training.
    comic :         bool
                    If True, also generates a strip comic style plot
                    for every sequence, original and reconstructed.
    """

    for i_batch, (x, y) in enumerate(zip(input_data, input_label)):

        x, y = Variable(x), Variable(y)
        x, y = x.to(config.device), y.to(config.device)

        batch_one_hot = utils.batch_one_hot(y, config.label_dim)
        y = batch_one_hot.to(config.device)

        x_recon = model(x, y)  # has shape [batch_size, seq_len, 159]
        _, seq_len, _ = x.shape

        x_formatted = x[0].reshape((seq_len, -1, 3))
        x_recon_formatted = x_recon[0].reshape((seq_len, -1, 3))

        if results_path is None:
            name = f"recon_{i_batch}_{purpose}_epoch_{epoch}_{config.run_name}.gif"
            filepath = os.path.join(
                os.path.abspath(os.getcwd()), "animations/" + config.run_name
            )
            fname = os.path.join(filepath, name)

        if results_path is not None:
            name = f"recon_{i_batch}_{purpose}_epoch_{epoch}_{config.run_name}.gif"
            fname = os.path.join(str(results_path), name)

        fname = animatestick(
            x_recon_formatted,
            fname=fname,
            ghost=x_formatted,
            dot_alpha=0.7,
            ghost_shift=0.2,
        )

    if comic:
        plotname = f"comic_{purpose}"
        comicname_recon = os.path.join(str(results_path), plotname + "_recon.png")
        comicname = os.path.join(str(results_path), plotname + ".png")

        draw_comic(x_recon_formatted, comicname_recon, recon=True)
        draw_comic(
            x_formatted,
            comicname,
        )

    if log_to_wandb:
        animation_artifact = wandb.Artifact("animation", type="video")
        animation_artifact.add_file(fname)
        wandb.log_artifact(animation_artifact)
        logging.info("ARTIFACT: logged reconstruction to wandb.")


def generate_rand(
    model,
    config,
    n_seq,
    y_given=None,
):
    """Generate a dance by sampling in the latent space.

    If no label y is given, then a random label is chosen.
    Choice of "label" does not actually condition the output
    for an entangled latent space.

    Parameters
    ----------
    model :     serialized object
                Model to evaluate.
    config :    dict
                Configuration for run.
    n_seq :     int
                Number of sequences to generate.
    y_given :   int
                Int associated to categorical label.


    Returns
    ----------
    x_create :  array
                Shape = [n_seq, seq_len, input_dim]
                Sampled sequences.
    """

    onehot_encoder = utils.make_onehot_encoder(config.label_dim)
    x_create = []
    for i in range(n_seq):
        if y_given is not None:
            y_onehot = onehot_encoder(y_given)
            y_onehot = y_onehot.reshape((1, y_onehot.shape[0]))
            y_onehot = y_onehot.to(config.device)

        else:
            y_rand = random.randint(0, config.label_dim - 1)
            y_onehot = onehot_encoder(y_rand)
            y_onehot = y_onehot.reshape((1, y_onehot.shape[0]))
            y_onehot = y_onehot.to(config.device)

        z_create = torch.randn(size=(1, config.latent_dim))
        z_create = z_create.to(config.device)

        x_create_one_seq = model.sample(z_create, y_onehot)
        x_create_one_seq_formatted = x_create_one_seq.reshape((config.seq_len, -1, 3))

        x_create = np.append(x_create, x_create_one_seq_formatted.cpu().data.numpy())

    x_create = x_create.reshape((n_seq, config.seq_len, config.input_dim))

    return x_create


def generate_cond(
    model, config, num_gen_cond_lab, encoded_data, encoded_labels, shuffle=False
):
    """Generates new sequences by sampling from high-density Efforts
    in latent space.

    While the latent space is entangled, we structure the generation
    to be sampled from a set of latent variables that were constructed
    from an equal amount of categorical labels.

    Parameters
    ----------
    model :             serialized object
                        Model to be evaluated.
    config :            dict
                        Configuration for the run.
    num_gen_cond_lab :  int
                        Number of latent variables to be constructed
                        with each label.
    encoded_data :      array
                        Shape = [n_seq, seq_len, input_dim]
                        Set of sequences to be encoded in latent space
                        for locating high density neighborhoods.
    encoded_labels :    array
                        Shape = [n_seq,]
                        Categorical labels associated to encoded_data
                        sequences.
    shuffle :           bool
                        True: re-shuffle produced sequences to not
                        categorize by label.

    Returns
    ----------
    gen_dance :         array
                        Shape = [label_dim * num_gen_cond_lab, seq_len, input_dim]
                        Generated sequences.
    gen_labels :        array
                        Shape = [label_dim * num_gen_cond_lab, ]
    """

    all_high, pca, z_center = get_high_neighb(
        model, config, encoded_data, encoded_labels
    )

    gen_dance = []
    for y in range(config.label_dim):
        one_label_seq = []
        for i in range(num_gen_cond_lab):
            # decode within high density tile
            tile_to_pick = np.random.randint(0, len(all_high[y]))
            tile = all_high[y][tile_to_pick]
            zx = np.random.uniform(tile[0], tile[0] + config.step_size[y])
            zy = np.random.uniform(tile[1], tile[1] + config.step_size[y])
            z = np.array((zx, zy))
            z = pca[y].inverse_transform(z) + z_center[y]

            z_within_tile = torch.tensor(z).reshape(1, -1).to(config.device).float()

            x_create = model.sample(
                z_within_tile,
                utils.one_hot(y, config.label_dim).reshape((1, 3)).to(config.device),
            )
            x_create_formatted = x_create.reshape((config.seq_len, -1, 3))

            one_label_seq = np.append(
                one_label_seq, x_create_formatted.cpu().data.numpy()
            )

        gen_dance = np.append(gen_dance, one_label_seq)

        gen_dance = np.array(gen_dance).reshape(-1, default_config.seq_len, 53, 3)

    gen_labels = []
    for i in range(config.label_dim):
        gen_label = i * np.ones(num_gen_cond_lab, dtype=int)
        gen_labels = np.concatenate((gen_labels, gen_label.astype(int)))

    gen_labels = torch.tensor(gen_labels.astype(int))

    if shuffle:
        shuffler = np.random.permutation(len(gen_labels))
        gen_labels = np.array(gen_labels)[shuffler]
        gen_dance = np.array(gen_dance)[shuffler]

    return gen_dance, gen_labels


def generate_and_save(
    model,
    config,
    epoch,
    num_artifacts,
    type,
    encoded_data=None,
    encoded_labels=None,
    y_given=None,
    log_to_wandb=False,
    results_path=None,
    comic=False,
    npy_output=False,
):
    """Generates a dance by randomly sampling the latent space and save
     the corresponding artifact. Choice of "label" does not actually
     condition the output for an entangled latent space.

    Parameters
    ----------
    model :             serialized object
                        Model to evaluate.
    config :            dict
                        Configuration for run.
    epoch :             int
                        Epoch to evaluate it at.
    num_artifacts :     int
                        Number of sequences to produce, either by random
                        sampling, or for each categorical label.
    type :              string
                        {"random", "cond"}
                        Determines the nature of the generation.
    encoded_data :      array
                        Shape = [n_seq, seq_len, input_dim]
                        Set of sequences to be encoded in latent space
                        for locating high density neighborhoods.
                        (only needed for conditional generation)
    encoded_labels :    array
                        Shape = [n_seq,]
                        Categorical labels associated to encoded_data
                        sequences.
                        (only needed for conditional generation)
    y_given :           int
                        Int associated to categorical label. ONLY use
                        for testing entanglement of latent space.

    log_to_wandb :      bool
                        If True, logs the animation file to wandb.
    results_path :      string
                        Filepath to folder for saving an artifact
                        not during training (i.e. only for 1 epoch).
                        Set to None for artifact generation during
                        training.
    comic :             bool
                        If True, also generates a strip comic style plot
                        for every sequence, original and reconstructed.
    npy_output :        bool
                        If True, also generates a npy file for every
                        sequence (saves pose data) made for animation.
    """

    if y_given is not None:
        x_create = generate_rand(model, config, num_artifacts, y_given)
    if y_given is None:
        if type == "random":
            x_create = generate_rand(model, config, num_artifacts)
        if type == "cond":
            x_create, _ = generate_cond(
                model,
                config,
                num_artifacts,
                encoded_data,
                encoded_labels,
                shuffle=False,
            )

    x_create_formatted = x_create.reshape((-1, config.seq_len, 53, 3))

    for i in range(len(x_create_formatted)):
        if results_path is None:
            name = f"create_{i}_epoch_{epoch}_{config.run_name}.gif"
            filepath = os.path.join(
                os.path.abspath(os.getcwd()), "animations/" + config.run_name
            )
            fname = os.path.join(filepath, name)
            npyname = os.path.join(
                filepath, f"data_{i}_epoch_{epoch}_{config.run_name}.npy"
            )

        if results_path is not None:
            name = f"create_{i}_epoch_{epoch}_{config.run_name}.gif"
            fname = os.path.join(str(results_path), name)
            plotname = f"comic_{i}_{epoch}_{config.run_name}.png"
            comicname = os.path.join(str(results_path), plotname)
            npyname = os.path.join(
                str(results_path), f"data_{i}_epoch_{epoch}_{config.run_name}.npy"
            )

        fname = animatestick(
            x_create_formatted[i],
            fname=fname,
            ghost=None,
            dot_alpha=0.7,
            ghost_shift=0.2,
            condition=y_given,
        )

        if comic:
            draw_comic(x_create_formatted[i], comicname, recon=True)

        if npy_output:
            np.save(npyname, x_create_formatted[i])

        if log_to_wandb:
            animation_artifact = wandb.Artifact("animation", type="video")
            animation_artifact.add_file(fname)
            wandb.log_artifact(animation_artifact)
            logging.info("ARTIFACT: logged random generation to wandb.")


def draw_comic(
    frames, comicname, figsize=None, window_size=0.8, dot_size=0, lw=0.8, recon=False
):
    """Generates a strip comic style plot of a dance sequence.

    Extracted poses are plotted in consecutive order along a
    single axis, left to right.

    Parameters
    ----------
    frames :        array
                    Shape = [seq_len, keypoints, 3]
                    Dance sequence to be plotted.
    comicname :     string
                    Filepath + name of comic for saving purposes.
    figsize :       tuple
                    Size of figure.
    window_size :   float
                    Sets size of box containing skeleton poses.
    dot_size :      float
                    Size of keypoints of each skeleton pose.
    lw :            float
                    Line width of each skeleton's segments.
    recon :         bool
                    If True, changes color of skeleton to signify
                    sequence to be a reconstruction.
    """
    if torch.is_tensor(frames):
        frames = frames.cpu().data.numpy()
    frames = frames[::4, :, :]

    if recon:
        cmap = "autumn"
    if not recon:
        cmap = "cool_r"

    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    ax.view_init(30, 0)
    shift_size = 0.6

    ax.set_xlim(-0.4 * window_size, 0.4 * window_size)
    ax.set_ylim(0, 6)
    ax.set_zlim(-0.1, 0.5)
    ax.set_box_aspect([1, 8, 0.8])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    cm = matplotlib.cm.get_cmap(cmap)

    n_frame = 0
    for iframe, frame in enumerate(frames):
        n_frame += 1
        ax.scatter(
            frame[:, 0],
            frame[:, 1] + 0.4 + n_frame * shift_size,
            frame[:, 2],
            c=frame[:, 2],
            cmap=cm,
            s=dot_size,
            depthshade=True,
        )

        zcolor = frame[:, 2] * 1

        for i, (g1, g2) in enumerate(skeleton_lines):
            g1_idx = [point_labels.index(l) for l in g1]
            g2_idx = [point_labels.index(l) for l in g2]

            if zcolor is not None:
                color = cm(0.5 * (zcolor[g1_idx].mean() + zcolor[g2_idx].mean()))
            else:
                color = None

            x1 = np.mean(frame[g1_idx], axis=0)
            x2 = np.mean(frame[g2_idx], axis=0)

            ax.plot(
                np.linspace(x1[0], x2[0], 10),
                np.linspace(x1[1], x2[1], 10) + iframe * shift_size,
                np.linspace(x1[2], x2[2], 10),
                color=color,
                lw=lw,
            )
        plt.savefig(comicname, dpi=500)


def get_high_neighb(model, config, labelled_data, labels):
    """Encode data in latent space and locate high density neighborhoods.

    Use Principle Component Analysis to search for high density neighborhoods in 2D.

    Parameters
    ---------
    model :         serialized object
                    Model to evaluate.
    config :        dict
                    Configuration for run.
    labelled_data : array
                    Shape = [n_seqs, seq_len, input_dim]
                    Input sequences to be encoded in the latent space.

    labels :        array
                    Shape = [n_seqs, ]
                    Labels associated to input sequences.

    Returns
    ---------
    all_high :          list
                        Coordinates for high density neighborhoods per
                        label.
    pca :               list
                        Principle Component Analysis transformation
                        per label.
    z_center :          list
                        Mean of latent variables sampled from high
                        density neighborhoods, per label.
    """
    z_0 = []
    z_1 = []
    z_2 = []
    batch = 0

    for i_batch, (x_batch, y_batch) in enumerate(zip(labelled_data, labels)):

        batch += 1

        x_batch, y_batch = Variable(x_batch), Variable(y_batch)
        x_batch, y_batch = x_batch.to(default_config.device), y_batch.to(
            default_config.device
        )

        y_labels = torch.squeeze(y_batch)
        for i in range(len(y_labels)):
            y_label = y_labels[i].item()
            one_hot = utils.one_hot(y_label, default_config.label_dim)
            y = one_hot.to(default_config.device).reshape(
                (1, 1, default_config.label_dim)
            )
            x = x_batch[i].reshape((1, config.seq_len, -1))

            z, _, _ = model.encode(x, y)

            if y_label == 0:
                z_0.append(z.cpu().data.numpy())
            if y_label == 1:
                z_1.append(z.cpu().data.numpy())
            if y_label == 2:
                z_2.append(z.cpu().data.numpy())

    z_0 = np.squeeze(np.array(z_0))
    z_1 = np.squeeze(np.array(z_1))
    z_2 = np.squeeze(np.array(z_2))

    z_0_center = np.mean(z_0, axis=0)
    z_1_center = np.mean(z_1, axis=0)
    z_2_center = np.mean(z_2, axis=0)
    z_center = [z_0_center, z_1_center, z_2_center]

    pca0 = PCA(n_components=2).fit(z_0 - z_0_center)
    z_0_transf = pca0.transform(z_0 - z_0_center)

    pca1 = PCA(n_components=2).fit(z_1 - z_1_center)
    z_1_transf = pca1.transform(z_1 - z_1_center)

    pca2 = PCA(n_components=2).fit(z_2 - z_2_center)
    z_2_transf = pca2.transform(z_2 - z_2_center)

    all_z = [z_0_transf, z_1_transf, z_2_transf]
    pca = [pca0, pca1, pca2]

    z_min, z_max = -8, 8

    grid_xs = [
        np.arange(z_min, z_max, config.step_size[0]),
        np.arange(z_min, z_max, config.step_size[1]),
        np.arange(z_min, z_max, config.step_size[2]),
    ]
    grid_ys = [
        -np.arange(z_min, z_max, config.step_size[0]),
        -np.arange(z_min, z_max, config.step_size[1]),
        -np.arange(z_min, z_max, config.step_size[2]),
    ]
    n_xs = len(grid_xs[0])
    n_ys = len(grid_ys[0])

    count = np.zeros((default_config.label_dim, n_xs, n_ys))

    for y in range(default_config.label_dim):
        for i, y_coord in enumerate(grid_ys[y]):
            for j, x_coord in enumerate(grid_xs[y]):
                for z in all_z[y]:
                    if x_coord < z[0] and z[0] < (x_coord + config.step_size[y]):
                        if y_coord < z[1] and z[1] < (y_coord + config.step_size[y]):
                            count[y, i, j] += 1

    sum_of_counts = np.sum(count, axis=0)
    density = [count[i] / sum_of_counts for i in range(default_config.label_dim)]

    # create array of high density tiles that we can sample from later
    all_high = []
    for y in range(default_config.label_dim):
        high = []
        for i, y_coord in enumerate(grid_ys[y]):
            for j, x_coord in enumerate(grid_xs[y]):
                if (
                    density[y][i, j] > config.density_thresh[y]
                    and count[y, i, j] > config.dances_per_tile[y]
                ):
                    high.append((x_coord, y_coord))
        all_high.append(high)

    return all_high, pca, z_center


def plot_latentspace(model, config, encoded_labeled_data, encoded_labels, path):
    """Make and save a plot of data encoded into 3D latent space.

    Also saves a plot of the explained variance of the Principle Components,
    to ensure reliability of the 3D latent space.

    Parameters
    ---------
    model :                 serialized object
                            Model to evaluate.
    config :                dict
                            Configuration for run.
    encoded_labelled_data : array
                            Shape = [n_seqs, seq_len, input_dim]
                            Input sequences to be encoded in the latent space.

    encoded_labels :        array
                            Shape = [n_seqs, ]
                            Labels associated to input sequences.
    path :                  string
                            Path to directory where latent space plots are saved.
    """
    x = torch.tensor(encoded_labeled_data.dataset).to(config.device)
    y = torch.tensor(encoded_labels.dataset).to(config.device)
    batch_one_hot = utils.batch_one_hot(y, default_config.label_dim)
    y = batch_one_hot.to(config.device)
    y_for_encoder = y.repeat((1, default_config.seq_len, 1))
    y_for_encoder = 0.33 * torch.ones_like(y_for_encoder).to(config.device)
    z, _, _ = model.encoder(torch.cat([x, y_for_encoder], dim=2).float())
    index = np.arange(0, len(y), 1.0)
    pca = PCA(n_components=10).fit(z.cpu().detach().numpy())
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    axs.plot(pca.explained_variance_ratio_)
    axs.set_xlabel("Number of Principal Components")
    axs.set_ylabel("Explained variance")
    z_transformed = pca.transform(z.cpu().data.numpy())

    plt.savefig(fname=path + f"/PC_comp_{default_config.load_from_checkpoint}.png")

    fig = plt.figure()
    axs = fig.add_subplot(projection="3d")

    sc = axs.scatter(
        z_transformed[:, 0],
        z_transformed[:, 1],
        z_transformed[:, 2],
        c=np.squeeze(encoded_labels.dataset),
        alpha=0.4,
        s=0.5,
    )
    lp = lambda i: plt.plot(
        [],
        [],
        [],
        color=sc.cmap(sc.norm(i)),
        ms=np.sqrt(5),
        mec="none",
        label="Laban Effort {:g}".format(i),
        ls="",
        marker="o",
    )[0]
    handles = [lp(i) for i in np.unique(np.squeeze(encoded_labels.dataset))]

    plt.legend(handles=handles)
    plt.savefig(fname=path + f"/encoded_{default_config.load_from_checkpoint}.png")

    # fig, axs = plt.subplots(1, 1, figsize=(8, 8))

    # axs.scatter(
    #     z_transformed[:, 0],
    #     z_transformed[:, 1],
    #     c=index,  # np.squeeze(labels_valid.dataset),
    #     alpha=0.2,
    #     s=0.1,
    # )
    # plt.savefig(
    #     f"evaluate/log_files/neighb/latent_space_{default_config.load_from_checkpoint}_neighbour_effort.png"
    # )


def plot_dist_one_move(model, config, path, n_one_moves):
    """Compute and plot distances between label-produced sequences.

    Tests entanglement of the latent space by comparing similitude of
    sequences produced with same latent variable, except for the label.

    Parameters
    ---------
    model :         serialized object
                    Model to be evaluated.
    config :        dict
                    Configuration for run.
    path :          string
                    Filepath for saving artifacts.
    n_one_moves :   int
                    Amount of sets of sequences to compare.
    """
    all_dist_01 = []
    all_dist_12 = []
    all_dist_02 = []
    for i in range(n_one_moves):
        all_moves, _ = generate_one_move(model, config)

        dist_01 = np.linalg.norm(all_moves[0] - all_moves[1])
        dist_02 = np.linalg.norm(all_moves[0] - all_moves[2])
        dist_12 = np.linalg.norm(all_moves[1] - all_moves[2])

        all_dist_01.append(dist_01)
        all_dist_12.append(dist_12)
        all_dist_02.append(dist_02)

    fig, ax = plt.subplots()
    x = np.arange(0, n_one_moves, 1)
    print(np.array(x).shape)
    print(np.array(all_dist_01).shape)
    ax.plot(x, all_dist_01, c="blue", label="Dist Low-Med")
    ax.plot(x, all_dist_12, c="orange", label="Dist Med-High")
    ax.plot(x, all_dist_02, c="green", label="Dist Low-High")
    plt.legend()
    plt.title(f"Distances between label-produced seq ({n_one_moves} sets)")
    plt.savefig(path + f"/histogram_{config.load_from_checkpoint}.png")


def generate_and_save_one_move(
    model,
    config,
    path,
):
    """Tests latent space entanglement.

    Generates label_dim dance animations using the
    samebody motion latent variable and different labels.
    Save the corresponding artifacts. Different movement
    means the labels have an impact on generation
    (disentangled latent space).

    Parameters
    ---------
    model :     serialized object
                Model to be evaluated.
    config :    dict
                Configuration for run.
    path :      string
                Filepath for saving artifacts
    """

    all_moves, all_labels = generate_one_move(model, config)

    for i, y in enumerate(all_labels):
        x_create_formatted = all_moves[y].reshape((config.seq_len, -1, 3))
        name = f"one_move_{y}.gif"
        fname = os.path.join(path, name)

        fname = animatestick(
            x_create_formatted,
            fname=fname,
            ghost=None,
            dot_alpha=0.7,
            ghost_shift=0.2,
            condition=y,
        )


def generate_one_move(
    model,
    config,
):
    """Repeat one mvoe for different labels.

    Generate a sequence from the same body-motion latent variable
    with all possible different labels.

    Parameters
    ---------
    model :         serialized object
                    Model to be evaluated.
    config :        dict
                    Configuration for run.

    Returns
    ---------
    all_moves :     array
                    Shape = [label_dim, seq_len, input_dim/3, 3]
                    Sequences generated.
    all_labels :    array
                    Shape = [label_dim,]
                    Labels associated to each sequence.
    """

    z_create = torch.randn(size=(1, config.latent_dim)).to(config.device)

    all_moves = []
    all_labels = []
    for y in range(config.label_dim):
        y_onehot = utils.one_hot(y, config.label_dim)

        y_onehot = y_onehot.reshape((1, y_onehot.shape[0]))
        y_onehot = y_onehot.to(config.device)

        x_create = model.sample(z_create, y_onehot)
        all_moves.append(x_create.cpu().data.numpy())
        all_labels.append(y)

    return all_moves, all_labels
