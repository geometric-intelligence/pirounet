"""Defines reconstruction loss."""

import default_config as config
import torch


def reconstruction_loss(x, x_recon):
    """Compute the reconstruction loss between two sequences.

    This is computed as the mean square errors on the joints'
    positions in 3D.

    Parameters
    ----------
    x :             array
                    Shape = [batch_size, seq_len, input_dim]
                    Input batch of sequences.
    x_recon :       array
                    Shape = [batch_size, seq_len, input_dim]
                    Batch of reconstructed sequences.

    Returns
    ----------
    recon_loss :    array
                    Shape = [batch_size, 1]
                    Reconstruction loss for each sequence
                    in batch.
    """
    assert x.ndim == x_recon.ndim == 3
    batch_size = x.shape[0]
    recon_loss = (x - x_recon) ** 2

    recon_loss = torch.mean(recon_loss, axis=(1, 2))

    assert recon_loss.shape == (batch_size,)
    return recon_loss


def kld(q_param):
    """Compute KL-divergence.
    The KL is defined as:
    KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                = -E[log p(z) - log q(z)]
    Formula in Keras was:
    # -0.5*K.mean(K.sum(1 + auto_log_var -
    # K.square(auto_mean) - K.exp(auto_log_var), axis=-1))
    Parameters
    ----------
    z : array-like
        Sample from q-distribuion
    q_param : tuple
        (mu, log_var) of the q-distribution
    Returns
    -------
    kl : KL(q||p)
    """
    z_mean, z_logvar = q_param
    kl = -(torch.sum(1 + z_logvar - torch.square(z_mean) - z_logvar.exp(), axis=-1))
    return kl


def graph_magnitude(x, x_recon):
    """Calculates graph magnitude diff between two batches of sequences.

    Graph magnitude is the sum of all segment magnitudes making up the
    skeleton. These segments' end points are made up of means between groups
    of points.

    Parameters
    ----------
    x :             array
                    Shape = [batch, seq_len, input_dim]
                    Input batch of sequences.

    x_recon :       array
                    Shape = [batch, seq_len, input_dim]
                    Reconstructed batch of sequences.


    Returns
    ----------
    total_graph_loss :  float
                        Sum of differences in graph magnitude
                        over all segments in all poses of the
                        batch.

    """
    print("start graph magnitude")
    x = x.reshape(x.shape[0], x.shape[1], -1, 3)
    x_recon = x_recon.reshape(x.shape[0], x.shape[1], -1, 3)

    x_lines = batch_getlines(x).to(
        config.device
    )  # shape [batch, seq_len, n_segments, 3, 2]
    x_recon_lines = batch_getlines(x_recon).to(config.device)

    x_limb_size = (
        x_lines[:, :, :, :, 0] - x_lines[:, :, :, :, 1]
    )  # [batch, seq_len, n_segments, 3]
    x_recon_limb_size = x_recon_lines[:, :, :, :, 0] - x_recon_lines[:, :, :, :, 1]

    print("get magnitudes 1")
    limb_magni = torch.sum(
        torch.square(x_limb_size), dim=3
    )  # [batch, seq_len, n_segments]
    limb_magni = torch.sqrt(limb_magni).to(config.device)
    print("get magnitudes 2")
    limb_magni_recon = torch.sum(torch.square(x_recon_limb_size), dim=3)
    limb_magni_recon = torch.sqrt(limb_magni_recon).to(config.device)
    print("subtract")
    graph_loss = limb_magni - limb_magni_recon
    print("take sum")
    total_graph_loss = torch.sum(graph_loss.reshape(-1)).to(config.device)

    return total_graph_loss


def batch_getlines(x):
    """Calculates coordinates for the lines of a batch.

    Parameters
    ----------
    x :             array
                    Shape = [batch, seq_len, keypoints (53), 3]
                    Batch of input sequences.

    Returns
    ----------
    batch_x_lines : array
                    Shape = [batch, seq_len, n_segments, 3, 2]
                    Returns 3D start and end points for every
                    segment in a skeleton by taking mean of
                    relevant keypoints.

    """
    print("get batch lines")
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
        (("LFHD",), ("LBHD",)),
        (("LBHD",), ("RBHD",)),
        (("RBHD",), ("RFHD",)),
        (("RFHD",), ("LFHD",)),
        (("LFHD",), ("ARIEL",)),
        (("LBHD",), ("ARIEL",)),
        (("RBHD",), ("ARIEL",)),
        (("RFHD",), ("ARIEL",)),
    ]
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

    skeleton_idxs = []
    for g1, g2 in skeleton_lines:
        entry = []
        entry.append([point_labels.index(line) for line in g1])
        entry.append([point_labels.index(line) for line in g2])
        skeleton_idxs.append(entry)

    batch_size = x.shape[0]
    seq_len = x.shape[1]

    batch_x_lines = torch.zeros((1, seq_len, len(skeleton_idxs), 3, 2)).to(
        config.device
    )
    print("batch loop")
    for i in range(batch_size):
        xline = torch.zeros((x[i].shape[0], len(skeleton_idxs), 3, 2)).to(config.device)
        for i, (g1, g2) in enumerate(skeleton_idxs):
            xline[:, i, :, 0] = torch.mean(x[i][:, g1], axis=1)
            xline[:, i, :, 1] = torch.mean(x[i][:, g2], axis=1)

        batch_x_lines = torch.cat((batch_x_lines, torch.unsqueeze(xline, 0)))
    batch_x_lines = batch_x_lines[1:, :, :, :].reshape((batch_size, seq_len, -1, 3, 2))

    return batch_x_lines
