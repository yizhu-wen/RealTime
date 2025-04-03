import math

import julius
import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.functional.filtering import highpass_biquad, treble_biquad

# class Loss(nn.Module):
#     def __init__(self, train_config):
#         super(Loss, self).__init__()
#         self.msg_loss = nn.MSELoss()
#         self.embedding_loss = nn.MSELoss()
#
#     def en_de_loss(self, x, w_x, msg, rec_msg):
#         embedding_loss = self.embedding_loss(x, w_x)
#         msg_loss = self.msg_loss(msg, rec_msg)
#         return embedding_loss, msg_loss

def basic_loudness(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """This is a simpler loudness function that is more stable.
    Args:
        waveform(torch.Tensor): audio waveform of dimension `(..., channels, time)`
        sample_rate (int): sampling rate of the waveform
    Returns:
        loudness loss as a scalar
    """

    if waveform.size(-2) > 5:
        raise ValueError("Only up to 5 channels are supported.")
    eps = torch.finfo(torch.float32).eps
    gate_duration = 0.4
    overlap = 0.75
    gate_samples = int(round(gate_duration * sample_rate))
    step = int(round(gate_samples * (1 - overlap)))

    # Apply K-weighting
    waveform = treble_biquad(waveform, sample_rate, 4.0, 1500.0, 1 / math.sqrt(2))
    waveform = highpass_biquad(waveform, sample_rate, 38.0, 0.5)

    # Compute the energy for each block
    energy = torch.square(waveform).unfold(-1, gate_samples, step)
    energy = torch.mean(energy, dim=-1)

    # Compute channel-weighted summation
    g = torch.tensor([1.0, 1.0, 1.0, 1.41, 1.41], dtype=waveform.dtype, device=waveform.device)
    g = g[: energy.size(-2)]

    energy_weighted = torch.sum(g.unsqueeze(-1) * energy, dim=-2)
    # loudness with epsilon for stability. Not as much precision in the very low loudness sections
    loudness = -0.691 + 10 * torch.log10(energy_weighted + eps)
    return loudness


def _unfold(a: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """Given input of size [*OT, T], output Tensor of size [*OT, F, K]
    with K the kernel size, by extracting frames with the given stride.
    This will pad the input so that `F = ceil(T / K)`.
    see https://github.com/pytorch/pytorch/issues/60466
    """
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(a, (0, tgt_length - length))
    strides = list(a.stride())
    assert strides[-1] == 1, "data should be contiguous"
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)


class TFLoudnessRatio(nn.Module):
    """TF-loudness ratio loss.

    Input should be [B, C, T], output is scalar.

    Args:
        sample_rate (int): Sample rate.
        segment (float or None): Evaluate on chunks of that many seconds. If None, evaluate on
            entire audio only.
        overlap (float): Overlap between chunks, i.e. 0.5 = 50 % overlap.
        n_bands (int): number of bands to separate
        temperature (float): temperature of the softmax step
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        segment: float = 0.5,
        overlap: float = 0.5,
        n_bands: int = 0,
        clip_min: float = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment = segment
        self.overlap = overlap
        self.clip_min = clip_min
        self.temperature = temperature
        if n_bands == 0:
            self.filter = None
        else:
            self.n_bands = n_bands
            self.filter = julius.SplitBands(sample_rate=sample_rate, n_bands=n_bands)

    def forward(self, out_sig: torch.Tensor, ref_sig: torch.Tensor) -> torch.Tensor:
        B, C, T = ref_sig.shape
        device = out_sig.device
        assert ref_sig.shape == out_sig.shape
        assert C == 1
        assert self.filter is not None

        self.filter.to(device)
        bands_ref = self.filter(ref_sig).view(B * self.n_bands, 1, -1)
        bands_out = self.filter(out_sig).view(B * self.n_bands, 1, -1)
        frame = int(self.segment * self.sample_rate)
        stride = int(frame * (1 - self.overlap))
        gt = _unfold(bands_ref, frame, stride).squeeze(1).contiguous().view(-1, 1, frame)
        est = _unfold(bands_out, frame, stride).squeeze(1).contiguous().view(-1, 1, frame)
        l_noise = basic_loudness(est - gt, sample_rate=self.sample_rate)  # watermark
        l_ref = basic_loudness(gt, sample_rate=self.sample_rate)  # ground truth
        l_ratio = (l_noise - l_ref).view(-1, B)
        loss = torch.nn.functional.softmax(l_ratio / self.temperature, dim=0) * l_ratio
        return loss.mean()


class Loss_identity(nn.Module):
    def __init__(self):
        super(Loss_identity, self).__init__()
        # self.msg_loss = nn.MSELoss()
        self.msg_loss = nn.BCEWithLogitsLoss()
        # self.msg_loss= nn.BCELoss()
        self.embedding_loss = nn.MSELoss()
        self.tfloudness_loss = TFLoudnessRatio(n_bands=16)
    
    def en_de_loss(self, x, w_x, msg, rec_msg):
        embedding_loss = self.embedding_loss(x, w_x)
        msg_loss = self.msg_loss(msg, rec_msg[0]) + self.msg_loss(msg, rec_msg[1])
        loudness_loss = self.tfloudness_loss(w_x.unsqueeze(1), x.unsqueeze(1))

        return embedding_loss, msg_loss, loudness_loss


def encodec_loss_fn(decoded_msgs, msg):
    bce_loss = nn.BCEWithLogitsLoss()
    # Initialize total_loss as a tensor on the correct device.
    total_loss = torch.tensor(0.0, device=msg.device)
    for decoded_msg in decoded_msgs:
        loss = bce_loss(decoded_msg, msg)
        total_loss += loss
    # Average the loss by dividing by the number of decoded messages.
    total_loss = total_loss / len(decoded_msgs) if decoded_msgs else total_loss
    return total_loss


# class Loss_identity_3(nn.Module):
#     def __init__(self, train_config):
#         super(Loss_identity_3, self).__init__()
#         self.msg_loss = nn.MSELoss()
#         # self.msg_loss = nn.CrossEntropyLoss()
#         self.embedding_loss = nn.MSELoss()
#
#     def en_de_loss(self, x, w_x, msg, rec_msg):
#         embedding_loss = self.embedding_loss(x, w_x)
#         # msg_loss = self.msg_loss(msg, rec_msg[0]) + self.msg_loss(msg, rec_msg[1]) + self.msg_loss(msg, rec_msg[2])
#         # msg_loss = self.msg_loss(msg, rec_msg[3]) + self.msg_loss(msg, rec_msg[3]) + self.msg_loss(msg, rec_msg[3])
#         msg_loss = self.msg_loss(msg, rec_msg[0]) + self.msg_loss(msg, rec_msg[1]) + self.msg_loss(msg, rec_msg[2]) + self.msg_loss(msg, rec_msg[3])
#         return embedding_loss, msg_loss
#
# class Loss_identity_3_2(nn.Module):
#     def __init__(self, train_config):
#         super(Loss_identity_3_2, self).__init__()
#         # self.msg_loss = nn.MSELoss()
#         self.msg_loss = nn.BCEWithLogitsLoss()
#         self.embedding_loss = nn.MSELoss()
#
#     def en_de_loss(self, x, w_x, msg, rec_msg):
#         embedding_loss = self.embedding_loss(x, w_x)
#         # msg_loss = self.msg_loss(msg, rec_msg[0]) + self.msg_loss(msg, rec_msg[1]) + self.msg_loss(msg, rec_msg[2])
#         # msg_loss = self.msg_loss(msg, rec_msg[3]) + self.msg_loss(msg, rec_msg[3]) + self.msg_loss(msg, rec_msg[3])
#         msg_loss =  self.msg_loss(rec_msg[0].squeeze(1), msg.squeeze(1)) + \
#                     self.msg_loss(rec_msg[1].squeeze(1), msg.squeeze(1)) + \
#                     self.msg_loss(rec_msg[2].squeeze(1), msg.squeeze(1)) + \
#                     self.msg_loss(rec_msg[3].squeeze(1), msg.squeeze(1))
#         return embedding_loss, msg_loss
#
# class Loss2(nn.Module):
#     def __init__(self, train_config):
#         super(Loss2, self).__init__()
#         # self.msg_loss = nn.MSELoss()
#         self.msg_loss = nn.BCEWithLogitsLoss()
#         self.embedding_loss = nn.MSELoss()
#
#     def en_de_loss(self, x, w_x, msg, rec_msg):
#         embedding_loss = self.embedding_loss(x, w_x)
#         msg_loss = self.msg_loss(rec_msg.squeeze(1), msg.squeeze(1))
#         return embedding_loss, msg_loss
#
# class Loss_identity_2(nn.Module):
#     def __init__(self, train_config):
#         super(Loss_identity_2, self).__init__()
#         self.msg_loss = nn.BCEWithLogitsLoss()
#         self.embedding_loss = nn.MSELoss()
#
#     def en_de_loss(self, x, w_x, msg, rec_msg):
#         embedding_loss = self.embedding_loss(x, w_x)
#         msg_loss = self.msg_loss(rec_msg[0].squeeze(1), msg.squeeze(1)) + self.msg_loss(rec_msg[1].squeeze(1), msg.squeeze(1))
#         return embedding_loss, msg_loss
#
# class Lossex(nn.Module):
#     def __init__(self, train_config):
#         super(Lossex, self).__init__()
#         self.msg_loss = nn.MSELoss()
#         # self.msg_loss = nn.CrossEntropyLoss()
#         self.embedding_loss = nn.MSELoss()
#
#     def en_de_loss(self, x, w_x, msg, rec_msg, no_msg, no_decoded):
#         embedding_loss = self.embedding_loss(x, w_x)
#         msg_loss = self.msg_loss(msg, rec_msg)
#         no_msg_loss = self.msg_loss(no_msg, no_decoded)
#         return embedding_loss, msg_loss, no_msg_loss
