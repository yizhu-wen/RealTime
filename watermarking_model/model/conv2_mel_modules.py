import torch
import torch.nn as nn
from distortions.frequency2 import fixed_STFT
from silero_vad import load_silero_vad
from torch.nn import LeakyReLU

from .blocks import (
    FCBlock,
    Conv2Encoder,
    WatermarkEmbedder,
    WatermarkExtracter,
    ReluBlock,
)

# from distortions.dl import distortion

# Optional: set up a small constant
EPS = 1e-9


def save_spectrum(y, flag="linear"):
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    # Directory to save figures
    root = "draw_figure"
    os.makedirs(root, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.specgram(
        y, Fs=16000, NFFT=320, noverlap=160, window=np.hanning(320), cmap="magma"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title("Amplitude Spectrogram")
    plt.tight_layout()
    plt.savefig(
        os.path.join(root, f"{flag}_amplitude_spectrogram.png"),
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close()


def save_spectrum_normal(y, flag="linear"):
    import numpy as np
    import os
    import librosa.display
    import matplotlib.pyplot as plt

    peak = np.max(np.abs(y))
    if peak > 1e-8:
        y = y / peak

    # Directory to save figures
    root = "draw_figure"
    os.makedirs(root, exist_ok=True)

    plt.figure(figsize=(10, 4))

    # Compute the spectrogram
    Pxx, freqs, bins, im = plt.specgram(
        y, Fs=16000, NFFT=320, noverlap=160, cmap="magma"
    )

    Pxx_dB = librosa.amplitude_to_db(Pxx, ref=np.max)

    # Clear previous plot and redraw with log values
    plt.clf()
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(bins, freqs, Pxx_dB, shading="auto", cmap="magma")

    plt.colorbar(format="%+2.0f dB")
    plt.title("Log Amplitude Spectrogram")
    plt.tight_layout()
    plt.savefig(
        os.path.join(root, f"{flag}_amplitude_spectrogram.png"),
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close()


def save_feature_map(feature_maps):
    import os
    import matplotlib.pyplot as plt

    feature_maps = feature_maps.cpu().numpy()
    root = "draw_figure"
    output_folder = os.path.join(root, "feature_map_or")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    n_channels = feature_maps.shape[0]
    for channel_idx in range(n_channels):
        fig, ax = plt.subplots()
        ax.imshow(feature_maps[channel_idx, :, :], cmap="gray")
        ax.axis("off")
        output_file = os.path.join(
            output_folder, f"feature_map_channel_{channel_idx + 1}.png"
        )
        plt.savefig(output_file, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)


def save_waveform(a_tensor, flag="original"):
    import os
    import soundfile

    root = "draw_figure"
    y = a_tensor.detach().cpu().numpy()
    soundfile.write(os.path.join(root, flag + "_waveform.wav"), y, samplerate=16000)
    # D = librosa.stft(y)
    # spectrogram = np.abs(D)
    # img=librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=22050, x_axis='time', y_axis='log', y_coords=None);
    # plt.axis('off')
    # plt.savefig(os.path.join(root, flag + '_amplitude_spectrogram_from_waveform.png'), bbox_inches='tight', pad_inches=0.0)


# def freeze_model_and_submodules(model):
#     for param in model.parameters():
#         param.requires_grad = False
#
#     for module in model.children():
#         if isinstance(module, nn.Module):
#             freeze_model_and_submodules(module)


class MsgEmbedder(nn.Module):
    def __init__(self, process_config, train_config):
        super().__init__()
        self.hop_length = process_config["mel"]["hop_length"]
        self.sampling_rate = process_config["audio"]["or_sample_rate"]
        self.prefilling_amt = int(
            train_config["watermark"]["prefilling_amt_second"]
            * self.sampling_rate
            // self.hop_length
            + 1
        )
        self.nbits = train_config["watermark"]["length"]
        self.win_dim = int(
            (process_config["mel"]["n_fft"] / 2) + 1
        )  # 322 / 2 + 1 = 162
        self.embedding_table = torch.nn.Embedding(2 * self.nbits, self.win_dim // 2)
        self.W_M = torch.nn.Linear(self.nbits, self.prefilling_amt, bias=False)

    def forward(self, msg):
        # create indices to take from embedding layer
        indices = 2 * torch.arange(msg.shape[-1]).to(msg.device)  # k: 0 2 4 ... 2k
        indices = indices.repeat(msg.shape[0], 1).unsqueeze(1)  # b x k
        indices = (indices + msg).long()
        msg_embed = self.embedding_table(
            indices.long().squeeze(1)
        )  # b x k -> b x k x (self.win_dim/2)
        msg_embed = msg_embed.transpose(1, 2)  # [B, H, K]
        watermark_encoded = self.W_M(msg_embed).unsqueeze(1).repeat(1, 1, 2, 1)
        return watermark_encoded


class Encoder(nn.Module):
    def __init__(self, process_config, model_config, train_config):
        super(Encoder, self).__init__()
        self.name = "conv2"
        self.win_dim = int(
            (process_config["mel"]["n_fft"] / 2) + 1
        )  # 322 / 2 + 1 = 162
        self.add_carrier_noise = False
        self.block = model_config["conv2"]["block"]
        self.layers_CE = model_config["conv2"]["layers_CE"]
        self.EM_input_dim = model_config["conv2"]["hidden_dim"] + 3
        self.layers_EM = model_config["conv2"]["layers_EM"]
        self.n_fft = process_config["mel"]["n_fft"]
        self.hop_length = process_config["mel"]["hop_length"]
        self.win_length = process_config["mel"]["win_length"]
        self.sampling_rate = process_config["audio"]["or_sample_rate"]
        self.nbits = train_config["watermark"]["length"]
        self.prefilling_amt = int(
            train_config["watermark"]["prefilling_amt_second"]
            * self.sampling_rate
            // self.hop_length
            + 1
        )
        self.delay_amt = int(
            train_config["watermark"]["delay_amt_second"]
            * self.sampling_rate
            // self.hop_length
            + 1
        )
        self.future_amt = int(
            train_config["watermark"]["future_amt_second"]
            * self.sampling_rate
            // self.hop_length
            + 1
        )
        self.delay = True
        self.power = 1.0
        self.vad = load_silero_vad()
        self.vad_threshold = 0.50

        self.vocoder_step = model_config["structure"]["vocoder_step"]
        # MLP for the input wm
        self.msg_linear_in = FCBlock(
            self.nbits, self.win_dim // 2, activation=LeakyReLU(inplace=True)
        )

        # stft transform
        self.stft = fixed_STFT(
            process_config["mel"]["n_fft"],
            process_config["mel"]["hop_length"],
            process_config["mel"]["win_length"],
        )

        self.ENc = Conv2Encoder(
            input_channel=2,
            hidden_dim=model_config["conv2"]["hidden_dim"],
            block=self.block,
            n_layers=self.layers_CE,
        )

        self.EM = WatermarkEmbedder(
            input_channel=self.EM_input_dim,
            hidden_dim=model_config["conv2"]["hidden_dim"],
            block=self.block,
            n_layers=self.layers_EM,
        )

    def pad_w_zero_stft(self, input_stft, watermark_stft, voice_prefilling):
        """
        Pad the watermarked stft output with zeros on the left + right,
        respecting future_amt and chunk-based offsets.
        """
        chunk_size = (
            voice_prefilling if not self.delay else (voice_prefilling + self.delay_amt)
        )

        zeros_right_len = input_stft.shape[3] - watermark_stft.shape[3] - chunk_size
        if zeros_right_len < 0:
            # Edge case: won't happen if chunking logic is correct, but just to be safe
            zeros_right_len = 0

        zeros_left = torch.zeros_like(input_stft[:, :, :, :chunk_size])
        zeros_right = torch.zeros_like(input_stft[:, :, :, :zeros_right_len])

        actual_watermark = (
            torch.cat([zeros_left, watermark_stft, zeros_right], dim=3) + EPS
        )
        return actual_watermark

    def forward(self, x, msg, global_step):
        num_samples = x.shape[-1]
        _, _, stft_result = self.stft.transform(x)
        # Evaluate how many chunks we can process
        # 2s input + 0.5s calculation delay
        # 2.00*16000 = 32000
        # 32800 // hop_length + 1 = 201 center=True
        # 0.5s*16000 = 8000
        # 8000 // hop_length + 1 = 51 center=True
        # Predict future 0.5s watermark
        # 0.5*16000 = 8000
        # 8000 // hop_length + 1 =51
        max_start = stft_result.shape[-1] - (self.prefilling_amt + self.delay_amt)
        if int(max_start / self.delay_amt) <= 0:
            return None  # Not enough frames for a chunk

        watermark_encoded = (
            self.msg_linear_in(msg)
            .transpose(1, 2)
            .unsqueeze(1)
            .repeat(1, 1, 2, self.prefilling_amt)
        )
        list_of_watermarks = []
        for i in range(
            int(
                (stft_result.shape[-1] - (self.prefilling_amt + self.delay_amt))
                / self.future_amt
            )
        ):
            carrier_encoded = self.ENc(
                stft_result[
                    :,
                    :,
                    :,
                    i * self.future_amt : self.prefilling_amt + i * self.future_amt,
                ]
            )
            # torch.Size([B, 1, 81])
            # torch.Size([B, 81, 1])
            # torch.Size([B, 1, 81, 1])
            # torch.Size([B, 1, 162, 201])
            # watermark_encoded = self.msg_linear_in(msg).transpose(1, 2).unsqueeze(1).repeat(1, 1, 2,
            #                                                                                 carrier_encoded.shape[3])

            concatenated_feature = torch.cat(
                (
                    carrier_encoded,
                    stft_result[
                        :,
                        :,
                        :,
                        i * self.future_amt : self.prefilling_amt + i * self.future_amt,
                    ],
                    watermark_encoded,
                ),
                dim=1,
            )
            # [B, 2, bins, length]
            # Embed the watermark
            carrier_watermarked = self.EM(concatenated_feature)
            # Append both the watermark chunk and the pilot segment
            list_of_watermarks.append(carrier_watermarked)

        if len(list_of_watermarks) > 0:
            watermark = torch.cat(list_of_watermarks, dim=-1)
            all_watermark_stft = self.pad_w_zero_stft(
                stft_result, watermark, self.prefilling_amt
            )
            del list_of_watermarks
            mask = stft_result != 0
            all_watermark_stft = all_watermark_stft * mask + 0.0000001

            self.stft.num_samples = num_samples

            # Recompute magnitude & phase
            real_part = all_watermark_stft[:, 0, :, :]
            imag_part = all_watermark_stft[:, 1, :, :]
            spect = torch.sqrt(real_part**2 + imag_part**2)
            phase = torch.atan2(imag_part, real_part)

            y = self.stft.inverse(spect, phase).squeeze(1)
            del spect, phase, real_part, imag_part

            with torch.no_grad():
                # Get chunk-level speech probabilities for the batch.
                # The output shape should be [batch, num_chunks]
                batch_chunk_probs = self.vad.audio_forward(x, sr=self.sampling_rate)

            # Threshold the probabilities to obtain a binary mask per chunk.
            batch_chunk_mask = (batch_chunk_probs > self.vad_threshold).float()

            # Upsample the chunk-level mask to a sample-level mask.
            # Each chunk's decision is repeated for chunk_size samples.
            sample_masks = torch.repeat_interleave(batch_chunk_mask, 512, dim=1).to(
                y.device
            )

            # Since the upsampled mask might be longer than the actual audio length,
            # slice the mask to match the original number of samples.
            sample_length = x.shape[-1]
            sample_masks = sample_masks[:, :sample_length]

            # Apply the mask to the original audio to zero out non-speech regions.
            masked_y = y * sample_masks
            return masked_y, all_watermark_stft
        else:
            print("Not enough watermarking!!!!")
            return None


class Decoder(nn.Module):
    def __init__(self, process_config, model_config, train_config):
        super(Decoder, self).__init__()
        self.nbits = train_config["watermark"]["length"]
        self.win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.block = model_config["conv2"]["block"]
        self.EX = WatermarkExtracter(
            input_channel=2,
            hidden_dim=model_config["conv2"]["hidden_dim"],
            block=self.block,
        )
        self.stft = fixed_STFT(
            process_config["mel"]["n_fft"],
            process_config["mel"]["hop_length"],
            process_config["mel"]["win_length"],
        )
        self.msg_linear_out = FCBlock(self.win_dim // 2, self.nbits)

    def forward(self, y, global_step):
        # Process the main distorted signal y_d_d.
        _, _, stft_result = self.stft.transform(y)
        extracted_wm = self.EX(stft_result).squeeze(1)
        del stft_result
        low, high = extracted_wm.chunk(2, dim=1)
        low_msg = torch.mean(low, dim=2, keepdim=True).transpose(1, 2)
        high_msg = torch.mean(high, dim=2, keepdim=True).transpose(1, 2)
        msg_avg = (low_msg + high_msg) / 2
        msg = self.msg_linear_out(msg_avg)
        return msg


# class Decoder(nn.Module):
#     def __init__(self, process_config, model_config, train_config, msg_length):
#         super(Decoder, self).__init__()
#         self.robust = model_config["robust"]
#         # if self.robust:
#         #     self.dl = distortion()
#
#         # self.mel_transform = TacotronSTFT(filter_length=process_config["mel"]["n_fft"], hop_length=process_config["mel"]["hop_length"], win_length=process_config["mel"]["win_length"])
#         # self.vocoder = get_vocoder(device)
#         # self.vocoder_step = model_config["structure"]["vocoder_step"]
#
#         self.win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
#         self.hop_length = process_config["mel"]["hop_length"]
#         self.block = model_config["conv2"]["block"]
#         self.EX = WatermarkExtracter(input_channel=2, hidden_dim=model_config["conv2"]["hidden_dim"], block=self.block)
#         self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])
#         self.msg_linear_out = FCBlock(self.win_dim//2, msg_length)
#
#     def forward(self, y, global_step):
#         y_identity = y.clone()
#         y_d = y
#         # if global_step > self.vocoder_step:
#         #     y_mel = self.mel_transform.mel_spectrogram(y.squeeze(1))
#         #     # y = self.vocoder(y_mel)
#         #     y_d = (self.mel_transform.griffin_lim(magnitudes=y_mel)).unsqueeze(1)
#         # else:
#         #     y_d = y
#         if self.robust:
#             y_d_d = self.dl(y_d, self.robust)
#         else:
#             y_d_d = y_d
#         _, _, stft_result = self.stft.transform(y_d_d)
#         extracted_wm = self.EX(stft_result).squeeze(1)  # (B, win_dim, length)
#         # Explicitly split the 162-dim vector into two halves of 81-dim each
#         low, high = extracted_wm.chunk(2, dim=1)  # each has shape [B, win_dim / 2, length]
#         low_msg = torch.mean(low, dim=2, keepdim=True).transpose(1,2)
#         high_msg = torch.mean(high, dim=2, keepdim=True).transpose(1, 2)
#         msg_avg = (low_msg + high_msg) / 2  # Average the two halves -> shape: [B, 1, 81]
#         # msg = torch.mean(extracted_wm, dim=2, keepdim=True).transpose(1,2)
#         msg = self.msg_linear_out(msg_avg)
#
#         _, _, stft_result_identity = self.stft.transform(y_identity)
#         extracted_wm_identity = self.EX(stft_result_identity).squeeze(1)
#         low_identity, high_identity = extracted_wm_identity.chunk(2, dim=1)  # each has shape [B, win_dim / 2, length]
#         low_msg_identity = torch.mean(low_identity, dim=2, keepdim=True).transpose(1, 2)
#         high_msg_identity = torch.mean(high_identity, dim=2, keepdim=True).transpose(1, 2)
#         msg_avg_identity = (low_msg_identity + high_msg_identity) / 2  # Average the two halves -> shape: [B, 1, 81]
#         # msg_identity = torch.mean(extracted_wm_identity,dim=2, keepdim=True).transpose(1,2)
#         msg_identity = self.msg_linear_out(msg_avg_identity)
#         return msg, msg_identity
#
#     def test_forward(self, y):
#         spect, phase, stft_result= self.stft.transform(y)
#         extracted_wm = self.EX(stft_result).squeeze(1)
#         msg = torch.mean(extracted_wm,dim=2, keepdim=True).transpose(1,2)
#         msg = self.msg_linear_out(msg)
#         return msg


class Discriminator(nn.Module):
    def __init__(self, process_config):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            ReluBlock(2, 16, 3, 1, 1),
            ReluBlock(16, 32, 3, 1, 1),
            ReluBlock(32, 64, 3, 1, 1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.linear = nn.Linear(64, 1)
        self.stft = fixed_STFT(
            process_config["mel"]["n_fft"],
            process_config["mel"]["hop_length"],
            process_config["mel"]["win_length"],
        )

    def forward(self, x):
        _, _, stft_result = self.stft.transform(x)
        x = self.conv(stft_result)
        x = x.squeeze(2).squeeze(2)
        x = self.linear(x)
        return x
