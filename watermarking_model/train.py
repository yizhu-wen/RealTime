import argparse
import logging
import os
import random
from collections import defaultdict
from io import BytesIO
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from rich.progress import track
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import wandb
from dataset.data import WavDataset as MyDataset
from dataset.data import collate_fn
from model.conv2_mel_modules import Encoder, Decoder, Discriminator
from model.loss import Loss_identity
from utils.optimizer import my_step
from utils.tools import save_op
from watermarking_model.distortions.dl import AudioEffects, get_my_encodec_audio_effect


def save_spectrogram_to_buffer(signal, sample_rate=16000):
    buf = BytesIO()
    plt.figure(figsize=(10, 4))
    plt.specgram(
        np.maximum(signal.cpu().detach().numpy(), 1e-10),
        Fs=sample_rate,
        NFFT=320,
        noverlap=160,
        window=np.hanning(320),
        cmap="magma",
        vmin=-100,
    )
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.0)
    plt.close()
    buf.seek(0)  # reset buffer pointer to beginning
    return buf


def buffer_to_wandb_image(buffer, caption=""):
    # Convert the buffer to a PIL Image, then to a numpy array.
    img = Image.open(buffer)
    np_img = np.array(img)
    return wandb.Image(np_img, caption=caption)


def normalize_audio(y: torch.Tensor) -> torch.Tensor:
    """Normalize an audio tensor so its maximum absolute value is 1."""
    peak = torch.max(torch.abs(y))
    if peak.item() > 1e-8:
        y = y / peak
    return y


# Set random seed for reproducibility
def set_random_seed(seed: int):
    random.seed(seed)  # For Python random module
    np.random.seed(seed)  # For NumPy
    torch.manual_seed(seed)  # For PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # For PyTorch (GPU)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables optimization for reproducibility


def generate_random_msg(batch_size, msg_length, device):
    # random [0, 1], mapped to [-1, 1]
    return (
        torch.randint(0, 2, (batch_size, 1, msg_length), device=device).float() * 2
    ) - 1


# Set random seed for reproducibility
SEED = 2022
set_random_seed(SEED)

logging_mark = "#" * 20
# warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(configs):
    logging.info("main function")
    process_config, model_config, train_config = configs

    pre_step = 0
    # ---------------- get train dataset
    train_audios = MyDataset(
        process_config=process_config, train_config=train_config, flag="train"
    )
    val_audios = MyDataset(
        process_config=process_config, train_config=train_config, flag="val"
    )
    dev_audios = MyDataset(
        process_config=process_config, train_config=train_config, flag="test"
    )

    batch_size = train_config["optimize"]["batch_size"]
    assert batch_size < len(train_audios)
    train_audios_loader = DataLoader(
        train_audios,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,  # Enable pin memory for faster data transfer to GPU
        num_workers=20,  # Enable multiple workers for data loading
        persistent_workers=True,  # Keep workers alive between epochs
    )
    val_audios_loader = DataLoader(
        val_audios,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=20,
        persistent_workers=True,
    )
    dev_audios_loader = DataLoader(
        dev_audios,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=20,
        persistent_workers=True,
    )

    # Initialize wandb only if enabled in config
    if train_config["wandb"]["enabled"]:
        wandb.login(key=train_config["wandb"]["key"])
        wandb.init(
            project=train_config["wandb"]["project"],
            name=train_config["wandb"]["name"],
            config={
                "dataset": "LibriSpeech",
                "data_size": "whole size divided by {}".format(
                    train_config["iter"]["data_divider"]
                ),
                "batch_size": train_config["optimize"]["batch_size"],
                "learning_rate": train_config["optimize"]["lr"],
                "delay_amt_second": train_config["watermark"]["delay_amt_second"],
                "future_amt_second": train_config["watermark"]["future_amt_second"],
                "nbits": train_config["watermark"]["length"],
                "epochs": train_config["iter"]["epoch"],
                "save_circle": train_config["iter"]["save_circle"],
                "show_circle": train_config["iter"]["show_circle"],
                "val_circle": train_config["iter"]["val_circle"],
            },
        )
        test_loss_summary_table = wandb.Table(
            columns=[
                "test_wav_loss",
                "test_msg_loss",
                "test_loudness_loss",
                "test_acc",
                "test_avg_snr",
                "test_d_loss_on_encoded",
                "test_d_loss_on_cover",
            ]
        )
        # val_audio_table = wandb.Table(
        #     columns=[
        #         "Epoch",
        #         "Original Audio",
        #         "Watermarked Audio",
        #         "Watermark Audio",
        #         "Original Amplitude Spectrogram",
        #         "Original Phase Spectrogram",
        #         "Watermarked Amplitude Spectrogram",
        #         "Watermarked Phase Spectrogram",
        #         "Watermark Amplitude Spectrogram",
        #         "Watermark Phase Spectrogram",
        #     ]
        # )
        test_audio_table = wandb.Table(
            columns=[
                "Original Audio",
                "Watermarked Audio",
                "Watermark Audio",
                "Original Amplitude Spectrogram",
                "Watermarked Amplitude Spectrogram",
                "Watermark Amplitude Spectrogram",
            ]
        )
    else:
        test_loss_summary_table = None

    # ---------------- build model
    encoder = Encoder(process_config, model_config, train_config).to(device)
    decoder = Decoder(process_config, model_config, train_config).to(device)
    # adv
    if train_config["adv"]:
        discriminator = Discriminator(process_config).to(device)
        d_op = Adam(
            params=discriminator.parameters(),  # No need for chain() with single model
            betas=train_config["optimize"]["betas"],
            eps=train_config["optimize"]["eps"],
            weight_decay=train_config["optimize"]["weight_decay"],
            lr=train_config["optimize"]["lr"],
        )
        lr_sched_d = StepLR(
            d_op,
            step_size=train_config["optimize"]["step_size"],
            gamma=train_config["optimize"]["gamma"],
        )

    # ---------------- optimizer
    en_de_op = Adam(
        params=chain(encoder.parameters(), decoder.parameters()),
        betas=train_config["optimize"]["betas"],
        eps=train_config["optimize"]["eps"],
        weight_decay=train_config["optimize"]["weight_decay"],
        lr=train_config["optimize"]["lr"],
    )
    lr_sched = StepLR(
        en_de_op,
        step_size=train_config["optimize"]["step_size"],
        gamma=train_config["optimize"]["gamma"],
    )

    # ---------------- Loss
    loss = Loss_identity()

    # ---------------- Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)

    # ---------------- train
    logging.info(logging_mark + "\t" + "Begin Training" + "\t" + logging_mark)
    epoch_num = train_config["iter"]["epoch"]
    msg_length = train_config["watermark"]["length"]
    save_circle = train_config["iter"]["save_circle"]
    show_circle = train_config["iter"]["show_circle"]
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    lambda_b = train_config["optimize"]["lambda_b"]
    num_save_img = train_config["iter"]["num_save_img"]
    sample_rate = process_config["audio"]["or_sample_rate"]
    encodec_cfg = train_config["audio_effects"]["encodec"]
    aug_weights = train_config["aug_weights"]
    all_aug_names = list(aug_weights.keys())

    encodec_effects = get_my_encodec_audio_effect(encodec_cfg, sample_rate)

    other_augs = {
        "speed": AudioEffects.speed,
        "updownresample": AudioEffects.updownresample,
        "echo": AudioEffects.echo,
        "pink_noise": AudioEffects.pink_noise,
        "lowpass_filter": AudioEffects.lowpass_filter,
        "highpass_filter": AudioEffects.highpass_filter,
        "bandpass_filter": AudioEffects.bandpass_filter,
        "smooth": AudioEffects.smooth,
        "boost_audio": AudioEffects.boost_audio,
        "duck_audio": AudioEffects.duck_audio,
        "mp3_compression": AudioEffects.mp3_compression,
        "aac_compression": AudioEffects.aac_compression,
        "identity": AudioEffects.identity,
        "mel_griffin_lim": AudioEffects.mel_griffin_lim,
    }

    augmentations = {**other_augs, **encodec_effects}

    global_step = 0
    train_len = len(train_audios_loader)
    # At the beginning of each epoch, initialize a dictionary to store accuracies:
    train_epoch_acc_dict = defaultdict(list)
    val_epoch_acc_dict = defaultdict(list)
    test_epoch_acc_dict = defaultdict(list)
    for ep in range(1, epoch_num + 1):
        encoder.train()
        decoder.train()
        discriminator.train()
        step = 0
        logging.info("Epoch {}/{}".format(ep, epoch_num))
        train_aug_avg_msg_loss = 0
        train_aug_avg_acc = 0
        train_avg_snr = 0
        train_avg_wav_loss = 0
        train_avg_loudness_loss = 0
        train_avg_d_loss_on_encoded = 0
        train_avg_d_loss_on_cover = 0
        for sample in track(train_audios_loader):
            other_losses: dict = {}
            global_step += 1
            step += 1
            b = sample["matrix"].shape[0]
            # ---------------- build watermark
            wav_matrix = sample["matrix"].to(device)
            msg = generate_random_msg(wav_matrix.size(0), msg_length, device)
            watermark, carrier_watermarked = encoder(wav_matrix, msg, global_step)
            y_wm = wav_matrix + watermark

            # Initialize loss aggregation variables.
            num_augs_applied = 0
            batch_acc_dict = {}  # To store per-augmentation accuracy
            for aug_name, prob in aug_weights.items():
                # Sample a random number and apply this augmentation if it falls under the probability.
                # Skip mp3 and aac during training
                if aug_name in ["mp3_compression", "aac_compression"]:
                    continue

                if random.random() < prob:
                    # print(f"Applying augmentation: {aug_name}")
                    # Get the augmentation function from the dictionary.
                    aug_func = augmentations[aug_name]
                    # Apply the augmentation; you can pass additional parameters if needed.
                    augmented_audio = aug_func(y_wm.unsqueeze(1))
                    decoded = decoder(augmented_audio.squeeze(1), global_step)
                    msg_loss = loss.msg_loss_func(decoded, msg)
                    other_losses[f"{aug_name}_msg_loss"] = msg_loss
                    num_augs_applied += 1
                    aug_acc = (
                        (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                    ).item()
                    # Save the per-augmentation accuracy in a dictionary.
                    batch_acc_dict[aug_name] = aug_acc

            # For logging, update the epoch-level dictionary with batch accuracies:
            for aug, acc in batch_acc_dict.items():
                train_epoch_acc_dict[aug].append(acc)

            other_loss = torch.tensor(0.0, device=device)
            for name, o_loss in other_losses.items():
                other_loss += o_loss / num_augs_applied
            losses = loss.en_de_loss(wav_matrix, y_wm)

            if global_step < pre_step:
                sum_loss = lambda_m * losses[1]
            else:
                sum_loss = lambda_e * losses[0] + lambda_e * losses[1] + 0 * other_loss

            # adv
            if train_config["adv"]:
                lambda_a = lambda_m = train_config["optimize"][
                    "lambda_a"
                ]  # modify weights of m and a for better convergence
                g_target_label_encoded = torch.full((b, 1), 1, device=device).float()
                d_on_encoded_for_enc = discriminator(y_wm)
                # target label for encoded images should be 'cover', because we want to fool the discriminator
                g_loss_adv = F.binary_cross_entropy_with_logits(
                    d_on_encoded_for_enc, g_target_label_encoded
                )
                sum_loss += lambda_a * g_loss_adv

            sum_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                chain(encoder.parameters(), decoder.parameters()), max_norm=1.0
            )

            my_step(en_de_op, lr_sched, global_step, train_len)

            if train_config["adv"]:
                d_target_label_cover = torch.full((b, 1), 1, device=device).float()
                d_on_cover = discriminator(wav_matrix)
                d_loss_on_cover = F.binary_cross_entropy_with_logits(
                    d_on_cover, d_target_label_cover
                )

                d_target_label_encoded = torch.full((b, 1), 0, device=device).float()
                d_on_encoded = discriminator(y_wm.detach())
                # target label for encoded images should be 'encoded', because we want discriminator fight with encoder
                d_loss_on_encoded = F.binary_cross_entropy_with_logits(
                    d_on_encoded, d_target_label_encoded
                )

                d_loss = d_loss_on_cover + d_loss_on_encoded
                d_loss.backward()

                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

                my_step(d_op, lr_sched_d, global_step, train_len)

                train_avg_d_loss_on_cover += d_loss_on_cover.item()
                train_avg_d_loss_on_encoded += d_loss_on_encoded.item()

            decoder_acc = sum(batch_acc_dict.values()) / len(batch_acc_dict)
            zero_tensor = torch.zeros(wav_matrix.shape).to(device)
            snr = 10 * torch.log10(
                mse_loss(wav_matrix.detach(), zero_tensor)
                / mse_loss(wav_matrix.detach(), y_wm.detach())
            )
            norm2 = mse_loss(wav_matrix.detach(), zero_tensor)

            # Update averages
            train_aug_avg_acc += decoder_acc
            train_avg_snr += snr.item()
            train_avg_wav_loss += losses[0].item()
            train_avg_loudness_loss += losses[1].item()
            train_aug_avg_msg_loss += other_loss.item()

            if step % show_circle == 0:
                logging.info("-" * 100)
                logging.info(
                    "step:{} - wav_loss:{:.8f} - tfloudness_loss:{:.8f} - aug_msg_loss:{:.8f} - acc:{:.8f} - snr:{:.8f} - norm:{:.8f} - d_loss_on_encoded:{:.8f} - d_loss_on_cover:{:.8f}".format(
                        step,
                        losses[0],
                        losses[1],
                        other_loss,
                        decoder_acc,
                        snr,
                        norm2,
                        d_loss_on_encoded.item(),
                        d_loss_on_cover.item(),
                    )
                )

            # --- Memory Cleanup ---
            # Delete all intermediate variables that are no longer needed
            del wav_matrix, msg, watermark, carrier_watermarked, y_wm, decoded, losses
            if train_config["adv"]:
                del d_target_label_cover, d_on_cover, d_loss_on_cover
                del d_target_label_encoded, d_on_encoded, d_loss_on_encoded, d_loss

        train_epoch_avg_acc = {
            f"train/{aug}_accuracy": sum(acc_list) / len(acc_list)
            for aug, acc_list in train_epoch_acc_dict.items()
        }

        train_avg_wav_loss /= step
        train_aug_avg_msg_loss /= step
        train_aug_avg_acc /= step
        train_avg_snr /= step
        train_avg_loudness_loss /= step
        train_avg_d_loss_on_encoded /= step
        train_avg_d_loss_on_cover /= step

        train_metrics = {
            "train/wav_loss": train_avg_wav_loss,
            "train/msg_loss": train_aug_avg_msg_loss,
            "train/loudness_loss": train_avg_loudness_loss,
            "train/aug_acc": train_aug_avg_acc,
            "train/snr": train_avg_snr,
            "train/d_loss_on_encoded": train_avg_d_loss_on_encoded,
            "train/d_loss_on_cover": train_avg_d_loss_on_cover,
        }

        # if ep % save_circle == 0 or ep == 1 or ep == 2:
        if ep % save_circle == 0:
            if not model_config["structure"]["ab"]:
                path = os.path.join(train_config["path"]["ckpt"], "pth")
            else:
                path = os.path.join(train_config["path"]["ckpt"], "pth_ab")
            save_op(path, ep, encoder, decoder, en_de_op)
            # shutil.copyfile(os.path.realpath(__file__), os.path.join(path, os.path.basename(os.path.realpath(__file__)))) # save training scripts

        # ---------------- validation
        with torch.inference_mode():
            encoder.eval()
            decoder.eval()
            discriminator.eval()

            val_aug_avg_msg_loss = 0
            val_aug_avg_acc = 0
            val_avg_snr = 0
            val_avg_wav_loss = 0
            val_avg_loudness_loss = 0
            val_avg_d_loss_on_encoded = 0
            val_avg_d_loss_on_cover = 0
            count = 0
            for sample in track(val_audios_loader):
                other_losses: dict = {}
                count += 1
                b = sample["matrix"].shape[0]
                # ---------------- build watermark
                wav_matrix = sample["matrix"].to(device)
                msg = generate_random_msg(wav_matrix.size(0), msg_length, device)
                watermark, carrier_watermarked = encoder(wav_matrix, msg, global_step)
                y_wm = wav_matrix + watermark
                # Initialize loss aggregation variables.
                num_augs_applied = 0
                batch_acc_dict = {}  # To store per-augmentation accuracy
                for aug_name, prob in aug_weights.items():
                    # Sample a random number and apply this augmentation if it falls under the probability.
                    if random.random() < prob:
                        print(f"Applying augmentation: {aug_name}")
                        # Get the augmentation function from the dictionary.
                        aug_func = augmentations[aug_name]
                        # Apply the augmentation; you can pass additional parameters if needed.
                        augmented_audio = aug_func(y_wm.unsqueeze(1))
                        decoded = decoder(augmented_audio.squeeze(1), global_step)
                        msg_loss = loss.msg_loss_func(msg, decoded)
                        other_losses[f"{aug_name}_msg_loss"] = msg_loss
                        num_augs_applied += 1
                        aug_acc = (
                            (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                        ).item()
                        # Save the per-augmentation accuracy in a dictionary.
                        batch_acc_dict[aug_name] = aug_acc

                # For logging, update the epoch-level dictionary with batch accuracies:
                for aug, acc in batch_acc_dict.items():
                    val_epoch_acc_dict[aug].append(acc)

                other_loss = torch.tensor(0.0, device=device)
                for name, o_loss in other_losses.items():
                    other_loss += o_loss / num_augs_applied

                losses = loss.en_de_loss(wav_matrix, y_wm)

                # adv
                if train_config["adv"]:
                    lambda_a = lambda_m = train_config["optimize"]["lambda_a"]
                    g_target_label_encoded = torch.full(
                        (b, 1), 1, device=device
                    ).float()
                    d_on_encoded_for_enc = discriminator(y_wm)
                    g_loss_adv = F.binary_cross_entropy_with_logits(
                        d_on_encoded_for_enc, g_target_label_encoded
                    )
                if train_config["adv"]:
                    d_target_label_cover = torch.full((b, 1), 1, device=device).float()
                    d_on_cover = discriminator(wav_matrix)
                    d_loss_on_cover = F.binary_cross_entropy_with_logits(
                        d_on_cover, d_target_label_cover
                    )

                    d_target_label_encoded = torch.full(
                        (b, 1), 0, device=device
                    ).float()
                    d_on_encoded = discriminator(y_wm.detach())
                    d_loss_on_encoded = F.binary_cross_entropy_with_logits(
                        d_on_encoded, d_target_label_encoded
                    )

                decoder_acc = sum(batch_acc_dict.values()) / len(batch_acc_dict)
                zero_tensor = torch.zeros(wav_matrix.shape).to(device)
                snr = 10 * torch.log10(
                    mse_loss(wav_matrix.detach(), zero_tensor)
                    / mse_loss(wav_matrix.detach(), y_wm.detach())
                )

                val_aug_avg_acc += decoder_acc
                val_avg_snr += snr.item()
                val_avg_wav_loss += losses[0].item()
                val_avg_loudness_loss += losses[1].item()
                val_aug_avg_msg_loss += other_loss.item()
                val_avg_d_loss_on_cover += d_loss_on_cover
                val_avg_d_loss_on_encoded += d_loss_on_encoded

            val_avg_wav_loss /= count
            val_aug_avg_msg_loss /= count
            val_aug_avg_acc /= count
            val_avg_snr /= count
            val_avg_loudness_loss /= count
            val_avg_d_loss_on_encoded /= count
            val_avg_d_loss_on_cover /= count

            logging.info("#e" * 60)
            logging.info(
                "epoch:{} - wav_loss:{:.8f} - msg_loss:{:.8f} - tfloudness_loss:{:.8f} - acc:{:.8f} - snr:{:.8f} - d_loss_on_encoded:{:.8f} - d_loss_on_cover:{:.8f}".format(
                    ep,
                    val_avg_wav_loss,
                    val_aug_avg_msg_loss,
                    val_avg_loudness_loss,
                    val_aug_avg_acc,
                    val_avg_snr,
                    val_avg_d_loss_on_encoded.item(),
                    val_avg_d_loss_on_cover.item(),
                )
            )

        val_epoch_avg_acc = {
            f"train/{aug}_accuracy": sum(acc_list) / len(acc_list)
            for aug, acc_list in val_epoch_acc_dict.items()
        }

        val_metrics = {
            "val/wav_loss": val_avg_wav_loss,
            "val/msg_loss": val_aug_avg_msg_loss,
            "val/tfloudness_loss": val_avg_loudness_loss,
            "val/acc": val_aug_avg_acc,
            "val/snr": val_avg_snr,
            "val/d_loss_on_encoded": val_avg_d_loss_on_encoded,
            "val/d_loss_on_cover": val_avg_d_loss_on_cover,
        }

        if train_config["wandb"]["enabled"]:
            wandb.log(
                {
                    **train_metrics,
                    **val_metrics,
                    **train_epoch_avg_acc,
                    **val_epoch_avg_acc,
                }
            )
        # Clear for the next epoch:
        train_epoch_acc_dict.clear()
        val_epoch_acc_dict.clear()

    with torch.inference_mode():
        encoder.eval()
        decoder.eval()
        discriminator.eval()

        test_aug_avg_msg_loss = 0
        test_aug_avg_acc = 0
        test_avg_snr = 0
        test_avg_wav_loss = 0
        test_avg_loudness_loss = 0
        test_avg_d_loss_on_encoded = 0
        test_avg_d_loss_on_cover = 0
        count = 0
        for sample in track(dev_audios_loader):
            count += 1
            b = sample["matrix"].shape[0]
            # ---------------- build watermark
            wav_matrix = sample["matrix"].to(device)
            msg = generate_random_msg(wav_matrix.size(0), msg_length, device)
            watermark, carrier_watermarked = encoder(wav_matrix, msg, global_step)
            y_wm = wav_matrix + watermark
            # Initialize loss aggregation variables.
            num_augs_applied = 0
            batch_acc_dict = {}  # To store per-augmentation accuracy
            for aug_name, prob in aug_weights.items():
                # Sample a random number and apply this augmentation if it falls under the probability.
                if random.random() < prob:
                    print(f"Applying augmentation: {aug_name}")
                    # Get the augmentation function from the dictionary.
                    aug_func = augmentations[aug_name]
                    # Apply the augmentation; you can pass additional parameters if needed.
                    augmented_audio = aug_func(y_wm.unsqueeze(1))
                    decoded = decoder(augmented_audio.squeeze(1), global_step)
                    msg_loss = loss.msg_loss_func(msg, decoded)
                    other_losses[f"{aug_name}_msg_loss"] = msg_loss
                    num_augs_applied += 1
                    aug_acc = (
                        (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
                    ).item()
                    # Save the per-augmentation accuracy in a dictionary.
                    batch_acc_dict[aug_name] = aug_acc

            other_loss = torch.tensor(0.0, device=device)
            for name, o_loss in other_losses.items():
                other_loss += o_loss / num_augs_applied

            losses = loss.en_de_loss(wav_matrix, y_wm)
            # adv
            if train_config["adv"]:
                lambda_a = lambda_m = train_config["optimize"]["lambda_a"]
                g_target_label_encoded = torch.full((b, 1), 1, device=device).float()
                d_on_encoded_for_enc = discriminator(y_wm)
                g_loss_adv = F.binary_cross_entropy_with_logits(
                    d_on_encoded_for_enc, g_target_label_encoded
                )
            if train_config["adv"]:
                d_target_label_cover = torch.full((b, 1), 1, device=device).float()
                d_on_cover = discriminator(wav_matrix)
                d_loss_on_cover = F.binary_cross_entropy_with_logits(
                    d_on_cover, d_target_label_cover
                )

                d_target_label_encoded = torch.full((b, 1), 0, device=device).float()
                d_on_encoded = discriminator(y_wm.detach())
                d_loss_on_encoded = F.binary_cross_entropy_with_logits(
                    d_on_encoded, d_target_label_encoded
                )

            decoder_acc = sum(batch_acc_dict.values()) / len(batch_acc_dict)
            zero_tensor = torch.zeros(wav_matrix.shape).to(device)
            snr = 10 * torch.log10(
                mse_loss(wav_matrix.detach(), zero_tensor)
                / mse_loss(wav_matrix.detach(), y_wm.detach())
            )

            test_aug_avg_acc += decoder_acc
            test_avg_snr += snr.item()
            test_avg_wav_loss += losses[0].item()
            test_avg_loudness_loss += losses[1].item()
            test_aug_avg_msg_loss += other_loss.item()
            test_avg_d_loss_on_cover += d_loss_on_cover
            test_avg_d_loss_on_encoded += d_loss_on_encoded

            # Initialize wandb only if enabled in config
            # if train_config["wandb"]["enabled"]:
            #     if count <= num_save_img:
            #         with tempfile.TemporaryDirectory() as tmpdir:
            #             # Normalize each audio signal:
            #             y_norm = normalize_audio(wav_matrix[0].cpu().detach())
            #             y_wm_norm = normalize_audio(y_wm[0].cpu().detach())
            #             wm_norm = normalize_audio(watermark[0].cpu().detach())
            #
            #             original_buf = save_spectrogram_to_buffer(y_norm)
            #             watermarked_buf = save_spectrogram_to_buffer(y_wm_norm)
            #             watermark_buf = save_spectrogram_to_buffer(wm_norm)
            #
            #             test_audio_table.add_data(
            #                 wandb.Audio(wav_matrix[0].cpu().detach().numpy(), sample_rate=sample_rate),
            #                 wandb.Audio(y_wm[0].cpu().detach().numpy(), sample_rate=sample_rate),
            #                 wandb.Audio(watermark[0].cpu().detach().numpy(), sample_rate=sample_rate),
            #                 buffer_to_wandb_image(original_buf),
            #                 buffer_to_wandb_image(watermarked_buf),
            #                 buffer_to_wandb_image(watermark_buf))

        test_avg_wav_loss /= count
        test_aug_avg_msg_loss /= count
        test_aug_avg_acc /= count
        test_avg_snr /= count
        test_avg_loudness_loss /= count
        test_avg_d_loss_on_encoded /= count
        test_avg_d_loss_on_cover /= count

        logging.info("#test" * 20)
        logging.info(
            "Test: wav_loss:{:.8f} - msg_loss:{:.8f} - tfloudness_loss:{:.8f} - acc:{:.8f} - snr:{:.8f} - d_loss_on_encoded:{:.8f} - d_loss_on_cover:{:.8f}".format(
                test_avg_wav_loss,
                test_aug_avg_msg_loss,
                test_avg_loudness_loss,
                test_aug_avg_acc,
                test_avg_snr,
                test_avg_d_loss_on_encoded.item(),
                test_avg_d_loss_on_cover.item(),
            )
        )

        # if train_config["wandb"]["enabled"]:
        #     # Log the audio_table to wandb
        #     wandb.log({"test_audio_table": test_audio_table})

        test_epoch_avg_acc = {
            f"train/{aug}_accuracy": sum(acc_list) / len(acc_list)
            for aug, acc_list in test_epoch_acc_dict.items()
        }

        if train_config["wandb"]["enabled"] and test_loss_summary_table is not None:
            test_loss_summary_table.add_data(
                test_avg_wav_loss,
                test_aug_avg_msg_loss,
                test_avg_loudness_loss,
                test_aug_avg_acc,
                test_avg_snr,
                test_avg_d_loss_on_encoded,
                test_avg_d_loss_on_cover,
            )

            wandb.log(
                {
                    "test_loss_summary_table": test_loss_summary_table,
                    **test_epoch_avg_acc,
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--process_config",
        type=str,
        help="path to process.yaml",
        default=r"./config/process.yaml",
    )
    parser.add_argument(
        "-m",
        "--model_config",
        type=str,
        help="path to model.yaml",
        default=r"./config/model.yaml",
    )
    parser.add_argument(
        "-t",
        "--train_config",
        type=str,
        help="path to train.yaml",
        default=r"./config/train.yaml",
    )
    args = parser.parse_args()

    # Read Config
    process_config = yaml.load(open(args.process_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (process_config, model_config, train_config)

    main(configs)
