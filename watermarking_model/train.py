import os
import torch
import yaml
import logging
import argparse
import wandb
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from rich.progress import track
from torch.utils.data import DataLoader
from model.loss import Loss_identity, encodec_loss_fn
from utils.tools import save_op
from utils.optimizer import my_step
from itertools import chain
import librosa.display
from torch.nn.functional import mse_loss
from dataset.data import collate_fn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from model.conv2_mel_modules import Encoder, Decoder, Discriminator
from dataset.data import WavDataset as MyDataset
import random
import shutil
from PIL import Image
import torch.backends.cudnn as cudnn

def save_spectrogram_to_buffer(signal, sample_rate=16000):
    buf = BytesIO()
    plt.figure(figsize=(10, 4))
    plt.specgram(
        np.maximum(signal.cpu().detach().numpy(), 1e-10),
        Fs=sample_rate,
        NFFT=320,
        noverlap=160,
        window=np.hanning(320),
        cmap='magma',
        vmin=-100,
    )
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.0)
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
    return torch.randint(0, 2, (batch_size, 1, msg_length), device=device).float()

# Set random seed for reproducibility
SEED = 2022
set_random_seed(SEED)

logging_mark = "#"*20
# warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(configs):
    logging.info('main function')
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
                "test_loss",
                "test_wav_loss",
                "test_msg_loss",
                "test_loudness_loss",
                "test_acc_1",
                "test_acc_2",
                "test_avg_snr",
                "test_d_loss_on_encoded",
                "test_d_loss_on_cover",
                "test_encodec_loss",
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
            lr = train_config["optimize"]["lr"]
        )
        lr_sched_d = StepLR(d_op, step_size=train_config["optimize"]["step_size"], gamma=train_config["optimize"]["gamma"])

    # ---------------- optimizer
    en_de_op = Adam(
            params=chain(encoder.parameters(), decoder.parameters()),
            betas=train_config["optimize"]["betas"],
            eps=train_config["optimize"]["eps"],
            weight_decay=train_config["optimize"]["weight_decay"],
            lr = train_config["optimize"]["lr"]
        )
    lr_sched = StepLR(en_de_op, step_size=train_config["optimize"]["step_size"], gamma=train_config["optimize"]["gamma"])

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
    global_step = 0
    train_len = len(train_audios_loader)
    for ep in range(1, epoch_num+1):
        encoder.train()
        decoder.train()
        discriminator.train()
        step = 0
        logging.info('Epoch {}/{}'.format(ep, epoch_num))

        train_avg_acc = [0, 0]
        train_avg_snr = 0
        train_avg_wav_loss = 0
        train_avg_msg_loss = 0
        train_avg_loudness_loss = 0
        train_avg_d_loss_on_encoded = 0
        train_avg_d_loss_on_cover = 0
        train_avg_encodec_loss = 0
        for sample in track(train_audios_loader):
            global_step += 1
            step += 1
            b = sample["matrix"].shape[0]
            # ---------------- build watermark
            wav_matrix = sample["matrix"].to(device)
            msg = generate_random_msg(wav_matrix.size(0), msg_length, device)
            watermark, carrier_watermarked = encoder(wav_matrix, msg, global_step)
            y_wm = wav_matrix + watermark
            decoded, msg_encodec_list = decoder(y_wm, global_step)
            losses = loss.en_de_loss(wav_matrix, y_wm, msg, decoded)
            encodec_loss = encodec_loss_fn(msg_encodec_list, msg)
            #lamda_e = 1.
            #lamda_m = 10
            if global_step < pre_step:
                sum_loss = lambda_m*losses[1]
            else:
                sum_loss = lambda_e*losses[0] + lambda_m*losses[1] + lambda_b*losses[2] + encodec_loss
            
            # adv
            if train_config["adv"]:
                lambda_a = lambda_m = train_config["optimize"]["lambda_a"] # modify weights of m and a for better convergence
                g_target_label_encoded = torch.full((b, 1), 1, device=device).float()
                d_on_encoded_for_enc = discriminator(y_wm)
                # target label for encoded images should be 'cover', because we want to fool the discriminator
                g_loss_adv = F.binary_cross_entropy_with_logits(d_on_encoded_for_enc, g_target_label_encoded)
                sum_loss += lambda_a*g_loss_adv

            sum_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                chain(encoder.parameters(), decoder.parameters()), max_norm=1.0
            )

            my_step(en_de_op, lr_sched, global_step, train_len)
            
            if train_config["adv"]:
                d_target_label_cover = torch.full((b, 1), 1, device=device).float()
                d_on_cover = discriminator(wav_matrix)
                d_loss_on_cover = F.binary_cross_entropy_with_logits(d_on_cover, d_target_label_cover)

                d_target_label_encoded = torch.full((b, 1), 0, device=device).float()
                d_on_encoded = discriminator(y_wm.detach())
                # target label for encoded images should be 'encoded', because we want discriminator fight with encoder
                d_loss_on_encoded = F.binary_cross_entropy_with_logits(d_on_encoded, d_target_label_encoded)

                d_loss = d_loss_on_cover + d_loss_on_encoded
                d_loss.backward()

                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

                my_step(d_op, lr_sched_d, global_step, train_len)

                train_avg_d_loss_on_cover += d_loss_on_cover.item()
                train_avg_d_loss_on_encoded += d_loss_on_encoded.item()

            decoder_acc = [((decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(),
                           ((decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item()]
            zero_tensor = torch.zeros(wav_matrix.shape).to(device)
            snr = 10 * torch.log10(mse_loss(wav_matrix.detach(), zero_tensor) / mse_loss(wav_matrix.detach(), y_wm.detach()))
            norm2 = mse_loss(wav_matrix.detach(), zero_tensor)

            # Update averages
            train_avg_acc[0] += decoder_acc[0]
            train_avg_acc[1] += decoder_acc[1]
            train_avg_snr += snr.item()
            train_avg_wav_loss += losses[0].item()
            train_avg_msg_loss += losses[1].item()
            train_avg_loudness_loss += losses[2].item()
            train_avg_encodec_loss += encodec_loss.item()

            if step % show_circle == 0:
                logging.info('-' * 100)
                logging.info("step:{} - wav_loss:{:.8f} - msg_loss:{:.8f} - tfloudness_loss:{:.8f} - encodec_loss:{:.8f} - acc:[{:.8f},{:.8f}] - snr:{:.8f} - norm:{:.8f} - d_loss_on_encoded:{:.8f} - d_loss_on_cover:{:.8f}".format(
                    step, losses[0], losses[1], losses[2], encodec_loss.item(), decoder_acc[0], decoder_acc[1],
                    snr, norm2, d_loss_on_encoded.item(), d_loss_on_cover.item()))

            # --- Memory Cleanup ---
            # Delete all intermediate variables that are no longer needed
            del wav_matrix, msg, watermark, carrier_watermarked, y_wm, decoded, msg_encodec_list
            del losses, encodec_loss
            if train_config["adv"]:
                del d_target_label_cover, d_on_cover, d_loss_on_cover
                del d_target_label_encoded, d_on_encoded, d_loss_on_encoded, d_loss

        train_avg_acc[0] /= step
        train_avg_acc[1] /= step
        train_avg_snr /= step
        train_avg_wav_loss /= step
        train_avg_msg_loss /= step
        train_avg_loudness_loss /= step
        train_avg_d_loss_on_encoded /= step
        train_avg_d_loss_on_cover /= step
        train_avg_encodec_loss /= step

        train_metrics = {
            "train/wav_loss": train_avg_wav_loss,
            "train/msg_loss": train_avg_msg_loss,
            "train/loudness_loss": train_avg_loudness_loss,
            "train/acc_1": train_avg_acc[0],
            "train/acc_2": train_avg_acc[1],
            "train/snr": train_avg_snr,
            "train/d_loss_on_encoded": train_avg_d_loss_on_encoded,
            "train/d_loss_on_cover": train_avg_d_loss_on_cover,
            "train/encodec_loss": train_avg_encodec_loss,
        }

        # if ep % save_circle == 0 or ep == 1 or ep == 2:
        if ep % save_circle == 0:
            if not model_config["structure"]["ab"]:
                path = os.path.join(train_config["path"]["ckpt"], 'pth')
            else:
                path = os.path.join(train_config["path"]["ckpt"], 'pth_ab')
            save_op(path, ep, encoder, decoder, en_de_op)
            # shutil.copyfile(os.path.realpath(__file__), os.path.join(path, os.path.basename(os.path.realpath(__file__)))) # save training scripts

        # ---------------- validation
        with torch.inference_mode():
            encoder.eval()
            decoder.eval()
            discriminator.eval()
            val_avg_acc = [0, 0]
            val_avg_snr = 0
            val_avg_wav_loss = 0
            val_avg_msg_loss = 0
            val_avg_loudness_loss = 0
            val_avg_d_loss_on_encoded = 0
            val_avg_d_loss_on_cover = 0
            val_avg_encodec_loss = 0
            count = 0
            for sample in track(val_audios_loader):
                count += 1
                b = sample["matrix"].shape[0]
                # ---------------- build watermark
                wav_matrix = sample["matrix"].to(device)
                msg = generate_random_msg(wav_matrix.size(0), msg_length, device)
                watermark, carrier_watermarked = encoder(wav_matrix, msg, global_step)
                y_wm = wav_matrix + watermark
                decoded, msg_encodec_list = decoder(y_wm, global_step)
                losses = loss.en_de_loss(wav_matrix, y_wm, msg, decoded)
                encodec_loss = encodec_loss_fn(msg_encodec_list, msg)
                # adv
                if train_config["adv"]:
                    lambda_a = lambda_m = train_config["optimize"]["lambda_a"]
                    g_target_label_encoded = torch.full((b, 1), 1, device=device).float()
                    d_on_encoded_for_enc = discriminator(y_wm)
                    g_loss_adv = F.binary_cross_entropy_with_logits(d_on_encoded_for_enc, g_target_label_encoded)
                if train_config["adv"]:
                    d_target_label_cover = torch.full((b, 1), 1, device=device).float()
                    d_on_cover = discriminator(wav_matrix)
                    d_loss_on_cover = F.binary_cross_entropy_with_logits(d_on_cover, d_target_label_cover)

                    d_target_label_encoded = torch.full((b, 1), 0, device=device).float()
                    d_on_encoded = discriminator(y_wm.detach())
                    d_loss_on_encoded = F.binary_cross_entropy_with_logits(d_on_encoded, d_target_label_encoded)
                
                decoder_acc = [((decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(),
                               ((decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item()]
                zero_tensor = torch.zeros(wav_matrix.shape).to(device)
                snr = 10 * torch.log10(mse_loss(wav_matrix.detach(), zero_tensor) / mse_loss(wav_matrix.detach(), y_wm.detach()))
                val_avg_acc[0] += decoder_acc[0]
                val_avg_acc[1] += decoder_acc[1]
                val_avg_snr += snr
                val_avg_wav_loss += losses[0].item()
                val_avg_msg_loss += losses[1].item()
                val_avg_loudness_loss += losses[2].item()
                val_avg_d_loss_on_cover += d_loss_on_cover
                val_avg_d_loss_on_encoded += d_loss_on_encoded
                val_avg_encodec_loss += encodec_loss
            val_avg_acc[0] /= count
            val_avg_acc[1] /= count
            val_avg_snr /= count
            val_avg_wav_loss /= count
            val_avg_msg_loss /= count
            val_avg_loudness_loss /= count
            val_avg_d_loss_on_encoded /= count
            val_avg_d_loss_on_cover /= count
            val_avg_encodec_loss /= count
            logging.info('#e' * 60)
            logging.info("epoch:{} - wav_loss:{:.8f} - msg_loss:{:.8f} - tfloudness_loss:{:.8f} - encodec_loss:{:.8f} - acc:[{:.8f},{:.8f}] - snr:{:.8f} - d_loss_on_encoded:{:.8f} - d_loss_on_cover:{:.8f}".format(
                ep, val_avg_wav_loss, val_avg_msg_loss, val_avg_loudness_loss, val_avg_encodec_loss, val_avg_acc[0], val_avg_acc[1], val_avg_snr, val_avg_d_loss_on_encoded.item(), val_avg_d_loss_on_cover.item()))

        val_metrics = {
            "val/wav_loss": val_avg_wav_loss,
            "val/msg_loss": val_avg_msg_loss,
            "val/tfloudness_loss": val_avg_loudness_loss,
            "val/acc_1": val_avg_acc[0],
            "val/acc_2": val_avg_acc[1],
            "val/snr": val_avg_snr,
            "val/d_loss_on_encoded": val_avg_d_loss_on_encoded,
            "val/d_loss_on_cover": val_avg_d_loss_on_cover,
            "val/encodec_loss": val_avg_encodec_loss,
        }

        if train_config["wandb"]["enabled"]:
            wandb.log({**train_metrics, **val_metrics})

    with torch.inference_mode():
        encoder.eval()
        decoder.eval()
        discriminator.eval()
        test_avg_acc = [0, 0]
        test_avg_snr = 0
        test_avg_wav_loss = 0
        test_avg_msg_loss = 0
        test_avg_loudness_loss = 0
        test_avg_d_loss_on_encoded = 0
        test_avg_d_loss_on_cover = 0
        test_avg_encodec_loss = 0
        count = 0
        for sample in track(dev_audios_loader):
            count += 1
            b = sample["matrix"].shape[0]
            # ---------------- build watermark
            wav_matrix = sample["matrix"].to(device)
            msg = generate_random_msg(wav_matrix.size(0), msg_length, device)
            watermark, carrier_watermarked = encoder(wav_matrix, msg, global_step)
            y_wm = wav_matrix + watermark
            decoded, msg_encodec_list = decoder(y_wm, global_step)
            losses = loss.en_de_loss(wav_matrix, y_wm, msg, decoded)
            encodec_loss = encodec_loss_fn(msg_encodec_list, msg)
            # adv
            if train_config["adv"]:
                lambda_a = lambda_m = train_config["optimize"]["lambda_a"]
                g_target_label_encoded = torch.full((b, 1), 1, device=device).float()
                d_on_encoded_for_enc = discriminator(y_wm)
                g_loss_adv = F.binary_cross_entropy_with_logits(d_on_encoded_for_enc, g_target_label_encoded)
            if train_config["adv"]:
                d_target_label_cover = torch.full((b, 1), 1, device=device).float()
                d_on_cover = discriminator(wav_matrix)
                d_loss_on_cover = F.binary_cross_entropy_with_logits(d_on_cover, d_target_label_cover)

                d_target_label_encoded = torch.full((b, 1), 0, device=device).float()
                d_on_encoded = discriminator(y_wm.detach())
                d_loss_on_encoded = F.binary_cross_entropy_with_logits(d_on_encoded, d_target_label_encoded)

            decoder_acc = [((decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(),
                           ((decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item()]
            zero_tensor = torch.zeros(wav_matrix.shape).to(device)
            snr = 10 * torch.log10(
                mse_loss(wav_matrix.detach(), zero_tensor) / mse_loss(wav_matrix.detach(), y_wm.detach()))
            test_avg_acc[0] += decoder_acc[0]
            test_avg_acc[1] += decoder_acc[1]
            test_avg_snr += snr
            test_avg_wav_loss += losses[0]
            test_avg_msg_loss += losses[1]
            test_avg_loudness_loss += losses[2]
            test_avg_d_loss_on_cover += d_loss_on_cover
            test_avg_d_loss_on_encoded += d_loss_on_encoded
            test_avg_encodec_loss += encodec_loss
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

        test_avg_acc[0] /= count
        test_avg_acc[1] /= count
        test_avg_snr /= count
        test_avg_wav_loss /= count
        test_avg_msg_loss /= count
        test_avg_loudness_loss /= count
        test_avg_d_loss_on_encoded /= count
        test_avg_d_loss_on_cover /= count
        test_avg_encodec_loss /= count
        test_loss = test_avg_acc[0] + test_avg_acc[1] + test_avg_snr + test_avg_wav_loss + test_avg_msg_loss + test_avg_d_loss_on_encoded + test_avg_d_loss_on_cover
        logging.info('#test' * 20)
        logging.info("Test: wav_loss:{:.8f} - msg_loss:{:.8f} - tfloudness_loss:{:.8f} - encodec_loss:{:.8f} - acc:[{:.8f},{:.8f}] - snr:{:.8f} - d_loss_on_encoded:{:.8f} - d_loss_on_cover:{:.8f}".format(
            test_avg_wav_loss, test_avg_msg_loss, test_avg_loudness_loss, test_avg_encodec_loss, test_avg_acc[0], test_avg_acc[1], test_avg_snr, test_avg_d_loss_on_encoded.item(), test_avg_d_loss_on_cover.item()))

        # if train_config["wandb"]["enabled"]:
        #     # Log the audio_table to wandb
        #     wandb.log({"test_audio_table": test_audio_table})

        if train_config["wandb"]["enabled"] and test_loss_summary_table is not None:
            test_loss_summary_table.add_data(
                test_loss,
                test_avg_wav_loss,
                test_avg_msg_loss,
                test_avg_loudness_loss,
                test_avg_acc[0],
                test_avg_acc[1],
                test_avg_snr,
                test_avg_d_loss_on_encoded,
                test_avg_d_loss_on_cover,
                test_avg_encodec_loss,
            )

            wandb.log(
                {
                    "test_loss_summary_table": test_loss_summary_table,
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
        "-m", "--model_config", type=str, help="path to model.yaml", default=r"./config/model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, help="path to train.yaml", default=r"./config/train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    process_config = yaml.load(open(args.process_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (process_config, model_config, train_config)

    main(configs)
