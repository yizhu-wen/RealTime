# from encodec import EncodecModel
import inspect
import io
import logging
import random
import re
import typing as tp
from functools import partial

import julius
import omegaconf
import torch
import torchaudio
from distortions.frequency import TacotronSTFT
from julius import fft_conv1d, resample_frac
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def compress(
    wav: torch.Tensor,
    sr: int,
    target_format: tp.Literal["mp3", "ogg", "flac"] = "mp3",
    bitrate: str = "128k",
) -> tp.Tuple[torch.Tensor, int]:
    """Convert audio wave form to a specified lossy format: mp3, ogg, flac

    Args:
        wav (torch.Tensor): Input wav tensor.
        sr (int): Sampling rate.
        target_format (str): Compression format (e.g., 'mp3').
        bitrate (str): Bitrate for compression.

    Returns:
        Tuple of compressed WAV tensor and sampling rate.
    """

    # Extract the bit rate from string (e.g., '128k')
    match = re.search(r"\d+(\.\d+)?", str(bitrate))
    parsed_bitrate = float(match.group()) if match else None
    assert parsed_bitrate, f"Invalid bitrate specified (got {parsed_bitrate})"
    try:
        # Create a virtual file instead of saving to disk
        buffer = io.BytesIO()

        torchaudio.save(
            buffer,
            wav,
            sr,
            format=target_format,
            bits_per_sample=parsed_bitrate,
        )
        # Move to the beginning of the file
        buffer.seek(0)
        compressed_wav, sr = torchaudio.load(buffer)
        return compressed_wav, sr

    except RuntimeError:
        logger.warning(
            f"compression failed skipping compression: {format} {parsed_bitrate}"
        )
        return wav, sr


def get_mp3(wav_tensor: torch.Tensor, sr: int, bitrate: str = "128k") -> torch.Tensor:
    """Convert a batch of audio files to MP3 format, maintaining the original shape.

    This function takes a batch of audio files represented as a PyTorch tensor, converts
    them to MP3 format using the specified bitrate, and returns the batch in the same
    shape as the input.

    Args:
        wav_tensor (torch.Tensor): Batch of audio files represented as a tensor.
            Shape should be (batch_size, channels, length).
        sr (int): Sampling rate of the audio.
        bitrate (str): Bitrate for MP3 conversion, default is '128k'.

    Returns:
        torch.Tensor: Batch of audio files converted to MP3 format, with the same
            shape as the input tensor.
    """
    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape

    # Flatten tensor for conversion and move to CPU
    wav_tensor_flat = wav_tensor.view(1, -1).cpu()

    # Convert to MP3 format with specified bitrate
    wav_tensor_flat, _ = compress(wav_tensor_flat, sr, bitrate=bitrate)

    # Reshape back to original batch format and trim or pad if necessary
    wav_tensor = wav_tensor_flat.view(batch_size, channels, -1)
    compressed_length = wav_tensor.shape[-1]
    if compressed_length > original_length:
        wav_tensor = wav_tensor[:, :, :original_length]  # Trim excess frames
    elif compressed_length < original_length:
        padding = torch.zeros(
            batch_size, channels, original_length - compressed_length, device=device
        )
        wav_tensor = torch.cat((wav_tensor, padding), dim=-1)  # Pad with zeros

    # Move tensor back to the original device
    return wav_tensor.to(device)


def get_aac(
    wav_tensor: torch.Tensor,
    sr: int,
    bitrate: str = "128k",
    lowpass_freq: tp.Optional[int] = None,
) -> torch.Tensor:
    """Converts a batch of audio tensors to AAC format and then back to tensors.

    This function first saves the input tensor batch as WAV files, then uses FFmpeg to convert
    these WAV files to AAC format. Finally, it loads the AAC files back into tensors.

    Args:
        wav_tensor (torch.Tensor): A batch of audio files represented as a tensor.
                                   Shape should be (batch_size, channels, length).
        sr (int): Sampling rate of the audio.
        bitrate (str): Bitrate for AAC conversion, default is '128k'.
        lowpass_freq (Optional[int]): Frequency for a low-pass filter. If None, no filter is applied.

    Returns:
        torch.Tensor: Batch of audio files converted to AAC and back, with the same
                      shape as the input tensor.
    """
    import tempfile
    import subprocess

    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape

    # Parse the bitrate value from the string
    match = re.search(r"\d+(\.\d+)?", bitrate)
    parsed_bitrate = (
        match.group() if match else "128"
    )  # Default to 128 if parsing fails

    # Flatten tensor for conversion and move to CPU
    wav_tensor_flat = wav_tensor.view(1, -1).cpu()

    with tempfile.NamedTemporaryFile(
        suffix=".wav"
    ) as f_in, tempfile.NamedTemporaryFile(suffix=".aac") as f_out:
        input_path, output_path = f_in.name, f_out.name

        # Save the tensor as a WAV file
        torchaudio.save(input_path, wav_tensor_flat, sr, backend="ffmpeg")

        # Prepare FFmpeg command for AAC conversion
        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ar",
            str(sr),
            "-b:a",
            f"{parsed_bitrate}k",
            "-c:a",
            "aac",
        ]
        if lowpass_freq is not None:
            command += ["-cutoff", str(lowpass_freq)]
        command.append(output_path)

        try:
            # Run FFmpeg and suppress output
            subprocess.run(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            # Load the AAC audio back into a tensor
            aac_tensor, _ = torchaudio.load(output_path, backend="ffmpeg")
        except Exception as exc:
            raise RuntimeError(
                "Failed to run command "
                ".join(command)} "
                "(Often this means ffmpeg is not installed or the encoder is not supported, "
                "make sure you installed an older version ffmpeg<5)"
            ) from exc

    original_length_flat = batch_size * channels * original_length
    compressed_length_flat = aac_tensor.shape[-1]

    # Trim excess frames
    if compressed_length_flat > original_length_flat:
        aac_tensor = aac_tensor[:, :original_length_flat]

    # Pad the shortedn frames
    elif compressed_length_flat < original_length_flat:
        padding = torch.zeros(
            1, original_length_flat - compressed_length_flat, device=device
        )
        aac_tensor = torch.cat((aac_tensor, padding), dim=-1)

    # Reshape and adjust length to match original tensor
    wav_tensor = aac_tensor.view(batch_size, channels, -1)
    compressed_length = wav_tensor.shape[-1]

    assert compressed_length == original_length, (
        "AAC-compressed audio does not have the same frames as original one. "
        "One reason can be ffmpeg is not  installed and used as proper backed "
        "for torchaudio, or the AAC encoder is not correct. Run "
        "`torchaudio.utils.ffmpeg_utils.get_audio_encoders()` and make sure we see entry for"
        "AAC in the output."
    )
    return wav_tensor.to(device)


def select_audio_effects(
    audio_effects: tp.Dict,
    weights: tp.Optional[tp.Dict] = None,
    mode: str = "all",
    max_length: tp.Optional[int] = None,
):
    """Samples a subset of audio effects methods from the `AudioEffects` class.

    This function allows you to select a subset of audio effects
    based on the chosen selection mode and optional weights.

    Args:
        audio_effects (dict): A dictionary of available audio augmentations, usually
            obtained from the output of the 'get_audio_effects' function.
        weights (dict): A dictionary mapping augmentation names to their corresponding
            probabilities of being selected. This argument is used when 'mode' is set
            to "weighted." If 'weights' is None, all augmentations have equal
            probability of being selected.
        mode (str): The selection mode, which can be one of the following:
            - "all": Select all available augmentations.
            - "weighted": Select augmentations based on their probabilities in the
              'weights' dictionary.
        max_length (int): The maximum number of augmentations to select. If 'max_length'
            is None, no limit is applied.

    Returns:
        dict: A subset of the 'audio_effects' dictionary containing the selected audio
        augmentations.

    Note:
        - In "all" mode, all available augmentations are selected.
        - In "weighted" mode, augmentations are selected with a probability
          proportional to their weights specified in the 'weights' dictionary.
        - If 'max_length' is set, the function limits the number of selected
          augmentations.
        - If no augmentations are selected or 'audio_effects' is empty, the function
          defaults to including an "identity" augmentation.
        - The "identity" augmentation means that no audio effect is applied.
    """
    if mode == "all":  # original code
        out = audio_effects
    elif mode == "weighted":
        # Probability proportionnal to weights
        assert weights is not None
        out = {
            name: value
            for name, value in audio_effects.items()
            if random.random() < weights.get(name, 1.0)
        }
    else:
        raise ValueError(f"Unknown mode {mode}")
    if max_length is not None:
        # Help having a deterministic limit of the gpu memory usage
        random_keys = random.sample(list(out.keys()), max_length)
        out = {key: out[key] for key in random_keys}
    if len(out) == 0:  # Check not to return empty dict
        out = {"identity": AudioEffects.identity}
    return out


def get_audio_effects(cfg: omegaconf.DictConfig):
    """Automatically pull the list all effects available in this class based on the parameters from the cfg

    Returns:
        dict: A dict of names and pointers to all methods in this class.
    """
    assert hasattr(cfg, "audio_effects")
    cfg_audio_effects = dict(cfg["audio_effects"])
    return {
        name: partial(value, **cfg_audio_effects.get(name, {}))
        for name, value in inspect.getmembers(AudioEffects)
        if inspect.isfunction(value)
    }


def audio_effect_return(
    tensor: torch.Tensor, mask: tp.Optional[torch.Tensor]
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the mask if it was in the input otherwise only the output tensor"""
    if mask is None:
        return tensor
    else:
        return tensor, mask


def generate_pink_noise(length: int) -> torch.Tensor:
    """Generate pink noise using Voss-McCartney algorithm with PyTorch."""
    num_rows = 16
    array = torch.randn(num_rows, length // num_rows + 1)
    reshaped_array = torch.cumsum(array, dim=1)
    reshaped_array = reshaped_array.reshape(-1)
    reshaped_array = reshaped_array[:length]
    # Normalize
    pink_noise = reshaped_array / torch.max(torch.abs(reshaped_array))
    return pink_noise


# def compress_with_encodec(
#     tensor: torch.Tensor,
#     n_q: int,
#     model,
#     sample_rate: int,
#     mask: tp.Optional[torch.Tensor] = None,
# ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
#     """Special augmentation function that compresses and decompresses wav tensor
#     using a compression model with the n_q codebooks
#     """
#
#     model.to(tensor.device)
#     model.set_num_codebooks(n_q)
#     codes, scale = model.encode(
#         julius.resample_frac(tensor, old_sr=sample_rate, new_sr=model.sample_rate)
#     )
#     compressed = model.decode(codes=codes, scale=scale)
#     return audio_effect_return(
#         tensor=julius.resample_frac(
#             compressed, old_sr=model.sample_rate, new_sr=sample_rate
#         ),
#         mask=mask,
#     )


def compress_with_encodec(
    tensor: torch.Tensor,
    n_bandwidth: int,
    model,
    sample_rate: int,
    mask: tp.Optional[torch.Tensor] = None,
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Special augmentation function that compresses and decompresses wav tensor
    using a compression model with the n_q codebooks
    """

    model.to(tensor.device)
    model.set_target_bandwidth(n_bandwidth)
    compressed = model(
        julius.resample_frac(tensor, old_sr=sample_rate, new_sr=model.sample_rate)
    )
    return audio_effect_return(compressed, mask=None)


def apply_compression_skip_grad(tensor: torch.Tensor, compression_fn, **kwargs):
    """Applies a specified compression function to the audio tensor.
    Whire carrying over the grads to the output tensor with skip through estimator
    this is a straight through estimator to make mp3/aac compression differentiable
    see more: Yin et al. 2019 https://arxiv.org/pdf/1903.05662.pdf

    Args:
        tensor (torch.Tensor): The input audio tensor.
        compression_fn (function): The compression function to apply.
        **kwargs: Additional keyword arguments for the compression function.

    Returns:
        torch.Tensor: The output tensor after applying compression and straight through estimator.
    """
    compressed = compression_fn(tensor.detach(), **kwargs)

    # Trim compressed output if needed
    compressed = compressed[:, :, : tensor.size(-1)]

    # Straight through estimator for differentiable compression
    out = tensor + (compressed - tensor).detach()

    # Check that gradients are not broken
    if out.requires_grad:
        assert (
            out.grad_fn
        ), "The computation graph might be broken due to compression augmentation."

    return out


class AudioEffects:
    @staticmethod
    def speed(
        tensor: torch.Tensor,
        speed_range: tuple = (0.8, 1.2),
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Function to change the speed of a batch of audio data.
        The output will have a different length !

        Args:
            audio_batch (torch.Tensor): The batch of audio data in torch tensor format.
            speed (float): The speed to change the audio to.

        Returns:
            torch.Tensor: The batch of audio data with the speed changed.
        """
        speed = torch.FloatTensor(1).uniform_(*speed_range)
        new_sr = int(sample_rate * 1 / speed)
        resampled_tensor = julius.resample.resample_frac(tensor, sample_rate, new_sr)
        if mask is None:
            return resampled_tensor
        else:
            return resampled_tensor, torch.nn.functional.interpolate(
                mask, size=resampled_tensor.size(-1), mode="nearest-exact"
            )

    @staticmethod
    def updownresample(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        intermediate_freq: int = 32000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        orig_shape = tensor.shape
        # upsample
        tensor = resample_frac(tensor, sample_rate, intermediate_freq)
        # downsample
        tensor = resample_frac(tensor, intermediate_freq, sample_rate)

        assert tensor.shape == orig_shape
        return audio_effect_return(tensor=tensor, mask=mask)

    @staticmethod
    def echo(
        tensor: torch.Tensor,
        volume_range: tuple = (0.1, 0.5),
        duration_range: tuple = (0.1, 0.5),
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Attenuating the audio volume by a factor of 0.4, delaying it by 100ms,
        and then overlaying it with the original.

        Args:
            tensor: 3D Tensor representing the audio signal [bsz, channels, frames]
            volumne range: volume range of the echo signal
            duration range: duration range of the echo signal
            sample_rate: Sample rate of the audio signal.
        Returns:
            Audio signal with reverb.
        """

        # Create a simple impulse response
        # Duration of the impulse response in seconds
        duration = torch.FloatTensor(1).uniform_(*duration_range)
        volume = torch.FloatTensor(1).uniform_(*volume_range)

        n_samples = int(sample_rate * duration)
        impulse_response = torch.zeros(n_samples).type(tensor.type()).to(tensor.device)

        # Define a few reflections with decreasing amplitude
        impulse_response[0] = 1.0  # Direct sound

        impulse_response[int(sample_rate * duration) - 1] = (
            volume  # First reflection after 100ms
        )

        # Add batch and channel dimensions to the impulse response
        impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)

        # Convolve the audio signal with the impulse response
        reverbed_signal = fft_conv1d(tensor, impulse_response)

        # Normalize to the original amplitude range for stability
        reverbed_signal = (
            reverbed_signal
            / torch.max(torch.abs(reverbed_signal))
            * torch.max(torch.abs(tensor))
        )

        # Ensure tensor size is not changed
        tmp = torch.zeros_like(tensor)
        tmp[..., : reverbed_signal.shape[-1]] = reverbed_signal
        reverbed_signal = tmp

        return audio_effect_return(tensor=reverbed_signal, mask=mask)

    @staticmethod
    def random_noise(
        waveform: torch.Tensor,
        noise_std: float = 0.001,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Add Gaussian noise to the waveform."""
        noise = torch.randn_like(waveform) * noise_std
        noisy_waveform = waveform + noise
        return audio_effect_return(tensor=noisy_waveform, mask=mask)

    @staticmethod
    def pink_noise(
        waveform: torch.Tensor,
        noise_std: float = 0.01,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Add pink background noise to the waveform."""
        noise = generate_pink_noise(waveform.shape[-1]) * noise_std
        noise = noise.to(waveform.device)
        # Assuming waveform is of shape (bsz, channels, length)
        noisy_waveform = waveform + noise.unsqueeze(0).unsqueeze(0).to(waveform.device)
        return audio_effect_return(tensor=noisy_waveform, mask=mask)

    @staticmethod
    def lowpass_filter(
        waveform: torch.Tensor,
        cutoff_freq: float = 5000,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Filter the lowpass frequency from the waveform"""
        return audio_effect_return(
            tensor=julius.lowpass_filter(waveform, cutoff=cutoff_freq / sample_rate),
            mask=mask,
        )

    @staticmethod
    def highpass_filter(
        waveform: torch.Tensor,
        cutoff_freq: float = 500,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Filter the highpass frequency from the waveform"""
        return audio_effect_return(
            tensor=julius.highpass_filter(waveform, cutoff=cutoff_freq / sample_rate),
            mask=mask,
        )

    @staticmethod
    def bandpass_filter(
        waveform: torch.Tensor,
        cutoff_freq_low: float = 300,
        cutoff_freq_high: float = 8000,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Apply a bandpass filter to the waveform by cascading
        a high-pass filter followed by a low-pass filter.

        Args:
            waveform (torch.Tensor): Input audio waveform.
            low_cutoff (float): Lower cutoff frequency.
            high_cutoff (float): Higher cutoff frequency.
            sample_rate (int): The sample rate of the waveform.

        Returns:
            torch.Tensor: Filtered audio waveform.
        """

        return audio_effect_return(
            tensor=julius.bandpass_filter(
                waveform,
                cutoff_low=cutoff_freq_low / sample_rate,
                cutoff_high=cutoff_freq_high / sample_rate,
            ),
            mask=mask,
        )

    @staticmethod
    def smooth(
        tensor: torch.Tensor,
        window_size_range: tuple = (2, 10),
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Smooths the input tensor (audio signal) using a moving average filter with the
        given window size.

        Args:
            tensor (torch.Tensor): Input audio tensor. Assumes tensor shape is (batch_size,
            channels, time).
            window_size (int): Size of the moving average window.
            mask: Masks for the input wave

        Returns:
            torch.Tensor: Smoothed audio tensor.
        """

        window_size = int(torch.FloatTensor(1).uniform_(*window_size_range))
        # Create a uniform smoothing kernel
        kernel = torch.ones(1, 1, window_size).type(tensor.type()) / window_size
        kernel = kernel.to(tensor.device)

        smoothed = fft_conv1d(tensor, kernel)
        # Ensure tensor size is not changed
        tmp = torch.zeros_like(tensor)
        tmp[..., : smoothed.shape[-1]] = smoothed
        smoothed = tmp

        return audio_effect_return(tensor=smoothed, mask=mask)

    @staticmethod
    def boost_audio(
        tensor: torch.Tensor,
        amount: float = 20,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Filter the lowpass frequency from the waveform"""
        return audio_effect_return(tensor=tensor * (1 + amount / 100), mask=mask)

    @staticmethod
    def duck_audio(
        tensor: torch.Tensor,
        amount: float = 20,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Mask input wav with some ducked signnals"""
        return audio_effect_return(tensor=tensor * (1 - amount / 100), mask=mask)

    @staticmethod
    def identity(
        tensor: torch.Tensor, mask: tp.Optional[torch.Tensor] = None
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return audio_effect_return(tensor=tensor, mask=mask)

    @staticmethod
    def mp3_compression(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        bitrate: str = "128k",
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Compress audio using MP3 algorithm
        Args:
            tensor (torch.Tensor): The input audio tensor.
            sample_rate (int): The sample rate of the audio.
            bitrate (str): The bitrate for MP3 compression.

        Returns:
            torch.Tensor: The output tensor after applying MP3 compression.
        """
        out = apply_compression_skip_grad(
            tensor, get_mp3, sr=sample_rate, bitrate=bitrate
        )
        return audio_effect_return(tensor=out, mask=mask)

    @staticmethod
    def aac_compression(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        bitrate: str = "128k",
        lowpass_freq: tp.Optional[int] = None,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Applies AAC compression to an audio tensor.

        Args:
            tensor (torch.Tensor): The input audio tensor.
            sample_rate (int): The sample rate of the audio.
            bitrate (str): The bitrate for AAC compression.
            lowpass_freq (Optional[int]): The frequency for a low-pass filter.

        Returns:
            torch.Tensor: The output tensor after applying AAC compression.
        """
        out = apply_compression_skip_grad(
            tensor, get_aac, sr=sample_rate, bitrate=bitrate, lowpass_freq=lowpass_freq
        )
        return audio_effect_return(tensor=out, mask=mask)

    @staticmethod
    def mel_griffin_lim(y, mask: tp.Optional[torch.Tensor] = None):
        mel_transform = TacotronSTFT(
            filter_length=1024, hop_length=256, win_length=1024
        )
        y_mel = mel_transform.mel_spectrogram(y.squeeze(1))
        y_d = (mel_transform.griffin_lim(magnitudes=y_mel)).unsqueeze(1)
        return audio_effect_return(tensor=y_d, mask=mask)


def encodec_transform(
    encodec,
    y: torch.Tensor,
    n_bandwidth: int,
    sample_rate: int,
    mask: tp.Optional[torch.Tensor] = None,
):
    """
    Apply an encodec-based compression augmentation.

    Args:
        encodec: An instance of an Encodec model.
        y (torch.Tensor): Input audio tensor, shape (B, T) or (B, 1, T).
        n_bandwidth (int): Target bandwidth (or quantization level) to set.
        sample_rate (int): Original sampling rate of y.
        mask (Optional[torch.Tensor]): Optional mask to be passed through.

    Returns:
        The processed audio tensor (and mask if provided) after compression/decompression.
    """
    # Ensure input is in shape (B, 1, T)
    if y.dim() == 2:  # (B, T)
        y = y.unsqueeze(1)  # -> (B, 1, T)
    elif y.dim() != 3:
        raise ValueError(
            f"Expected input of shape (B, T) or (B, 1, T), but got {y.shape}"
        )

    # Set the target bandwidth on the model.
    encodec.set_target_bandwidth(n_bandwidth)

    # Resample from the original sample_rate to the model's expected sample rate.
    # (Here, julius.resample_frac is assumed to be a function you have that resamples audio.)
    y_resampled = julius.resample_frac(
        y, old_sr=sample_rate, new_sr=encodec.sample_rate
    )

    # Run the model: this performs the compression (encoding) and decompression (decoding).
    compressed = encodec(y_resampled)

    # Resample the output back to the original sample_rate.
    y_d = julius.resample_frac(
        compressed, old_sr=encodec.sample_rate, new_sr=sample_rate
    )

    return audio_effect_return(tensor=y_d, mask=mask)


def get_my_encodec_audio_effect(encodec_cfg: DictConfig, sr: int) -> tp.Dict:
    """
    Construct custom encodec-based compression augmentation using your own transformation.

    Args:
        encodec_cfg (DictConfig): A configuration that should contain:
            - ckpt: The model checkpoint identifier for EncodecModel.
            - n_qs (or bandwidth values): A list of target bandwidth values (e.g., [4, 8, 16]).
        sr (int): The original sample rate of the input audio.

    Returns:
        A dict mapping a string key (e.g., "encodec_bw=4") to a partial function that calls
        encodec_transform with the model and a fixed target bandwidth.
    """
    # Import your Encodec model from your custom package.
    from encodec import (
        EncodecModel,
    )  # ensure that you have your custom import configured

    # Load the model using the provided checkpoint.
    model = EncodecModel.encodec_model_24khz()

    # Build a dictionary where, for every target bandwidth value in encodec_cfg.n_qs,
    # we create a partial function with the corresponding bandwidth.

    return {
        f"encodec_bw={bw}": partial(
            encodec_transform,
            model,  # pass the loaded model
            n_bandwidth=bw,  # set the target bandwidth for this augmentation
            sample_rate=sr,  # the original sample rate to use for resampling
        )
        for bw in encodec_cfg["n_bandwidths"]
    }
