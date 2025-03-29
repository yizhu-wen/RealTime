#################################################################### tacotron2's mel-frequency-spectrum ####################################################################
import numpy as np
import librosa
import scipy.io.wavfile
import scipy.signal


def _log(x, base):
    if base == 10:
        return np.log10(x)
    return np.log(x)

def _exp(x, base):
    if base == 10:
        return np.power(10, x)
    return np.exp(x)

class tacotron_mel():
    def __init__(self):
        self.preemphasis=0.0
        self.do_amp_to_db_mel=True
        self.fft_size = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.stft_pad_mode = "reflect"
        self.spec_gain=20
        log_func="np.log"
        if log_func == "np.log":
            self.base = np.e
        elif log_func == "np.log10":
            self.base = 10
        self.mel_basis = self._build_mel_basis()
        self.inv_mel_basis = np.linalg.pinv(self._build_mel_basis())
        self.mel_fmax=8000
        self.sample_rate=22050
        self.num_mels = 80
        self.mel_fmin=0.0
        self.signal_norm=True
        self.ref_level_db=20
        self.min_level_db = 0
        self.symmetric_norm = True
        self.max_norm = 1.0
        self.clip_norm = True
        self.power = 1.5
        self.griffin_lim_iters = 60
        clip_norm: bool = True
        stats_path: str = None


    def _build_mel_basis(
        self,
    ) -> np.ndarray:
        """Build melspectrogram basis.

        Returns:
            np.ndarray: melspectrogram basis.
        """
        if self.mel_fmax is not None:
            assert self.mel_fmax <= self.sample_rate // 2
        return librosa.filters.mel(
            self.sample_rate, self.fft_size, n_mels=self.num_mels, fmin=self.mel_fmin, fmax=self.mel_fmax
        )

    def _stft(self, y: np.ndarray) -> np.ndarray:
        """Librosa STFT wrapper.

        Args:
            y (np.ndarray): Audio signal.

        Returns:
            np.ndarray: Complex number array.
        """
        return librosa.stft(
            y=y,
            n_fft=self.fft_size,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode=self.stft_pad_mode,
            window="hann",
            center=True,
        )

    def apply_preemphasis(self, x: np.ndarray) -> np.ndarray:
        """Apply pre-emphasis to the audio signal. Useful to reduce the correlation between neighbouring signal values.

        Args:
            x (np.ndarray): Audio signal.

        Raises:
            RuntimeError: Preemphasis coeff is set to 0.

        Returns:
            np.ndarray: Decorrelated audio signal.
        """
        if self.preemphasis == 0:
            raise RuntimeError(" [!] Preemphasis is set 0.0.")
        return scipy.signal.lfilter([1, -self.preemphasis], [1], x)

    def _amp_to_db(self, x: np.ndarray) -> np.ndarray:
        """Convert amplitude values to decibels.

        Args:
            x (np.ndarray): Amplitude spectrogram.

        Returns:
            np.ndarray: Decibels spectrogram.
        """
        return self.spec_gain * _log(np.maximum(1e-5, x), self.base)

    def _linear_to_mel(self, spectrogram: np.ndarray) -> np.ndarray:
        """Project a full scale spectrogram to a melspectrogram.

        Args:
            spectrogram (np.ndarray): Full scale spectrogram.

        Returns:
            np.ndarray: Melspectrogram
        """
        return np.dot(self.mel_basis, spectrogram)

    def normalize(self, S: np.ndarray) -> np.ndarray:
        """Normalize values into `[0, self.max_norm]` or `[-self.max_norm, self.max_norm]`

        Args:
            S (np.ndarray): Spectrogram to normalize.

        Raises:
            RuntimeError: Mean and variance is computed from incompatible parameters.

        Returns:
            np.ndarray: Normalized spectrogram.
        """
        # pylint: disable=no-else-return
        S = S.copy()
        if self.signal_norm:
            # mean-var scaling
            if hasattr(self, "mel_scaler"):
                if S.shape[0] == self.num_mels:
                    return self.mel_scaler.transform(S.T).T
                elif S.shape[0] == self.fft_size / 2:
                    return self.linear_scaler.transform(S.T).T
                else:
                    raise RuntimeError(" [!] Mean-Var stats does not match the given feature dimensions.")
            # range normalization
            S -= self.ref_level_db  # discard certain range of DB assuming it is air noise
            S_norm = (S - self.min_level_db) / (-self.min_level_db)
            if self.symmetric_norm:
                S_norm = ((2 * self.max_norm) * S_norm) - self.max_norm
                if self.clip_norm:
                    S_norm = np.clip(
                        S_norm, -self.max_norm, self.max_norm  # pylint: disable=invalid-unary-operand-type
                    )
                return S_norm
            else:
                S_norm = self.max_norm * S_norm
                if self.clip_norm:
                    S_norm = np.clip(S_norm, 0, self.max_norm)
                return S_norm
        else:
            return S

    def melspectrogram(self, y: np.ndarray) -> np.ndarray:
        """Compute a melspectrogram from a waveform."""
        if self.preemphasis != 0:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        if self.do_amp_to_db_mel:
            S = self._amp_to_db(self._linear_to_mel(np.abs(D)))
        else:
            S = self._linear_to_mel(np.abs(D))
        return self.normalize(S).astype(np.float32)

    
    def denormalize(self, S: np.ndarray) -> np.ndarray:
        """Denormalize spectrogram values.

        Args:
            S (np.ndarray): Spectrogram to denormalize.

        Raises:
            RuntimeError: Mean and variance are incompatible.

        Returns:
            np.ndarray: Denormalized spectrogram.
        """
        # pylint: disable=no-else-return
        S_denorm = S.copy()
        if self.signal_norm:
            # mean-var scaling
            if hasattr(self, "mel_scaler"):
                if S_denorm.shape[0] == self.num_mels:
                    return self.mel_scaler.inverse_transform(S_denorm.T).T
                elif S_denorm.shape[0] == self.fft_size / 2:
                    return self.linear_scaler.inverse_transform(S_denorm.T).T
                else:
                    raise RuntimeError(" [!] Mean-Var stats does not match the given feature dimensions.")
            if self.symmetric_norm:
                if self.clip_norm:
                    S_denorm = np.clip(
                        S_denorm, -self.max_norm, self.max_norm  # pylint: disable=invalid-unary-operand-type
                    )
                S_denorm = ((S_denorm + self.max_norm) * -self.min_level_db / (2 * self.max_norm)) + self.min_level_db
                return S_denorm + self.ref_level_db
            else:
                if self.clip_norm:
                    S_denorm = np.clip(S_denorm, 0, self.max_norm)
                S_denorm = (S_denorm * -self.min_level_db / self.max_norm) + self.min_level_db
                return S_denorm + self.ref_level_db
        else:
            return S_denorm 
    

    def _db_to_amp(self, x: np.ndarray) -> np.ndarray:
        """Convert decibels spectrogram to amplitude spectrogram.

        Args:
            x (np.ndarray): Decibels spectrogram.

        Returns:
            np.ndarray: Amplitude spectrogram.
        """
        return _exp(x / self.spec_gain, self.base)

    def apply_inv_preemphasis(self, x: np.ndarray) -> np.ndarray:
        """Reverse pre-emphasis."""
        if self.preemphasis == 0:
            raise RuntimeError(" [!] Preemphasis is set 0.0.")
        return scipy.signal.lfilter([1], [1, -self.preemphasis], x)

    def _griffin_lim(self, S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        if not np.isfinite(y).all():
            print(" [!] Waveform is not finite everywhere. Skipping the GL.")
            return np.array([0.0])
        for _ in range(self.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def inv_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        # import pdb
        # pdb.set_trace()
        """Convert a spectrogram to a waveform using Griffi-Lim vocoder."""
        S = self.denormalize(spectrogram)
        S = self._db_to_amp(S)
        # Reconstruct phase
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        return self._griffin_lim(S**self.power)





#################################################################### tacotron2's mel-frequency-spectrum ####################################################################
import torch
from scipy.signal import get_window
from librosa.util import pad_center, tiny
import torch.nn.functional as F
from torch.autograd import Variable
import librosa.util as librosa_util
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C

def window_sumsquare(window, n_frames, hop_length=160, win_length=320,
                     n_fft=320, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, size=n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x

# class STFT(torch.nn.Module):
#     """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
#     def __init__(self, filter_length=800, hop_length=200, win_length=800,
#                  window='hann'):
#         super(STFT, self).__init__()
#         self.filter_length = filter_length
#         self.hop_length = hop_length
#         self.win_length = win_length
#         self.window = window
#         self.forward_transform = None
#         scale = self.filter_length / self.hop_length
#         fourier_basis = np.fft.fft(np.eye(self.filter_length))
#
#         cutoff = int((self.filter_length / 2 + 1))
#         fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
#                                    np.imag(fourier_basis[:cutoff, :])])
#
#         forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
#         inverse_basis = torch.FloatTensor(
#             np.linalg.pinv(scale * fourier_basis).T[:, None, :])
#
#         if window is not None:
#             assert(filter_length >= win_length)
#             # get window and zero center pad it to filter_length
#             fft_window = get_window(window, win_length, fftbins=True)
#             fft_window = pad_center(fft_window, size=filter_length)
#             fft_window = torch.from_numpy(fft_window).float()
#
#             # window the bases
#             forward_basis *= fft_window
#             inverse_basis *= fft_window
#
#         self.register_buffer('forward_basis', forward_basis.float())
#         self.register_buffer('inverse_basis', inverse_basis.float())
#
#     def transform(self, input_data):
#         num_batches = input_data.size(0)
#         num_samples = input_data.size(1)
#
#         self.num_samples = num_samples
#
#         # similar to librosa, reflect-pad the input
#         input_data = input_data.view(num_batches, 1, num_samples)
#         input_data = F.pad(
#             input_data.unsqueeze(1),
#             (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
#             mode='reflect')
#         input_data = input_data.squeeze(1)
#
#         forward_transform = F.conv1d(
#             input_data,
#             Variable(self.forward_basis, requires_grad=False),
#             stride=self.hop_length,
#             padding=0)
#
#         cutoff = int((self.filter_length / 2) + 1)
#         real_part = forward_transform[:, :cutoff, :]
#         imag_part = forward_transform[:, cutoff:, :]
#
#         magnitude = torch.sqrt(real_part**2 + imag_part**2)
#         phase = torch.autograd.Variable(
#             torch.atan2(imag_part.data, real_part.data))
#
#         return magnitude, phase
#
#     def inverse(self, magnitude, phase):
#         recombine_magnitude_phase = torch.cat(
#             [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)
#
#         inverse_transform = F.conv_transpose1d(
#             recombine_magnitude_phase,
#             Variable(self.inverse_basis, requires_grad=False),
#             stride=self.hop_length,
#             padding=0)
#
#         if self.window is not None:
#             window_sum = window_sumsquare(
#                 self.window, magnitude.size(-1), hop_length=self.hop_length,
#                 win_length=self.win_length, n_fft=self.filter_length,
#                 dtype=np.float32)
#             # remove modulation effects
#             approx_nonzero_indices = torch.from_numpy(
#                 np.where(window_sum > tiny(window_sum))[0])
#             window_sum = torch.autograd.Variable(
#                 torch.from_numpy(window_sum), requires_grad=False)
#             window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
#             inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
#
#             # scale by hop ratio
#             inverse_transform *= float(self.filter_length) / self.hop_length
#
#         inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
#         inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]
#
#         return inverse_transform
#
#     def forward(self, input_data):
#         self.magnitude, self.phase = self.transform(input_data)
#         reconstruction = self.inverse(self.magnitude, self.phase)
#         return reconstruction

class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        # assert(torch.min(y.data) >= -1)
        # assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        # magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output



class fixed_STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=320, hop_length=160, win_length=320,
                 window='hann'):
        super(fixed_STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        # num_batches = input_data.size(0)
        # num_samples = input_data.size(1)

        # self.num_samples = num_samples

        # # similar to librosa, reflect-pad the input
        # input_data = input_data.view(num_batches, 1, num_samples)

        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        # Create stft_result with shape [b, 2, fre_bins, frame]
        stft_result = torch.stack([real_part, imag_part], dim=1)

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase, stft_result

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum.to("cuda:0") if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        # inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :self.num_samples]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase, self.stft_result = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction

import torch
import torch.nn.functional as F
from scipy.signal import get_window  # Only if you need a non-standard window


import torch
import torch.nn as nn
from scipy.signal import get_window

# class fixed_STFT(nn.Module):
#     def __init__(self, n_fft=320, hop_length=160, win_length=320, window='hann'):
#         """
#         Initializes the fixed_STFT module.
#
#         Parameters
#         ----------
#         n_fft : int
#             FFT size.
#         hop_length : int
#             Number of samples between successive frames.
#         win_length : int
#             Length of the window.
#         window : str or torch.Tensor
#             Window specification. If a string is provided, a window will be generated.
#         """
#         super(fixed_STFT, self).__init__()
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.win_length = win_length
#         self.window = window
#         # Will be set during transform (used for ISTFT length)
#         self.num_samples = None
#
#     def _get_window(self, device, dtype):
#         """
#         Internal helper to generate a window tensor on the correct device and type.
#         """
#         if isinstance(self.window, str):
#             if self.window.lower() == 'hann':
#                 # Ensure dtype is real (float32 or float64)
#                 real_dtype = torch.float32 if dtype in [torch.complex64, torch.float32] else torch.float64
#                 win = torch.hann_window(self.win_length, device=device, dtype=real_dtype)
#             else:
#                 win_np = get_window(self.window, self.win_length, fftbins=True)
#                 win = torch.tensor(win_np, device=device, dtype=dtype)
#         elif isinstance(self.window, torch.Tensor):
#             win = self.window.to(device=device, dtype=dtype)
#         else:
#             raise ValueError("window must be a string or a torch.Tensor")
#
#         # Ensure window is correctly shaped
#         if win.dim() != 1 or win.shape[0] != self.win_length:
#             raise ValueError(f"Window has incorrect shape {win.shape}, expected ({self.win_length},)")
#
#         return win
#
#     def transform(self, input_signal):
#         """
#         Computes the STFT of the input signal.
#
#         Parameters
#         ----------
#         input_signal : torch.Tensor
#             1D or 2D tensor representing the time-domain signal(s). Expected shape (T) or (B, T).
#
#         Returns
#         -------
#         stft_result : torch.Tensor
#             The complex STFT output.
#         """
#         # Ensure input has a batch dimension
#         if input_signal.dim() == 1:
#             input_signal = input_signal.unsqueeze(0)
#
#         # Save the original signal length for ISTFT reconstruction
#         self.num_samples = input_signal.shape[-1]
#
#         # Apply reflect padding before STFT
#         x_padded = F.pad(input_signal.unsqueeze(1), (self.n_fft // 2, self.n_fft // 2), mode='reflect').squeeze(1)
#         # Get the appropriate window
#         win = self._get_window(input_signal.device, input_signal.dtype)
#
#         stft_result = torch.stft(
#             x_padded,
#             n_fft=self.n_fft,
#             hop_length=self.hop_length,
#             win_length=self.win_length,
#             window=win,
#             center=False,
#             return_complex=False  # Returns a complex tensor
#         )
#         # stft_result shape: (B, n_fft//2+1, T, 2)
#         return stft_result
#
#     def inverse(self, magnitude, phase):
#         """
#         Reconstructs the time-domain signal using the inverse STFT.
#
#         Parameters
#         ----------
#         magnitude : torch.Tensor
#             Magnitude of the STFT.
#         phase : torch.Tensor
#             Phase of the STFT.
#
#         Returns
#         -------
#         reconstructed_signal : torch.Tensor
#             The reconstructed time-domain signal.
#         """
#         # Recombine magnitude and phase into a complex tensor
#         stft_complex = torch.polar(magnitude, phase)
#
#         # Get the window on the same device and type as the complex tensor
#         win = self._get_window(stft_complex.device, stft_complex.dtype)
#
#         # Reconstruct the time-domain signal using ISTFT
#         reconstructed_signal = torch.istft(
#             stft_complex,
#             n_fft=self.n_fft,
#             hop_length=self.hop_length,
#             win_length=self.win_length,
#             window=win,
#             center=False
#         )
#
#         if reconstructed_signal.shape[-1] > self.num_samples:
#             reconstructed_signal = reconstructed_signal[:, :self.num_samples]
#         return reconstructed_signal
#
#     def forward(self, input_signal):
#         """
#         Full forward pass: computes the STFT and then reconstructs the signal via ISTFT.
#
#         Parameters
#         ----------
#         input_signal : torch.Tensor
#             Input time-domain signal(s).
#
#         Returns
#         -------
#         reconstructed_signal : torch.Tensor
#             The reconstructed time-domain signal.
#         """
#         magnitude, phase = self.transform(input_signal)
#         reconstructed_signal = self.inverse(magnitude, phase)
#         return reconstructed_signal

