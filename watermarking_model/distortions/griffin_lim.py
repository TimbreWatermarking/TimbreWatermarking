import numpy as np
import scipy.signal


def _exp(x, base):
    if base == 10:
        return np.power(10, x)
    return np.exp(x)


class griffin_lim():

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


    def _mel_to_linear(self, mel_spec: np.ndarray) -> np.ndarray:
        """Convert a melspectrogram to full scale spectrogram."""
        return np.maximum(1e-10, np.dot(self.inv_mel_basis, mel_spec))

    
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


    def inv_melspectrogram(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Convert a melspectrogram to a waveform using Griffi-Lim vocoder."""
        D = self.denormalize(mel_spectrogram)
        S = self._db_to_amp(D)
        S = self._mel_to_linear(S)  # Convert back to linear
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        return self._griffin_lim(S**self.power)