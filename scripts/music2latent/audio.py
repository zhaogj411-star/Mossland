import torch
import torch.nn.functional as F
import numpy as np


class AudioProcessor:
    def __init__(
        self,
        alpha_rescale=1.0,
        beta_rescale=1.0,
        hop_size=256,
        fac=4,
        center_pad=False,
    ):
        self.alpha_rescale = alpha_rescale
        self.beta_rescale = beta_rescale
        self.hop_size = hop_size
        self.fac = fac
        self.center_pad = bool(center_pad)

    def normalize_complex(self, x):
        """归一化复数张量"""
        return (
            self.beta_rescale
            * (x.abs() ** self.alpha_rescale).to(torch.complex64)
            * torch.exp(1j * torch.angle(x).to(torch.complex64))
        )

    def denormalize_complex(self, x):
        """反归一化复数张量"""
        x = x / self.beta_rescale
        return (x.abs() ** (1.0 / self.alpha_rescale)).to(torch.complex64) * torch.exp(
            1j * torch.angle(x).to(torch.complex64)
        )

    def wv2complex(self, wv):
        """将波形转换为复数频谱"""
        X = self.stft(wv)
        return X[..., : self.hop_size * 2, :]

    def wv2realimag(self, wv):
        """将波形转换为实部和虚部表示"""
        X = self.wv2complex(wv)
        X = self.normalize_complex(X)
        if X.ndim == 3:
            return torch.stack((torch.real(X), torch.imag(X)), dim=1)
        if X.ndim == 4:
            realimag = torch.stack((torch.real(X), torch.imag(X)), dim=2)
            batch, channels, parts, freq, frames = realimag.shape
            return realimag.reshape(batch, channels * parts, freq, frames)
        raise ValueError(f"Expected waveform STFT with 3 or 4 dims, got {tuple(X.shape)}")

    def realimag2wv(self, x):
        """将实部和虚部表示转换回波形"""
        x = torch.nn.functional.pad(x, (0, 0, 0, 1))
        if x.ndim == 4 and x.shape[1] > 2:
            batch, repr_channels, freq, frames = x.shape
            if repr_channels % 2 != 0:
                raise ValueError(f"Expected even representation channels, got {repr_channels}")
            x = x.reshape(batch, repr_channels // 2, 2, freq, frames)
            real = x[:, :, 0]
            imag = x[:, :, 1]
        else:
            real, imag = torch.chunk(x, 2, -3)
            real = real.squeeze(-3)
            imag = imag.squeeze(-3)
        X = torch.complex(real, imag)
        X = self.denormalize_complex(X)
        return self.istft(X).clamp(-1.0, 1.0)

    def to_representation_encoder(self, x):
        """编码器的表示转换"""
        return self.wv2realimag(x)

    def to_representation(self, x, hop):
        """将输入转换为标准表示形式"""
        return self.wv2realimag(x)

    def to_waveform(self, x, hop):
        """将表示形式转换回波形"""
        return self.realimag2wv(x)

    def overlap_and_add(self, signal, frame_step):
        """重叠相加算法"""
        outer_dimensions = signal.shape[:-2]
        outer_rank = torch.numel(torch.tensor(outer_dimensions))

        def full_shape(inner_shape):
            s = torch.cat(
                [torch.tensor(outer_dimensions), torch.tensor(inner_shape)], 0
            )
            s = list(s)
            s = [int(el) for el in s]
            return s

        frame_length = signal.shape[-1]
        frames = signal.shape[-2]

        output_length = frame_length + frame_step * (frames - 1)
        segments = -(-frame_length // frame_step)

        signal = torch.nn.functional.pad(
            signal, (0, segments * frame_step - frame_length, 0, segments)
        )
        shape = full_shape([frames + segments, segments, frame_step])
        signal = torch.reshape(signal, shape)

        perm = torch.cat(
            [
                torch.arange(0, outer_rank),
                torch.tensor([el + outer_rank for el in [1, 0, 2]]),
            ],
            0,
        )
        perm = list(perm)
        perm = [int(el) for el in perm]
        signal = torch.permute(signal, perm)

        shape = full_shape([(frames + segments) * segments, frame_step])
        signal = torch.reshape(signal, shape)
        signal = signal[..., : (frames + segments - 1) * segments, :]

        shape = full_shape([segments, (frames + segments - 1), frame_step])
        signal = torch.reshape(signal, shape)
        signal = signal.sum(-3)

        shape = full_shape([(frames + segments - 1) * frame_step])
        signal = torch.reshape(signal, shape)
        signal = signal[..., :output_length]

        return signal

    def inverse_stft_window(self, frame_length, frame_step, forward_window):
        """计算ISTFT的逆窗口函数"""
        denom = forward_window**2
        overlaps = -(-frame_length // frame_step)
        denom = F.pad(denom, (0, overlaps * frame_step - frame_length))
        denom = torch.reshape(denom, [overlaps, frame_step])
        denom = torch.sum(denom, 0, keepdim=True)
        denom = torch.tile(denom, [overlaps, 1])
        denom = torch.reshape(denom, [overlaps * frame_step])
        return forward_window / denom[:frame_length]

    def istft(self, SP):
        """逆短时傅里叶变换，确保与frame方法的边界处理一致"""
        x = torch.fft.irfft(SP, dim=-2)
        window = torch.hann_window(self.fac * self.hop_size, device=SP.device)
        window = self.inverse_stft_window(
            self.fac * self.hop_size, self.hop_size, window
        )
        x = x * window.unsqueeze(-1)
        
        # 重叠相加重建信号
        reconstructed = self.overlap_and_add(x.transpose(-1, -2), self.hop_size)
        
        if self.center_pad:
            frame_length = self.fac * self.hop_size
            frame_step = self.hop_size
            pad_left = (frame_length - frame_step) // 2
            pad_right = (frame_length - frame_step) // 2
            if reconstructed.shape[-1] > pad_left + pad_right:
                reconstructed = reconstructed[
                    ..., pad_left : reconstructed.shape[-1] - pad_right
                ]
        
        return reconstructed

    def frame(
        self, signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1
    ):
        """将信号分帧；默认使用官方 Music2Latent 的 no-center-pad 行为。"""
        if self.center_pad:
            pad_left = (frame_length - frame_step) // 2
            pad_right = (frame_length - frame_step) // 2
            if axis == -1:
                signal = F.pad(signal, (pad_left, pad_right), "reflect")
            else:
                pad_dims = [0] * (signal.ndim * 2)
                pad_dims[-(axis + 1) * 2 - 1] = pad_left
                pad_dims[-(axis + 1) * 2] = pad_right
                signal = F.pad(signal, pad_dims, "reflect")

        if pad_end:
            signal_length = signal.shape[axis]
            frames_overlap = frame_length - frame_step
            rest_samples = np.abs(signal_length - frames_overlap) % np.abs(
                frame_length - frames_overlap
            )
            pad_size = int(frame_length - rest_samples)
            if pad_size != 0:
                pad_axis = [0] * signal.ndim
                pad_axis[axis] = pad_size
                signal = F.pad(signal, pad_axis, "constant", pad_value)
        
        frames = signal.unfold(axis, frame_length, frame_step)
        return frames

    def stft(self, wv):
        """短时傅里叶变换"""
        window = torch.hann_window(self.fac * self.hop_size, device=wv.device)
        framed_signals = self.frame(wv, self.fac * self.hop_size, self.hop_size)
        framed_signals = framed_signals * window
        return torch.fft.rfft(framed_signals, n=None, dim=-1, norm=None).transpose(
            -1, -2
        )
