import torch
import numpy as np
import soundfile as sf
import torch.nn.functional as F
import einops

from .hparams import *
from .hparams_inference import *
from .utils import is_path, distribute, is_integer, download_model
from .models import UNet
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

if mixed_precision:
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


class EncoderDecoder:
    """Codec wrapper for encoding waveforms to latents and decoding them back.

    Handles model loading, device placement, and batching utilities. Public API
    mirrors previous releases while using the new codicodec architecture.
    """
    def __init__(self, load_path_inference=None, device=None):
        # Ensure checkpoint is available locally (downloaded from HF Hub if needed)
        download_model()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.load_path_inference = load_path_inference
        if load_path_inference is None:
            self.load_path_inference = load_path_inference_default
        self.get_models()
        self.latents_per_timestep = num_latents
        self.bottleneck_channels = bottleneck_channels

    def latents2dim(self, latents, desired_channels=64):
        """Reshape latent channels to desired size while preserving content."""
        assert desired_channels%self.bottleneck_channels==0, f"Desired channels must be divisible by original number of channels = {self.bottleneck_channels}"
        return einops.rearrange(latents, '... (l d) c -> ... l (d c)', d=desired_channels//self.bottleneck_channels)

    def dim2latents(self, latents):
        """Inverse of latents2dim()."""
        return einops.rearrange(latents, '... l (d c) -> ... (l d) c', c=self.bottleneck_channels)

    def get_models(self):
        """Instantiate UNet and load weights from checkpoint."""
        gen = UNet().to(self.device)
        gen.eval()
        checkpoint = torch.load(self.load_path_inference, map_location=self.device, weights_only=False)
        gen.load_state_dict(checkpoint['gen_state_dict'], strict=True)
        self.gen = gen

    def encode(self, path_or_audio, max_batch_size=None, discrete=False, preprocess_on_gpu=True, desired_channels=64, fix_batch_size=False):
        '''
        path_or_audio: path of audio sample to encode or numpy array of waveform to encode
        max_batch_size: maximum inference batch size for encoding: tune it depending on the available GPU memory

        WARNING! if input is numpy array of stereo waveform, it must have shape [audio_channels, waveform_samples]

        Returns latents with shape [audio_channels, dim, length]
        '''
        if max_batch_size is None:
            max_batch_size = max_batch_size_encode
        if discrete:
            raise RuntimeError(
                "Discrete RVQ export is not supported by this legacy inference wrapper. "
                "Use UNet.quantize_representation() from the Hydra model path."
            )
        # Continuous encoding. encoder_forward already bounds the latent to (-1, 1)
        # via an internal tanh (replacing the old FSQ.bound), so no external atanh
        # rescale is applied here -- the returned latent is the model's native latent.
        latents = encode_audio_inference(path_or_audio, self, max_batch_size, device=self.device, preprocess_on_gpu=preprocess_on_gpu, fix_batch_size=fix_batch_size)
        # reshape to desired channels
        out = self.latents2dim(latents, desired_channels=desired_channels)
        return out

    def decode(self, latent, mode='full', max_batch_size=None, denoising_steps=None, preprocess_on_gpu=True):
        '''
        latent: numpy array of latents to decode with shape [audio_channels, dim, length]
        max_batch_size: maximum inference batch size for decoding: tune it depending on the available GPU memory

        Returns numpy array of decoded waveform with shape [waveform_samples, audio_channels]
        '''
        # if dtype of latents is int32 or int64, then set discrete to True
        discrete = is_integer(latent)
        if max_batch_size is None:
            max_batch_size = max_batch_size_decode
        if discrete:
            latents = self.gen.latent_from_codes(latent.to(self.device) if torch.is_tensor(latent) else torch.from_numpy(latent).to(self.device))
        else:
            # Continuous latent is already the model's native (-1, 1) latent; just
            # regroup channels back to the decoder's [.., num_latents, bottleneck] layout.
            inv = latent if torch.is_tensor(latent) else torch.from_numpy(latent)
            latents = self.dim2latents(inv.to(self.device))
        return decode_latent_inference(latents, self, mode, max_batch_size, denoising_steps=denoising_steps, device=self.device, preprocess_on_gpu=preprocess_on_gpu)

    def reset(self):
        """Compatibility no-op; stateful autoregressive decoding is not used."""

    def decode_next(self, latents, max_batch_size=None, denoising_steps=None, discrete=False, preprocess_on_gpu=True):
        raise RuntimeError("decode_next is no longer supported; pass the full latent sequence to decode().")






# Encode audio sample for inference
# Parameters:
#   audio_path: path of audio sample
#   model: trained consistency model
#   device: device to run the model on
# Returns:
#   latent: compressed latent representation with shape [audio_channels, latent_length, dim]
@torch.no_grad()
def encode_audio_inference(audio_path, trainer, max_batch_size_encode, device='cuda', preprocess_on_gpu=False, fix_batch_size=False):
    trainer.gen = trainer.gen.to(device)
    trainer.gen.eval()
    squeeze_batch_dimensions = False
    if is_path(audio_path):
        audio, sr = sf.read(audio_path, dtype='float32', always_2d=True)
        audio = np.transpose(audio, [1,0])
    else:
        audio = audio_path
        sr = None
        if len(audio.shape)==1:
            squeeze_batch_dimensions = True
            # check if audio is numpy array, then use np.expand_dims, if it is a pytorch tensor, then use torch.unsqueeze
            if isinstance(audio, np.ndarray):
                audio = np.expand_dims(audio, 0)
                if stereo:
                    audio = np.repeat(audio, 2, axis=0)
            else:
                audio = torch.unsqueeze(audio, 0)
                if stereo:
                    audio = torch.repeat_interleave(audio, 2, dim=0)
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    if preprocess_on_gpu:
        audio = audio.to(device)
    else:
        audio = audio.cpu()
    audio_channels = audio.shape[-2]
    if audio_channels==1 and stereo:
        audio = torch.cat([audio, audio], -2)

    if stereo and len(audio.shape)==2:
        squeeze_batch_dimensions = True
        audio = torch.unsqueeze(audio, 0)
    if len(audio.shape)>3:
        raise ValueError("Input audio shape is not valid. It should be [waveform_samples], [audio_channels, waveform_samples] or [batch_size, audio_channels, waveform_samples]")

    batch_size = audio.shape[0]
    repr_encoder = trainer.gen.audio_processor.to_representation_encoder(audio)
    del audio

    if repr_encoder.shape[-1]%spec_length!=0:
        pad = spec_length-(repr_encoder.shape[-1]%spec_length)
        repr_encoder = F.pad(repr_encoder, (0,pad))

    if repr_encoder.shape[-1]>spec_length:
        repr_encoder = torch.split(repr_encoder, spec_length, dim=-1)
        repr_encoder = torch.cat(repr_encoder, dim=0)

    device = next(trainer.gen.parameters()).device
    if fix_batch_size:
        original_batch_size = repr_encoder.shape[0]
        # make sure that batch size is exactly divisible by max_batch_size_encode
        if repr_encoder.shape[0]%max_batch_size_encode!=0:
            rem = torch.zeros(max_batch_size_encode-(repr_encoder.shape[0]%max_batch_size_encode), *repr_encoder.shape[1:], device=repr_encoder.device, dtype=repr_encoder.dtype)
            repr_encoder = torch.cat([repr_encoder, rem], 0)
        latent = distribute(trainer.gen.encoder_forward_fast, repr_encoder, max_batch_size_encode, device)
        latent = latent[:original_batch_size]
    else:
        latent = distribute(trainer.gen.encoder_forward, repr_encoder, max_batch_size_encode, device)

    del repr_encoder
    # split samples
    latent = torch.split(latent, batch_size, 0)
    latent = torch.stack(latent, -3)
    if latent.shape[0]==1 and squeeze_batch_dimensions:
        latent = latent.squeeze(0)
    return latent



# Decode latent representation for inference, use the same framework as in encode_audio_inference, but in reverse order for decoding
# Parameters:
#   latent: compressed latent representation with shape [batch_size, timesteps, latents_per_timestep, dim] or [timesteps, latents_per_timestep, dim]
#   model: trained consistency model
#   device: device to run the model on
# Returns:
#   audio: numpy array of decoded waveform with shape [waveform_samples, audio_channels]
@torch.no_grad()
def decode_latent_inference(latent, trainer, mode='full', max_batch_size_decode=None, denoising_steps=None, device='cuda', preprocess_on_gpu=False):
    trainer.gen = trainer.gen.to(device)
    trainer.gen.eval()
    # check if latent is numpy array, then convert to tensor
    if isinstance(latent, np.ndarray):
        latent = torch.from_numpy(latent)
    if preprocess_on_gpu:
        latent = latent.to(device)
    else:
        latent = latent.cpu()
    squeeze_batch_dimensions = False
    # if latent has only 3 dimensions, add a third dimension as axis 0
    if len(latent.shape)==3:
        squeeze_batch_dimensions = True
        latent = torch.unsqueeze(latent, 0)
    latent = torch.cat(torch.unbind(latent, -3), -2)
    if mode not in (None, "full", "parallel"):
        raise ValueError(f"mode={mode!r} is not supported; use the full-sequence decode interface.")
    original_length = (latent.shape[-2]//num_latents)*spec_length
    repr = trainer.gen.decode(latent, denoising_steps=denoising_steps)
    if not preprocess_on_gpu:
        repr = repr.cpu()
    repr = trainer.gen.audio_processor.to_waveform(repr[..., :original_length]).cpu()
    del latent
    if squeeze_batch_dimensions:
        repr = repr.squeeze(0)
    return repr
