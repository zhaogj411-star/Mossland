# torch compile path should be the path where this file is located
import os
torch_compile_cache_dir = os.path.dirname(os.path.abspath(__file__)) + '/torch_compile_cache'

# GENERAL/INFERENCE
mixed_precision = True                                                      # use mixed precision
seed = 42                                                                   # seed for Pytorch

stereo = True                                                               # if True, train on stereo data, if False, train on mono data

default_denoising_steps = 5                                                 # default number of denoising steps for inference

# stft spectrogram params
hop = 960
fac = 2
if stereo:
    stft_channels = 4
else:
    stft_channels = 2

sample_rate = 48000                                                         # sampling rate of input/output audio

# STFT normalization params
alpha_rescale = 0.65
beta_rescale = 0.34


# MODEL
dim = 512                                                                   # hidden transformer dimension
head_dim = 128                                                              # hidden dimension of each head in transformer
heads = dim//head_dim                                                       # number of heads in transformer
mlp_mult = 4                                                                # multiplier for hidden layer in transformer
pos_emb = 'learned'                                                         # if True, use positional embedding in transformer (alibi)
num_layers = 12                                                             # number of layers in diffusion backbone
num_layers_encoder = num_layers                                             # number of layers in encoder
cond_channels = 512                                                         # dimension of time embedding

num_latents = 128                                                           # latents per chunk; stacked = num_latents*bottleneck_channels = 512-dim
num_more_latents = 8                                                        # number of additional latents per patch of data_length//2 tokens, to be discarded
bottleneck_channels = 4
rvq_dim = 64                                                                # chunk stack (512) reshaped to 512/rvq_dim tokens; RVQ codes on rvq_dim

# frontend params
frontend_base_channels = 64
frontend_multipliers_list = [1, 2, 4, dim//frontend_base_channels]
frontend_layers_list = [3, 3, 3, 1]
frontend_encoder_layers_list = frontend_layers_list
frontend_freq_downsample_list = [0, 1, 0]

spec_length = 32
downsample_ratio = (4**frontend_freq_downsample_list.count(0))*(4**frontend_freq_downsample_list.count(1))*(2**frontend_freq_downsample_list.count(2))*(2**frontend_freq_downsample_list.count(3))
data_length = (hop*(fac//2)*spec_length)//downsample_ratio                  # sequence length of data used for training

sigma_data = 0.333                                                          # data std (TrigFlow prior scale), measured on NETEASE_SPIDER representation (hop=960,fac=2)
t_min = 0.006                                                               # TrigFlow time lower bound (near clean data)
t_max = 1.5666                                                              # TrigFlow time upper bound (near pure noise, ~pi/2)
