| Model | Source / Eval setting | Stereo | Representation | Compression Ratio | Bitrate | Bandwidth (kHz) | SI-SDR ↑ | ViSQOL ↑ | Mel distance ↓ | STFT distance ↓ | FAD_clap ↓ | FAD ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Music2Latent | MusicCaps-HF full (5355/5521); official SonyCSLParis/music2latent checkpoint | No | Continuous | 64x | - | 22.05 | -3.093 | 4.099 | 25.318 | 0.515 | 0.197 | 1.504 |
| Stable Audio 3 SAME-S | MusicCaps-HF full (5355/5521); official SAME-S autoencoder | Yes | Continuous | 4096-hop / 256-d | - | 22.05 | 8.771 | 3.781 | 26.794 | 0.345 | 0.126 | 1.857 |
| Stable Audio 3 SAME-L | MusicCaps-HF full (5355/5521); official SAME-L autoencoder | Yes | Continuous | 4096-hop / 256-d | - | 22.05 | 11.484 | 3.541 | 29.209 | 0.282 | 0.138 | 0.824 |
| CoDiCodec (AR) | CoDiCodec Table 2 / MusicCaps | Yes | Continuous | 128x | - | — | 未复现 | 未复现 | 未复现 | 未复现 | 未复现 | 未复现 |
| CoDiCodec (Par., s=3) | CoDiCodec Table 2 / MusicCaps | Yes | Continuous | 128x | - | — | 未复现 | 未复现 | 未复现 | 未复现 | 未复现 | 未复现 |
| CoDiCodec official checkpoint | MusicCaps-HF full (5355/5521); parallel default | Yes | Continuous | 128x | - | 22.05 | -1.129 | 4.186 | 20.839 | 0.490 | 0.092 | 0.497 |
| CoDiCodec paper-repro 10k (raw) | MusicCaps-HF full (5355/5521); local 1-10000.ckpt raw; full-clip chunked xfade | Yes | Continuous | 128x | - | 22.05 | -36.786 | 3.713 | 28.211 | 0.638 | 0.166 | 3.089 |
| CoDiCodec paper-repro 10k (EMA) | MusicCaps-HF full (5355/5521); local 1-10000.ckpt EMA; full-clip chunked xfade | Yes | Continuous | 128x | - | 22.05 | -37.183 | 3.766 | 27.849 | 0.643 | 0.171 | 3.251 |
| CoDiCodec paper-repro 18k (raw) | MusicCaps-HF full (5355/5521); local 2-18000.ckpt raw; full-clip chunked xfade | Yes | Continuous | 128x | - | 22.05 | -35.471 | 3.902 | 27.215 | 0.606 | 0.132 | 2.024 |
| CoDiCodec paper-repro 18k (EMA) | MusicCaps-HF full (5355/5521); local 2-18000.ckpt EMA; full-clip chunked xfade | Yes | Continuous | 128x | - | 22.05 | -35.566 | 3.891 | 27.295 | 0.606 | 0.136 | 2.073 |
| Mossland codec 370k (EMA) | MusicCaps-HF full (5355/5521); local 2026-06-12_12-46-36 last.ckpt EMA; EncoderDecoder parallel decode | Yes | Continuous | 128x | - | 22.05 | -9.930 | 3.013 | 28.207 | 0.522 | 0.098 | 0.490 |
| DAC | MusicCaps-HF full (5355/5521); DAC n_quantizers=3 | No | Discrete | - | 2.67 kbps | 22.05 | 3.474 | 4.054 | 22.709 | 0.691 | 0.102 | 2.720 |
| DAC | MusicCaps-HF full (5355/5521) | No | Discrete | - | 8 kbps | 22.05 | 9.726 | 4.272 | 22.412 | 0.557 | 0.075 | 0.714 |
| CoDiCodec (AR) | CoDiCodec Table 2 / MusicCaps | Yes | Discrete | - | 2.38 kbps | — | 未复现 | 未复现 | 未复现 | 未复现 | 未复现 | 未复现 |
| CoDiCodec (Par., s=3) | CoDiCodec Table 2 / MusicCaps | Yes | Discrete | - | 2.38 kbps | — | 未复现 | 未复现 | 未复现 | 未复现 | 未复现 | 未复现 |
| CoDiCodec (Par., s=4) | CoDiCodec Table 2 / MusicCaps | Yes | Discrete | - | 2.38 kbps | — | 未复现 | 未复现 | 未复现 | 未复现 | 未复现 | 未复现 |
| DAC / Proposed | MusicCaps-HF full (5355/5521); DAC n_quantizers=2 | — | Discrete | — | 1.78 kbps | 22.05 | 1.150 | 3.940 | 22.994 | 0.755 | 0.123 | 3.781 |
| DAC / Proposed | MusicCaps-HF full (5355/5521); DAC n_quantizers=3 | — | Discrete | — | 2.67 kbps | 22.05 | 3.474 | 4.054 | 22.709 | 0.691 | 0.102 | 2.720 |
| DAC / Proposed | MusicCaps-HF full (5355/5521); DAC n_quantizers=6 | — | Discrete | — | 5.33 kbps | 22.05 | 7.178 | 4.199 | 22.538 | 0.598 | 0.081 | 1.241 |
| DAC / Proposed | MusicCaps-HF full (5355/5521); DAC n_quantizers=9 | — | Discrete | — | 8 kbps | 22.05 | 9.726 | 4.272 | 22.412 | 0.557 | 0.075 | 0.714 |
| EnCodec | RVQGAN Table 3 / 44.1 kHz objective eval | — | Discrete | — | 1.5 kbps | — | 未复现 | 未复现 | 未复现 | 未复现 | 未复现 | 未复现 |
| EnCodec | MusicCaps-HF full (5355/5521) | — | Discrete | — | 3 kbps | 22.05 | 3.506 | 4.008 | 18.389 | 0.464 | 0.105 | 2.926 |
| EnCodec | MusicCaps-HF full (5355/5521) | — | Discrete | — | 6 kbps | 22.05 | 6.171 | 4.135 | 17.931 | 0.396 | 0.071 | 2.003 |
| EnCodec | MusicCaps-HF full (5355/5521) | — | Discrete | — | 12 kbps | 22.05 | 8.725 | 4.227 | 17.654 | 0.334 | 0.062 | 1.407 |
| EnCodec | MusicCaps-HF full (5355/5521) | — | Discrete | — | 24 kbps | 22.05 | 10.886 | 4.266 | 16.700 | 0.283 | 0.041 | 0.981 |
| Lyra | RVQGAN Table 3 / 44.1 kHz objective eval | — | Discrete | — | 9.2 kbps | — | 未复现 | 未复现 | 未复现 | 未复现 | 未复现 | 未复现 |
| Opus | MusicCaps-HF full (5355/5521) | — | Traditional codec | — | 8 kbps | 22.05 | 0.331 | 2.980 | 30.356 | 0.498 | 0.231 | 8.214 |
| Opus | MusicCaps-HF full (5355/5521) | — | Traditional codec | — | 14 kbps | 22.05 | 5.915 | 4.346 | 19.671 | 0.401 | 0.098 | 4.538 |
| Opus | MusicCaps-HF full (5355/5521) | — | Traditional codec | — | 24 kbps | 22.05 | 10.404 | 4.484 | 18.187 | 0.252 | 0.030 | 1.280 |
