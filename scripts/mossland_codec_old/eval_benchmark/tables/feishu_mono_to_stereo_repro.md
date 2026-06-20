| Model | Eval setting | Clips | FAD_clap ↓ | FAD ↓ | SI-SDR ↑ | SNR ↑ | LSD ↓ | MRSTFT ↓ | Fold-down SI-SDR ↑ | Stereo width ↑ | Channel corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mossland ckpt/mossland-codec0613 | MusicCaps-HF seed0+seed1 | 200 | 0.455 | 3.480 | -12.354 | -0.636 | 28.488 | 0.549 | -11.533 | 0.165 | 0.921 |
| DiffStereo official sampler seed0 | MusicCaps-HF seed0 | 100 | 0.319 | 6.057 | 0.806 | 0.186 | 31.863 | 1.743 | 4.916 | 0.455 | 0.651 |
| DiffStereo official sampler seed1 | MusicCaps-HF seed1 | 100 | 0.317 | 6.006 | 0.964 | 0.289 | 31.799 | 1.723 | 4.955 | 0.439 | 0.671 |
| Ambisonizer official checkpoint | MusicCaps-HF seed0 | 100 | 0.222 | 2.279 | 3.961 | 2.460 | 16.343 | 0.647 | 51.338 | 0.543 | 0.578 |
| s3a fallback decorrelator | MusicCaps-HF seed0 | 100 | 0.001 | 0.026 | 6.840 | 7.476 | 6.156 | 0.361 | 60.973 | 0.324 | 0.954 |
