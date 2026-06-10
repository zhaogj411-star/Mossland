from scripts.tools.music_info_extractor import MusicInfoExtractor
from tqdm import tqdm
import torch
from scripts.tools.multiprocessor import distribute_tasks_to_gpus
import logging
import os
import librosa
import numpy as np
import torch.nn.functional as F
import torch

logging.basicConfig(
    level=logging.INFO,
    filename="prepare.log",
    filemode="a",
    format="%(asctime)s - %(message)s",
)


def process(files):
    from scripts.factory import load_model
    from scripts.models.audio_autoencoder import AudioAutoencoder

    ckpt_path = "/home/gjzhao/workspace2024/heartstrings-version4/ckpt/vae48000.ckpt"
    model: AudioAutoencoder = load_model(ckpt_path)
    model.cuda()
    mie = MusicInfoExtractor(sample_rate=48000, device="cuda")
    target_dir = "/home/gjzhao/data/training_data/latent_48000_chroma"
    os.makedirs(target_dir, exist_ok=True)
    for file in tqdm(files):
        try:

            basename = os.path.basename(file)
            target_path = f"{target_dir}/{basename}.pt"
            if os.path.exists(target_path):
                logging.info(f"Skip {target_path}")
                continue
            mie.file_path = file
            wav = mie.demix(track_name="no_vocals")
            onset_env = librosa.onset.onset_strength(y=wav.numpy(), sr=48000,hop_length = 40960*5)
            chroma = librosa.feature.chroma_stft(
                S=np.abs(librosa.stft(wav.numpy(), hop_length=40960*5)), sr=48000
            )

            with torch.no_grad():
                latent = model.encode_audio(wav.unsqueeze(0).cuda(), chunked=True)
                onset_env = torch.from_numpy(onset_env)
                chroma = torch.from_numpy(chroma)
                onset_env = F.interpolate(onset_env[None], (latent.shape[-1],))[0]
                chroma = F.interpolate(chroma, (latent.shape[-1],))
                info = {
                    "latent": latent[0].cpu(),
                    "file": file,
                    "onset_env": onset_env / (onset_env.max() + 1e-6),
                    "chroma": torch.mean(chroma, dim=0),
                }
                torch.save(info, target_path)
                logging.info(f"Saved {target_path}")

        except Exception as e:
            logging.error(f"Error in {file}: {e}")
            continue


if __name__ == "__main__":
    import os

    from scripts.data.datasets import get_audio_filenames

    dir = ['']
    files = get_audio_filenames(['/home/gjzhao/data/music/电音','/home/gjzhao/data/music/古风','/home/gjzhao/data/music/无损音质 网易云音乐10W+纯音乐 （持续更新）','/home/gjzhao/data/music/【纯音音乐馆】【240+10P】它们不需要歌词,也可以打动你的心（2018.10.05更新）'])
    # process(files)
    devices = [4, 5, 6, 7]
    # group_num = len(devices)
    group_size = 100
    group_num = len(files) // group_size
    tasks = [(files[i::group_num],) for i in range(group_num)]
    distribute_tasks_to_gpus(process, tasks, devices, 1)
