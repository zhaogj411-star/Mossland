import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

class TextAudioDataset(Dataset):
    def __init__(self, csv_file, data_dir, num=100000, target_length=120):
        """
        文本音频数据集

        参数:
            csv_file (str): CSV文件路径，包含名称和标签列
            data_dir (str): 存放音频潜在表示的目录路径
            num (int): 数据集大小
            target_length (int): 目标音频长度
        """
        self.data_frame = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.num = num
        self.target_length = target_length

    def __len__(self):
        return self.num

    def random_crop(self, data):
        if len(data.shape) > 2:
            data = data.squeeze(0)
        data_length = data.shape[1]
        if data_length > self.target_length:
            start = random.randint(0, data_length - self.target_length)
            data = data[:, start : start + self.target_length]
        elif data_length < self.target_length:
            raise ValueError(f"数据长度 {data_length} 小于目标长度 {self.target_length}")
        return data, start, start + self.target_length

    def __getitem__(self, idx):
        idx = idx % len(self.data_frame)
        try:
            # 获取名称和标签
            name = self.data_frame.iloc[idx]['名称']
            label = self.data_frame.iloc[idx]['标签']

            # 构建音频文件路径
            audio_file_path = os.path.join(self.data_dir, f"{name}.mp3.pt")

            # 加载音频潜在表示
            audio_data = torch.load(audio_file_path)
            audio_latent = audio_data['latent']

            # 随机裁剪到目标长度
            audio_latent, start, end = self.random_crop(audio_latent)

            return audio_latent, label
        except Exception as e:
            return self.__getitem__(idx + 1)

if __name__ == "__main__":
    dataset = TextAudioDataset(csv_file="/home/gjzhao/workspace2025/circleland-ai/pixaboy_info_filtered.csv", data_dir="/home/gjzhao/data/training_data/music2latent_no_vocal",num=100000,target_length=120)
    print(len(dataset))
    audio_latent, label = dataset[0]
    print(audio_latent.shape)
    print(label)
