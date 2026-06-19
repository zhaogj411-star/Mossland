import torch

from scripts.data.datasets import PTDataset


def test_pt_dataset_accepts_exact_target_length_without_random_crop(tmp_path):
    pt_path = tmp_path / "sample.pt"
    torch.save({"audio": torch.zeros(2, 16), "path": "sample.wav"}, pt_path)

    dataset = PTDataset(
        folder_path=str(tmp_path),
        mainkey="audio",
        infokeys=["path"],
        target_length=16,
        if_random_crop=False,
    )

    audio, info = dataset[0]

    assert audio.shape == (2, 16)
    assert info["path"] == "sample.wav"


def test_pt_dataset_accepts_exact_target_length_with_random_crop(tmp_path):
    pt_path = tmp_path / "sample.pt"
    torch.save({"audio": torch.zeros(2, 16)}, pt_path)

    dataset = PTDataset(
        folder_path=str(tmp_path),
        mainkey="audio",
        infokeys=[],
        target_length=16,
        if_random_crop=True,
    )

    audio, _info = dataset[0]

    assert audio.shape == (2, 16)
