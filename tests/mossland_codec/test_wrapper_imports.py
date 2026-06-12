import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import get_class


def test_mossland_wrapper_and_callback_are_importable_for_hydra():
    module = importlib.import_module("scripts.mossland-codec.wrapper")

    assert hasattr(module, "MosslandCodecTrainingWrapper")
    assert hasattr(module, "MosslandCodecTrainingCallback")


def test_mossland_model_uses_task_conditioned_transformer_class_name():
    module = importlib.import_module("scripts.mossland-codec.models")

    assert hasattr(module, "MosslandCodecTransformer")
    assert not hasattr(module, "MosslandCodecUNet")
    assert not hasattr(module, "UNet")

    init_signature = inspect.signature(module.MosslandCodecTransformer)
    for name in ("sample_rate", "hop", "fac", "spec_length", "sigma_min", "sigma_max"):
        assert name in init_signature.parameters

    signature = inspect.signature(module.MosslandCodecTransformer.decoder_forward)
    assert "task_id" in signature.parameters
    assert "degradation_id" not in signature.parameters
    assert "channel_mode" not in signature.parameters

    assert hasattr(module.MosslandCodecTransformer, "prepare_audio_batch")
    generate_signature = inspect.signature(module.MosslandCodecTransformer.generate_waveform)
    assert "src" in generate_signature.parameters
    assert "task_id" in generate_signature.parameters
    assert "degradation_id" not in generate_signature.parameters
    assert "channel_mode" not in generate_signature.parameters


def test_hydra_targets_resolve_to_mossland_self_contained_classes():
    model_cls = get_class("scripts.mossland-codec.models.MosslandCodecTransformer")
    wrapper_cls = get_class("scripts.mossland-codec.wrapper.MosslandCodecTrainingWrapper")

    assert model_cls.__name__ == "MosslandCodecTransformer"
    assert wrapper_cls.__name__ == "MosslandCodecTrainingWrapper"


def test_mossland_wrapper_does_not_import_codicodec():
    source = Path("scripts/mossland-codec/wrapper.py").read_text(encoding="utf-8")

    assert "scripts.codicodec" not in source
    assert "CoDiCodecTrainingWrapper" not in source


def test_mossland_wrapper_only_consumes_task_payloads():
    source = Path("scripts/mossland-codec/wrapper.py").read_text(encoding="utf-8")

    assert "build_task_batch" not in source
    assert "sample_task_id" not in source
    assert "def generate_waveform" not in source
    assert "def _unpack_batch" not in source
    assert "def _task_from_payload" not in source
    assert "def _prepare_audio_batch" not in source
    assert "random_mix" not in source
    assert "model.generate_waveform" in source


def test_training_wrapper_reindexes_datamodule_dataset_on_due_train_step():
    module = importlib.import_module("scripts.mossland-codec.wrapper")
    calls = []

    class ReindexableDataset:
        def rebuild_index(self):
            calls.append("rebuild")

    prepared = ReindexableDataset()
    datamodule = SimpleNamespace(
        dataset=SimpleNamespace(dataset=prepared),
        train_dataset=SimpleNamespace(dataset=prepared),
    )
    wrapper = SimpleNamespace(index_data_every_step=2, _last_index_data_step=None)
    trainer = SimpleNamespace(global_step=1, datamodule=datamodule)

    assert module.MosslandCodecTrainingWrapper._maybe_rebuild_data_index(wrapper, trainer) == 0

    trainer.global_step = 2
    assert module.MosslandCodecTrainingWrapper._maybe_rebuild_data_index(wrapper, trainer) == 1
    assert calls == ["rebuild"]

    assert module.MosslandCodecTrainingWrapper._maybe_rebuild_data_index(wrapper, trainer) == 0
    assert calls == ["rebuild"]


def test_demo_callback_clears_cuda_cache_before_and_after_demo(monkeypatch, tmp_path):
    module = importlib.import_module("scripts.mossland-codec.wrapper")
    callback = module.MosslandCodecTrainingCallback(
        demo_dir=str(tmp_path),
        demo_num=1,
        demo_every=1000,
        sample_rate=44100,
        use_ema=False,
    )

    empty_cache_calls = []
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(module.torch.cuda, "empty_cache", lambda: empty_cache_calls.append("clear"))
    monkeypatch.setattr(module.torchaudio, "save", lambda *args, **kwargs: None)

    class FakeModel:
        def generate_waveform(self, src, **kwargs):
            return src.detach().cpu(), torch.zeros_like(src).detach().cpu()

        def prepare_audio_batch(self, audio):
            return audio

    payload = {
        "src": torch.ones(1, 2, 8),
        "target": torch.zeros(1, 2, 8),
        "task_id": "reconstruct",
    }
    trainer = SimpleNamespace(global_step=1, global_rank=0)
    lightning_module = SimpleNamespace(model=FakeModel())

    callback.on_train_batch_end(trainer, lightning_module, None, (payload, {}), 0)

    assert len(empty_cache_calls) == 2


def test_demo_callback_saves_quantized_and_continuous_demos(monkeypatch, tmp_path):
    module = importlib.import_module("scripts.mossland-codec.wrapper")
    callback = module.MosslandCodecTrainingCallback(
        demo_dir=str(tmp_path),
        demo_num=1,
        demo_every=1000,
        sample_rate=4,
        use_ema=False,
        silence_seconds=0.0,
    )

    saved_paths = []
    generate_calls = []
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        module.torchaudio,
        "save",
        lambda path, audio, sample_rate: saved_paths.append(Path(path).name),
    )

    class FakeModel:
        def generate_waveform(self, src, **kwargs):
            generate_calls.append(kwargs["dont_quantize"])
            value = 3.0 if kwargs["dont_quantize"] is False else 4.0
            return src.detach().cpu(), torch.full_like(src, value).detach().cpu()

        def prepare_audio_batch(self, audio):
            return audio

    payload = {
        "src": torch.ones(1, 2, 8),
        "target": torch.zeros(1, 2, 8),
        "task_id": "reconstruct",
    }
    trainer = SimpleNamespace(global_step=1, global_rank=0)
    lightning_module = SimpleNamespace(model=FakeModel())

    callback.on_train_batch_end(trainer, lightning_module, None, (payload, {}), 0)

    assert generate_calls == [False, True]
    assert saved_paths == [
        "1_0_reconstruct_rank0_quantized_src_target_generated.wav",
        "1_0_reconstruct_rank0_continuous_src_target_generated.wav",
    ]


def test_mossland_codec_does_not_use_hparams_modules():
    codec_dir = Path("scripts/mossland-codec")

    assert not (codec_dir / "hparams.py").exists()
    assert not (codec_dir / "hparams_inference.py").exists()

    for path in codec_dir.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert ".hparams" not in source
        assert "hparams_inference" not in source
        assert "import *" not in source


def test_mossland_codec_does_not_use_old_unet_names():
    codec_dir = Path("scripts/mossland-codec")

    for path in [*codec_dir.glob("*.py"), codec_dir / "README.md"]:
        source = path.read_text(encoding="utf-8")
        assert "UNet" not in source
        assert "unet" not in source


def test_mossland_experiment_config_points_to_self_contained_codec():
    config_dir = str((Path.cwd() / "scripts/configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="train", overrides=["experiment=mossland-codec"])

    assert cfg.model._target_ == "scripts.mossland-codec.models.MosslandCodecTransformer"
    assert cfg.wrapper._target_ == "scripts.mossland-codec.wrapper.MosslandCodecTrainingWrapper"
    assert cfg.model.sample_rate == 44100
    assert cfg.model.hop == 1024
    assert cfg.model.fac == 2
    assert cfg.model.spec_length == 32
    assert cfg.model.sigma_min == 0.002
    assert cfg.model.sigma_max == 80.0
    assert cfg.model.dim == 768
    assert cfg.model.head_dim == 128
    assert cfg.model.num_layers == 22
    assert cfg.model.num_layers_encoder == 22
    assert cfg.model.cond_channels == 768
    assert list(cfg.model.frontend_multipliers_list) == [1, 2, 4, 12]
    assert "source_root" not in cfg.data.dataset.dataset
    assert cfg.data.dataset.dataset.max_duration_seconds == 300
    assert cfg.data.dataset.dataset.crops_per_file == 16
    assert cfg.data.dataset.dataset.length == 1000000
    assert "index_data_every_step" not in cfg.data.dataset.dataset
    assert cfg.wrapper.index_data_every_step is None
    assert cfg.data.train_batch_size == 1
    assert cfg.data.train_batch_size * cfg.data.dataset.dataset.crops_per_file == 16
    assert cfg.data.num_workers == 6
    assert cfg.data.prefetch_factor == 2
    assert cfg.trainer.log_every_n_steps == 10
    assert "conditioning_scale" not in cfg.wrapper
    assert "active_tasks" not in cfg.wrapper
    assert "task_weights" not in cfg.wrapper
    assert "low_sample_rate" not in cfg.wrapper
    assert "random_mix_prob" not in cfg.wrapper


def test_task_payload_validation_rejects_raw_audio():
    module = importlib.import_module("scripts.mossland-codec.tasks")

    with pytest.raises(TypeError, match="MosslandTaskDataset"):
        module.MosslandTaskBatch.from_payload(torch.zeros(1, 2, 64))


def test_model_only_has_task_conditioning_embeddings():
    module = importlib.import_module("scripts.mossland-codec.models")

    assert not hasattr(module, "DEGRADATION_NAMES")
    assert not hasattr(module, "CHANNEL_MODE_NAMES")

    model = module.MosslandCodecTransformer(
        dim=8,
        head_dim=4,
        num_layers=1,
        num_layers_encoder=1,
        cond_channels=8,
        num_latents=2,
        num_more_latents=0,
        frontend_base_channels=4,
        frontend_multipliers_list=[1],
        frontend_layers_list=[1],
        frontend_encoder_layers_list=[1],
        frontend_freq_downsample_list=[0],
        spec_length=2,
        hop=16,
        fac=2,
    )

    assert hasattr(model, "task_embedding")
    assert not hasattr(model, "degradation_embedding")
    assert not hasattr(model, "channel_mode_embedding")
