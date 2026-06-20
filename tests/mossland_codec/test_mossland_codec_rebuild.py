from importlib import import_module
from pathlib import Path

import torch
from torch import nn
from hydra import compose, initialize_config_dir
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]


class TinyAudioDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        audio = torch.linspace(-1.0, 1.0, 64).repeat(2, 1)
        return audio, {"path": f"tiny-{index}.wav"}


class RecordingModel(nn.Module):
    sigma_min = 0.002
    sigma_max = 80.0
    rho = 7.0

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.task_ids = []

    def forward(self, latents, x, sigma=None, latent_override=None, task_id=None):
        self.task_ids.append(task_id)
        return x * self.weight


class IdentityAudioProcessor:
    fac = 4

    def to_representation_encoder(self, audio):
        return audio.unsqueeze(2)


class SourceRecordingNoRvqModel(nn.Module):
    sigma_min = 0.002
    sigma_max = 80.0
    rho = 7.0
    hop = 1
    freq_downsample_list = []
    has_quantizer = False
    task_names = (
        "reconstruct",
        "separate_vocals",
        "separate_accompaniment",
        "super_resolution",
        "mono_to_stereo",
    )

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.audio_processor = IdentityAudioProcessor()
        self.encoder_inputs = []
        self.latent_overrides = []

    def encoder(self, representation):
        self.encoder_inputs.append(representation.detach().clone())
        return representation + 10.0

    def forward(self, latents, x, sigma=None, latent_override=None, task_id=None):
        self.latent_overrides.append(
            None if latent_override is None else latent_override.detach().clone()
        )
        return x * self.weight


def test_mossland_codec_config_targets_mossland_rvq_package():
    with initialize_config_dir(
        config_dir=str(ROOT / "scripts" / "configs"),
        version_base=None,
    ):
        cfg = compose(config_name="train", overrides=["experiment=mossland-codec"])

    assert cfg.model._target_ == "scripts.mossland_codec.models.MosslandCodec"
    assert cfg.wrapper._target_ == "scripts.mossland_codec.wrapper.MosslandCodecTrainingWrapper"
    assert cfg.data.dataset._target_ == "scripts.mossland_codec.tasks.MosslandTaskDataset"
    assert "music2latent" not in cfg.model._target_.lower()
    assert cfg.model.data_channels == 4
    assert cfg.model.bottleneck_channels == 64
    assert cfg.model.quantizer_num_quantizers == 32
    assert list(cfg.model.task_names) == [
        "reconstruct",
        "separate_vocals",
        "separate_accompaniment",
        "super_resolution",
        "mono_to_stereo",
    ]
    assert cfg.data.train_batch_size == 1


def test_mossland_codec_debug_config_reads_preprocessed_task_pt():
    with initialize_config_dir(
        config_dir=str(ROOT / "scripts" / "configs"),
        version_base=None,
    ):
        cfg = compose(config_name="train", overrides=["experiment=mossland_codec_debug"])

    assert cfg.project_name == "mossland_codec_debug"
    assert cfg.model._target_ == "scripts.mossland_codec.models.MosslandCodec"
    assert cfg.wrapper._target_ == "scripts.mossland_codec.wrapper.MosslandCodecTrainingWrapper"
    assert cfg.data.dataset._target_ == "scripts.mossland_codec.tasks.MosslandTaskPTDataset"
    assert "music2latent" not in cfg.project_name.lower()
    assert cfg.data.train_batch_size == 16
    assert cfg.model.base_channels == 64
    assert cfg.model.data_channels == 4
    assert cfg.model.quantizer_num_quantizers == 32
    assert list(cfg.model.task_names) == [
        "reconstruct",
        "separate_vocals",
        "separate_accompaniment",
        "super_resolution",
        "mono_to_stereo",
    ]
    assert list(cfg.wrapper.train_n_quantizers_choices) == [8, 16, 32]
    assert list(cfg.callbacks.demo_callback.demo_n_quantizers_choices) == [8, 16, 32]
    assert cfg.callbacks.demo_callback.demo_num == 1
    assert cfg.callbacks.demo_callback.demo_start_step == 100
    assert cfg.callbacks.demo_callback.use_ema is False


def test_mossland_demo_callback_honors_start_step():
    wrapper = import_module("scripts.mossland_codec.wrapper")
    callback = wrapper.MosslandCodecTrainingCallback(
        demo_dir="unused",
        demo_every=100,
        demo_start_step=100,
    )

    assert not callback._should_run_demo(1)
    assert not callback._should_run_demo(99)
    assert callback._should_run_demo(100)
    assert callback._should_run_demo(200)
    callback.last_demo_step = 200
    assert not callback._should_run_demo(200)


def test_mossland_demo_callback_restores_training_mode_after_demo():
    wrapper = import_module("scripts.mossland_codec.wrapper")
    callback = wrapper.MosslandCodecTrainingCallback(
        demo_dir="unused",
        demo_every=100,
        demo_start_step=100,
        use_ema=False,
    )
    model = nn.Linear(2, 2)
    model.train()

    def save_demo(demo_model, *_args, **_kwargs):
        demo_model.eval()

    callback._save_demo_for_model = save_demo
    module = type("Module", (), {"model": model, "device": torch.device("cpu")})()
    trainer = type("Trainer", (), {"global_rank": 0, "global_step": 100})()
    payload = {
        "src": torch.zeros(1, 2, 8),
        "target": torch.zeros(1, 2, 8),
        "task_id": ("reconstruct",),
    }

    callback.on_train_batch_end(trainer, module, None, (payload, {}), 0)

    assert model.training


def test_mossland_demo_callback_restores_rng_after_demo():
    wrapper = import_module("scripts.mossland_codec.wrapper")
    callback = wrapper.MosslandCodecTrainingCallback(
        demo_dir="unused",
        demo_every=100,
        demo_start_step=100,
        use_ema=False,
    )
    model = nn.Linear(2, 2)

    def save_demo(*_args, **_kwargs):
        torch.rand(4)

    callback._save_demo_for_model = save_demo
    module = type("Module", (), {"model": model, "device": torch.device("cpu")})()
    trainer = type("Trainer", (), {"global_rank": 0, "global_step": 100})()
    payload = {
        "src": torch.zeros(1, 2, 8),
        "target": torch.zeros(1, 2, 8),
        "task_id": ("reconstruct",),
    }

    torch.manual_seed(123)
    expected_state = torch.random.get_rng_state()
    callback.on_train_batch_end(trainer, module, None, (payload, {}), 0)

    assert torch.equal(torch.random.get_rng_state(), expected_state)


def test_mossland_task_condition_embedding_uses_per_sample_task_ids():
    models = import_module("scripts.mossland_codec.models")
    model = models.MosslandCodec.__new__(models.MosslandCodec)
    nn.Module.__init__(model)
    model.task_names = ("reconstruct", "separate_vocals")
    model.task_to_idx = {"reconstruct": 0, "separate_vocals": 1}
    model.task_embedding = nn.Embedding(2, 4)
    model.emb_proj = nn.Identity()
    with torch.no_grad():
        model.task_embedding.weight.copy_(
            torch.tensor(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [10.0, 11.0, 12.0, 13.0],
                ]
            )
        )

    sigma_embedding = torch.zeros(2, 4)
    conditioned = model._condition_embedding(
        sigma_embedding,
        task_id=("reconstruct", "separate_vocals"),
    )

    assert torch.allclose(conditioned[0], model.task_embedding.weight[0])
    assert torch.allclose(conditioned[1], model.task_embedding.weight[1])


def test_mossland_wrapper_passes_task_id_to_consistency_model():
    wrapper_mod = import_module("scripts.mossland_codec.wrapper")
    model = RecordingModel()
    wrapper = wrapper_mod.MosslandCodecTrainingWrapper(
        model=model,
        use_ema=False,
        sigma_sampling="uniform",
        consistency_step_schedule="constant",
        fail_on_nonfinite=False,
    )

    target_representation = torch.zeros(2, 4, 8, 3)
    task_id = ("separate_vocals", "separate_accompaniment")
    wrapper._consistency_loss(
        target_representation,
        latent_override=None,
        task_id=task_id,
    )

    assert model.task_ids == [task_id, task_id]


def test_mossland_no_rvq_training_uses_source_latent_override():
    wrapper_mod = import_module("scripts.mossland_codec.wrapper")
    model = SourceRecordingNoRvqModel()
    wrapper = wrapper_mod.MosslandCodecTrainingWrapper(
        model=model,
        use_ema=False,
        sigma_sampling="uniform",
        consistency_step_schedule="constant",
        fail_on_nonfinite=False,
    )
    wrapper.log = lambda *args, **kwargs: None

    src = torch.ones(2, 2, 8)
    target = torch.zeros(2, 2, 8)
    payload = {
        "src": src,
        "target": target,
        "task_id": ("separate_vocals", "separate_accompaniment"),
    }

    loss = wrapper.training_step((payload, {}), 0)

    expected_source_representation = src.unsqueeze(2)
    expected_source_latent = expected_source_representation + 10.0
    assert torch.isfinite(loss)
    assert len(model.encoder_inputs) == 1
    assert torch.allclose(model.encoder_inputs[0], expected_source_representation)
    assert len(model.latent_overrides) == 2
    assert torch.allclose(model.latent_overrides[0], expected_source_latent)
    assert torch.allclose(model.latent_overrides[1], expected_source_latent)


def test_mossland_task_logging_writes_zero_for_absent_tasks():
    wrapper_mod = import_module("scripts.mossland_codec.wrapper")
    model = SourceRecordingNoRvqModel()
    wrapper = wrapper_mod.MosslandCodecTrainingWrapper(
        model=model,
        use_ema=False,
        sigma_sampling="uniform",
        consistency_step_schedule="constant",
        fail_on_nonfinite=False,
    )
    logged = {}

    def capture_log(name, value, **kwargs):
        if name.startswith("task/"):
            logged[name] = (float(value), kwargs)

    wrapper.log = capture_log
    payload = {
        "src": torch.ones(2, 2, 8),
        "target": torch.zeros(2, 2, 8),
        "task_id": ("separate_vocals", "separate_accompaniment"),
    }

    wrapper.training_step((payload, {}), 0)

    assert logged["task/separate_vocals"][0] == 0.5
    assert logged["task/separate_accompaniment"][0] == 0.5
    assert logged["task/reconstruct"][0] == 0.0
    assert logged["task/super_resolution"][0] == 0.0
    assert logged["task/mono_to_stereo"][0] == 0.0
    assert all(kwargs["sync_dist"] is True for _, kwargs in logged.values())


def test_mossland_task_pt_dataset_returns_preprocessed_payload(tmp_path):
    tasks = import_module("scripts.mossland_codec.tasks")
    src = torch.linspace(-1.0, 1.0, 64).repeat(2, 1)
    target = src * 0.5
    pt_path = tmp_path / "000000.pt"
    torch.save(
        {
            "src": src,
            "target": target,
            "task_id": "super_resolution",
            "source_path": "debug.wav",
            "sample_rate": 44100,
        },
        pt_path,
    )
    (tmp_path / "files.list").write_text(f"{pt_path.name}\n", encoding="utf-8")

    dataset = tasks.MosslandTaskPTDataset(
        root=tmp_path,
        index_file=tmp_path / "files.list",
        length=3,
    )
    payload, info = dataset[2]

    assert len(dataset) == 3
    assert set(payload) == {"src", "target", "task_id"}
    assert payload["task_id"] == "super_resolution"
    assert torch.allclose(payload["src"], src)
    assert torch.allclose(payload["target"], target)
    assert info["path"] == str(pt_path)
    assert info["source_path"] == "debug.wav"
    assert info["sample_rate"] == 44100


def test_mossland_task_dataset_returns_src_target_task_payload():
    tasks = import_module("scripts.mossland_codec.tasks")
    dataset = tasks.MosslandTaskDataset(
        TinyAudioDataset(),
        active_tasks=["mono_to_stereo"],
        sample_rate=44100,
    )

    payload, info = dataset[0]

    assert set(payload) == {"src", "target", "task_id"}
    assert payload["task_id"] == "mono_to_stereo"
    assert payload["src"].shape == payload["target"].shape == (2, 64)
    assert torch.allclose(payload["src"][0], payload["src"][1])
    assert info["path"] == "tiny-0.wav"


def test_new_mossland_codec_package_is_minimal():
    package_dir = ROOT / "scripts" / "mossland_codec"

    assert package_dir.exists()
    assert not (package_dir / "eval_benchmark").exists()
    assert not (package_dir / "transformer.py").exists()
    assert not (package_dir / "transformer_layers.py").exists()
