import importlib
import inspect
import copy
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def _install_fake_residual_vq(monkeypatch):
    class FakeResidualVQ(torch.nn.Module):
        def __init__(self, *, dim, num_quantizers, codebook_size, **kwargs):
            super().__init__()
            self.dim = dim
            self.num_quantizers = num_quantizers
            self.codebook_size = codebook_size
            self.kwargs = kwargs

        def forward(self, latents, **kwargs):
            n_quantizers = kwargs.get("n_q") or kwargs.get("n_quantizers")
            if n_quantizers is None:
                n_quantizers = self.num_quantizers
            quantized = latents + 0.125
            codes = torch.zeros(
                *latents.shape[:-1],
                int(n_quantizers),
                dtype=torch.long,
                device=latents.device,
            )
            loss = torch.ones(int(n_quantizers), device=latents.device) * 0.25
            return quantized, codes, loss

    fake_module = types.ModuleType("vector_quantize_pytorch")
    fake_module.ResidualVQ = FakeResidualVQ
    monkeypatch.setitem(sys.modules, "vector_quantize_pytorch", fake_module)
    return FakeResidualVQ


def _small_model(**overrides):
    module = importlib.import_module("scripts.music2latent.models")
    kwargs = {
        "sample_rate": 8000,
        "hop": 16,
        "fac": 4,
        "audio_channels": 2,
        "data_length": 4,
        "base_channels": 8,
        "layers_list": [1, 1],
        "multipliers_list": [1, 2],
        "attention_list": [0, 0],
        "freq_downsample_list": [0],
        "layers_list_encoder": [1, 1],
        "attention_list_encoder": [0, 0],
        "bottleneck_base_channels": 16,
        "num_bottleneck_layers": 1,
        "bottleneck_channels": 6,
        "cond_channels": 8,
        "heads": 2,
        "dropout_rate": 0.0,
        "min_res_dropout": 1,
        "normalization": True,
        "init_as_zero": True,
        "frequency_scaling": False,
        "pre_normalize_2d_to_1d": True,
        "pre_normalize_downsampling_encoder": True,
        "sigma_min": 0.002,
        "sigma_max": 80.0,
        "sigma_data": 0.5,
        "rho": 7.0,
    }
    kwargs.update(overrides)
    return module.Music2LatentUNet(**kwargs)


def test_model_constructor_uses_explicit_parameters():
    module = importlib.import_module("scripts.music2latent.models")

    assert not hasattr(module, "hparams")
    assert hasattr(module, "Music2LatentUNet")

    signature = inspect.signature(module.Music2LatentUNet)
    for name in (
        "sample_rate",
        "hop",
        "fac",
        "audio_channels",
        "data_length",
        "bottleneck_channels",
        "use_rvq",
        "rvq_num_quantizers",
        "rvq_codebook_size",
    ):
        assert name in signature.parameters


def test_model_exposes_continuous_and_rvq_latents(monkeypatch):
    fake_rvq = _install_fake_residual_vq(monkeypatch)
    model = _small_model(
        use_rvq=True,
        rvq_num_quantizers=3,
        rvq_codebook_size=17,
        rvq_kwargs={"decay": 0.9},
    )
    audio = torch.randn(2, 2, 64)
    representation = model.to_representation_encoder(audio)

    continuous = model.encode_latents(representation, quantize=False)
    quantized = model.encode_latents(representation, quantize=True, n_quantizers=2)

    assert isinstance(model.rvq, fake_rvq)
    assert model.rvq.kwargs["decay"] == 0.9
    assert continuous.latents.shape == quantized.continuous.shape
    assert quantized.latents.shape == continuous.latents.shape
    assert quantized.codes.shape == (*continuous.latents.shape[:-1], 2)
    assert quantized.loss.shape == ()
    assert not torch.equal(continuous.latents, quantized.latents)
    assert representation.shape[1] == 4
    assert model.to_waveform(representation).shape[:2] == (2, 2)


def test_training_wrapper_runs_one_loss_step(monkeypatch):
    _install_fake_residual_vq(monkeypatch)
    model = _small_model(use_rvq=True, rvq_num_quantizers=2, rvq_codebook_size=8)
    wrapper_module = importlib.import_module("scripts.music2latent.wrapper")
    wrapper = wrapper_module.Music2LatentTrainingWrapper(
        model=model,
        learning_rate=1e-4,
        rvq_loss_weight=0.5,
        rvq_dropout_prob=0.0,
        consistency_step_total_steps=10,
        fail_on_nonfinite=True,
    )
    batch = torch.randn(2, 2, 64)

    loss = wrapper.training_step(batch, 0)

    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_training_wrapper_logs_latent_std(monkeypatch):
    _install_fake_residual_vq(monkeypatch)
    model = _small_model(use_rvq=True, rvq_num_quantizers=2, rvq_codebook_size=8)
    wrapper_module = importlib.import_module("scripts.music2latent.wrapper")
    wrapper = wrapper_module.Music2LatentTrainingWrapper(
        model=model,
        learning_rate=1e-4,
        rvq_dropout_prob=0.0,
        consistency_step_total_steps=10,
        fail_on_nonfinite=True,
    )
    logged = {}

    def fake_log(name, value, **kwargs):
        logged[name] = value.detach() if torch.is_tensor(value) else value

    monkeypatch.setattr(wrapper, "log", fake_log)

    loss = wrapper.training_step(torch.randn(2, 2, 64), 0)

    assert torch.isfinite(loss)
    for name in ("latent/std", "latent/continuous_std", "latent/rvq_std"):
        assert name in logged
        assert torch.isfinite(logged[name])


def test_training_wrapper_detaches_latents_before_rvq_sidecar(monkeypatch):
    _install_fake_residual_vq(monkeypatch)
    model = _small_model(use_rvq=True, rvq_num_quantizers=2, rvq_codebook_size=8)
    model_module = importlib.import_module("scripts.music2latent.models")
    wrapper_module = importlib.import_module("scripts.music2latent.wrapper")
    wrapper = wrapper_module.Music2LatentTrainingWrapper(
        model=model,
        learning_rate=1e-4,
        rvq_dropout_prob=1.0,
        consistency_step_total_steps=10,
        fail_on_nonfinite=True,
    )
    observed = {}

    def fake_quantize_latents(continuous_latents, *, n_quantizers=None):
        observed["requires_grad"] = continuous_latents.requires_grad
        codes = torch.zeros(
            *continuous_latents.shape[:-1],
            2,
            dtype=torch.long,
            device=continuous_latents.device,
        )
        return model_module.LatentOutput(
            latents=continuous_latents + 0.125,
            continuous=continuous_latents,
            codes=codes,
            loss=continuous_latents.new_tensor(0.25),
        )

    monkeypatch.setattr(model, "quantize_latents", fake_quantize_latents)

    loss = wrapper.training_step(torch.randn(2, 2, 64), 0)

    assert torch.isfinite(loss)
    assert observed["requires_grad"] is False


def test_training_wrapper_pads_audio_to_valid_frame_count(monkeypatch):
    _install_fake_residual_vq(monkeypatch)
    model = _small_model(use_rvq=True, rvq_num_quantizers=2, rvq_codebook_size=8)
    wrapper_module = importlib.import_module("scripts.music2latent.wrapper")
    wrapper = wrapper_module.Music2LatentTrainingWrapper(
        model=model,
        learning_rate=1e-4,
        rvq_dropout_prob=0.0,
        consistency_step_total_steps=10,
        fail_on_nonfinite=True,
    )
    batch = torch.randn(2, 2, 96)

    audio = wrapper._audio_from_batch(batch)
    representation = model.to_representation(audio)
    loss = wrapper.training_step(batch, 0)

    assert representation.shape[-1] % model.time_downsample_factor == 0
    assert torch.isfinite(loss)


def test_pseudo_huber_loss_stays_finite_for_large_difference():
    diffusion = importlib.import_module("scripts.music2latent.diffusion")
    predicted = torch.full((1, 1, 1, 1), 1e20, dtype=torch.float32)
    target = torch.zeros_like(predicted)

    loss = diffusion.pseudo_huber_loss(predicted, target, delta=0.00054)

    assert torch.isfinite(loss).all()


def test_axial_local_resblock_starts_as_identity():
    model_module = importlib.import_module("scripts.music2latent.models")
    local_module = importlib.import_module("scripts.music2latent.local_transformer")
    reference = model_module.ResBlock(
        8,
        8,
        cond_channels=6,
        normalize=True,
        attention=True,
        heads=2,
        use_2d=True,
        min_res_dropout=1,
    )
    block = local_module.AxialLocalAttentionResBlock.from_resblock(
        reference,
        transformer_depth=1,
        dim_heads=4,
        time_window=(1, 1),
        freq_window=(1, 1),
        use_freq_axis=True,
        ff_mult=1,
        differential=False,
        init_as_zero=True,
    )
    x = torch.randn(2, 8, 8, 6)
    time_emb = torch.randn(2, 6)

    out = block(x, time_emb)

    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-6, rtol=0.0)


def test_music2latent_axiallocal_forward_shape():
    model_module = importlib.import_module("scripts.music2latent.models")
    audio_module = importlib.import_module("scripts.music2latent.audio")
    model = model_module.Music2Latent_axiallocal(
        audio_processor=audio_module.AudioProcessor(hop_size=16, fac=4, center_pad=False),
        sample_rate=8000,
        base_channels=8,
        layers_list=[1, 1],
        multipliers_list=[1, 2],
        attention_list=[0, 0],
        freq_downsample_list=[0],
        layers_list_encoder=[1, 1],
        attention_list_encoder=[0, 0],
        bottleneck_base_channels=16,
        num_bottleneck_layers=1,
        bottleneck_channels=6,
        cond_channels=8,
        heads=2,
        hop=16,
        data_channels=2,
        frequency_scaling=False,
        mixed_precision=False,
        min_res_dropout=1,
        axial_transformer_depth=1,
        axial_dim_heads=4,
        axial_time_window=[1, 1],
        axial_freq_window=[1, 1],
        axial_ff_mult=1,
        axial_differential=False,
        axial_min_channels=0,
    )
    representation = torch.randn(1, 2, 32, 8)
    latents = model.encoder(representation)

    out = model(latents, representation, sigma=1.0)

    assert latents.shape == (1, 6, 4)
    assert out.shape == representation.shape
    assert torch.isfinite(out).all()
    assert any(
        isinstance(layer, model_module.AxialLocalAttentionResBlock)
        for layer in list(model.down_layers) + list(model.up_layers)
    )


def test_music2latent_axiallocal_can_skip_high_resolution_resblocks():
    model_module = importlib.import_module("scripts.music2latent.models")
    audio_module = importlib.import_module("scripts.music2latent.audio")
    model = model_module.Music2Latent_axiallocal(
        audio_processor=audio_module.AudioProcessor(hop_size=16, fac=4, center_pad=False),
        sample_rate=8000,
        base_channels=8,
        layers_list=[1, 1],
        multipliers_list=[1, 2],
        attention_list=[0, 0],
        freq_downsample_list=[0],
        layers_list_encoder=[1, 1],
        attention_list_encoder=[0, 0],
        bottleneck_base_channels=16,
        num_bottleneck_layers=1,
        bottleneck_channels=6,
        cond_channels=8,
        heads=2,
        hop=16,
        data_channels=2,
        frequency_scaling=False,
        mixed_precision=False,
        min_res_dropout=1,
        axial_transformer_depth=1,
        axial_dim_heads=4,
        axial_time_window=[1, 1],
        axial_freq_window=[1, 1],
        axial_ff_mult=1,
        axial_differential=False,
        axial_min_channels=999,
    )

    assert model.axial_replaced_modules == 0
    assert not any(
        isinstance(layer, model_module.AxialLocalAttentionResBlock)
        for layer in list(model.down_layers) + list(model.up_layers)
    )


def test_music2latent_sametemporal_replaces_only_bottleneck_1d_convs():
    model_module = importlib.import_module("scripts.music2latent.models")
    audio_module = importlib.import_module("scripts.music2latent.audio")
    model = model_module.Music2Latent_sametemporal(
        audio_processor=audio_module.AudioProcessor(hop_size=16, fac=4, center_pad=False),
        sample_rate=8000,
        base_channels=8,
        layers_list=[1, 1],
        multipliers_list=[1, 2],
        attention_list=[0, 0],
        freq_downsample_list=[0],
        layers_list_encoder=[1, 1],
        attention_list_encoder=[0, 0],
        bottleneck_base_channels=16,
        num_bottleneck_layers=1,
        bottleneck_channels=6,
        cond_channels=8,
        heads=2,
        hop=16,
        data_channels=2,
        frequency_scaling=False,
        mixed_precision=False,
        min_res_dropout=1,
        same_transformer_depth=1,
        same_dim_heads=4,
        same_sliding_window=[1, 1],
        same_ff_mult=1,
    )
    representation = torch.randn(1, 2, 32, 8)
    latents = model.encoder(representation)

    out = model(latents, representation, sigma=1.0)

    assert latents.shape == (1, 6, 4)
    assert out.shape == representation.shape
    assert torch.isfinite(out).all()
    assert isinstance(model.encoder.conv_inp, torch.nn.Conv2d)
    assert isinstance(model.encoder.bottleneck_layers[0], model_module.ChannelLinear1d)
    assert isinstance(model.encoder.conv_out, model_module.ChannelLinear1d)
    assert isinstance(model.decoder.conv_inp, model_module.ChannelLinear1d)
    assert isinstance(model.decoder.conv_out_bottleneck, model_module.ChannelLinear1d)
    assert any(
        isinstance(layer, model_module.SameTemporalResBlock1d)
        for layer in model.encoder.bottleneck_layers
    )
    assert any(
        isinstance(layer, model_module.SameTemporalResBlock1d)
        for layer in model.decoder.bottleneck_layers
    )
    assert all(
        not isinstance(layer, model_module.SameTemporalResBlock1d)
        for layer in list(model.down_layers) + list(model.up_layers)
    )


def test_training_wrapper_reports_nonfinite_gradients(monkeypatch):
    _install_fake_residual_vq(monkeypatch)
    model = _small_model()
    wrapper_module = importlib.import_module("scripts.music2latent.wrapper")
    wrapper = wrapper_module.Music2LatentTrainingWrapper(model=model)
    first_param = next(wrapper.model.parameters())
    first_param.grad = torch.full_like(first_param, float("nan"))
    wrapper.trainer = SimpleNamespace(global_step=12)

    with pytest.raises(FloatingPointError, match="gradient"):
        wrapper.on_after_backward()


def test_hydra_experiment_instantiates_music2latent_components(monkeypatch):
    _install_fake_residual_vq(monkeypatch)
    with initialize_config_dir(
        config_dir="/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/Mossland/scripts/configs",
        version_base=None,
    ):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=music2latent",
                "model.base_channels=8",
                "model.layers_list=[1,1]",
                "model.layers_list_encoder=[1,1]",
                "model.multipliers_list=[1,2]",
                "model.attention_list=[0,0]",
                "model.attention_list_encoder=[0,0]",
                "model.freq_downsample_list=[0]",
                "model.bottleneck_base_channels=16",
                "model.num_bottleneck_layers=1",
                "model.bottleneck_channels=6",
                "model.cond_channels=8",
                "model.hop=16",
                "model.data_length=4",
                "model.sample_rate=8000",
                "model.audio_channels=2",
                "model.use_rvq=true",
                "model.rvq_num_quantizers=2",
                "model.rvq_codebook_size=8",
                "data.dataset.sample_size=112",
                "data.train_batch_size=1",
                "data.val_split=null",
                "trainer.devices=1",
            ],
        )

    model = instantiate(cfg.model)
    wrapper = instantiate(cfg.wrapper, model=model)

    assert model.use_rvq is True
    assert model.rvq_num_quantizers == 2
    assert model.rvq.kwargs["quantize_dropout"] is False
    assert model.rvq.kwargs["sync_codebook"] is True
    assert cfg.data.dataset._target_ == "scripts.data.datasets.SampleDataset"
    assert cfg.data.dataset.num_channels == 2
    assert cfg.data.dataset.index_file is not None
    assert str(cfg.data.dataset.index_file).startswith(str(cfg.data.dataset.dirs[0]))
    assert cfg.data.dataset.max_duration_seconds == 300
    assert wrapper.model is model
    assert wrapper.rvq_dropout_prob == 0.8
    assert wrapper.rvq_warmup_steps == 1000
    assert wrapper.rvq_loss_weight == 0.1


def test_music2latent_0p5b_stereo_keeps_original_latent_width():
    with initialize_config_dir(
        config_dir="/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/Mossland/scripts/configs",
        version_base=None,
    ):
        cfg = compose(
            config_name="train",
            overrides=["experiment=music2latent_rvq_0p5b_stereo"],
        )

    assert cfg.model.base_channels == 192
    assert cfg.model.bottleneck_base_channels == 1536
    assert cfg.model.bottleneck_channels == 64
    assert cfg.model.data_channels == 4


def test_sample_dataset_lazily_filters_overlong_audio(tmp_path, monkeypatch):
    datasets = importlib.import_module("scripts.data.datasets")
    root = tmp_path / "audio"
    root.mkdir()
    long_file = root / "long.mp3"
    short_file = root / "short.mp3"
    long_file.write_bytes(b"long")
    short_file.write_bytes(b"short")
    index_file = root / "index.txt"
    index_file.write_text(f"{long_file}\n{short_file}\n", encoding="utf-8")
    probed = []
    loaded = []

    def fake_run(cmd, capture_output, text, timeout, check):
        path = cmd[-1]
        probed.append(path)
        stdout = "301.0\n" if path == str(long_file) else "120.0\n"
        return SimpleNamespace(returncode=0, stdout=stdout)

    def fake_load_file(self, filename):
        loaded.append(filename)
        return torch.ones(2, 16)

    monkeypatch.setattr(datasets.subprocess, "run", fake_run)
    monkeypatch.setattr(datasets.SampleDataset, "load_file", fake_load_file)

    dataset = datasets.SampleDataset(
        dirs=[root],
        sample_size=8,
        sample_rate=8000,
        random_crop=False,
        audio_cache_dir=None,
        index_file=index_file,
        max_duration_seconds=300,
    )

    audio, info = dataset[0]

    assert probed == [str(long_file), str(short_file)]
    assert loaded == [str(short_file)]
    assert dataset.filenames == [str(short_file)]
    assert info["path"] == str(short_file)
    assert audio.shape == (2, 8)


def test_training_callback_exports_mp3_demo(monkeypatch, tmp_path):
    _install_fake_residual_vq(monkeypatch)
    saved = []

    def fake_save(path, tensor, sample_rate, **kwargs):
        saved.append((path, tuple(tensor.shape), sample_rate, kwargs.get("format")))

    wrapper_module = importlib.import_module("scripts.music2latent.wrapper")
    monkeypatch.setattr(wrapper_module.torchaudio, "save", fake_save)
    model = _small_model(use_rvq=True, rvq_num_quantizers=2, rvq_codebook_size=8)
    wrapper = wrapper_module.Music2LatentTrainingWrapper(
        model=model,
        use_ema=False,
        rvq_dropout_prob=0.0,
    )
    wrapper.ema = types.SimpleNamespace(ema_model=copy.deepcopy(model))
    callback = wrapper_module.Music2LatentTrainingCallback(
        demo_dir=tmp_path,
        demo_num=1,
        demo_every=1,
        sample_rate=8000,
        use_ema=True,
    )
    trainer = types.SimpleNamespace(global_step=1, global_rank=0)
    batch = torch.randn(1, 2, 144)

    callback.on_train_batch_end(trainer, wrapper, None, batch, 0)

    assert len(saved) == 2
    paths = [str(item[0]) for item in saved]
    assert any("_raw_rank0_" in path for path in paths)
    assert any("_ema_rank0_" in path for path in paths)
    for path, shape, sample_rate, fmt in saved:
        assert str(path).endswith(".mp3")
        assert shape[0] == 2
        assert sample_rate == 8000
        assert fmt == "mp3"


def test_training_callback_reconstructs_demo_from_noise(tmp_path):
    wrapper_module = importlib.import_module("scripts.music2latent.wrapper")

    class ProbeModel(torch.nn.Module):
        sigma_min = 0.002
        sigma_max = 80.0
        rho = 7.0

        def __init__(self):
            super().__init__()
            self.probe = torch.nn.Parameter(torch.zeros(()))
            self.calls = []

        def decoder_features(self, latents):
            return []

        def forward_generator(self, latents, x, sigma, pyramid_latents):
            self.calls.append((x.detach().clone(), sigma.detach().clone()))
            return torch.zeros_like(x)

        def to_waveform(self, representation):
            return representation[:, :2, 0, :]

    model = ProbeModel()
    callback = wrapper_module.Music2LatentTrainingCallback(
        demo_dir=tmp_path,
        demo_num=1,
        demo_every=1,
        sample_rate=8000,
        demo_denoising_steps=1,
    )
    representation = torch.ones(2, 4, 8, 5)
    latents = torch.randn(2, 5, 6)

    callback._reconstruct_from_latents(model, representation, latents)

    assert len(model.calls) == 1
    noisy_input, sigma = model.calls[0]
    assert torch.allclose(sigma, torch.full((2,), model.sigma_max))
    assert not torch.equal(noisy_input, representation)


def test_training_wrapper_mixes_rvq_latents_per_sample_and_keeps_encoder_grad(monkeypatch):
    wrapper_module = importlib.import_module("scripts.music2latent.wrapper")
    model_module = importlib.import_module("scripts.music2latent.models")

    class FakeAudioProcessor:
        fac = 4
        center_pad = False

        def to_representation_encoder(self, batch):
            return batch

    class FakeModel(torch.nn.Module):
        sigma_min = 0.002
        sigma_max = 80.0
        rho = 7.0
        freq_downsample_list = []
        hop = 1
        audio_processor = FakeAudioProcessor()
        has_quantizer = True

        def __init__(self):
            super().__init__()
            self.probe = torch.nn.Parameter(torch.zeros(()))
            self.detach_args = []
            self.latent_overrides = []

        def quantize_representation(self, representation, detach_encoder=True, n_quantizers=None):
            self.detach_args.append(detach_encoder)
            batch_size = representation.shape[0]
            continuous = torch.zeros(
                batch_size,
                1,
                1,
                dtype=representation.dtype,
                device=representation.device,
            ) + self.probe
            discrete = torch.arange(
                1,
                batch_size + 1,
                dtype=representation.dtype,
                device=representation.device,
            ).view(batch_size, 1, 1) + self.probe
            return model_module.QuantizedLatents(
                continuous=continuous,
                discrete=discrete,
                codes=torch.zeros(
                    batch_size,
                    1,
                    1,
                    dtype=torch.long,
                    device=representation.device,
                ),
                projected_latents=discrete,
                commitment_loss=representation.new_tensor(0.0),
                codebook_loss=representation.new_tensor(0.0),
                distill_loss=representation.new_tensor(0.0),
            )

        def forward(self, latents, x, sigma=None, latent_override=None):
            self.latent_overrides.append(latent_override.detach().clone())
            return x * 0.0

    model = FakeModel()
    wrapper = wrapper_module.Music2LatnetTrainingWrapper(
        model=model,
        use_ema=False,
        rvq_latent_train_prob=0.5,
        rvq_detach_encoder=False,
        train_n_quantizers=8,
        consistency_step_total_steps=10,
        sigma_sampling="uniform",
    )
    logged = {}

    def fake_log(name, value, **kwargs):
        logged[name] = value.detach() if torch.is_tensor(value) else value

    def fake_rand(*args, **kwargs):
        if args == (2,):
            return torch.tensor([0.0, 1.0], device=kwargs.get("device"))
        return torch.ones(*args, **kwargs) * 0.5

    monkeypatch.setattr(wrapper, "log", fake_log)
    monkeypatch.setattr(torch, "rand", fake_rand)

    loss = wrapper.training_step((torch.ones(2, 1, 1, 1), {}), 0)

    assert torch.isfinite(loss)
    assert model.detach_args == [False]
    assert torch.equal(model.latent_overrides[0][:, 0, 0], torch.tensor([1.0, 0.0]))
    assert torch.equal(model.latent_overrides[1][:, 0, 0], torch.tensor([1.0, 0.0]))
    assert logged["rvq/n_quantizers"] == 8.0
    assert torch.equal(logged["latent/source_discrete"], torch.tensor(0.5))
