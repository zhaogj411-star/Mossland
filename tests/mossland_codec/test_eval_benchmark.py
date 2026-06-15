import argparse
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch
import torchaudio


manifest = importlib.import_module("scripts.mossland-codec.eval_benchmark.manifest")
metrics = importlib.import_module("scripts.mossland-codec.eval_benchmark.metrics")
fad_backends = importlib.import_module("scripts.mossland-codec.eval_benchmark.fad_backends")
eval_run = importlib.import_module("scripts.mossland-codec.eval_benchmark.run")
aggregate_results = importlib.import_module("scripts.mossland-codec.eval_benchmark.aggregate_results")
build_musdb = importlib.import_module("scripts.mossland-codec.eval_benchmark.build_musdb_manifest")
build_musiccaps = importlib.import_module("scripts.mossland-codec.eval_benchmark.build_musiccaps_manifest")
build_audio_tasks = importlib.import_module(
    "scripts.mossland-codec.eval_benchmark.build_audio_task_manifest"
)
infer = importlib.import_module("scripts.mossland-codec.eval_benchmark.infer")
codec_reconstruct_infer = importlib.import_module(
    "scripts.mossland-codec.eval_benchmark.codec_reconstruct_infer"
)
baseline_common = importlib.import_module("scripts.mossland-codec.eval_benchmark.baselines.common")
encodec_baseline = importlib.import_module("scripts.mossland-codec.eval_benchmark.baselines.encodec_baseline")
dac_baseline = importlib.import_module("scripts.mossland-codec.eval_benchmark.baselines.dac_baseline")
codicodec_baseline = importlib.import_module(
    "scripts.mossland-codec.eval_benchmark.baselines.codicodec_baseline"
)
diffstereo_baseline = importlib.import_module(
    "scripts.mossland-codec.eval_benchmark.baselines.diffstereo_baseline"
)
sr_baseline = importlib.import_module("scripts.mossland-codec.eval_benchmark.baselines.sr_baseline")
nuwave2_baseline = importlib.import_module(
    "scripts.mossland-codec.eval_benchmark.baselines.nuwave2_baseline"
)
flowhigh_baseline = importlib.import_module(
    "scripts.mossland-codec.eval_benchmark.baselines.flowhigh_baseline"
)
snac_baseline = importlib.import_module("scripts.mossland-codec.eval_benchmark.baselines.snac_baseline")
separation_baseline = importlib.import_module(
    "scripts.mossland-codec.eval_benchmark.baselines.separation_baseline"
)
openunmix_baseline = importlib.import_module(
    "scripts.mossland-codec.eval_benchmark.baselines.openunmix_baseline"
)
msst_baseline = importlib.import_module(
    "scripts.mossland-codec.eval_benchmark.baselines.msst_baseline"
)
wavtokenizer_baseline = importlib.import_module(
    "scripts.mossland-codec.eval_benchmark.baselines.wavtokenizer_baseline"
)
opus_baseline = importlib.import_module("scripts.mossland-codec.eval_benchmark.baselines.opus_baseline")
download_musiccaps_audio = importlib.import_module(
    "scripts.mossland-codec.eval_benchmark.download_musiccaps_audio"
)
download_musiccaps_hf_audio = importlib.import_module(
    "scripts.mossland-codec.eval_benchmark.download_musiccaps_hf_audio"
)
download_public_dataset = importlib.import_module(
    "scripts.mossland-codec.eval_benchmark.download_public_dataset"
)
visqol_eval = importlib.import_module("scripts.mossland-codec.eval_benchmark.visqol_eval")
museval_eval = importlib.import_module("scripts.mossland-codec.eval_benchmark.museval_eval")
generate_tables = importlib.import_module("scripts.mossland-codec.eval_benchmark.generate_tables")


def test_manifest_reads_jsonl_with_relative_paths(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"placeholder")
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        '{"item_id":"a","task_id":"reconstruct","source_path":"audio.wav","duration_seconds":1}\n',
        encoding="utf-8",
    )

    items = manifest.read_manifest(path)

    assert items[0].item_id == "a"
    assert items[0].task_id == "reconstruct"
    assert items[0].source_path == audio
    assert items[0].duration_seconds == 1.0


def test_si_sdr_is_high_for_identical_audio():
    torch.manual_seed(0)
    audio = torch.randn(2, 2048)

    value = metrics.si_sdr(audio, audio)

    assert value > 70.0


def test_stereo_metrics_detect_mono_duplicate():
    mono = torch.randn(2048)
    stereo = torch.stack([mono, mono], dim=0)

    assert metrics.stereo_width(stereo) == 0.0
    assert math.isclose(metrics.channel_correlation(stereo), 1.0, rel_tol=1e-6)


def test_frechet_distance_zero_for_same_features():
    features = torch.randn(4, 8)

    assert metrics.frechet_distance(features, features) < 1e-8


def test_pair_metrics_adds_super_resolution_bands():
    reference = torch.randn(2, 4096)
    prediction = reference * 0.9

    values = metrics.pair_metrics(
        prediction,
        reference,
        sample_rate=44100,
        task_id="super_resolution",
        low_sample_rate=16000,
    )

    assert "lsd_lf" in values
    assert "lsd_hf" in values
    assert "hf_energy_ratio" in values


def test_pair_metrics_adds_mir_eval_bss_for_separation():
    if importlib.util.find_spec("mir_eval") is None:
        pytest.skip("mir_eval is not installed in this Python environment")
    torch.manual_seed(0)
    reference = torch.randn(2, 4096)
    prediction = reference + 0.01 * torch.randn(2, 4096)

    values = metrics.pair_metrics(
        prediction,
        reference,
        sample_rate=44100,
        task_id="separate_vocals",
    )

    assert "mir_eval_bss_images_sdr_db" in values
    assert "mir_eval_bss_images_isr_db" in values
    assert "mir_eval_bss_images_sir_db" in values
    assert "mir_eval_bss_images_sar_db" in values


def test_mel_proxy_fad_backend_returns_named_embeddings():
    backend = fad_backends.build_fad_backend("mel_proxy", device=None)
    audio = torch.randn(2, 4096)

    result = backend.pair(audio, audio, sample_rate=44100)

    assert result.metric_name == "fad_mel_proxy"
    assert result.backend_name == "mel_log_stats_proxy"
    assert result.reference_embedding.shape == result.prediction_embedding.shape
    assert result.reference_embedding.numel() > 0


def test_vggish_fad_backend_uses_mean_frame_embedding(monkeypatch, tmp_path):
    class FakeVggish(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU())

        def forward(self, waveform, sample_rate):
            assert sample_rate == 16000
            assert waveform.shape[0] >= 16000
            return torch.tensor([[1.0, 3.0], [5.0, 7.0]])

    def fake_load(*args, **kwargs):
        return FakeVggish()

    monkeypatch.setattr(torch.hub, "load", fake_load)
    backend = fad_backends.VggishFadBackend(
        model_name=str(tmp_path),
        cache_dir=tmp_path / "cache",
        device="cpu",
    )

    embedding = backend.embed(torch.zeros(2, 1024), sample_rate=44100)

    assert backend.metric_name == "fad_vggish"
    torch.testing.assert_close(embedding, torch.tensor([3.0, 5.0]))


def test_summarize_fad_groups_by_task_and_low_sample_rate():
    rows = [
        {
            "task_id": "super_resolution",
            "low_sample_rate": 16000,
            "reference_fad_embedding": [0.0, 1.0],
            "prediction_fad_embedding": [0.1, 1.1],
        },
        {
            "task_id": "super_resolution",
            "low_sample_rate": 16000,
            "reference_fad_embedding": [1.0, 0.0],
            "prediction_fad_embedding": [1.1, 0.1],
        },
    ]

    summary = fad_backends.summarize_fad(rows, "fad_test")

    assert "super_resolution@16000" in summary
    assert summary["super_resolution@16000"]["count"] == 2.0
    assert summary["super_resolution@16000"]["fad_test"] >= 0.0
    assert rows[0]["reference_fad_embedding"] == [0.0, 1.0]


def test_eval_run_select_items_supports_sharding_and_max_items(tmp_path):
    items = [
        manifest.EvalItem(
            item_id=f"item-{index}",
            task_id="reconstruct",
            source_path=tmp_path / f"item-{index}.wav",
        )
        for index in range(10)
    ]

    selected = eval_run.select_items(items, num_shards=3, shard_id=1, max_items=2)

    assert [item.item_id for item in selected] == ["item-1", "item-4"]


def test_eval_run_select_items_rejects_invalid_shard(tmp_path):
    items = [
        manifest.EvalItem(
            item_id="item-0",
            task_id="reconstruct",
            source_path=tmp_path / "item-0.wav",
        )
    ]

    with pytest.raises(ValueError, match="--shard-id"):
        eval_run.select_items(items, num_shards=2, shard_id=2)


def test_aggregate_results_merges_rows_and_recomputes_fad(tmp_path):
    rows = [
        {
            "item_id": "a",
            "task_id": "reconstruct",
            "metrics": {"snr_db": 1.0},
            "fad_metric_name": "fad_test",
            "reference_fad_embedding": [0.0, 1.0],
            "prediction_fad_embedding": [0.1, 1.1],
        },
        {
            "item_id": "b",
            "task_id": "reconstruct",
            "metrics": {"snr_db": 3.0},
            "fad_metric_name": "fad_test",
            "reference_fad_embedding": [1.0, 0.0],
            "prediction_fad_embedding": [1.1, 0.1],
        },
    ]
    first = tmp_path / "shard0.jsonl"
    second = tmp_path / "shard1.jsonl"
    manifest.write_jsonl([rows[0]], first)
    manifest.write_jsonl([rows[1]], second)

    summary = aggregate_results.aggregate_results([first, second], tmp_path / "merged")

    assert summary["reconstruct"]["count"] == 2.0
    assert summary["reconstruct"]["snr_db/mean"] == 2.0
    assert summary["reconstruct"]["fad_test"] >= 0.0
    assert (tmp_path / "merged" / "results.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert (tmp_path / "merged" / "summary.md").exists()


def test_generate_tables_reads_summary_and_visqol_csv(tmp_path):
    summary_dir = tmp_path / "runs" / "example" / "eval_clap"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps({"reconstruct": {"count": 2.0, "fad_clap": 0.25}}),
        encoding="utf-8",
    )
    assert (
        generate_tables.metric_from_summary(
            tmp_path,
            generate_tables.SummaryRef("runs/example/eval_clap/summary.json", "reconstruct"),
            "fad_clap",
        )
        == 0.25
    )

    visqol_dir = tmp_path / "runs" / "visqol" / "model"
    visqol_dir.mkdir(parents=True)
    (visqol_dir / "visqol_results.csv").write_text("moslqo\n1.0\n3.0\n", encoding="utf-8")
    visqol = generate_tables.read_visqol_summary(tmp_path, "runs/visqol/model")
    assert visqol["count"] == 2.0
    assert visqol["visqol_moslqo/mean"] == 2.0

    generate_tables.write_table(
        [generate_tables.TableRow(label="row", data={"model": "m", "fad_clap": 0.25})],
        ["model", "fad_clap"],
        tmp_path / "tables" / "example",
    )
    assert (tmp_path / "tables" / "example.csv").exists()
    assert "| m | 0.25 |" in (tmp_path / "tables" / "example.md").read_text(encoding="utf-8")


def test_build_musdb_manifest_writes_accompaniment(tmp_path):
    track = tmp_path / "musdb18hq" / "test" / "Track A"
    track.mkdir(parents=True)
    sample_rate = 44100
    stems = {
        "vocals": torch.full((2, 32), 0.1),
        "drums": torch.full((2, 32), 0.2),
        "bass": torch.full((2, 32), 0.3),
        "other": torch.full((2, 32), 0.4),
    }
    mixture = sum(stems.values()).clamp(-1, 1)
    torchaudio.save(str(track / "mixture.wav"), mixture, sample_rate)
    for name, audio in stems.items():
        torchaudio.save(str(track / f"{name}.wav"), audio, sample_rate)

    rows = build_musdb.build_rows(
        root=tmp_path / "musdb18hq",
        split="test",
        sample_rate=sample_rate,
        max_tracks=1,
        duration_seconds=10.0,
        accompaniment_root=tmp_path / "derived",
    )

    assert [row["task_id"] for row in rows] == ["separate_vocals", "separate_accompaniment"]
    accompaniment_path = Path(rows[1]["accompaniment_path"])
    assert accompaniment_path.exists()
    accompaniment, sr = torchaudio.load(str(accompaniment_path))
    assert sr == sample_rate
    torch.testing.assert_close(accompaniment, (stems["drums"] + stems["bass"] + stems["other"]).clamp(-1, 1))


def test_build_musdb_manifest_selects_non_silent_window(tmp_path):
    track = tmp_path / "musdb18hq" / "test" / "Track B"
    track.mkdir(parents=True)
    sample_rate = 8000
    length = sample_rate * 3
    vocals = torch.zeros(2, length)
    vocals[:, sample_rate : 2 * sample_rate] = 0.1
    drums = torch.zeros(2, length)
    drums[:, sample_rate : 2 * sample_rate] = 0.2
    bass = torch.zeros(2, length)
    other = torch.zeros(2, length)
    mixture = (vocals + drums + bass + other).clamp(-1, 1)
    for name, audio in {
        "mixture": mixture,
        "vocals": vocals,
        "drums": drums,
        "bass": bass,
        "other": other,
    }.items():
        torchaudio.save(str(track / f"{name}.wav"), audio, sample_rate)

    rows = build_musdb.build_rows(
        root=tmp_path / "musdb18hq",
        split="test",
        sample_rate=sample_rate,
        max_tracks=1,
        duration_seconds=1.0,
        accompaniment_root=tmp_path / "derived",
        select_non_silent=True,
        window_hop_seconds=1.0,
        min_source_rms=1e-4,
    )

    assert len(rows) == 2
    assert {row["start_seconds"] for row in rows} == {1.0}
    assert rows[0]["metadata"]["window_selection"] == "first_non_silent_vocals_and_accompaniment"


def test_build_musiccaps_manifest_can_mark_audio_as_preclipped(tmp_path):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    (audio_root / "abc.wav").write_bytes(b"placeholder")
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        '{"ytid":"abc","start_s":30,"end_s":40,"caption":"clip"}\n',
        encoding="utf-8",
    )

    rows, missing = build_musiccaps.build_rows(
        metadata_path=metadata,
        audio_root=audio_root,
        sample_rate=44100,
        max_items=0,
        audio_is_clipped=True,
    )

    assert not missing
    assert rows[0]["start_seconds"] == 0.0
    assert rows[0]["duration_seconds"] == 10.0
    assert rows[0]["metadata"]["original_start_s"] == 30.0


def test_build_audio_task_manifest_derives_sr_and_stereo_rows(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"placeholder")
    source = manifest.EvalItem(
        item_id="clip",
        task_id="reconstruct",
        source_path=audio,
        reference_path=audio,
        duration_seconds=10.0,
        sample_rate=44100,
        metadata={"dataset": "unit"},
    )

    rows = build_audio_tasks.build_rows(
        [source],
        tasks=("reconstruct", "super_resolution", "mono_to_stereo"),
        sr_rates=(16000, 24000),
        stereo_seeds=(0, 1),
        max_items=0,
        duration_seconds=None,
    )

    assert [row["task_id"] for row in rows] == [
        "reconstruct",
        "super_resolution",
        "super_resolution",
        "mono_to_stereo",
        "mono_to_stereo",
    ]
    assert rows[1]["low_sample_rate"] == 16000
    assert rows[2]["low_sample_rate"] == 24000
    assert rows[3]["seed"] == 0
    assert rows[4]["seed"] == 1
    assert rows[0]["metadata"]["derived_from_item_id"] == "clip"


def test_public_dataset_no_proxy_session_ignores_environment():
    session = download_public_dataset.build_session(no_proxy=True)

    assert session.trust_env is False


def test_public_dataset_verify_size(tmp_path):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"abc")

    assert download_public_dataset.verify_size(path, 3)
    assert not download_public_dataset.verify_size(path, 4)
    assert download_public_dataset.verify_size(path, None)


def test_public_dataset_download_rejects_short_file(tmp_path):
    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"abc"

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    with pytest.raises(RuntimeError, match="downloaded size mismatch"):
        download_public_dataset.download_file(
            Session(),
            "https://example.test/archive.zip",
            tmp_path / "archive.zip",
            expected_size=4,
        )


def test_generate_prediction_reuses_existing_output(tmp_path):
    item = manifest.EvalItem(
        item_id="clip",
        task_id="reconstruct",
        source_path=tmp_path / "source.wav",
        sample_rate=44100,
        seed=0,
    )
    output_path = infer.prediction_path_for_item(item, tmp_path / "predictions")
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"existing")

    class Model:
        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(()))

        def generate_waveform(self, *args, **kwargs):
            raise AssertionError("existing prediction should be reused")

    result = infer.generate_prediction(Model(), item, tmp_path / "predictions")

    assert result == output_path
    assert output_path.read_bytes() == b"existing"


def test_codec_reconstruct_infer_uses_parallel_decode_and_crops_to_source(tmp_path):
    source_path = tmp_path / "source.wav"
    sample_rate = 44100
    source = torch.linspace(-0.5, 0.5, 1024).repeat(2, 1)
    torchaudio.save(str(source_path), source, sample_rate)
    item = manifest.EvalItem(
        item_id="clip",
        task_id="reconstruct",
        source_path=source_path,
        sample_rate=sample_rate,
        seed=0,
    )

    class Codec:
        max_batch_size_encode = 3
        max_batch_size_decode = 5

        def encode(self, audio, **kwargs):
            assert audio.shape == source.shape
            assert kwargs["desired_channels"] == 64
            return torch.ones(2, 4, 8)

        def decode(self, latents, **kwargs):
            assert latents.shape == (2, 4, 8)
            assert kwargs["mode"] == "parallel"
            assert kwargs["task_id"] == "reconstruct"
            return torch.ones(2, 1200)

    prediction_path, metadata = codec_reconstruct_infer.generate_codec_reconstruction(
        Codec(),
        item,
        tmp_path / "predictions",
    )

    prediction, sr = torchaudio.load(str(prediction_path))
    assert sr == sample_rate
    assert prediction.shape == source.shape
    assert metadata["inference_mode"] == "codec_parallel_reconstruct"
    assert metadata["prediction_frames"] == source.shape[-1]


def test_musiccaps_hf_row_helpers(tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"wav")

    assert download_musiccaps_hf_audio.row_youtube_id({"youtube_id": "abc"}) == "abc"
    assert download_musiccaps_hf_audio.row_audio_bytes({"audio": {"bytes": b"data"}}) == b"data"
    assert download_musiccaps_hf_audio.row_audio_bytes({"audio": {"path": str(audio_path)}}) == b"wav"


def test_musiccaps_timeout_preserves_bytes_output(monkeypatch):
    import subprocess

    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(["yt-dlp"], timeout=1, output=b"stdout", stderr=b"stderr")

    monkeypatch.setattr(subprocess, "run", timeout_run)

    with pytest.raises(TimeoutError, match="stdout"):
        download_musiccaps_audio.run_command(["yt-dlp"], env={}, timeout_seconds=1)


def test_baseline_prediction_path_groups_by_baseline_and_task(tmp_path):
    item = manifest.EvalItem(
        item_id="clip",
        task_id="reconstruct",
        source_path=tmp_path / "source.wav",
        seed=3,
    )

    path = baseline_common.baseline_prediction_path(tmp_path, "encodec", item)

    assert path == tmp_path / "encodec" / "reconstruct" / "clip_seed3.wav"


def test_write_predicted_manifest_uses_absolute_prediction_paths(tmp_path):
    item = manifest.EvalItem(
        item_id="clip",
        task_id="reconstruct",
        source_path=tmp_path / "source.wav",
    )
    prediction = tmp_path / "pred.wav"
    output = tmp_path / "manifest.jsonl"

    baseline_common.write_predicted_manifest([item], [prediction], output)
    read_back = manifest.read_manifest(output)

    assert read_back[0].prediction_path == prediction.resolve()


def test_nuwave2_super_resolution_items_filters_task(tmp_path):
    audio = tmp_path / "audio.wav"
    torchaudio.save(str(audio), torch.zeros(1, 64), 48000)
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"item_id":"a","task_id":"reconstruct","source_path":"audio.wav"}',
                '{"item_id":"b","task_id":"super_resolution","source_path":"audio.wav","low_sample_rate":16000}',
                '{"item_id":"c","task_id":"super_resolution","source_path":"audio.wav","low_sample_rate":24000}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items = nuwave2_baseline.super_resolution_items(path, max_items=1)

    assert [item.item_id for item in items] == ["b"]


def test_nuwave2_band_mask_marks_low_frequency_bins():
    hparams = argparse.Namespace(audio=argparse.Namespace(filter_length=1024, sampling_rate=48000))

    band = nuwave2_baseline.band_mask(16000, hparams, "cpu")

    assert band.shape == (1, 513)
    assert int(band.sum().item()) == 171
    assert band[0, 0].item() == 1
    assert band[0, -1].item() == 0


def test_nuwave2_istft_complex_compat_converts_real_imag_tensor(monkeypatch):
    calls = {}

    def fake_istft(input_tensor, *args, **kwargs):
        calls["is_complex"] = torch.is_complex(input_tensor)
        return torch.zeros(4)

    monkeypatch.setattr(torch, "istft", fake_istft)
    real_imag = torch.zeros(3, 5, 2)

    with nuwave2_baseline.istft_complex_compat():
        result = torch.istft(real_imag, n_fft=8)

    assert result.shape == (4,)
    assert calls["is_complex"] is True


def test_encodec_reconstruct_manifest_filters_reconstruct(tmp_path):
    audio = tmp_path / "audio.wav"
    torchaudio.save(str(audio), torch.zeros(2, 32), 44100)
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"item_id":"a","task_id":"reconstruct","source_path":"audio.wav"}',
                '{"item_id":"b","task_id":"super_resolution","source_path":"audio.wav","low_sample_rate":16000}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items = encodec_baseline.reconstruct_manifest_items(path, max_items=0)

    assert [item.item_id for item in items] == ["a"]


def test_encodec_padding_matches_segment_grid():
    class Model:
        segment_length = 48000
        segment_stride = 47520

    audio = torch.zeros(2, 48000 + 47520 + 160)

    padded, original_length = encodec_baseline.pad_for_encodec_segments(audio, Model())

    assert original_length == audio.shape[-1]
    assert padded.shape[-1] == 48000 + 2 * 47520


def test_dac_official_weights_url_uses_release_maps():
    class Utils:
        __MODEL_LATEST_TAGS__ = {("44khz", "8kbps"): "0.0.1"}
        __MODEL_URLS__ = {("44khz", "0.0.1", "8kbps"): "https://example.test/weights.pth"}

    class DacModule:
        utils = Utils

    url = dac_baseline._official_weights_url(DacModule, "44khz", "8kbps", "latest")

    assert url == "https://example.test/weights.pth"


def test_opus_manifest_filters_reconstruct(tmp_path):
    audio = tmp_path / "audio.wav"
    torchaudio.save(str(audio), torch.zeros(2, 32), 44100)
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"item_id":"a","task_id":"reconstruct","source_path":"audio.wav"}',
                '{"item_id":"b","task_id":"super_resolution","source_path":"audio.wav","low_sample_rate":16000}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items = opus_baseline.opus_manifest_items(path, max_items=0)

    assert [item.item_id for item in items] == ["a"]


def test_opus_ffmpeg_command_uses_requested_bitrate(tmp_path):
    commands = opus_baseline.ffmpeg_encode_decode_command(
        tmp_path / "in.wav",
        tmp_path / "tmp.opus",
        tmp_path / "out.wav",
        bitrate_kbps=14,
        sample_rate=44100,
    )

    assert commands[0][commands[0].index("-b:a") + 1] == "14k"
    assert commands[0][commands[0].index("-c:a") + 1] == "libopus"
    assert commands[1][commands[1].index("-ar") + 1] == "44100"


def test_snac_manifest_filters_reconstruct(tmp_path):
    audio = tmp_path / "audio.wav"
    torchaudio.save(str(audio), torch.zeros(2, 32), 44100)
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"item_id":"a","task_id":"reconstruct","source_path":"audio.wav"}',
                '{"item_id":"b","task_id":"mono_to_stereo","source_path":"audio.wav"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items = snac_baseline.snac_manifest_items(path, max_items=0)

    assert [item.item_id for item in items] == ["a"]


def test_snac_helpers_downmix_and_fit_length():
    audio = torch.tensor([[1.0, -1.0, 0.5], [-1.0, 1.0, 0.5]])

    mono = snac_baseline._downmix_mono(audio)
    padded = snac_baseline._fit_length(mono, 5)

    torch.testing.assert_close(mono, torch.tensor([[0.0, 0.0, 0.5]]))
    assert padded.shape == (1, 5)
    torch.testing.assert_close(snac_baseline._fit_length(padded, 2), torch.zeros(1, 2))


def test_snac_model_sample_rate_accepts_official_field_name():
    class Model:
        sampling_rate = 32000

    assert snac_baseline._model_sample_rate(Model()) == 32000


def test_codicodec_manifest_filters_reconstruct_or_reference_audio(tmp_path):
    audio = tmp_path / "audio.wav"
    reference = tmp_path / "reference.wav"
    torchaudio.save(str(audio), torch.zeros(2, 32), 44100)
    torchaudio.save(str(reference), torch.zeros(2, 32), 44100)
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"item_id":"a","task_id":"reconstruct","source_path":"audio.wav"}',
                (
                    '{"item_id":"b","task_id":"super_resolution","source_path":"audio.wav",'
                    '"reference_path":"reference.wav","low_sample_rate":16000}'
                ),
                '{"item_id":"c","task_id":"mono_to_stereo","source_path":"audio.wav"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items = codicodec_baseline.codicodec_manifest_items(path, max_items=0)

    assert [item.item_id for item in items] == ["a", "b"]


def test_diffstereo_detects_lfs_pointer(tmp_path):
    pointer = tmp_path / "model_epoch_80000.pt"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:b32c13d8b54b15fab6c4c7e65da924c8349ae247b1f2bfe5b0a7b7549a4143e0\n"
        "size 402045986\n",
        encoding="utf-8",
    )

    assert diffstereo_baseline._is_lfs_pointer(pointer)


def test_diffstereo_manifest_filters_mono_to_stereo(tmp_path):
    audio = tmp_path / "audio.wav"
    torchaudio.save(str(audio), torch.zeros(2, 32), 44100)
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"item_id":"a","task_id":"mono_to_stereo","source_path":"audio.wav"}',
                '{"item_id":"b","task_id":"reconstruct","source_path":"audio.wav"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items = diffstereo_baseline.mono_to_stereo_items(path, max_items=0)

    assert [item.item_id for item in items] == ["a"]


def test_sr_manifest_filters_super_resolution(tmp_path):
    audio = tmp_path / "audio.wav"
    torchaudio.save(str(audio), torch.zeros(2, 32), 44100)
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"item_id":"a","task_id":"super_resolution","source_path":"audio.wav","low_sample_rate":16000}',
                '{"item_id":"b","task_id":"reconstruct","source_path":"audio.wav"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items = sr_baseline._super_resolution_items(path, max_items=0)

    assert [item.item_id for item in items] == ["a"]


def test_sr_fit_length_pads_and_truncates():
    audio = torch.ones(1, 4)

    assert sr_baseline._fit_length(audio, 2).shape[-1] == 2
    padded = sr_baseline._fit_length(audio, 6)
    assert padded.shape[-1] == 6
    assert padded[..., -2:].sum() == 0


def test_flowhigh_manifest_filters_super_resolution(tmp_path):
    audio = tmp_path / "audio.wav"
    torchaudio.save(str(audio), torch.zeros(2, 32), 48000)
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"item_id":"a","task_id":"reconstruct","source_path":"audio.wav"}',
                '{"item_id":"b","task_id":"super_resolution","source_path":"audio.wav","low_sample_rate":16000}',
                '{"item_id":"c","task_id":"super_resolution","source_path":"audio.wav","low_sample_rate":24000}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items = flowhigh_baseline.super_resolution_items(path, max_items=1)

    assert [item.item_id for item in items] == ["b"]


def test_flowhigh_prepare_input_downmixes_and_downsamples(tmp_path):
    sample_rate = 48000
    source = tmp_path / "source.wav"
    audio = torch.stack([torch.ones(sample_rate), -torch.ones(sample_rate)], dim=0)
    torchaudio.save(str(source), audio, sample_rate)
    item = manifest.EvalItem(
        item_id="clip",
        task_id="super_resolution",
        source_path=source,
        sample_rate=sample_rate,
        low_sample_rate=8000,
    )

    low_audio, sr_in, target_length = flowhigh_baseline.prepare_flowhigh_input(item, 48000)

    assert sr_in == 8000
    assert target_length == 48000
    assert low_audio.shape == (8000,)
    assert abs(float(low_audio.max())) < 1e-6


def test_flowhigh_call_generate_accepts_sr_in_signature():
    class Model:
        def generate(self, audio, sr_in, target_sampling_rate, timestep):
            assert audio.shape == (4,)
            assert sr_in == 16000
            assert target_sampling_rate == 48000
            assert timestep == 1
            return torch.zeros(1, 4)

    result = flowhigh_baseline.call_flowhigh_generate(
        Model(),
        torch.zeros(4).numpy(),
        sr_in=16000,
        target_sample_rate=48000,
        timestep=1,
    )

    assert result.shape == (1, 4)


def test_flowhigh_run_item_writes_fake_model_output_48k_mono(tmp_path):
    sample_rate = 48000
    source = tmp_path / "source.wav"
    audio = torch.stack([torch.full((sample_rate,), 0.25), torch.full((sample_rate,), 0.75)], dim=0)
    torchaudio.save(str(source), audio, sample_rate)
    item = manifest.EvalItem(
        item_id="clip",
        task_id="super_resolution",
        source_path=source,
        sample_rate=sample_rate,
        low_sample_rate=16000,
    )

    class FakeFlowHigh:
        def generate(self, audio, sr, target_sampling_rate, timestep):
            assert audio.shape == (16000,)
            assert sr == 16000
            assert target_sampling_rate == 48000
            assert timestep == 1
            return torch.stack(
                [torch.full((target_sampling_rate + 100,), 0.2), torch.full((target_sampling_rate + 100,), 0.6)],
                dim=0,
            )

    output_path = tmp_path / "out.wav"
    flowhigh_baseline.run_flowhigh_item(
        FakeFlowHigh(),
        item,
        output_path,
        argparse.Namespace(target_sample_rate=48000, timestep=1),
    )

    prediction, sr = torchaudio.load(str(output_path))
    assert sr == 48000
    assert prediction.shape == (1, 48000)
    torch.testing.assert_close(prediction.mean(), torch.tensor(0.4), atol=1e-4, rtol=0)


def test_separation_manifest_filters_source_separation_tasks(tmp_path):
    audio = tmp_path / "audio.wav"
    torchaudio.save(str(audio), torch.zeros(2, 32), 44100)
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"item_id":"a","task_id":"separate_vocals","source_path":"audio.wav"}',
                '{"item_id":"b","task_id":"separate_accompaniment","source_path":"audio.wav"}',
                '{"item_id":"c","task_id":"reconstruct","source_path":"audio.wav"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items = separation_baseline.separation_manifest_items(path, max_items=0)

    assert [item.item_id for item in items] == ["a", "b"]


def test_separation_grouped_items_pairs_same_mixture_window(tmp_path):
    mixture = tmp_path / "mixture.wav"
    torchaudio.save(str(mixture), torch.zeros(2, 32), 44100)
    items = [
        manifest.EvalItem(
            item_id="track__vocals",
            task_id="separate_vocals",
            source_path=mixture,
            start_seconds=1.0,
            duration_seconds=10.0,
        ),
        manifest.EvalItem(
            item_id="track__accompaniment",
            task_id="separate_accompaniment",
            source_path=mixture,
            start_seconds=1.0,
            duration_seconds=10.0,
        ),
    ]

    groups = separation_baseline.grouped_items(items)

    assert len(groups) == 1
    assert sorted(item.task_id for item in groups[0]) == ["separate_accompaniment", "separate_vocals"]


def test_separation_adapter_writes_predictions_with_fake_demucs(monkeypatch, tmp_path):
    sample_rate = 8000
    mixture = tmp_path / "mixture.wav"
    torchaudio.save(str(mixture), torch.ones(2, sample_rate), sample_rate)
    items = [
        manifest.EvalItem(
            item_id="track__vocals",
            task_id="separate_vocals",
            source_path=mixture,
            sample_rate=sample_rate,
        ),
        manifest.EvalItem(
            item_id="track__accompaniment",
            task_id="separate_accompaniment",
            source_path=mixture,
            sample_rate=sample_rate,
        ),
    ]

    def fake_run_demucs_cli(input_wav, output_root, args):
        output_dir = Path(output_root) / args.model / Path(input_wav).stem
        output_dir.mkdir(parents=True)
        torchaudio.save(str(output_dir / "vocals.wav"), torch.full((2, sample_rate), 0.25), sample_rate)
        torchaudio.save(str(output_dir / "no_vocals.wav"), torch.full((2, sample_rate), 0.75), sample_rate)

    monkeypatch.setattr(separation_baseline, "run_demucs_cli", fake_run_demucs_cli)
    args = argparse.Namespace(
        output_dir=tmp_path / "out",
        overwrite=False,
        progress_every=0,
        temp_dir=None,
        keep_temp="",
        clean_demucs_output=False,
        model="htdemucs",
    )

    prediction_paths = separation_baseline.separate_items(items, args)

    assert [path.name for path in prediction_paths] == ["track__vocals_seed0.wav", "track__accompaniment_seed0.wav"]
    vocals, vocals_sr = torchaudio.load(str(prediction_paths[0]))
    accompaniment, accompaniment_sr = torchaudio.load(str(prediction_paths[1]))
    assert vocals_sr == sample_rate
    assert accompaniment_sr == sample_rate
    torch.testing.assert_close(vocals, torch.full((2, sample_rate), 0.25))
    torch.testing.assert_close(accompaniment, torch.full((2, sample_rate), 0.75))


def test_openunmix_manifest_filters_source_separation_tasks(tmp_path):
    audio = tmp_path / "audio.wav"
    torchaudio.save(str(audio), torch.zeros(2, 32), 44100)
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"item_id":"a","task_id":"separate_vocals","source_path":"audio.wav"}',
                '{"item_id":"b","task_id":"separate_accompaniment","source_path":"audio.wav"}',
                '{"item_id":"c","task_id":"reconstruct","source_path":"audio.wav"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items = openunmix_baseline.separation_manifest_items(path, max_items=0)

    assert [item.item_id for item in items] == ["a", "b"]


def test_openunmix_estimate_for_accompaniment_sums_non_vocal_stems():
    estimates = {
        "vocals": torch.full((1, 2, 8), 0.1),
        "drums": torch.full((1, 2, 8), 0.2),
        "bass": torch.full((1, 2, 8), 0.3),
        "other": torch.full((1, 2, 8), 0.4),
    }

    accompaniment = openunmix_baseline.estimate_for_task(estimates, "separate_accompaniment")

    torch.testing.assert_close(accompaniment, torch.full((2, 8), 0.9))


def test_openunmix_adapter_writes_predictions_with_fake_separator(monkeypatch, tmp_path):
    sample_rate = 8000
    mixture = tmp_path / "mixture.wav"
    torchaudio.save(str(mixture), torch.ones(2, sample_rate), sample_rate)
    items = [
        manifest.EvalItem(
            item_id="track__vocals",
            task_id="separate_vocals",
            source_path=mixture,
            sample_rate=sample_rate,
        ),
        manifest.EvalItem(
            item_id="track__accompaniment",
            task_id="separate_accompaniment",
            source_path=mixture,
            sample_rate=sample_rate,
        ),
    ]

    class FakeSeparator:
        pass

    def fake_load_openunmix_separator(args):
        separator = FakeSeparator()
        separator.sample_rate = sample_rate
        return separator

    def fake_run_openunmix_separate(audio, rate, separator, args):
        assert audio.shape == (2, sample_rate)
        assert rate == sample_rate
        assert separator.sample_rate == sample_rate
        return {
            "vocals": torch.full((1, 2, sample_rate), 0.25),
            "drums": torch.full((1, 2, sample_rate), 0.2),
            "bass": torch.full((1, 2, sample_rate), 0.3),
            "other": torch.full((1, 2, sample_rate), 0.25),
        }

    monkeypatch.setattr(openunmix_baseline, "load_openunmix_separator", fake_load_openunmix_separator)
    monkeypatch.setattr(openunmix_baseline, "run_openunmix_separate", fake_run_openunmix_separate)
    args = argparse.Namespace(
        output_dir=tmp_path / "out",
        overwrite=False,
        progress_every=0,
    )

    prediction_paths = openunmix_baseline.separate_items(items, args)

    assert [path.name for path in prediction_paths] == ["track__vocals_seed0.wav", "track__accompaniment_seed0.wav"]
    vocals, vocals_sr = torchaudio.load(str(prediction_paths[0]))
    accompaniment, accompaniment_sr = torchaudio.load(str(prediction_paths[1]))
    assert vocals_sr == sample_rate
    assert accompaniment_sr == sample_rate
    torch.testing.assert_close(vocals, torch.full((2, sample_rate), 0.25))
    torch.testing.assert_close(accompaniment, torch.full((2, sample_rate), 0.75))


def test_msst_manifest_filters_source_separation_tasks(tmp_path):
    audio = tmp_path / "audio.wav"
    torchaudio.save(str(audio), torch.zeros(2, 32), 44100)
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"item_id":"a","task_id":"separate_vocals","source_path":"audio.wav"}',
                '{"item_id":"b","task_id":"separate_accompaniment","source_path":"audio.wav"}',
                '{"item_id":"c","task_id":"reconstruct","source_path":"audio.wav"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items = msst_baseline.separation_manifest_items(path, max_items=0)

    assert [item.item_id for item in items] == ["a", "b"]


def test_msst_estimate_for_accompaniment_sums_musdb_non_vocal_stems():
    estimates = {
        "vocals": torch.full((2, 8), 0.1),
        "bass": torch.full((2, 8), 0.2),
        "drums": torch.full((2, 8), 0.3),
        "other": torch.full((2, 8), 0.4),
    }

    accompaniment = msst_baseline.estimate_for_task(estimates, "separate_accompaniment")

    torch.testing.assert_close(accompaniment, torch.full((2, 8), 0.9))


def test_msst_parse_args_accepts_required_baseline_options(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    output_dir = tmp_path / "out"
    checkpoint = tmp_path / "model.ckpt"
    config = tmp_path / "config.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "msst_baseline.py",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--repo-dir",
            str(tmp_path / "repo"),
            "--model-type",
            "mel_band_roformer",
            "--config-path",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--progress-every",
            "1",
            "--download-missing",
            "--no-proxy",
        ],
    )

    args = msst_baseline.parse_args()

    assert args.model_type == "mel_band_roformer"
    assert args.config_path == str(config)
    assert args.checkpoint == str(checkpoint)
    assert args.device == "cpu"
    assert args.download_missing is True
    assert args.no_proxy is True


def test_msst_known_pretrained_scnet_is_registered():
    assert "scnet" in msst_baseline.SUPPORTED_MODEL_TYPES
    info = msst_baseline.KNOWN_PRETRAINED["scnet"]
    assert info["config_name"] == "config_musdb18_scnet_xl_more_wide_v5.yaml"
    assert info["checkpoint_name"] == "model_scnet_ep_36_sdr_10.0891.ckpt"


def test_msst_adapter_writes_predictions_with_fake_inference(monkeypatch, tmp_path):
    sample_rate = 8000
    mixture = tmp_path / "mixture.wav"
    torchaudio.save(str(mixture), torch.ones(2, sample_rate), sample_rate)
    items = [
        manifest.EvalItem(
            item_id="track__vocals",
            task_id="separate_vocals",
            source_path=mixture,
            sample_rate=sample_rate,
        ),
        manifest.EvalItem(
            item_id="track__accompaniment",
            task_id="separate_accompaniment",
            source_path=mixture,
            sample_rate=sample_rate,
        ),
    ]

    def fake_ensure_config_and_checkpoint(args):
        return tmp_path / "config.yaml", tmp_path / "model.ckpt"

    def fake_run_msst_cli(input_dir, store_dir, config_path, checkpoint_path, args):
        input_wavs = sorted(Path(input_dir).glob("*.wav"))
        assert len(input_wavs) == 1
        output_dir = Path(store_dir) / input_wavs[0].stem
        output_dir.mkdir(parents=True)
        torchaudio.save(str(output_dir / "vocals.wav"), torch.full((2, sample_rate), 0.25), sample_rate)
        torchaudio.save(str(output_dir / "bass.wav"), torch.full((2, sample_rate), 0.2), sample_rate)
        torchaudio.save(str(output_dir / "drums.wav"), torch.full((2, sample_rate), 0.3), sample_rate)
        torchaudio.save(str(output_dir / "other.wav"), torch.full((2, sample_rate), 0.25), sample_rate)

    monkeypatch.setattr(msst_baseline, "ensure_config_and_checkpoint", fake_ensure_config_and_checkpoint)
    monkeypatch.setattr(msst_baseline, "run_msst_cli", fake_run_msst_cli)
    args = argparse.Namespace(
        output_dir=tmp_path / "out",
        overwrite=False,
        progress_every=0,
        temp_dir=None,
        keep_temp="",
        model_type="bs_roformer",
    )

    prediction_paths = msst_baseline.separate_items(items, args)

    assert [path.name for path in prediction_paths] == ["track__vocals_seed0.wav", "track__accompaniment_seed0.wav"]
    vocals, vocals_sr = torchaudio.load(str(prediction_paths[0]))
    accompaniment, accompaniment_sr = torchaudio.load(str(prediction_paths[1]))
    assert vocals_sr == sample_rate
    assert accompaniment_sr == sample_rate
    torch.testing.assert_close(vocals, torch.full((2, sample_rate), 0.25))
    torch.testing.assert_close(accompaniment, torch.full((2, sample_rate), 0.75))


def test_wavtokenizer_manifest_filters_reconstruct(tmp_path):
    audio = tmp_path / "audio.wav"
    torchaudio.save(str(audio), torch.zeros(2, 32), 44100)
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"item_id":"a","task_id":"reconstruct","source_path":"audio.wav"}',
                '{"item_id":"b","task_id":"super_resolution","source_path":"audio.wav","low_sample_rate":16000}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items = wavtokenizer_baseline.reconstruct_manifest_items(path, max_items=0)

    assert [item.item_id for item in items] == ["a"]


def test_wavtokenizer_prepare_model_audio_downmixes_and_resamples():
    audio = torch.stack([torch.ones(4410), -torch.ones(4410)], dim=0)

    prepared = wavtokenizer_baseline.prepare_model_audio(audio, source_sample_rate=44100)

    assert prepared.shape[0] == 1
    assert prepared.shape[-1] == 2400
    assert prepared.abs().max() == 0


def test_wavtokenizer_reconstruct_item_writes_fake_model_output(tmp_path):
    sample_rate = 8000
    source = tmp_path / "source.wav"
    torchaudio.save(str(source), torch.ones(2, sample_rate), sample_rate)
    item = manifest.EvalItem(
        item_id="clip",
        task_id="reconstruct",
        source_path=source,
        sample_rate=sample_rate,
    )

    class FakeWavTokenizer:
        def encode_infer(self, audio, bandwidth_id):
            assert audio.shape == (1, wavtokenizer_baseline.WAVTOKENIZER_SAMPLE_RATE)
            assert bandwidth_id.tolist() == [0]
            return audio * 0.5, torch.zeros(1, 1, dtype=torch.long)

        def decode(self, features, bandwidth_id):
            return features.unsqueeze(0)

    output_path = tmp_path / "out.wav"
    wavtokenizer_baseline.reconstruct_item(
        FakeWavTokenizer(),
        item,
        output_path,
        argparse.Namespace(device="cpu", bandwidth_id=0),
    )

    prediction, sr = torchaudio.load(str(output_path))
    assert sr == sample_rate
    assert prediction.shape == (1, sample_rate)


def test_visqol_select_items_filters_prediction_and_bucket(tmp_path):
    source = tmp_path / "source.wav"
    prediction = tmp_path / "prediction.wav"
    items = [
        manifest.EvalItem(
            item_id="a",
            task_id="super_resolution",
            source_path=source,
            prediction_path=prediction,
            low_sample_rate=16000,
        ),
        manifest.EvalItem(
            item_id="b",
            task_id="super_resolution",
            source_path=source,
            low_sample_rate=24000,
        ),
    ]

    selected = visqol_eval.select_items(items, tasks={"super_resolution@16000"})

    assert [item.item_id for item in selected] == ["a"]
    assert visqol_eval.bucket_key(selected[0]) == "super_resolution@16000"


def test_visqol_select_items_supports_sharding(tmp_path):
    prediction = tmp_path / "prediction.wav"
    items = [
        manifest.EvalItem(
            item_id=f"item-{index}",
            task_id="reconstruct",
            source_path=tmp_path / f"source-{index}.wav",
            prediction_path=prediction,
        )
        for index in range(7)
    ]

    selected = visqol_eval.select_items(items, num_shards=3, shard_id=1)

    assert [item.item_id for item in selected] == ["item-1", "item-4"]


def test_visqol_prepare_pair_writes_48k_mono_wavs(tmp_path):
    reference_path = tmp_path / "reference.wav"
    prediction_path = tmp_path / "prediction.wav"
    sample_rate = 44100
    torchaudio.save(str(reference_path), torch.zeros(2, sample_rate), sample_rate)
    torchaudio.save(str(prediction_path), torch.zeros(2, sample_rate // 2), sample_rate)
    item = manifest.EvalItem(
        item_id="clip",
        task_id="reconstruct",
        source_path=reference_path,
        reference_path=reference_path,
        prediction_path=prediction_path,
        sample_rate=sample_rate,
    )

    reference_wav, degraded_wav = visqol_eval.prepare_visqol_pair(item, tmp_path / "visqol")
    reference, ref_sr = torchaudio.load(str(reference_wav))
    degraded, deg_sr = torchaudio.load(str(degraded_wav))

    assert ref_sr == 48000
    assert deg_sr == 48000
    assert reference.shape[0] == 1
    assert degraded.shape[0] == 1
    assert reference.shape[-1] == degraded.shape[-1]


def test_visqol_run_writes_summary_with_fake_binary(monkeypatch, tmp_path):
    reference_path = tmp_path / "reference.wav"
    prediction_path = tmp_path / "prediction.wav"
    torchaudio.save(str(reference_path), torch.zeros(2, 48000), 48000)
    torchaudio.save(str(prediction_path), torch.zeros(2, 48000), 48000)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "item_id": "clip",
                "task_id": "reconstruct",
                "source_path": str(reference_path),
                "reference_path": str(reference_path),
                "prediction_path": str(prediction_path),
                "sample_rate": 48000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    visqol_bin = tmp_path / "visqol"
    visqol_bin.write_text("fake", encoding="utf-8")

    def fake_run(command, check, text, capture_output):
        results_arg = next(arg for arg in command if arg.startswith("--results_csv="))
        results_path = Path(results_arg.split("=", 1)[1])
        results_path.write_text(
            "reference,degraded,moslqo\nref.wav,deg.wav,4.25\n",
            encoding="utf-8",
        )

        class Completed:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return Completed()

    monkeypatch.setattr(visqol_eval.subprocess, "run", fake_run)

    visqol_eval.run_visqol(
        argparse.Namespace(
            manifest=manifest_path,
            output_dir=tmp_path / "out",
            visqol_bin=visqol_bin,
            similarity_model=None,
            tasks=None,
            max_items=0,
            sample_rate=48000,
            use_speech_mode=False,
            overwrite_wavs=False,
        )
    )

    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))

    assert summary["reconstruct"]["visqol_moslqo/mean"] == 4.25


def test_museval_group_items_pairs_vocals_and_accompaniment(tmp_path):
    prediction = tmp_path / "prediction.wav"
    items = [
        manifest.EvalItem(
            item_id="musdb__test__Track__vocals",
            task_id="separate_vocals",
            source_path=tmp_path / "mixture.wav",
            vocals_path=tmp_path / "vocals.wav",
            prediction_path=prediction,
        ),
        manifest.EvalItem(
            item_id="musdb__test__Track__accompaniment",
            task_id="separate_accompaniment",
            source_path=tmp_path / "mixture.wav",
            accompaniment_path=tmp_path / "accompaniment.wav",
            prediction_path=prediction,
        ),
    ]

    grouped = museval_eval.group_items(items)

    assert list(grouped) == ["musdb__test__Track"]
    assert sorted(grouped["musdb__test__Track"]) == ["accompaniment", "vocals"]


def test_museval_imp_compat_allows_future_past_import():
    old = sys.modules.pop("imp", None)
    try:
        museval_eval.install_imp_compat()
        import imp

        assert imp.reload is importlib.reload
    finally:
        if old is not None:
            sys.modules["imp"] = old


def test_museval_evaluate_track_with_fake_backend(monkeypatch, tmp_path):
    sample_rate = 8000
    mixture = tmp_path / "mixture.wav"
    vocals = tmp_path / "vocals.wav"
    accompaniment = tmp_path / "accompaniment.wav"
    pred_vocals = tmp_path / "pred_vocals.wav"
    pred_accompaniment = tmp_path / "pred_accompaniment.wav"
    for path in [mixture, vocals, accompaniment, pred_vocals, pred_accompaniment]:
        torchaudio.save(str(path), torch.ones(2, sample_rate), sample_rate)

    items = {
        "vocals": manifest.EvalItem(
            item_id="track__vocals",
            task_id="separate_vocals",
            source_path=mixture,
            vocals_path=vocals,
            prediction_path=pred_vocals,
            sample_rate=sample_rate,
        ),
        "accompaniment": manifest.EvalItem(
            item_id="track__accompaniment",
            task_id="separate_accompaniment",
            source_path=mixture,
            accompaniment_path=accompaniment,
            prediction_path=pred_accompaniment,
            sample_rate=sample_rate,
        ),
    }

    class FakeMuseval:
        @staticmethod
        def evaluate(references, estimates, win, hop, mode, padding):
            assert mode == "v4"
            assert win == sample_rate
            assert len(references) == 2
            values = torch.tensor([[1.0, 3.0], [5.0, 7.0]]).numpy()
            return values, values + 10.0, values + 20.0, values + 30.0

    rows = museval_eval.evaluate_track(
        FakeMuseval,
        "track",
        items,
        win_seconds=1.0,
        hop_seconds=1.0,
    )

    assert [row["target"] for row in rows] == ["vocals", "accompaniment"]
    assert rows[0]["metrics"]["museval_v4_sdr_db"] == 2.0
    assert rows[1]["metrics"]["museval_v4_sar_db_frame_median"] == 35.0


def test_museval_rejects_obvious_full_track_length_mismatch(tmp_path):
    sample_rate = 8000
    mixture = tmp_path / "mixture.wav"
    vocals = tmp_path / "vocals.wav"
    accompaniment = tmp_path / "accompaniment.wav"
    pred_vocals = tmp_path / "pred_vocals.wav"
    pred_accompaniment = tmp_path / "pred_accompaniment.wav"
    for path in [mixture, vocals, accompaniment]:
        torchaudio.save(str(path), torch.ones(2, sample_rate * 4), sample_rate)
    for path in [pred_vocals, pred_accompaniment]:
        torchaudio.save(str(path), torch.ones(2, sample_rate // 2), sample_rate)

    items = {
        "vocals": manifest.EvalItem(
            item_id="track__vocals",
            task_id="separate_vocals",
            source_path=mixture,
            vocals_path=vocals,
            prediction_path=pred_vocals,
            sample_rate=sample_rate,
        ),
        "accompaniment": manifest.EvalItem(
            item_id="track__accompaniment",
            task_id="separate_accompaniment",
            source_path=mixture,
            accompaniment_path=accompaniment,
            prediction_path=pred_accompaniment,
            sample_rate=sample_rate,
        ),
    }

    class FakeMuseval:
        @staticmethod
        def evaluate(*args, **kwargs):
            raise AssertionError("length guard should reject before museval is called")

    with pytest.raises(ValueError, match="length mismatch"):
        museval_eval.evaluate_track(
            FakeMuseval,
            "track",
            items,
            win_seconds=1.0,
            hop_seconds=1.0,
        )


def test_museval_auto_length_check_allows_window_manifest_truncation(tmp_path):
    sample_rate = 8000
    mixture = tmp_path / "mixture.wav"
    vocals = tmp_path / "vocals.wav"
    accompaniment = tmp_path / "accompaniment.wav"
    pred_vocals = tmp_path / "pred_vocals.wav"
    pred_accompaniment = tmp_path / "pred_accompaniment.wav"
    for path in [mixture, vocals, accompaniment]:
        torchaudio.save(str(path), torch.ones(2, sample_rate * 4), sample_rate)
    for path in [pred_vocals, pred_accompaniment]:
        torchaudio.save(str(path), torch.ones(2, sample_rate // 2), sample_rate)

    items = {
        "vocals": manifest.EvalItem(
            item_id="track__vocals",
            task_id="separate_vocals",
            source_path=mixture,
            vocals_path=vocals,
            prediction_path=pred_vocals,
            duration_seconds=4.0,
            sample_rate=sample_rate,
        ),
        "accompaniment": manifest.EvalItem(
            item_id="track__accompaniment",
            task_id="separate_accompaniment",
            source_path=mixture,
            accompaniment_path=accompaniment,
            prediction_path=pred_accompaniment,
            duration_seconds=4.0,
            sample_rate=sample_rate,
        ),
    }

    class FakeMuseval:
        @staticmethod
        def evaluate(references, estimates, win, hop, mode, padding):
            assert references[0].shape[0] == sample_rate // 2
            assert estimates[0].shape[0] == sample_rate // 2
            values = torch.tensor([[1.0], [2.0]]).numpy()
            return values, values, values, values

    rows = museval_eval.evaluate_track(
        FakeMuseval,
        "track",
        items,
        win_seconds=1.0,
        hop_seconds=1.0,
    )

    assert len(rows) == 2
    assert rows[0]["museval_evaluated_seconds"] == 0.5
