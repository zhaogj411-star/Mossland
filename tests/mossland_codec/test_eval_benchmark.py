from __future__ import annotations

import importlib
import sys

import torch
import torchaudio


manifest = importlib.import_module("scripts.mossland-codec.eval_benchmark.manifest")
model_adapters = importlib.import_module("scripts.mossland-codec.eval_benchmark.model_adapters")


def test_generate_adapter_prediction_reuses_existing_output(tmp_path):
    item = manifest.EvalItem(
        item_id="clip",
        task_id="reconstruct",
        source_path=tmp_path / "source.wav",
        sample_rate=44100,
        seed=0,
    )
    output_path = tmp_path / "predictions" / "reconstruct" / "clip_seed0.wav"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"existing")

    class Adapter:
        def predict(self, *args, **kwargs):
            raise AssertionError("existing prediction should be reused")

    result = model_adapters.generate_adapter_prediction(
        Adapter(),
        item,
        tmp_path / "predictions",
        context=model_adapters.AdapterContext(output_dir=tmp_path / "predictions"),
    )

    assert result == output_path
    assert output_path.read_bytes() == b"existing"


def test_custom_adapter_target_can_be_factory(tmp_path, monkeypatch):
    module_path = tmp_path / "dummy_adapter.py"
    module_path.write_text(
        "\n".join(
            [
                "import torch",
                "",
                "class Adapter:",
                "    def __init__(self, context):",
                "        self.scale = float(context.options['scale'])",
                "",
                "    def predict(self, item, source, target, context):",
                "        return source * self.scale",
                "",
                "def create_adapter(context):",
                "    return Adapter(context)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("dummy_adapter", None)

    context = model_adapters.AdapterContext(options={"scale": 0.5})
    adapter = model_adapters.load_prediction_adapter(
        adapter_target="dummy_adapter:create_adapter",
        context=context,
    )

    source = torch.ones(2, 8)
    item = manifest.EvalItem(
        item_id="clip",
        task_id="reconstruct",
        source_path=tmp_path / "source.wav",
    )
    torch.testing.assert_close(adapter.predict(item, source, None, context), source * 0.5)


def test_generate_adapter_prediction_writes_audio(tmp_path):
    source_path = tmp_path / "source.wav"
    sample_rate = 44100
    source = torch.linspace(-0.25, 0.25, 1024).repeat(2, 1)
    torchaudio.save(str(source_path), source, sample_rate)
    item = manifest.EvalItem(
        item_id="clip",
        task_id="reconstruct",
        source_path=source_path,
        sample_rate=sample_rate,
        seed=0,
    )

    class Adapter:
        def predict(self, item, source, target, context):
            return source

    output = model_adapters.generate_adapter_prediction(
        Adapter(),
        item,
        tmp_path / "predictions",
        context=model_adapters.AdapterContext(output_dir=tmp_path / "predictions"),
    )

    prediction, sr = torchaudio.load(str(output))
    assert sr == sample_rate
    assert prediction.shape == source.shape
