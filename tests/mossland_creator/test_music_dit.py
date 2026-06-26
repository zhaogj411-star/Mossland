import torch
from megatron.energon import WorkerConfig
from megatron.energon.task_encoder.base import get_stateless

from scripts.mossland_creator.music_dit.core import contract as C
from scripts.mossland_creator.music_dit.data.taskencoder import (
    MusicDiffusionTaskEncoder,
    cook_music,
)
from scripts.mossland_creator.music_dit.diffusion import MusicFlowMatchingEulerSampler
from scripts.mossland_creator.music_dit.models.music_dit import MusicDiT
from scripts.mossland_creator.music_dit.wrapper import MusicDiTTrainingWrapper


def test_cook_music_is_stateless_and_handles_crude_keys():
    assert get_stateless(cook_music)
    sample = {
        "__key__": "k0",
        "__restore_key__": ("k0",),
        "pth": {"mixture": torch.randn(8, 40)},
        "pickle": torch.randn(20, 12),
        "json": {"caption": "test"},
    }

    cooked = cook_music(sample)

    assert cooked["latent_bank"] is sample["pth"]
    assert cooked["text"] is sample["pickle"]
    assert cooked["json"] is sample["json"]


def test_music_task_encoder_handles_current_shard_schema():
    encoder = MusicDiffusionTaskEncoder(
        seq_length=32,
        patch_t=1,
        text_embedding_padding_size=16,
        task_specs=[{"name": "text2music", "weight": 1.0, "target_key": "mixture"}],
    )
    sample = {
        "__key__": "k0",
        "__restore_key__": ("k0",),
        "latent_bank": {
            "mixture": torch.randn(8, 40),
            "vocals": torch.randn(8, 40),
            "accompaniment": torch.randn(8, 40),
        },
        "text": torch.randn(20, 12),
        "json": {"caption": "test"},
    }
    worker = WorkerConfig(rank=0, world_size=1, num_workers=0)
    worker.worker_activate(sample_index=0)
    try:
        encoded = encoder.encode_sample(sample)
    finally:
        worker.worker_deactivate()
    batch = encoder.batch([encoded, encoded])

    assert batch[C.KEY_LATENT].shape == (2, 32, 8)
    assert batch[C.KEY_CONTEXT].shape == (2, 32, 8)
    assert batch[C.KEY_COND_MASK].shape == (2, 32, 1)
    assert batch[C.KEY_CROSSATTN].shape == (2, 16, 12)
    assert batch[C.KEY_CROSSATTN_MASK].shape == (2, 16)
    assert batch[C.KEY_TASK] == ["text2music", "text2music"]


def test_music_dit_wrapper_loss_smoke():
    model = MusicDiT(
        latent_channels=8,
        patch_t=1,
        crossattn_emb_size=12,
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        use_context_conditioning=True,
        sigma_data=0.5,
    )
    wrapper = MusicDiTTrainingWrapper(
        model=model,
        learning_rate=1e-4,
        lr_schedule="constant",
        fail_on_nonfinite=True,
    )
    batch = {
        C.KEY_LATENT: torch.randn(2, 32, 8),
        C.KEY_CONTEXT: torch.randn(2, 32, 8),
        C.KEY_COND_MASK: torch.ones(2, 32, 1),
        C.KEY_CROSSATTN: torch.randn(2, 16, 12),
        C.KEY_CROSSATTN_MASK: torch.ones(2, 16),
        C.KEY_LOSS_MASK: torch.ones(2, 32),
        C.KEY_POS_IDS: torch.arange(32).view(1, 32, 1).expand(2, -1, -1),
        C.KEY_SEQLEN_Q: torch.tensor([32, 32], dtype=torch.int32),
        C.KEY_SEQLEN_KV: torch.tensor([16, 16], dtype=torch.int32),
        C.KEY_LATENT_SHAPE: torch.tensor([[8, 32], [8, 32]], dtype=torch.int32),
    }
    loss, metrics = wrapper.compute_loss(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "loss/mse" in metrics
    assert torch.isfinite(metrics["loss/mse"])


def test_music_dit_flow_matching_wrapper_loss_smoke():
    model = MusicDiT(
        latent_channels=8,
        patch_t=1,
        crossattn_emb_size=12,
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        use_context_conditioning=True,
        sigma_data=0.5,
    )
    wrapper = MusicDiTTrainingWrapper(
        model=model,
        learning_rate=1e-4,
        lr_schedule="constant",
        objective_name="flow_matching",
        flow_matching_estimator_target="conditional_vector_field",
        fail_on_nonfinite=True,
    )
    batch = {
        C.KEY_LATENT: torch.randn(2, 32, 8),
        C.KEY_CONTEXT: torch.randn(2, 32, 8),
        C.KEY_COND_MASK: torch.ones(2, 32, 1),
        C.KEY_CROSSATTN: torch.randn(2, 16, 12),
        C.KEY_CROSSATTN_MASK: torch.ones(2, 16),
        C.KEY_LOSS_MASK: torch.ones(2, 32),
        C.KEY_POS_IDS: torch.arange(32).view(1, 32, 1).expand(2, -1, -1),
        C.KEY_SEQLEN_Q: torch.tensor([32, 32], dtype=torch.int32),
        C.KEY_SEQLEN_KV: torch.tensor([16, 16], dtype=torch.int32),
        C.KEY_LATENT_SHAPE: torch.tensor([[8, 32], [8, 32]], dtype=torch.int32),
    }
    loss, metrics = wrapper.compute_loss(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "flow/time_mean" in metrics
    assert torch.isfinite(metrics["flow/time_mean"])


def test_music_flow_matching_euler_sampler_smoke():
    model = MusicDiT(
        latent_channels=8,
        patch_t=1,
        crossattn_emb_size=12,
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        use_context_conditioning=True,
        sigma_data=0.5,
    )
    sampler = MusicFlowMatchingEulerSampler(
        net=model,
        estimator_target="conditional_vector_field",
        num_steps=4,
    )
    batch = {
        C.KEY_CONTEXT: torch.randn(2, 32, 8),
        C.KEY_COND_MASK: torch.ones(2, 32, 1),
        C.KEY_CROSSATTN: torch.randn(2, 16, 12),
        C.KEY_CROSSATTN_MASK: torch.ones(2, 16),
        C.KEY_POS_IDS: torch.arange(32).view(1, 32, 1).expand(2, -1, -1),
        C.KEY_SEQLEN_Q: torch.tensor([32, 32], dtype=torch.int32),
    }

    out = sampler.sample(batch, shape=(2, 32, 8))

    assert out.shape == (2, 32, 8)
    assert torch.isfinite(out).all()
