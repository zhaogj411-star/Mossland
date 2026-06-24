import importlib

import torch

from vector_quantize_pytorch import vector_quantize_pytorch as vq_backend

from scripts.codec_common.quantize import ResidualVectorQuantize as SharedResidualVectorQuantize


def _assert_sync_toggle(rvq_cls):
    rvq = rvq_cls(input_dim=4, n_codebooks=2, codebook_size=8)
    rvq.set_distributed_sync(True)

    for layer in rvq.vq.layers:
        codebook = layer._codebook
        assert codebook.use_ddp is True
        assert codebook.sample_fn is vq_backend.sample_vectors_distributed
        assert codebook.replace_sample_fn is vq_backend.sample_vectors_distributed
        assert codebook.kmeans_all_reduce_fn is vq_backend.distributed.all_reduce
        assert codebook.all_reduce_fn is vq_backend.distributed.all_reduce

    rvq.set_distributed_sync(False)

    for layer in rvq.vq.layers:
        codebook = layer._codebook
        assert codebook.use_ddp is False
        assert codebook.sample_fn is vq_backend.batched_sample_vectors
        assert codebook.replace_sample_fn is vq_backend.batched_sample_vectors
        assert codebook.kmeans_all_reduce_fn is vq_backend.noop
        assert codebook.all_reduce_fn is vq_backend.noop


def test_shared_rvq_distributed_sync_toggle():
    _assert_sync_toggle(SharedResidualVectorQuantize)


def test_legacy_hyphen_rvq_distributed_sync_toggle():
    module = importlib.import_module("scripts.mossland-codec.quantize")
    _assert_sync_toggle(module.ResidualVectorQuantize)


def test_initialized_codebook_count_tracks_kmeans_initialization():
    rvq = SharedResidualVectorQuantize(input_dim=4, n_codebooks=3, codebook_size=8)
    assert rvq.initialized_codebook_count() == 0

    rvq(torch.randn(2, 4, 16))

    assert rvq.initialized_codebook_count() == 3
