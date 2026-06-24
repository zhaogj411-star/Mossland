import torch

from scripts.mossland_codec_same.audio import AudioProcessor
from scripts.mossland_codec_same.models import MosslandCodecSame


def _small_model(**overrides):
    kwargs = dict(
        audio_processor=AudioProcessor(
            alpha_rescale=0.65,
            beta_rescale=0.34,
            hop_size=512,
            fac=4,
            center_pad=False,
        ),
        sample_rate=44100,
        base_channels=8,
        data_channels=4,
        hop=512,
        bottleneck_channels=16,
        bottleneck_base_channels=32,
        num_bottleneck_layers=1,
        quantizer_num_quantizers=0,
        heads=2,
        cond_channels=32,
        layers_list=[1, 1, 1, 1, 1],
        layers_list_encoder=[1, 1, 1, 1, 1],
        multipliers_list=[1, 2, 4, 4, 4],
        freq_downsample_list=[1, 0, 0, 0],
        same_transformer_depth=1,
        same_transformer_min_channels=32,
    )
    kwargs.update(overrides)
    return MosslandCodecSame(**kwargs)


def test_same_codec_encoder_decoder_forward_shapes():
    model = _small_model()
    representation = torch.randn(1, 4, 1024, 64)
    latent = model.encoder(representation)
    assert latent.shape == (1, 16, 8)

    time_emb = model._condition_embedding(model.emb(torch.zeros(1)), "reconstruct")
    pyramid = model.decoder(latent, time_emb=time_emb)
    assert [tuple(item.shape) for item in pyramid] == [
        (1, 8, 1024, 64),
        (1, 16, 256, 64),
        (1, 32, 128, 32),
        (1, 32, 64, 16),
        (1, 32, 32, 8),
    ]

    predicted = model(
        representation,
        representation,
        sigma=torch.tensor([1.0]),
        latent_override=latent,
        task_id="reconstruct",
    )
    assert predicted.shape == representation.shape
    assert torch.isfinite(predicted).all()


def test_same_codec_quantizer_path_shapes():
    model = _small_model(
        quantizer_num_quantizers=2,
        quantizer_codebook_size=8,
        quantizer_kmeans_init=False,
    )
    representation = torch.randn(2, 4, 1024, 64)
    quantized = model.quantize_representation(
        representation,
        detach_encoder=False,
        n_quantizers=1,
    )
    assert quantized.continuous.shape == (2, 16, 8)
    assert quantized.discrete.shape == (2, 16, 8)
    assert quantized.codes.shape[0] == 2
    assert quantized.codes.shape[-1] == 8
