import pytest
import torch

from src.models.vit_vqgan.layers import Attention2d, PairwiseBatchedDistanceLayer, ShiftedPatchTokenizationLayer


@pytest.fixture(scope='module')
def images_batch():
    return torch.randn(4, 3, 32, 32)


def test_pairwise_batched_distance_layer():
    a = torch.randn(3, 5, 24)
    b = torch.randn(4, 24)

    distance_layer = PairwiseBatchedDistanceLayer()
    distances = distance_layer(a, b)

    assert torch.isfinite(distances).all()
    assert distances.size() == torch.Size([3, 5, 4])

    expected_distances = torch.empty(3, 5, 4)

    for i, a_mat in enumerate(a):
        for j, a_vec in enumerate(a_mat):
            for k, b_vec in enumerate(b):
                expected_distances[i, j, k] = torch.sum((a_vec - b_vec) ** 2)

    assert torch.isclose(expected_distances, distances).all()


def test_shifted_patch_tokenization_layer(images_batch):
    with pytest.raises(AssertionError):
        ShiftedPatchTokenizationLayer(10, 32, 11)

    layer = ShiftedPatchTokenizationLayer(10, 32, 8)

    output = layer(images_batch)

    assert torch.isfinite(output).all()
    assert output.size() == torch.Size([4, 10, 4, 4])

def test_attention_no_embs(images_batch):
    attention = Attention2d(3, 8, 16)

    output = attention(images_batch)
    assert torch.isfinite(output).all()
    assert output.size() == torch.Size([4, 3, 32, 32])


def test_attention_with_embs(images_batch):
    attention = Attention2d(3, 8, 16, num_patches=32)

    output = attention(images_batch)
    assert torch.isfinite(output).all()
    assert output.size() == torch.Size([4, 3, 32, 32])
