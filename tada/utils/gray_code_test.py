import pytest
import torch

from .gray_code import decode_gray_code_to_time, encode_time_with_gray_code


@pytest.mark.parametrize("num_frames", [34, 100, 200])
def test_gray_code(num_frames: int):
    gray_bits = encode_time_with_gray_code(torch.tensor([num_frames]), 8)
    time_len_rec = decode_gray_code_to_time(gray_bits, 8)
    assert num_frames == time_len_rec
