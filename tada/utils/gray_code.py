import torch


def int_to_gray_code(values: torch.Tensor) -> torch.Tensor:
    """Convert integer values to their Gray code equivalents.

    Gray code formula: gray = value XOR (value >> 1)
    This ensures adjacent integers differ by only one bit.

    Args:
        values: Integer tensor of shape (...)

    Returns:
        Gray code integers of same shape
    """
    return values ^ (values >> 1)


def gray_code_to_int(gray: torch.Tensor) -> torch.Tensor:
    """Convert Gray code integers back to binary integers.

    Args:
        gray: Gray code integer tensor of shape (...)

    Returns:
        Binary integer tensor of same shape
    """
    # Iteratively XOR with right-shifted versions
    binary = gray
    shift = 1
    while shift < 32:  # Assuming 32-bit integers
        binary = binary ^ (binary >> shift)
        shift <<= 1
    return binary


def encode_time_with_gray_code(
    num_frames: torch.LongTensor,
    num_bits: int,
) -> torch.Tensor:
    """Convert time values to Gray code bit representation as floats in {-1, 1}.

    Gray code ensures adjacent time values differ by only 1 bit, providing
    smoother continuous representation for the diffusion head.

    Args:
        num_frames: Integer time values of shape (...,)
        num_time_bits: Number of bits to use for Gray code representation
        num_time_classes: Maximum number of time classes for clamping

    Returns:
        Gray code bits as floats of shape (..., num_time_bits) with values in {-1, 1}
    """
    num_time_classes = 2**num_bits
    # Clamp input to valid range
    num_frames = num_frames.clamp(min=0, max=num_time_classes - 1)

    # Step 1: Convert time indices to Gray code equivalents (integer)
    gray_code = int_to_gray_code(num_frames)

    # Step 2: Convert Gray code integer to bit representation (0s and 1s)
    # Extract binary bits from the gray code integer (NOT applying binary_to_gray again)
    gray_bits = torch.zeros(*gray_code.shape, num_bits, dtype=torch.long, device=gray_code.device)
    for i in range(num_bits):
        gray_bits[..., num_bits - 1 - i] = (gray_code >> i) & 1

    # Convert from {0, 1} to {-1, 1}
    return gray_bits.float() * 2.0 - 1.0


def decode_gray_code_to_time(
    gray_bits: torch.Tensor,
    num_bits: int,
) -> torch.LongTensor:
    """Convert Gray code bit representation back to time values.

    This is the inverse operation of encode_time_with_gray_code.

    Args:
        gray_bits: Gray code bits as floats of shape (..., num_time_bits) with values in {-1, 1}
        num_bits: Number of bits in the Gray code representation

    Returns:
        Integer time values of shape (...,)
    """
    # Step 1: Convert from {-1, 1} to {0, 1}
    gray_bits_binary = ((gray_bits + 1.0) / 2.0).round().long()

    # Step 2: Convert bit representation to Gray code integer
    # Reconstruct the integer from bits
    gray_code = torch.zeros(*gray_bits_binary.shape[:-1], dtype=torch.long, device=gray_bits.device)
    for i in range(num_bits):
        gray_code += gray_bits_binary[..., num_bits - 1 - i] << i

    # Step 3: Convert Gray code integer to regular integer
    num_frames = gray_code_to_int(gray_code)

    return num_frames
