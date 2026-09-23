import random
from collections.abc import Sequence

_SALT_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_"


def get_salt_string(
    salt_alphabet: Sequence[str] = _SALT_ALPHABET, length: int = 10
) -> str:
    return "".join(random.choices(salt_alphabet, k=length))
