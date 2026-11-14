from __future__ import annotations

from cryptography.fernet import Fernet


def generate_master_key() -> str:
    return Fernet.generate_key().decode()


class SecretCipher:
    """Encrypts tenant LLM keys at rest. The master key never reaches the database."""

    def __init__(self, master_key: str) -> None:
        self._fernet = Fernet(master_key.encode())

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        return self._fernet.decrypt(ciphertext.encode()).decode()
