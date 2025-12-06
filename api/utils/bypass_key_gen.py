"""
Generate a random BYPASS_KEY and add/replace it in .env file.

This script generates a cryptographically secure random key that can be
used to bypass origin restrictions when testing the API locally or from
non-whitelisted domains.

Usage:
    python api/utils/bypass_key_gen.py

The generated key should be:
    1. Added to your .env file (done automatically by this script)
    2. Added to Vercel environment variables for production
    3. Used in requests as: ?bypass_key=<your_key>

Security Note:
    Keep this key secret. Anyone with the key can bypass origin restrictions.
"""

import secrets
import os

ENV_FILE = "../../.env"
KEY_NAME = "BYPASS_KEY"


def generate_key() -> str:
    """
    Generate a cryptographically secure random key.

    Returns:
        A 32-byte URL-safe base64-encoded string
    """
    return secrets.token_urlsafe(32)


def update_env_file(key_name: str, key_value: str) -> None:
    """
    Add or replace a key in the .env file.

    If the key exists, it will be replaced. Otherwise, it will be appended.

    Args:
        key_name: Name of the environment variable
        key_value: Value to set
    """
    env_path = os.path.join(os.path.dirname(__file__), ENV_FILE)

    lines = []
    key_found = False

    # Read existing .env if it exists
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()

        # Check if key exists and replace it
        for i, line in enumerate(lines):
            if line.startswith(f"{key_name}="):
                lines[i] = f"{key_name}={key_value}\n"
                key_found = True
                break

    # If key not found, append it
    if not key_found:
        # Ensure newline before adding if file doesn't end with one
        if lines and not lines[-1].endswith('\n'):
            lines[-1] += '\n'
        lines.append(f"{key_name}={key_value}\n")

    # Write back to .env
    with open(env_path, 'w') as f:
        f.writelines(lines)


def main():
    new_key = generate_key()
    update_env_file(KEY_NAME, new_key)
    print(f"Generated new {KEY_NAME}: {new_key}")
    print(f"Updated {ENV_FILE}")


if __name__ == "__main__":
    main()