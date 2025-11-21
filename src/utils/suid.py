"""Session ID utility functions."""

import uuid


def get_suid() -> str:
    """
    Generate a unique session ID (SUID) using UUID4.

    The value is a canonical RFC 4122 UUID (hex groups separated by
    hyphens) generated with uuid.uuid4().

    Returns:
        str: A UUID4 string suitable for use as a session identifier.
    """
    return str(uuid.uuid4())


def check_suid(suid: str) -> bool:
    """
    Check if given string is a proper session ID.

    Returns True if the string is a valid UUID or a llama-stack conversation ID.

    Parameters:
        suid (str | bytes): UUID value to validate — accepts a UUID string,
        its byte representation, or a llama-stack conversation ID (conv_xxx).

    Notes:
        Validation is performed by:
        1. For llama-stack conversation IDs starting with 'conv_':
           - Strips the 'conv_' prefix
           - Validates the remaining part is a valid hexadecimal UUID-like string
           - Converts to UUID format by inserting hyphens at standard positions
        2. For standard UUIDs: attempts to construct uuid.UUID(suid)
        Invalid formats or types result in False.
    """
    try:
        # Accept llama-stack conversation IDs (conv_<hex> format)
        if isinstance(suid, str) and suid.startswith("conv_"):
            # Extract the hex string after 'conv_'
            hex_part = suid[5:]  # Remove 'conv_' prefix

            # Verify it's a valid hex string of appropriate length
            # UUID without hyphens is 32 hex characters
            if len(hex_part) != 32:
                return False

            # Verify all characters are valid hex
            try:
                int(hex_part, 16)
            except ValueError:
                return False

            # Convert to UUID format with hyphens: 8-4-4-4-12
            uuid_str = f"{hex_part[:8]}-{hex_part[8:12]}-{hex_part[12:16]}-{hex_part[16:20]}-{hex_part[20:]}"

            # Validate it's a proper UUID
            uuid.UUID(uuid_str)
            return True

        # accepts strings and bytes only for UUID validation
        uuid.UUID(suid)
        return True
    except (ValueError, TypeError):
        return False
