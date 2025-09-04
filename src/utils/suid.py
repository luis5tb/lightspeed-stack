"""Session ID utility functions."""

import uuid


def get_suid() -> str:
    """Generate a unique session ID (SUID) using UUID4.

    Returns:
        A unique session ID.
    """
    return str(uuid.uuid4())


def check_suid(suid: str) -> bool:
    """Check if given string is a proper session ID or response ID.

    Args:
        suid: The string to check.

    Returns True if the string is a valid UUID or response ID, False otherwise.
    """
    try:
        # Check for responses API format (resp-<uuid>)
        if suid.startswith("resp-"):
            # Strip the "resp-" prefix and validate the UUID part
            uuid_part = suid[5:]  # Remove "resp-" prefix
            uuid.UUID(uuid_part)
            return True
        
        # Check for regular UUID format (backward compatibility)
        uuid.UUID(suid)
        return True
    except (ValueError, TypeError):
        return False
