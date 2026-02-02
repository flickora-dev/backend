"""
Chat input validation and sanitization utilities
"""
import re
import html
from typing import Tuple

# Maximum message length
MAX_MESSAGE_LENGTH = 2000

# Patterns that might indicate prompt injection attempts
INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)',
    r'disregard\s+(all\s+)?(previous|above|prior)',
    r'forget\s+(everything|all)',
    r'you\s+are\s+now\s+in\s+',
    r'new\s+instructions?:',
    r'system\s*:\s*',
    r'<\s*system\s*>',
    r'\[system\]',
    r'assistant\s*:\s*',
    r'</?(system|assistant|user)>',
    r'jailbreak',
    r'bypass\s+(restrictions?|filters?|rules?)',
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def sanitize_message(message: str) -> str:
    """
    Sanitize user message to prevent XSS and other injection attacks.

    Args:
        message: Raw user message

    Returns:
        Sanitized message safe for storage and display
    """
    if not message:
        return ""

    # Escape HTML entities
    message = html.escape(message)

    # Remove null bytes
    message = message.replace('\x00', '')

    # Normalize whitespace (but preserve newlines for formatting)
    message = re.sub(r'[ \t]+', ' ', message)
    message = re.sub(r'\n{3,}', '\n\n', message)

    # Strip leading/trailing whitespace
    message = message.strip()

    return message


def validate_message(message: str) -> Tuple[bool, str]:
    """
    Validate user message for length and content.

    Args:
        message: User message to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not message:
        return False, "Message cannot be empty"

    if not isinstance(message, str):
        return False, "Message must be a string"

    # Check length
    if len(message) > MAX_MESSAGE_LENGTH:
        return False, f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters"

    if len(message.strip()) == 0:
        return False, "Message cannot be empty or whitespace only"

    return True, ""


def check_prompt_injection(message: str) -> Tuple[bool, str]:
    """
    Check if message contains potential prompt injection patterns.

    This is a heuristic check - it may have false positives.
    Log suspicious messages but don't block legitimate users.

    Args:
        message: User message to check

    Returns:
        Tuple of (is_suspicious, matched_pattern)
    """
    message_lower = message.lower()

    for pattern in COMPILED_PATTERNS:
        match = pattern.search(message_lower)
        if match:
            return True, match.group()

    return False, ""


def prepare_user_message_for_llm(message: str) -> str:
    """
    Prepare user message before sending to LLM.
    Adds safety wrapper to reduce prompt injection effectiveness.

    Args:
        message: Sanitized user message

    Returns:
        Message with safety wrapper
    """
    # Wrap user input clearly to help LLM distinguish it from instructions
    # This makes prompt injection less effective
    wrapped = f"""[USER MESSAGE START]
{message}
[USER MESSAGE END]

Respond helpfully to the user's message above about movies."""

    return wrapped
