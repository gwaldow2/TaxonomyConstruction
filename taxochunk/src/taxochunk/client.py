"""Convenience constructor for an OpenAI-compatible client.

Lets a user spin up a client without importing ``openai`` themselves:

    from taxochunk import openai_client
    client = openai_client("http://localhost:8000/v1", "EMPTY")

Works against the hosted OpenAI API or any compatible server (vLLM, TGI, ...).
"""

__all__ = ["openai_client"]


def openai_client(base_url="http://localhost:8000/v1", api_key="EMPTY", **kwargs):
    """Return an ``openai.OpenAI`` client pointed at ``base_url``.

    Parameters
    ----------
    base_url : str
        Endpoint, e.g. ``"https://api.openai.com/v1"`` or a local server.
    api_key : str
        API key (any non-empty string is fine for most local servers).
    **kwargs :
        Forwarded to ``openai.OpenAI`` (e.g. ``timeout``, ``organization``).
    """
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "openai is required for openai_client(); install it with "
            "`pip install \"taxochunk[openai]\"` (or pip install openai)."
        ) from e
    return OpenAI(base_url=base_url, api_key=api_key, **kwargs)
