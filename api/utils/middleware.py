"""
Security middleware for origin-based access control.

This module provides middleware to restrict API access to allowed origins
(yashashm.dev domains) while allowing bypass for testing with a secret key.

Environment Variables:
    - BYPASS_KEY: Optional secret key to bypass origin checks for testing

Usage:
    The middleware is applied in app.py:
        app.add_middleware(SecurityMiddleware)

    To bypass for testing, add ?bypass_key=<your_key> to the request URL.
"""

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


def bypass_middleware(key: str | None) -> bool:
    """
    Check if the request should bypass origin restrictions.

    Args:
        key: The bypass key from request query params

    Returns:
        True if bypass is allowed, False otherwise
    """
    bypass_key = os.environ.get('BYPASS_KEY')
    # Only check if BYPASS_KEY is configured
    if bypass_key is None:
        return False
    # Check if provided key matches
    return key is not None and key == bypass_key


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce origin-based access control.

    Allows requests from:
        - https://yashashm.dev
        - https://ask.yashashm.dev
        - Requests with valid bypass_key query parameter

    Public endpoints (no origin check):
        - / (root redirect)
        - /api/healthCheck
    """

    async def dispatch(self, request, call_next):
        if bypass_middleware(key=request.query_params.get("bypass_key")):
            return await call_next(request)

        if request.url.path == '/' or request.url.path == '/api/healthCheck':
            return await call_next(request)

        allowed_origin = [
            "https://yashashm.dev",
            "https://ask.yashashm.dev",
        ]

        origin = request.headers.get("Origin")
        if origin not in allowed_origin:
            return JSONResponse({"detail": "Forbidden"}, status_code=403)

        return await call_next(request)
