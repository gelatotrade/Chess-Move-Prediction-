"""
Chess Move Prediction Server

FastAPI-based backend for the Chrome extension overlay.
"""

from .app import app, run_server

__all__ = ['app', 'run_server']
