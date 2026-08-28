"""Langfuse tracing setup for ADK agents.

Import this module before constructing any `google.adk.agents.Agent` -
`GoogleADKInstrumentor().instrument()` patches ADK's classes, so anything
built before this import runs untraced. Every entrypoint (CLI, UI, tests)
should import `langfuse` from here rather than calling `get_client()` again,
so instrumentation only ever runs once per process.
"""

from dotenv import load_dotenv

load_dotenv()

from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from langfuse import get_client

GoogleADKInstrumentor().instrument()

langfuse = get_client()
