"""
Entry point for running the multi-agent data analyzer as a module.

Usage:
    python -m multi_agent_analyzer <csv_file> [options]
"""

from .terminal_interface import main

if __name__ == "__main__":
    main()