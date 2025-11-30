"""Adapters for integrating hippocampus with external systems.

This subpackage contains adapters that bridge between external systems
(like Monty) and the core hippocampus module. The adapters are the ONLY
place where external dependencies are imported.
"""

from tbp.hippocampus.adapters.monty_adapter import MontyAdapter

__all__ = ["MontyAdapter"]
