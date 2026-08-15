"""
Drug-Shortage Early-Warning System (retrospective signal-recovery PoC).

This package flags and RANKS drugs by their risk of entering a shortage within a
configurable forward window, and backtests those flags against historical onsets
by LEAD TIME. It never issues a procurement or clinical verdict: the model
proposes and a human disposes.

See README.md for honest framing, scope, and limitations.
"""
__version__ = "1.0.0"
