"""Backward-compatibility shim — use ``profile_encodings`` directly.

This module re-exports everything from the ``profile_encodings`` package
so that existing imports (e.g. in ``not_used/`` files) keep working.
"""

import warnings

warnings.warn(
    "profile_encodings_helper is deprecated — import from profile_encodings instead",
    DeprecationWarning,
    stacklevel=2,
)

from profile_encodings import (  # noqa: F401, E402
    ResidualConvBlock,
    CNNProfileEncoder,
    DilatedProfileEncoder,
    AttentionProfileEncoder,
    MultiResolutionProfileEncoder,
    FourierProfileEncoder,
    FourierProfileEncoderWithContext,
)
