"""
Import scrapers that self-register into the registry on import.
This ensures registry-based sources are available to the scheduler.
"""

# Register sources that provide fixtures and/or odds/probabilities
from . import matchoutlook  # noqa: F401
from . import soccer365     # noqa: F401
from . import synth         # noqa: F401
from . import sstats        # noqa: F401
from . import forebet       # noqa: F401
from . import statarea      # noqa: F401

# Optionally keep placeholders disabled until implemented
# from . import bigsoccer  # sentiment (disabled by default)
# from . import eaglepredict  # external_model (disabled by default)
