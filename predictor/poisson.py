from typing import Dict, Any
import math

# Very basic Poisson-based home/draw/away probability placeholder
# Inputs should include average goals for/against, etc.

def match_probabilities(stats: Dict[str, Any]) -> Dict[str, float]:
    # Placeholder: equal teams => ~33/33/33
    return {"home": 0.34, "draw": 0.32, "away": 0.34}
