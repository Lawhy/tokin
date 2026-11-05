from . import mini_swe_agent, opencode, pi
from .mini_swe_agent import PrebuiltMiniSweAgent
from .opencode import PrebuiltOpenCode
from .pi import PrebuiltPi
from .prebuilt import Prebuilt

PREBUILT: dict[str, Prebuilt] = {m.PREBUILT.name: m.PREBUILT for m in (mini_swe_agent, pi, opencode)}

__all__ = ["PREBUILT", "Prebuilt", "PrebuiltMiniSweAgent", "PrebuiltOpenCode", "PrebuiltPi"]
