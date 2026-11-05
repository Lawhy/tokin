from . import mini_swe_agent, opencode, pi
from .bundle import Bundle
from .mini_swe_agent import BundledMiniSweAgent
from .opencode import BundledOpenCode
from .pi import BundledPi

BUNDLES: dict[str, Bundle] = {m.BUNDLE.name: m.BUNDLE for m in (mini_swe_agent, pi, opencode)}

__all__ = ["BUNDLES", "Bundle", "BundledMiniSweAgent", "BundledOpenCode", "BundledPi"]
