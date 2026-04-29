from xdsl.universe import Universe

from gem.xdsl_gem_dialect import GEM

GEM_UNIVERSE = Universe(all_dialects={"gem": lambda: GEM})
