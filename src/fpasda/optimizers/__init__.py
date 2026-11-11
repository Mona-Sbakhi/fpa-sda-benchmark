from .fpa import FPA
from .sda import SDA
from .hybrid import HybridFPASDA
from .others import DE, GA, PSO, RandomSearch
from .advanced import LSHADE, AdaptivePlantHybrid, OpenAIES, ZOSGD
from .memetic import Memetic

REGISTRY = {"fpa": FPA, "sda": SDA, "hybrid": HybridFPASDA,
            "random": RandomSearch, "pso": PSO, "de": DE, "ga": GA,
            "lshade": LSHADE, "aph": AdaptivePlantHybrid, "openai_es": OpenAIES, "zo_sgd": ZOSGD, "memetic": Memetic}


def build(spec: dict, pop_size: int, iterations: int, bounds):
    spec = dict(spec)
    cls = REGISTRY[spec.pop("type")]
    return cls(pop_size=pop_size, iterations=iterations, bounds=bounds, **spec)
