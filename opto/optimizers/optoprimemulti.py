from opto.optimizers.optoprime import OptoPrime
from opto.optimizers.optoprimemulti_base import OptoPrimeMultiMixin
from opto.utils.llm import LLMFactory  # Backwards-compatible import location used by existing tests/users.


class OptoPrimeMulti(OptoPrimeMultiMixin, OptoPrime):
    """Multi-candidate OptoPrime optimizer built on the V1 OptoPrime contract."""
