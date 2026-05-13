from typing import Union

from opto.optimizers.optoprime_v2 import OptoPrimeV2
from opto.optimizers.optoprimemulti_base import OptoPrimeMultiMixin


class OptoPrimeMultiV2(OptoPrimeMultiMixin, OptoPrimeV2):
    """Multi-candidate OptoPrime optimizer built on the V2 OptoPrime contract."""

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        verbose: Union[bool, str] = False,
        max_tokens: int = 4096,
    ) -> str:
        """V2-compatible public LLM helper.

        Multi-candidate internals call ``_call_llm_responses`` directly. This
        public method intentionally keeps OptoPrimeV2's single-string return
        contract.
        """
        responses = self._call_llm_responses(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            verbose=verbose,
            max_tokens=max_tokens,
            num_responses=1,
            temperature=0.0,
        )
        return responses[0] if responses else ""
