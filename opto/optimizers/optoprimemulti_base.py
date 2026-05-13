from __future__ import annotations

import json
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Union


class OptoPrimeMultiMixin:
    """Shared multi-candidate machinery for OptoPrime-style optimizers.

    The concrete optimizer decides the prompt/extraction contract through its
    normal OptoPrime base class. This mixin only fans out LLM calls, selects a
    candidate, and feeds parsed variable updates back into
    ``construct_update_dict``.
    """

    def __init__(
        self,
        *args,
        num_responses: int = 3,
        temperature_min_max: Optional[List[float]] = None,
        selector: Optional[Callable[[List[Any]], Any]] = None,
        generation_technique: str = "temperature_variation",
        selection_technique: str = "best_of_n",
        experts_list: Optional[List[str]] = None,
        llm_profiles: Optional[List[str]] = None,
        llm_weights: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.temperature_min_max = list(temperature_min_max or [0.0, 1.0])
        self.candidates: List[Any] = []
        self.candidate_details: List[Dict[str, Any]] = []
        self.selected_candidate: Any = None
        self.selected_candidate_details: Optional[Dict[str, Any]] = None
        self.num_responses = num_responses
        self.selector = selector
        self.generation_technique = generation_technique
        self.selection_technique = selection_technique
        self.experts_list = experts_list
        self.llm_profiles = llm_profiles
        self.llm_weights = llm_weights or ([1.0] * len(llm_profiles) if llm_profiles else None)
        self._llm_instances: Dict[str, Any] = {}

    def _get_llm_for_profile(self, profile: Optional[str] = None):
        if profile is None:
            return self.llm
        if profile not in self._llm_instances:
            try:
                import sys
                import opto.utils.llm as llm_module

                owner_module = sys.modules.get(self.__class__.__module__)
                owner_factory = getattr(owner_module, "LLMFactory", None)
                if owner_factory is not None and getattr(owner_factory, "__module__", None) != "opto.utils.llm":
                    factory = owner_factory
                else:
                    factory = llm_module.LLMFactory
                self._llm_instances[profile] = factory.get_llm(profile)
            except Exception as exc:
                warnings.warn(
                    f"Failed to create LLM for profile '{profile}': {exc}. Falling back to default LLM.",
                    stacklevel=2,
                )
                return self.llm
        return self._llm_instances[profile]

    def _get_llms_for_generation(self, num_responses: int) -> List[Any]:
        if not self.llm_profiles:
            return [self.llm] * num_responses
        return [self._get_llm_for_profile(self.llm_profiles[i % len(self.llm_profiles)]) for i in range(num_responses)]

    def _llm_response_format(self) -> Optional[Dict[str, str]]:
        return {"type": "json_object"} if getattr(self, "use_json_object_format", False) else None

    def _extract_contents(self, response: Any) -> List[str]:
        if response is None:
            return []
        if isinstance(response, str):
            return [response]
        if isinstance(response, list):
            contents: List[str] = []
            for item in response:
                contents.extend(self._extract_contents(item))
            return contents
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if choices is None:
            return [str(response)]

        contents = []
        for choice in choices:
            message = getattr(choice, "message", None)
            if message is not None and hasattr(message, "content"):
                contents.append(message.content)
            elif isinstance(choice, dict):
                message = choice.get("message", {})
                contents.append(message.get("content", str(choice)) if isinstance(message, dict) else str(choice))
            else:
                contents.append(str(choice))
        return [content for content in contents if isinstance(content, str)]

    def _call_llm_responses(
        self,
        system_prompt: str,
        user_prompt: str,
        verbose: Union[bool, str] = False,
        max_tokens: int = 4096,
        num_responses: int = 1,
        temperature: float = 0.0,
        llm: Any = None,
    ) -> List[str]:
        if verbose not in (False, "output"):
            print("Prompt\n", system_prompt + user_prompt)

        active_llm = llm or self.llm
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response_format = self._llm_response_format()

        def invoke(include_response_format: bool):
            kwargs = {
                "messages": messages,
                "max_tokens": max_tokens,
                "n": num_responses,
                "temperature": temperature,
            }
            if include_response_format and response_format is not None:
                kwargs["response_format"] = response_format
            if hasattr(active_llm, "create"):
                return active_llm.create(**kwargs)
            return active_llm(**kwargs)

        try:
            response = invoke(include_response_format=True)
        except Exception as exc:
            if response_format is None:
                if verbose:
                    print(f"ERROR {exc}")
                return []
            try:
                response = invoke(include_response_format=False)
            except Exception as inner_exc:
                if verbose:
                    print(f"ERROR {inner_exc}")
                return []

        contents = self._extract_contents(response)
        if verbose:
            print("LLM responses:\n", contents)
        return contents

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        verbose: Union[bool, str] = False,
        max_tokens: int = 4096,
        num_responses: int = 1,
        temperature: float = 0.0,
        llm: Any = None,
    ) -> List[str]:
        """V1-compatible multi-response LLM helper."""
        return self._call_llm_responses(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            verbose=verbose,
            max_tokens=max_tokens,
            num_responses=num_responses,
            temperature=temperature,
            llm=llm,
        )

    def _parallel_call_llm(self, arg_dicts: List[Dict[str, Any]]) -> List[str]:
        if not arg_dicts:
            return []
        outputs: List[Optional[str]] = [None] * len(arg_dicts)
        with ThreadPoolExecutor(max_workers=min(8, len(arg_dicts))) as pool:
            futures = {pool.submit(self._call_llm_responses, **kwargs): idx for idx, kwargs in enumerate(arg_dicts)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    responses = future.result()
                    if responses:
                        outputs[idx] = responses[0]
                except Exception as exc:
                    if arg_dicts[idx].get("verbose"):
                        print(f"[parallel-llm] worker {idx} failed: {exc}")
        return [output for output in outputs if output is not None]

    def _candidate_text(self, candidate: Any) -> str:
        if candidate is None:
            return ""
        if isinstance(candidate, str):
            return candidate.strip()
        if isinstance(candidate, dict):
            if "raw" in candidate or "text" in candidate:
                return str(candidate.get("raw", candidate.get("text", ""))).strip()
            return str(candidate).strip()
        return str(candidate).strip()

    def _candidate_variables_from_extraction(self, extracted: Any) -> Dict[str, Any]:
        if not isinstance(extracted, dict):
            return {}
        if isinstance(extracted.get("variables"), dict):
            return extracted["variables"]
        if isinstance(extracted.get("suggestion"), dict):
            return extracted["suggestion"]
        return {
            key: value
            for key, value in extracted.items()
            if key not in {"reasoning", "answer", "raw", "text", "parsed", "valid", "terminate"}
        }

    def _parse_candidate(self, candidate: Any) -> Dict[str, Any]:
        if isinstance(candidate, dict) and any(key in candidate for key in ("variables", "parsed", "terminate", "valid")):
            variables = self._candidate_variables_from_extraction(candidate)
            return {
                "raw": candidate.get("raw", self._candidate_text(candidate)),
                "parsed": candidate.get("parsed", candidate),
                "variables": variables,
                "reasoning": candidate.get("reasoning", ""),
                "valid": bool(candidate.get("valid", variables)),
                "terminate": bool(candidate.get("terminate", False)),
            }
        if isinstance(candidate, dict):
            variables = self._candidate_variables_from_extraction(candidate)
            return {
                "raw": candidate,
                "parsed": candidate,
                "variables": variables,
                "reasoning": candidate.get("reasoning", ""),
                "valid": bool(variables),
                "terminate": False,
            }

        raw_text = self._candidate_text(candidate)
        if raw_text.upper() == "TERMINATE":
            return {"raw": raw_text, "parsed": {}, "variables": {}, "reasoning": "", "valid": False, "terminate": True}
        try:
            extracted = self.extract_llm_suggestion(raw_text)
        except Exception:
            extracted = {}
        variables = self._candidate_variables_from_extraction(extracted)
        return {
            "raw": raw_text,
            "parsed": extracted,
            "variables": variables,
            "reasoning": extracted.get("reasoning", "") if isinstance(extracted, dict) else "",
            "valid": bool(variables),
            "terminate": False,
        }

    def _resolve_candidate_details(self, candidate: Any, candidates: List[Any], verbose: bool = False) -> Dict[str, Any]:
        details = self._parse_candidate(candidate)
        if details["valid"]:
            return details
        for fallback in reversed(candidates):
            fallback_details = self._parse_candidate(fallback)
            if fallback_details["valid"]:
                if verbose:
                    print("Falling back to the last parseable candidate.")
                return fallback_details
        return details

    def _build_update_dict_from_candidate(self, candidate: Any, candidates: List[Any], verbose: bool = False):
        details = self._resolve_candidate_details(candidate, candidates, verbose=verbose)
        if not details["valid"]:
            return {}, details
        return self.construct_update_dict(details["variables"]), details

    def generate_candidates(
        self,
        summary: Any,
        system_prompt: str,
        user_prompt: str,
        verbose: Union[bool, str] = False,
        mask: Any = None,
        max_tokens: Optional[int] = None,
        num_responses: int = 3,
        generation_technique: str = "temperature_variation",
        temperature_min_max: Optional[List[float]] = None,
        experts_list: Optional[List[str]] = None,
    ) -> List[str]:
        del mask
        num_responses = num_responses or self.num_responses
        max_tokens = max_tokens or self.max_tokens
        temp_range = list(temperature_min_max or self.temperature_min_max or [0.0, 1.0])
        temp_min, temp_max = (temp_range[0], temp_range[1]) if len(temp_range) >= 2 else (0.0, 1.0)
        generation_technique = (generation_technique or "temperature_variation").lower()
        candidates: List[str] = []

        if generation_technique == "multi_llm" and self.llm_profiles:
            llms = self._get_llms_for_generation(num_responses)
            arg_dicts = [
                {
                    "system_prompt": f"{system_prompt}\n\n[LLM profile: {self.llm_profiles[idx % len(self.llm_profiles)]}]",
                    "user_prompt": user_prompt,
                    "verbose": verbose,
                    "max_tokens": max_tokens,
                    "num_responses": 1,
                    "temperature": temp_min,
                    "llm": llm,
                }
                for idx, llm in enumerate(llms)
            ]
            candidates.extend(self._parallel_call_llm(arg_dicts))
        elif generation_technique == "self_refinement":
            for _ in range(num_responses):
                meta_prompt = system_prompt
                if candidates:
                    meta_prompt = (
                        f"{system_prompt}\n\nRefine the previous candidate for the same problem. "
                        "Preserve the exact output format specified above. Suggest changes only for "
                        "trainable variables or trainable code. Never modify fixed inputs.\n"
                        f"PREVIOUS_CANDIDATE:\n<<<\n{candidates[-1]}\n>>>"
                    )
                response = self._call_llm_responses(
                    system_prompt=meta_prompt,
                    user_prompt=user_prompt,
                    verbose=verbose,
                    max_tokens=max_tokens,
                    num_responses=1,
                    temperature=temp_min,
                )
                if response:
                    candidates.append(response[0])
        elif generation_technique == "iterative_alternatives":
            for _ in range(num_responses):
                meta_prompt = system_prompt
                if candidates:
                    previous = "\n".join(f"CANDIDATE {idx + 1}:\n<<<\n{candidate}\n>>>" for idx, candidate in enumerate(candidates))
                    meta_prompt = (
                        f"{system_prompt}\n\nGenerate a materially different new candidate for the same problem. "
                        "Preserve the exact output format specified above. Suggest changes only for "
                        "trainable variables or trainable code. Never modify fixed inputs.\n"
                        f"{previous}"
                    )
                response = self._call_llm_responses(
                    system_prompt=meta_prompt,
                    user_prompt=user_prompt,
                    verbose=verbose,
                    max_tokens=max_tokens,
                    num_responses=1,
                    temperature=temp_min,
                )
                if response:
                    candidates.append(response[0])
        elif generation_technique == "multi_experts":
            experts = self._expert_list(system_prompt, user_prompt, num_responses, verbose, max_tokens, experts_list)
            arg_dicts = [
                {
                    "system_prompt": (
                        f"You are a `{expert}`. Provide your strongest candidate solution for the problem below. "
                        f"Follow the exact output format specified below.\n{self.output_format_prompt}"
                    ),
                    "user_prompt": f"PROBLEM:\n\n{user_prompt}",
                    "verbose": verbose,
                    "max_tokens": max_tokens,
                    "num_responses": 1,
                    "temperature": 0.0,
                }
                for expert in experts[:num_responses]
            ]
            candidates.extend(self._parallel_call_llm(arg_dicts))

        if not candidates or generation_technique == "temperature_variation":
            if generation_technique not in {"temperature_variation", "multi_llm", "self_refinement", "iterative_alternatives", "multi_experts"} and verbose:
                print(f"Unknown generation technique: {generation_technique}; defaulting to temperature_variation")
            temperatures = [temp_max - i * (temp_max - temp_min) / max(1, num_responses - 1) for i in range(num_responses)]
            arg_dicts = [
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "verbose": verbose,
                    "max_tokens": max_tokens,
                    "num_responses": 1,
                    "temperature": temperature,
                }
                for temperature in temperatures
            ]
            candidates.extend(self._parallel_call_llm(arg_dicts))

        if self.log is not None:
            self.log.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": candidates,
                    "generation_technique": generation_technique,
                    "llm_profiles": self.llm_profiles,
                }
            )
            self.summary_log.append({"problem_instance": self.problem_instance(summary) if summary is not None else {}, "summary": summary})
        return candidates

    def _expert_list(
        self,
        system_prompt: str,
        user_prompt: str,
        num_responses: int,
        verbose: Union[bool, str],
        max_tokens: int,
        experts_list: Optional[List[str]],
    ) -> List[str]:
        experts: List[str] = []
        if isinstance(experts_list, list) and all(isinstance(expert, str) for expert in experts_list):
            experts = list(experts_list)
        else:
            expert_json = self._call_llm_responses(
                system_prompt=(
                    "Generate a JSON object with key `experts`, where `experts` is an array "
                    "of complementary expert persona strings that would help optimize the problem."
                ),
                user_prompt=f"NUMBER_OF_EXPERTS={num_responses}\nPROBLEM:\n<<<\n{system_prompt}\n{user_prompt}\n>>>",
                verbose=verbose,
                max_tokens=max_tokens,
                num_responses=1,
                temperature=0.0,
            )
            if expert_json:
                try:
                    parsed = json.loads(expert_json[0])
                    if isinstance(parsed, dict):
                        if isinstance(parsed.get("experts"), list):
                            parsed = parsed["experts"]
                        elif len(parsed) == 1 and isinstance(next(iter(parsed.values())), list):
                            parsed = next(iter(parsed.values()))
                        else:
                            parsed = []
                    if isinstance(parsed, list):
                        experts = [str(expert) for expert in parsed]
                except Exception:
                    experts = []
        default_experts = ["Algorithm Expert", "Performance Optimizer", "Prompt Engineer", "Compiler Specialist", "Critical Reviewer"]
        while len(experts) < num_responses:
            experts.append(default_experts[len(experts) % len(default_experts)])
        return experts

    def select_candidate(self, candidates: List[Any], selection_technique: str = "best_of_n", problem_summary: str = "") -> Any:
        if not candidates:
            return {}
        if len(candidates) == 1:
            return candidates[0]
        selection_technique = (selection_technique or "last_of_n").lower()
        candidate_texts = [self._candidate_text(candidate) for candidate in candidates]
        if selection_technique in {"moa", "mixture_of_agents"}:
            return self._select_moa(candidates, candidate_texts, problem_summary)
        if selection_technique in {"bestofn", "best_of_n"}:
            return self._select_bestofn(candidates, candidate_texts, problem_summary)
        if selection_technique == "majority":
            return self._select_majority(candidates, candidate_texts)
        return candidates[-1]

    def _select_moa(self, candidates: List[Any], candidate_texts: List[str], summary: Optional[str] = None) -> Any:
        system_prompt = (
            "You are an expert at synthesizing multiple candidate updates into one stronger candidate. "
            f"Follow the exact output format specified below.\n{self.output_format_prompt}"
        )
        prefix = f"PROBLEM:\n{summary}\n\n" if summary else ""
        user_prompt = prefix + "".join(f"CANDIDATE {idx}:\n{text}\n\n" for idx, text in enumerate(candidate_texts, start=1))
        response = self._call_llm_responses(system_prompt=system_prompt, user_prompt=user_prompt, num_responses=1, temperature=0.0)
        return response[0] if response else candidates[-1]

    def _select_bestofn(self, candidates: List[Any], candidate_texts: List[str], summary: Optional[str] = None) -> Any:
        system_prompt = (
            "You are an expert evaluator of candidate optimizer updates. Choose or synthesize the most promising candidate update. "
            f"Follow the exact output format specified below.\n{self.output_format_prompt}"
        )
        prefix = f"PROBLEM:\n{summary}\n\n" if summary else ""
        user_prompt = prefix + "".join(f"CANDIDATE {idx}:\n{text}\n\n" for idx, text in enumerate(candidate_texts, start=1))
        response = self._call_llm_responses(system_prompt=system_prompt, user_prompt=user_prompt, num_responses=1, temperature=0.0)
        return response[0] if response else candidates[-1]

    def _select_majority(self, candidates: List[Any], candidate_texts: List[str]) -> Any:
        if len(candidate_texts) <= 1:
            return candidates[0]
        most_common_text, count = Counter(candidate_texts).most_common(1)[0]
        if count > 1:
            return candidates[candidate_texts.index(most_common_text)]
        try:
            import numpy as np
            from sklearn.cluster import AgglomerativeClustering
        except Exception:
            return candidates[-1]

        n = len(candidate_texts)
        distance = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                similarity = SequenceMatcher(None, candidate_texts[i], candidate_texts[j]).ratio()
                distance[i, j] = distance[j, i] = 1 - similarity
        try:
            clusterer = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="complete", distance_threshold=0.2)
        except TypeError:
            clusterer = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage="complete", distance_threshold=0.2)
        labels = clusterer.fit(distance).labels_
        if len(set(labels)) == 1:
            return candidates[-1]
        top_label = Counter(labels).most_common(1)[0][0]
        indices = [idx for idx, label in enumerate(labels) if label == top_label]
        sub_distances = distance[np.ix_(indices, indices)]
        medoid_idx = indices[int(sub_distances.sum(axis=1).argmin())]
        return candidates[medoid_idx]

    def _step(
        self,
        verbose: Union[bool, str] = False,
        mask: Any = None,
        num_responses: Optional[int] = None,
        temperature_min_max: Optional[List[float]] = None,
        selector: Optional[Callable[[List[Any]], Any]] = None,
        generation_technique: Optional[str] = None,
        selection_technique: Optional[str] = None,
        experts_list: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> Dict[Any, Any]:
        del args, kwargs
        summary = self.summarize()
        system_prompt, user_prompt = self.construct_prompt(summary, mask=mask)
        if hasattr(self, "replace_symbols") and hasattr(self, "prompt_symbols"):
            system_prompt = self.replace_symbols(system_prompt, self.prompt_symbols)
            user_prompt = self.replace_symbols(user_prompt, self.prompt_symbols)

        num_responses = num_responses or self.num_responses
        selector = selector or self.selector
        generation_technique = generation_technique or self.generation_technique
        selection_technique = selection_technique or self.selection_technique
        experts_list = experts_list or self.experts_list
        problem_summary = f"{system_prompt}\n\n{user_prompt}"

        self.candidates = self.generate_candidates(
            summary=summary,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            verbose=verbose,
            mask=mask,
            num_responses=num_responses,
            temperature_min_max=temperature_min_max or self.temperature_min_max,
            generation_technique=generation_technique,
            experts_list=experts_list,
        )
        self.candidate_details = [self._parse_candidate(candidate) for candidate in self.candidates]

        if not self.candidates:
            self.selected_candidate = None
            self.selected_candidate_details = None
            return {}
        if all(details["terminate"] for details in self.candidate_details):
            self.selected_candidate = "TERMINATE"
            self.selected_candidate_details = {"raw": "TERMINATE", "parsed": {}, "variables": {}, "reasoning": "", "valid": False, "terminate": True}
            return {}

        self.selected_candidate = selector(self.candidates) if selector and callable(selector) else self.select_candidate(
            candidates=self.candidates,
            selection_technique=selection_technique,
            problem_summary=problem_summary,
        )
        update_dict, details = self._build_update_dict_from_candidate(self.selected_candidate, self.candidates, verbose=bool(verbose))
        self.selected_candidate_details = details
        if verbose:
            print(
                "Generated candidates:",
                self.candidates,
                "\nSelected candidate:",
                self.selected_candidate,
                "\nSelected candidate details:",
                details,
                "\nUpdate dict:",
                update_dict,
            )
        return update_dict
