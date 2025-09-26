# opto/trainer/algorithms/gepa_algorithms.py
# GEPA (+Merge) algorithms for Trace
# - GEPAUCBSearch: subclass of UCBSearchAlgorithm
# - GEPABeamPareto: subclass of BeamsearchAlgorithm (Pareto select + single-parent incremental)
# - GEPAAlgorithmBase: subclass of AlgorithmBase (minimal GEPA loop)
#
# All default to OptoPrimeV2 if optimizer=None.

from __future__ import annotations
import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from opto.optimizers.optoprime_v2 import OptoPrimeV2
from opto.trace.nodes import ParameterNode
from opto.trainer.algorithms.UCBsearch import UCBSearchAlgorithm
from opto.trainer.algorithms.beamsearch_algorithm import BeamsearchAlgorithm
from opto.trainer.algorithms.algorithm import Trainer as AlgorithmBase
from opto.trainer.algorithms.basic_algorithms import (
    evaluate,
    batchify,
    standard_optimization_step,
)
from opto.trainer.utils import async_run
from opto.optimizers.utils import print_color


# ----------------------------- Utilities ----------------------------- #

@dataclass
class Candidate:
    params: Dict[ParameterNode, Any]
    eval_vector: List[float]          # per-instance scores on fixed Pareto subset
    mean: float
    id: int
    parent_ids: Tuple[int, ...] = field(default_factory=tuple)
    ancestors: set = field(default_factory=set)
    created_iter: int = 0
    wins: int = 0                      # updated by Pareto accounting
    meta: Dict[str, Any] = field(default_factory=dict)  # freeform

def _eval_on_subset(agent, guide, xs, infos, *, num_threads: Optional[int], desc: str) -> List[float]:
    return evaluate(agent, guide, xs, infos, min_score=None, num_threads=num_threads, description=desc)

def _compute_pareto_counts(cands: List[Candidate]) -> None:
    """
    "Best-for-at-least-one-instance" winners.
    For each position m in eval vectors, find argmax candidate and credit a win.
    """
    if not cands:
        return
    L = len(cands[0].eval_vector)
    # Reset
    for c in cands:
        c.wins = 0
    # Credit wins
    for m in range(L):
        best_idx = None
        best_val = -float("inf")
        for i, c in enumerate(cands):
            v = c.eval_vector[m] if m < len(c.eval_vector) else -float("inf")
            if v > best_val:
                best_val, best_idx = v, i
        if best_idx is not None:
            cands[best_idx].wins += 1

def _pareto_sample(cands: List[Candidate], *, temperature: float = 1.0, rng: random.Random) -> Candidate:
    """
    Sample a parent from union of per-instance winners, proportional to wins^1/T.
    """
    if not cands:
        raise ValueError("Empty candidate buffer.")
    _compute_pareto_counts(cands)
    wins = np.array([max(1, c.wins) for c in cands], dtype=float)  # avoid zero
    if temperature <= 0:
        # Deterministic pick
        return cands[int(wins.argmax())]
    weights = wins ** (1.0 / max(1e-6, temperature))
    probs = weights / (weights.sum() if weights.sum() > 0 else 1.0)
    idx = rng.choices(range(len(cands)), weights=probs, k=1)[0]
    return cands[idx]

def _uniform_merge_params(a: Dict[ParameterNode, Any], b: Dict[ParameterNode, Any], rng: random.Random) -> Dict[ParameterNode, Any]:
    """
    Simple, robust "crossover": per-parameter uniform pick between parents.
    (System-aware enough for prompt/code params, cheap, and safe.)
    """
    keys = set(a.keys()) | set(b.keys())
    merged: Dict[ParameterNode, Any] = {}
    for p in keys:
        if p in a and p in b:
            merged[p] = copy.deepcopy(a[p] if rng.random() < 0.5 else b[p])
        elif p in a:
            merged[p] = copy.deepcopy(a[p])
        else:
            merged[p] = copy.deepcopy(b[p])
    return merged

def _maybe_merge(buffer: List[Candidate],
                 *,
                 agent,
                 guide,
                 pareto_inputs: List[Any],
                 pareto_infos: List[Any],
                 num_threads: Optional[int],
                 rng: random.Random,
                 tried_pairs: set,
                 max_tries: int = 8) -> Optional[Candidate]:
    """
    Try merging two non-lineage candidates once; return merged if better than both parents' mean, else None.
    """
    if len(buffer) < 2:
        return None
    # Prefer winners
    _compute_pareto_counts(buffer)
    pool = sorted(buffer, key=lambda c: (c.wins, c.mean), reverse=True)

    # Try a few distinct pairs
    for _ in range(max_tries):
        i, j = rng.sample(range(len(pool)), 2)
        a, b = pool[i], pool[j]
        if a.id == b.id:
            continue
        if a.id in b.ancestors or b.id in a.ancestors:
            continue  # avoid direct ancestry
        key = tuple(sorted((a.id, b.id)))
        if key in tried_pairs:
            continue
        tried_pairs.add(key)

        merged_params = _uniform_merge_params(a.params, b.params, rng)
        # Evaluate merged on Pareto subset
        original_params = {p: copy.deepcopy(p.data) for p in agent.parameters()}
        try:
            # load params to agent
            from opto.optimizers.optimizer import Optimizer  # type: ignore
            # We only need the parameters dict projection; we can set via optimizer.update if available
            # But we don't have an optimizer here; use ParameterNode._set
            for p, v in merged_params.items():
                p._set(v)

            vec = _eval_on_subset(agent, guide, pareto_inputs, pareto_infos, num_threads=num_threads,
                                  desc="GEPA+Merge: evaluating merged")
            mean = float(np.mean(vec)) if all(s is not None for s in vec) else -float("inf")
        finally:
            # restore original
            for p, v in original_params.items():
                p._set(v)

        if mean > max(a.mean, b.mean):
            merged = Candidate(params=merged_params,
                               eval_vector=vec,
                               mean=mean,
                               id=-1,  # to be set by caller
                               parent_ids=(a.id, b.id),
                               ancestors=set(a.ancestors) | set(b.ancestors) | {a.id, b.id},
                               created_iter=0)
            return merged
    return None


def _ensure_optimizer(agent, optimizer):
    if optimizer is not None:
        return optimizer
    params = [p for p in agent.parameters()]  # List[ParameterNode]
    return OptoPrimeV2(parameters=params)


def _train_step_generate_child(agent, guide, optimizer, train_xs, train_infos, *, verbose=False, num_threads=None):
    """
    Single-parent, incremental evolution "mutation": run forward on a minibatch to get batched feedback,
    then optimizer.step(bypassing=True) to obtain a new candidate param dict (without applying).
    """
    use_async = num_threads is not None and num_threads > 1
    if use_async:
        outputs = async_run([lambda a,x,g,info: standard_optimization_step(a, x, g, info)] * len(train_xs),
                            args_list=[(agent, x, guide, info) for x, info in zip(train_xs, train_infos)],
                            max_workers=num_threads,
                            description="GEPA forward (mutate parent)")
        # outputs: List[(target, score, feedback)]
    else:
        outputs = [standard_optimization_step(agent, x, guide, info) for x, info in zip(train_xs, train_infos)]

    scores, targets, feedbacks = [], [], []
    for target, score, feedback in outputs:
        scores.append(score)
        targets.append(target)
        feedbacks.append(feedback)

    target_batch = batchify(*targets)
    feedback_batch = batchify(*feedbacks).data

    optimizer.zero_feedback()
    optimizer.backward(target_batch, feedback_batch)
    try:
        update_dict = optimizer.step(bypassing=True, verbose=("output" if verbose else False))
        if not isinstance(update_dict, dict) or len(update_dict) == 0:
            # Fallback: treat current as child (rare)
            update_dict = {p: copy.deepcopy(p.data) for p in optimizer.parameters}
    except Exception as e:
        print_color(f"[GEPA] optimizer.step error: {e}", "red")
        update_dict = {}
    return update_dict, (None if not scores or any(s is None for s in scores) else float(np.mean(scores)))


def _apply_params(optimizer, param_dict: Dict[ParameterNode, Any]):
    """Load param dict into the agent via optimizer.update (preserves projections)."""
    optimizer.update(param_dict)


# ======================= Variant 1: GEPA + Merge (UCB subclass) ======================= #

class GEPAUCBSearch(UCBSearchAlgorithm):
    """
    GEPA (+Merge) implemented atop UCBSearchAlgorithm.
    Differences vs base UCB:
      - Fixed Pareto subset (D_pareto) and per-instance vectors kept for each candidate
      - Parent selection = Pareto "best-for-at-least-one" sampling (wins-weighted); UCB used only for eviction fallback
      - Single-parent incremental mutation via a minibatch
      - Optional periodic Merge crossover (uniform per-parameter) with desirability checks
    """

    def __init__(self,
                 agent,
                 optimizer=None,
                 *,
                 max_buffer_size: int = 16,
                 ucb_exploration_factor: float = 0.8,
                 rng_seed: int = 7,
                 logger=None,
                 num_threads: Optional[int] = None):
        optimizer = _ensure_optimizer(agent, optimizer)
        super().__init__(agent, optimizer,
                         max_buffer_size=max_buffer_size,
                         ucb_exploration_factor=ucb_exploration_factor,
                         logger=logger,
                         num_threads=num_threads)
        self.rng = random.Random(rng_seed)
        self._pareto_inputs: List[Any] = []
        self._pareto_infos: List[Any] = []
        self._id_counter = 0

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def _evaluate_on_pareto(self, params_dict: Dict[ParameterNode, Any], guide, *, num_threads) -> Tuple[List[float], float]:
        original_params = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        try:
            _apply_params(self.optimizer, params_dict)
            vec = _eval_on_subset(self.agent, guide, self._pareto_inputs, self._pareto_infos,
                                  num_threads=num_threads, desc="GEPA: evaluate on Pareto subset")
            mean = float(np.mean(vec)) if all(s is not None for s in vec) else -float("inf")
            return vec, mean
        finally:
            _apply_params(self.optimizer, original_params)

    def _select_pareto_parent(self, cand_buffer: List[Candidate]) -> Candidate:
        return _pareto_sample(cand_buffer, temperature=1.0, rng=self.rng)

    def train(self,
              guide,
              train_dataset: Dict[str, List[Any]],
              *,
              validation_dataset: Optional[Dict[str, List[Any]]] = None,
              pareto_subset_size: int = 24,
              num_search_iterations: int = 120,
              train_batch_size: int = 2,
              merge_every: int = 6,
              log_frequency: Optional[int] = None,
              save_frequency: Optional[int] = None,
              save_path: str = "checkpoints/gepa_ucb_agent.pkl",
              verbose: bool = False,
              num_threads: Optional[int] = None) -> Tuple[Dict[str, Any], float]:
        """
        GEPA search loop with Pareto sampling + (optional) Merge.
        """
        num_threads = num_threads or self.num_threads
        log_frequency = log_frequency or 5
        validate_ds = validation_dataset or train_dataset

        # Fix a Pareto subset (small, stable) to compute per-instance vectors
        assert len(validate_ds["inputs"]) > 0, "Empty dataset."
        idxs = np.random.choice(len(validate_ds["inputs"]),
                                min(pareto_subset_size, len(validate_ds["inputs"])),
                                replace=False)
        self._pareto_inputs = [validate_ds["inputs"][i] for i in idxs]
        self._pareto_infos  = [validate_ds["infos"][i]  for i in idxs]

        buffer: List[Candidate] = []
        tried_merges: set = set()

        # Seed with current params
        base_params = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        v0, m0 = self._evaluate_on_pareto(base_params, guide, num_threads=num_threads)
        buffer.append(Candidate(params=base_params, eval_vector=v0, mean=m0, id=self._next_id(), ancestors=set()))
        print_color(f"[GEPA] Seed candidate mean={m0:.4f}", "cyan")

        metrics = {"best_means": [], "new_child_means": [], "merge_accepts": 0, "total_merges": 0}

        for it in range(1, num_search_iterations + 1):
            # Select parent by Pareto winners
            parent = self._select_pareto_parent(buffer)
            _apply_params(self.optimizer, parent.params)

            # Sample train minibatch
            train_size = min(train_batch_size, len(train_dataset["inputs"]))
            tr_idxs = np.random.choice(len(train_dataset["inputs"]), train_size, replace=False)
            train_xs   = [train_dataset["inputs"][i] for i in tr_idxs]
            train_info = [train_dataset["infos"][i]  for i in tr_idxs]

            # Generate child via one incremental step
            update_dict, train_batch_mean = _train_step_generate_child(
                self.agent, guide, self.optimizer, train_xs, train_info, verbose=verbose, num_threads=num_threads
            )
            if not update_dict:
                print_color("[GEPA] Empty child update; skipping.", "yellow")
                continue

            # Evaluate child on Pareto subset
            child_vec, child_mean = self._evaluate_on_pareto(update_dict, guide, num_threads=num_threads)
            child = Candidate(params=update_dict,
                              eval_vector=child_vec,
                              mean=child_mean,
                              id=self._next_id(),
                              parent_ids=(parent.id,),
                              ancestors=set(parent.ancestors) | {parent.id},
                              created_iter=it)
            buffer.append(child)
            metrics["new_child_means"].append(child_mean)
            print_color(f"[GEPA] iter {it}: child mean={child_mean:.4f} (train-batch≈{train_batch_mean})", "green")

            # Optional Merge
            if merge_every and (it % merge_every == 0):
                metrics["total_merges"] += 1
                merged = _maybe_merge(buffer,
                                      agent=self.agent, guide=guide,
                                      pareto_inputs=self._pareto_inputs,
                                      pareto_infos=self._pareto_infos,
                                      num_threads=num_threads,
                                      rng=self.rng,
                                      tried_pairs=tried_merges)
                if merged is not None:
                    merged.id = self._next_id()
                    merged.created_iter = it
                    buffer.append(merged)
                    metrics["merge_accepts"] += 1
                    print_color(f"[GEPA] Merge accepted: mean={merged.mean:.4f}", "magenta")

            # Keep buffer bounded: remove the candidate with lowest (wins, mean)
            if len(buffer) > self.max_buffer_size:
                _compute_pareto_counts(buffer)
                buffer.sort(key=lambda c: (c.wins, c.mean))
                evicted = buffer.pop(0)
                print_color(f"[GEPA] Evicted cand#{evicted.id} (wins={evicted.wins}, mean={evicted.mean:.4f})", "yellow")

            # Track & log
            best = max(buffer, key=lambda c: c.mean)
            metrics["best_means"].append(best.mean)
            if it % log_frequency == 0:
                self.logger.log("GEPA best mean", best.mean, it, color="green")

            # Save best candidate snapshot (optional)
            if save_frequency and it % save_frequency == 0:
                _apply_params(self.optimizer, best.params)
                self.save_agent(save_path, it)

        # Load best into the agent and return
        best = max(buffer, key=lambda c: c.mean) if buffer else buffer[0]
        _apply_params(self.optimizer, best.params)
        return metrics, float(best.mean)


# ================= Variant 2: Beamsearch subclass with Pareto select ================= #

class GEPABeamPareto(BeamsearchAlgorithm):
    """
    BeamsearchAlgorithm retrofit:
      - override select() to a Pareto "best-for-at-least-one" selector
      - replace deep beam expansion with GEPA’s single-parent incremental evolution
    """

    def __init__(self,
                 agent,
                 optimizer=None,
                 *,
                 rng_seed: int = 11,
                 logger=None,
                 num_threads: Optional[int] = None):
        optimizer = _ensure_optimizer(agent, optimizer)
        super().__init__(agent, optimizer, num_threads=num_threads, logger=logger)
        self.rng = random.Random(rng_seed)

    # We keep a Pareto select helper that returns (selected_params, wins, scores)
    def select(self,
               candidates: List[Dict[ParameterNode, Any]],
               validate_guide,
               validation_mini_dataset,
               beam_width: int,
               num_threads: int = None,
               min_score: float = None,
               return_scores: bool = False):
        """
        Override to Pareto union-of-winners on the mini validation batch.
        """
        # Evaluate each candidate to a vector on the mini validation
        cand_objs: List[Candidate] = []
        current_params = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        try:
            for idx, params in enumerate(candidates):
                _apply_params(self.optimizer, params)
                vec = evaluate(self.agent,
                               validate_guide,
                               validation_mini_dataset['inputs'],
                               validation_mini_dataset['infos'],
                               min_score=min_score,
                               num_threads=num_threads,
                               description=f"Validating candidate {idx+1}/{len(candidates)} (Pareto)")
                mean = float(np.mean(vec)) if all(s is not None for s in vec) else -float("inf")
                cand_objs.append(Candidate(params=params, eval_vector=vec, mean=mean, id=idx))
        finally:
            _apply_params(self.optimizer, current_params)

        # Compute wins and select top "beam_width" by (wins, mean)
        _compute_pareto_counts(cand_objs)
        cand_objs.sort(key=lambda c: (c.wins, c.mean), reverse=True)
        selected = cand_objs[: min(beam_width, len(cand_objs))]
        sel_params = [c.params for c in selected]
        sel_scores = [c.mean for c in selected]
        if return_scores:
            return sel_params, sel_scores
        return sel_params

    # Replace beam "train" with GEPA-style incremental loop (keeps BeamsearchAlgorithm API)
    def train(self,
              guide,
              train_dataset,
              *,
              validate_dataset=None,
              pareto_subset_size: int = 24,
              num_search_iterations: int = 120,
              train_batch_size: int = 2,
              merge_every: int = 6,
              log_frequency: Optional[int] = None,
              save_frequency: Optional[int] = None,
              save_path: str = "checkpoints/gepa_beam_agent.pkl",
              verbose: bool = False,
              num_threads: Optional[int] = None):
        num_threads = num_threads or self.num_threads
        log_frequency = log_frequency or 5
        validate_ds = validate_dataset or train_dataset

        # Fix Pareto subset for this run
        idxs = np.random.choice(len(validate_ds["inputs"]),
                                min(pareto_subset_size, len(validate_ds["inputs"])),
                                replace=False)
        pareto_inputs = [validate_ds["inputs"][i] for i in idxs]
        pareto_infos  = [validate_ds["infos"][i]  for i in idxs]

        # Seed buffer
        buffer: List[Candidate] = []
        base_params = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        # Evaluate seed
        current_params = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        try:
            _apply_params(self.optimizer, base_params)
            vec = evaluate(self.agent, guide, pareto_inputs, pareto_infos,
                           min_score=None, num_threads=num_threads,
                           description="GEPA(beam): seed evaluation")
        finally:
            _apply_params(self.optimizer, current_params)
        m0 = float(np.mean(vec)) if all(s is not None for s in vec) else -float("inf")
        buffer.append(Candidate(params=base_params, eval_vector=vec, mean=m0, id=0, ancestors=set()))
        tried_merges: set = set()

        best_mean = m0
        for it in range(1, num_search_iterations + 1):
            # Pareto-select parent and mutate
            _compute_pareto_counts(buffer)
            parent = _pareto_sample(buffer, temperature=1.0, rng=self.rng)
            _apply_params(self.optimizer, parent.params)

            # Make a child
            k = min(train_batch_size, len(train_dataset["inputs"]))
            tr = np.random.choice(len(train_dataset["inputs"]), k, replace=False)
            train_xs = [train_dataset["inputs"][i] for i in tr]
            train_in = [train_dataset["infos"][i]  for i in tr]

            update_dict, _ = _train_step_generate_child(self.agent, guide, self.optimizer, train_xs, train_in,
                                                        verbose=verbose, num_threads=num_threads)
            if not update_dict:
                continue

            # Evaluate child on Pareto subset
            current_params = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
            try:
                _apply_params(self.optimizer, update_dict)
                vec = evaluate(self.agent, guide, pareto_inputs, pareto_infos, min_score=None,
                               num_threads=num_threads, description="GEPA(beam): child eval")
            finally:
                _apply_params(self.optimizer, current_params)
            mean = float(np.mean(vec)) if all(s is not None for s in vec) else -float("inf")
            buffer.append(Candidate(params=update_dict, eval_vector=vec, mean=mean, id=len(buffer),
                                    parent_ids=(parent.id,), ancestors=set(parent.ancestors) | {parent.id}))
            best_mean = max(best_mean, mean)
            if it % log_frequency == 0:
                self.logger.log("GEPA(beam) best mean", best_mean, it, color="green")

            # Periodic merge
            if merge_every and it % merge_every == 0:
                merged = _maybe_merge(buffer,
                                      agent=self.agent, guide=guide,
                                      pareto_inputs=pareto_inputs, pareto_infos=pareto_infos,
                                      num_threads=num_threads, rng=self.rng, tried_pairs=tried_merges)
                if merged is not None:
                    merged.id = len(buffer)
                    buffer.append(merged)

            # Trim buffer softly (keep top by (wins, mean))
            if len(buffer) > 16:
                _compute_pareto_counts(buffer)
                buffer.sort(key=lambda c: (c.wins, c.mean), reverse=True)
                buffer[:] = buffer[:16]

            # Optional save
            if save_frequency and it % save_frequency == 0:
                best = max(buffer, key=lambda c: c.mean)
                _apply_params(self.optimizer, best.params)
                self.save_agent(save_path, it)

        best = max(buffer, key=lambda c: c.mean)
        _apply_params(self.optimizer, best.params)
        return {"best_mean": best.mean}, float(best.mean)


# =================== Variant 3: Minimal GEPA on AlgorithmBase =================== #

class GEPAAlgorithmBase(AlgorithmBase):
    """
    Lightweight GEPA (+Merge) with only AlgorithmBase dependency.
    Useful when you want the simplest control loop with your own logging/saving.
    """

    def __init__(self,
                 agent,
                 optimizer=None,
                 *,
                 rng_seed: int = 13,
                 logger=None,
                 num_threads: Optional[int] = None):
        super().__init__(agent, num_threads=num_threads, logger=logger)
        self.optimizer = _ensure_optimizer(agent, optimizer)
        self.rng = random.Random(rng_seed)

    def train(self,
              guide,
              train_dataset,
              *,
              validate_dataset=None,
              pareto_subset_size: int = 24,
              num_iters: int = 100,
              train_batch_size: int = 2,
              merge_every: int = 5,
              num_threads: Optional[int] = None,
              save_path: Optional[str] = None):
        num_threads = num_threads or self.num_threads
        validate_ds = validate_dataset or train_dataset

        # Pareto subset
        idxs = np.random.choice(len(validate_ds["inputs"]),
                                min(pareto_subset_size, len(validate_ds["inputs"])),
                                replace=False)
        xsP = [validate_ds["inputs"][i] for i in idxs]
        isP = [validate_ds["infos"][i]  for i in idxs]

        # Seed
        buffer: List[Candidate] = []
        base_params = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        original = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
        try:
            _apply_params(self.optimizer, base_params)
            vec = evaluate(self.agent, guide, xsP, isP, min_score=None, num_threads=num_threads,
                           description="GEPA(base): seed eval")
        finally:
            _apply_params(self.optimizer, original)
        m0 = float(np.mean(vec)) if all(s is not None for s in vec) else -float("inf")
        buffer.append(Candidate(params=base_params, eval_vector=vec, mean=m0, id=0, ancestors=set()))
        tried_merges: set = set()

        for it in range(1, num_iters + 1):
            # Parent select
            _compute_pareto_counts(buffer)
            parent = _pareto_sample(buffer, temperature=1.0, rng=self.rng)
            _apply_params(self.optimizer, parent.params)

            # Child
            k = min(train_batch_size, len(train_dataset["inputs"]))
            tr = np.random.choice(len(train_dataset["inputs"]), k, replace=False)
            tx = [train_dataset["inputs"][i] for i in tr]
            ti = [train_dataset["infos"][i]  for i in tr]
            update_dict, _ = _train_step_generate_child(self.agent, guide, self.optimizer, tx, ti,
                                                        verbose=False, num_threads=num_threads)
            if not update_dict:
                continue

            # Eval child
            original = {p: copy.deepcopy(p.data) for p in self.optimizer.parameters}
            try:
                _apply_params(self.optimizer, update_dict)
                vec = evaluate(self.agent, guide, xsP, isP, min_score=None, num_threads=num_threads,
                               description="GEPA(base): child eval")
            finally:
                _apply_params(self.optimizer, original)
            mean = float(np.mean(vec)) if all(s is not None for s in vec) else -float("inf")
            buffer.append(Candidate(params=update_dict, eval_vector=vec, mean=mean, id=len(buffer),
                                    parent_ids=(parent.id,), ancestors=set(parent.ancestors) | {parent.id}))

            # Merge
            if merge_every and it % merge_every == 0:
                merged = _maybe_merge(buffer,
                                      agent=self.agent, guide=guide,
                                      pareto_inputs=xsP, pareto_infos=isP,
                                      num_threads=num_threads, rng=self.rng, tried_pairs=tried_merges)
                if merged is not None:
                    merged.id = len(buffer)
                    buffer.append(merged)

            # Keep compact buffer
            if len(buffer) > 16:
                _compute_pareto_counts(buffer)
                buffer.sort(key=lambda c: (c.wins, c.mean), reverse=True)
                buffer[:] = buffer[:16]

            # Log
            best = max(buffer, key=lambda c: c.mean)
            if self.logger:
                self.logger.log("GEPA(base) best mean", best.mean, it, color="green")

            # Optional save
            if save_path and it % 10 == 0:
                _apply_params(self.optimizer, best.params)
                self.save_agent(save_path, it)

        # Load best into agent
        best = max(buffer, key=lambda c: c.mean)
        _apply_params(self.optimizer, best.params)
        return {"best_mean": best.mean}, float(best.mean)

