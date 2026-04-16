import re
import numpy as np
import cvxpy as cp
from opto.trace.utils import dedent

def np_random(seed: int | None = None) -> tuple[np.random.Generator, int]:
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        if isinstance(seed, int) is False:
            raise Exception(f"Seed must be a python integer, actual type: {type(seed)}")
        else:
            raise Exception(f"Seed must be greater or equal to zero, actual value: {seed}")
    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    return rng, np_seed


def _norm_term(x: np.ndarray, norm_coef: float, norm_kind: str) -> float:
    if norm_coef == 0.0:
        return 0.0
    if norm_kind == "l2sq":
        return float(norm_coef * (x[0] ** 2 + x[1] ** 2))
    if norm_kind == "l2":
        return float(norm_coef * np.sqrt(x[0] ** 2 + x[1] ** 2))
    if norm_kind == "l1":
        return float(norm_coef * (abs(x[0]) + abs(x[1])))
    raise ValueError("norm_kind must be one of {'l2sq','l2','l1'}")


def _rosenbrock_cubic_global_min(a: float, b: float, lam: float) -> tuple[np.ndarray, float]:
    """
    For f(u,v)=(a-u)^2 + b(v-u^2)^2 + lam(u^2+v^2), b>0, lam>=0.
    Returns (x_star, f_star).
    """
    # lam == 0: classic Rosenbrock minimum at (a, a^2) with value 0.
    if lam == 0.0:
        x_star = np.array([a, a ** 2], dtype=float)
        f_star = 0.0
        return x_star, f_star

    # Solve cubic: c3*u^3 + (1+lam)*u - a = 0 with c3 = 2*b*lam/(b+lam)
    c3 = 2.0 * b * lam / (b + lam)
    c1 = 1.0 + lam
    roots = np.roots([c3, 0.0, c1, -a])

    best = None
    for r in roots:
        if abs(r.imag) < 1e-10:
            u = float(r.real)
            v = (b / (b + lam)) * u * u
            x = np.array([u, v], dtype=float)
            # evaluate full objective
            base = (a - u) ** 2 + b * (v - u * u) ** 2
            f = float(base + lam * (u * u + v * v))
            if best is None or f < best[1]:
                best = (x, f)

    if best is None:
        raise RuntimeError("Unexpected: cubic had no real root.")
    return best


# ---------------------------
# SOS / moment relaxation for Six-Hump Camel on a box
# ---------------------------

def _monomials_upto_degree(k: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for deg in range(k + 1):
        for i in range(deg + 1):
            j = deg - i
            out.append((i, j))
    return out

def _add_mono(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] + b[0], a[1] + b[1])

def _build_moment_matrix(y: dict[tuple[int,int], cp.Expression], basis: list[tuple[int,int]]) -> cp.Expression:
    m = len(basis)
    blocks = []
    for i in range(m):
        row = []
        for j in range(m):
            row.append(y[_add_mono(basis[i], basis[j])])
        blocks.append(row)
    return cp.bmat(blocks)

def _build_localizing_matrix_linear(
    y: dict[tuple[int,int], cp.Expression],
    basis: list[tuple[int,int]],
    g_lin: dict[tuple[int,int], float],  # g(x,y) = c00 + c10 x + c01 y
) -> cp.Expression:
    m = len(basis)
    blocks = []
    for i in range(m):
        row = []
        for j in range(m):
            a = _add_mono(basis[i], basis[j])
            expr = 0
            for beta, c in g_lin.items():
                expr += c * y[_add_mono(a, beta)]
            row.append(expr)
        blocks.append(row)
    return cp.bmat(blocks)

def _six_hump_coeffs(lam_l2sq: float = 0.0) -> dict[tuple[int,int], float]:
    """
    Base six-hump camel:
      f(x,y) = 4x^2 -2.1 x^4 + (1/3) x^6 + x y - 4y^2 + 4y^4

    With l2sq regularizer:
      f(x,y) + lam*(x^2 + y^2)
    => add lam to the (2,0) and (0,2) coefficients.
    """
    lam = float(lam_l2sq)
    return {
        (2, 0): 4.0 + lam,
        (4, 0): -2.1,
        (6, 0): 1.0 / 3.0,
        (1, 1): 1.0,
        (0, 2): -4.0 + lam,
        (0, 4): 4.0,
    }


def six_hump_sos_certificate_on_box(
    bound: float = 2.0,
    order_d: int = 3,
    solver: str = "SCS",
    verbose: bool = False,
    lam_l2sq: float = 0.0,
) -> tuple[float, str]:
    """
    Moment relaxation (Lasserre) order d for Six-Hump Camel (+ optional l2sq) on [-bound, bound]^2.
    Returns (lower_bound, status). lower_bound is the SOS certificate gamma.
    """
    if order_d < 3:
        raise ValueError("For degree-6 polynomial, use order_d >= 3")

    coeff = _six_hump_coeffs(lam_l2sq=lam_l2sq)
    max_deg = 2 * order_d
    all_monos = _monomials_upto_degree(max_deg)

    y: dict[tuple[int,int], cp.Variable] = {m: cp.Variable() for m in all_monos}
    constraints = [y[(0, 0)] == 1.0]

    basis_d = _monomials_upto_degree(order_d)
    M = _build_moment_matrix(y, basis_d)
    constraints.append(M >> 0)

    basis_d1 = _monomials_upto_degree(order_d - 1)
    g_list = [
        {(0,0): bound, (1,0): -1.0, (0,1): 0.0},
        {(0,0): bound, (1,0):  1.0, (0,1): 0.0},
        {(0,0): bound, (1,0): 0.0, (0,1): -1.0},
        {(0,0): bound, (1,0): 0.0, (0,1):  1.0},
    ]
    for g in g_list:
        L = _build_localizing_matrix_linear(y, basis_d1, g)
        constraints.append(L >> 0)

    obj = cp.Minimize(sum(c * y[m] for m, c in coeff.items()))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver, verbose=verbose)

    return float(prob.value), str(prob.status)

class LossLandscapeBase:
    def __init__(
        self,
        callable_func,
        x_low,
        x_high,
        optimal_sol,
        feedback=0,
        seed=None,
        precision_digit=2,
        horizon=10,
        # multi-objective / regularization knobs
        norm_coef: float = 0.0,
        norm_kind: str = "l2sq",
        # done criterion uses certificate
        done_tol: float = 1e-2,
    ):
        self.x_low = x_low
        self.x_high = x_high

        self._np_random = None
        self.stop_keywords = ["reach", "stay", "stop"]

        # base (unregularized) function
        self.base_func = callable_func

        self.norm_coef = float(norm_coef)
        self.norm_kind = str(norm_kind)
        self.done_tol = float(done_tol)

        # wrapped function used everywhere in env: base + norm
        def augmented(x: np.ndarray) -> float:
            x = np.asarray(x, dtype=float)
            return float(self.base_func(x) + _norm_term(x, self.norm_coef, self.norm_kind))

        self.callable_func = augmented

        self.prev_x = None
        self.left_attempts = horizon

        self.optimal_sol = optimal_sol
        self.precision_digit = precision_digit
        self.horizon = horizon
        self._seed = self.seed(seed)

        # subclass sets this (certificate-based) in _init_certificate()
        self.certificate_y: float | None = None
        self.certificate_meta: dict = {}

        self._init_certificate()
        if self.certificate_y is None:
            raise RuntimeError("Subclass must set self.certificate_y in _init_certificate().")

        # Use certificate as min_y for reward range + done checks
        self.min_y = float(self.certificate_y)

        self.reward_range = (self.get_min_reward(), -self.min_y)

        if self.norm_coef != 0.0:
            norm_desc = {
                "l2sq": "||x||_2^2 (squared L2 norm)",
                "l2": "||x||_2 (L2 norm)",
                "l1": "||x||_1 (L1 norm)",
            }.get(self.norm_kind, self.norm_kind)

            objective_line = (
                f"Your goal is to minimize the total objective:\n"
                f"  y(x) = f(x) + {self.norm_coef} * {norm_desc}\n"
                f"where f(x) is the base function output and x is a 2D vector."
            )
        else:
            objective_line = (
                "Your goal is to minimize the function output:\n"
                "  y(x) = f(x)\n"
                "where f(x) is the base function output and x is a 2D vector."
            )

        self.docstring = dedent(f"""
        You are trying to minimize an objective by choosing the input x.

        {objective_line}

        You get to observe y once you choose x, where x is a 2-dimensional vector:
          x = [x1, x2], with real-valued coordinates.

        The allowed range for x1 and x2 is [{self.x_low}, {self.x_high}].
        Please do not choose x outside of this range.

        You have {self.horizon} attempts.
        You can choose to stop at any time by outputting a message containing one of: {", ".join(self.stop_keywords)}.

        Output format:
        x = [x1, x2]
        """).strip()

        self.called_reset = False

    def _init_certificate(self) -> None:
        """
        Subclasses must set:
          self.certificate_y: float  (target min value / certificate)
        Optionally:
          self.certificate_meta: dict with info (solver status, x*, etc.)
        """
        raise NotImplementedError

    def get_min_reward(self):
        # conservative: evaluate on corners of box for reward lower bound
        x_range = [self.x_low, self.x_high]
        y_vals = [self.callable_func(np.array([x_range[i], x_range[j]])) for i in range(2) for j in range(2)]
        y_max = max(y_vals)
        return -float(y_max)

    def get_optimal_solution(self):
        return self.optimal_sol

    def reset(self, **kwargs):
        if "seed" in kwargs:
            self._seed = self.seed(kwargs["seed"])

        x = self.np_random.uniform(self.x_low, self.x_high, size=2)
        x = np.round(x, self.precision_digit)
        self.prev_x = x

        y = self.callable_func(x)
        self.left_attempts = self.horizon

        # obs = f"x={x.tolist()}\nFunction outputs y = {y}\nYou have {self.left_attempts} attempts left!\n"
        loss_line, info = self._format_loss_report(x)
        obs = loss_line
        obs += "Please output the next x that will make this function output the smallest y.\n"
        obs += "Format: x = [x1, x2]\n"
        obs += "Output:"

        self.called_reset = True
        return obs

    def seed(self, seed=None):
        self._np_random, seed = np_random(seed)
        return [seed]

    @property
    def np_random(self):
        if self._np_random is None:
            self.seed()
        return self._np_random  # type: ignore

    def text_extract(self, text):
        for stop_word in self.stop_keywords:
            if stop_word in text:
                return None, True

        pattern = r"\[(-?\d+\.?\d*(?:e[-+]?\d+)?),\s*(-?\d+\.?\d*(?:e[-+]?\d+)?)\]"
        match = re.search(pattern, text)
        if match is None:
            return None, False
        numbers = [float(g) for g in match.groups()]
        return np.array(numbers, dtype=float), False

    def _is_success(self, loss: float) -> bool:
        # Done criterion: close to certificate/guarantee.
        # Note: certificate_y is a lower bound for SOS cases; if it's tight, this is meaningful.
        return abs(float(loss) - float(self.certificate_y)) <= self.done_tol

    def _eval_losses(self, x: np.ndarray) -> tuple[float, float, float]:
        x = np.asarray(x, dtype=float)
        base = float(self.base_func(x))
        reg = float(_norm_term(x, self.norm_coef, self.norm_kind))
        total = base + reg
        return base, reg, total

    def _format_loss_report(self, x: np.ndarray) -> tuple[str, dict]:
        base, reg, total = self._eval_losses(x)
        info = {
            "base_loss": base,
            "reg_loss": reg,
            "total_loss": total,
            "certificate_y": float(self.certificate_y),
            "gap": float(total - float(self.certificate_y)),
        }

        if self.norm_coef != 0.0:
            # optional: report the raw norm too (not multiplied by coef)
            if self.norm_kind == "l2sq":
                norm_val = float(x[0] ** 2 + x[1] ** 2)
            elif self.norm_kind == "l2":
                norm_val = float(np.sqrt(x[0] ** 2 + x[1] ** 2))
            elif self.norm_kind == "l1":
                norm_val = float(abs(x[0]) + abs(x[1]))
            else:
                norm_val = None

            info["norm_value"] = norm_val
            info["norm_kind"] = self.norm_kind
            info["norm_coef"] = float(self.norm_coef)

            line = (
                f"Function outputs total y = {total}\n"
                f"  base f(x) = {base}\n"
                f"  regularizer = {reg}  (coef={self.norm_coef}, kind={self.norm_kind}, norm={norm_val})\n"
            )
        else:
            line = f"Function outputs y = {total}\n"

        return line, info

    def step(self, action):
        if not self.called_reset:
            raise Exception("must call env.reset() first before step()")

        x, stop = self.text_extract(action)

        if x is None and stop is False:
            feedback = (
                    f"You entered an invalid action: {action}"
                    + f" Please enter a valid action within ({self.x_low, self.x_high})"
            )
            return None, -1, True, {
                "feedback": feedback,
                "success": False,
                "base_loss": None,
                "reg_loss": None,
                "total_loss": None,
                "certificate_y": float(self.certificate_y),
                "gap": None,
            }

        if stop:
            base, reg, total = self._eval_losses(self.prev_x)
            success = self._is_success(total)
            feedback = f"You have chosen to stop at {self.prev_x}."
            feedback += " You have reached the (certified) minimum!" if success else " You have not reached the (certified) minimum!"
            return None, total, True, {
                "feedback": feedback,
                "success": success,
                "base_loss": base,
                "reg_loss": reg,
                "total_loss": total,
                "certificate_y": float(self.certificate_y),
                "gap": float(total - float(self.certificate_y)),
            }

        if np.any(x < self.x_low) or np.any(x > self.x_high):
            base, reg, total = self._eval_losses(self.prev_x)
            feedback = f"x must be within [{self.x_low}, {self.x_high}]. You gave {x.tolist()}."
            return None, total, True, {
                "feedback": feedback,
                "success": False,
                "base_loss": base,
                "reg_loss": reg,
                "total_loss": total,
                "certificate_y": float(self.certificate_y),
                "gap": float(total - float(self.certificate_y)),
            }

        base, reg, total = self._eval_losses(x)

        if self._is_success(total):
            feedback = f"Function outputs y: {total}\nYou have reached the (certified) minimum!"
            return feedback, -total, True, {
                "feedback": feedback,
                "success": True,
                "base_loss": base,
                "reg_loss": reg,
                "total_loss": total,
                "certificate_y": float(self.certificate_y),
                "gap": float(total - float(self.certificate_y)),
            }

        loss_line, info = self._format_loss_report(x)
        obs = loss_line
        obs += f"You have {self.left_attempts} attempts left!\n"
        obs += "Please output the next x that will make this function output the smallest y.\n"
        obs += "Format: x = [x1, x2]\n"
        obs += "Output:"

        self.prev_x = x
        self.left_attempts -= 1

        r = np.clip(float(-total), self.get_min_reward(), -self.min_y)
        feedback = f"You chose {action}. Choose different numbers such that you can minimize y."
        return obs, r, False, {
            "feedback": feedback,
            "success": False,
            "base_loss": base,
            "reg_loss": reg,
            "total_loss": total,
            "certificate_y": float(self.certificate_y),
            "gap": float(total - float(self.certificate_y)),
        }


class Rosenbrock(LossLandscapeBase):
    def __init__(
        self,
        a=1.0,
        b=1.0,
        feedback=0,
        seed=None,
        horizon=10,
        precision_digit=2,
        norm_coef: float = 1.0,
        norm_kind: str = "l2sq",
        done_tol: float = 1e-2,
    ):
        self.a = float(a)
        self.b = float(b)

        def base(x: np.ndarray) -> float:
            return float((self.a - x[0]) ** 2 + self.b * (x[1] - x[0] ** 2) ** 2)

        super().__init__(
            callable_func=base,
            x_low=-5,
            x_high=10,
            optimal_sol=np.ones(2),
            feedback=feedback,
            seed=seed,
            precision_digit=precision_digit,
            horizon=horizon,
            norm_coef=norm_coef,
            norm_kind=norm_kind,
            done_tol=done_tol,
        )

    def _init_certificate(self) -> None:
        if self.norm_kind != "l2sq":
            raise ValueError("Rosenbrock cubic certificate requires norm_kind='l2sq'.")

        if self.norm_coef < 0:
            raise ValueError("For a meaningful global certificate, norm_coef should be >= 0.")

        x_star, f_star = _rosenbrock_cubic_global_min(self.a, self.b, self.norm_coef)

        self.certificate_y = float(f_star)
        self.optimal_sol = x_star
        self.certificate_meta = {"method": "cubic", "x_star": x_star, "f_star": float(f_star)}


class SixHumpCamel(LossLandscapeBase):
    def __init__(
        self,
        feedback=0,
        seed=None,
        horizon=10,
        precision_digit=4,
        norm_coef: float = 1.0,
        norm_kind: str = "l2sq",
        done_tol: float = 1e-3,
        sos_solver: str = "SCS",
        sos_order_d: int = 3,
        sos_verbose: bool = False,
    ):
        self.sos_solver = sos_solver
        self.sos_order_d = sos_order_d
        self.sos_verbose = sos_verbose

        def base(x: np.ndarray) -> float:
            u, v = float(x[0]), float(x[1])
            return float((4 - 2.1 * u ** 2 + (u ** 4) / 3) * u ** 2 + u * v + (-4 + 4 * v ** 2) * v ** 2)

        super().__init__(
            callable_func=base,
            x_low=-2,
            x_high=2,
            optimal_sol=[np.array([0.0898, -0.7126]), np.array([-0.0898, 0.7126])],
            feedback=feedback,
            seed=seed,
            precision_digit=precision_digit,
            horizon=horizon,
            norm_coef=norm_coef,
            norm_kind=norm_kind,
            done_tol=done_tol,
        )

    def _init_certificate(self) -> None:
        if self.norm_coef != 0.0 and self.norm_kind != "l2sq":
            raise ValueError(
                "SixHumpCamel SOS certificate supports norm_coef==0 or norm_kind=='l2sq'. "
                "For l1/l2 you need epigraph variables."
            )

        gamma, status = six_hump_sos_certificate_on_box(
            bound=2.0,
            order_d=self.sos_order_d,
            solver=self.sos_solver,
            verbose=self.sos_verbose,
            lam_l2sq=self.norm_coef,
        )
        self.certificate_y = float(gamma)
        self.certificate_meta = {
            "method": "moment_sdp",
            "gamma": float(gamma),
            "status": status,
            "lam_l2sq": float(self.norm_coef),
            "bound": 2.0,
            "order_d": self.sos_order_d,
            "solver": self.sos_solver,
        }


# ============ Multi-objective test harness (Approach 1: BasicSearch + ObjectiveConfig) ============
from opto import trace
from opto.trainer.guide import Guide
from opto.trainer.loggers import TensorboardLogger
from opto import trainer
from opto.trainer.objectives import ObjectiveConfig
from opto.trainer.examples.basic_algorithms import BasicSearchAlgorithm as SearchAlgorithm
from typing import Tuple
from copy import copy


class RewardGuide(Guide):
    """
    Multi-objective metrics:

      - base_loss: minimize
      - reg_loss: minimize

    (The trainer's ObjectiveConfig decides how to combine/compare.)
    """

    def __init__(self, env: LossLandscapeBase):
        self.env = env

    def _score_action_on_env_copy(self, action: str):
        env_copy = copy.deepcopy(self.env)
        obs, reward, done, info = env_copy.step(action)
        return obs, reward, done, info

    def get_feedback(self, query: str, response: str, reference=None, **kwargs) -> Tuple[float, str]:
        # Legacy scalar path: advances the real env.
        obs, reward, done, info = self.env.step(response)
        return float(reward), ((obs + "\n\n") if obs else "") + info.get("feedback", "")

    def get_score_dict(self, query: str, response: str, reference=None, **kwargs) -> dict[str, float]:
        # Vector score path for trainer-side selection:
        obs, reward, done, info = self._score_action_on_env_copy(response)

        base_loss = info.get("base_loss")
        reg_loss = info.get("reg_loss")

        # If action invalid, your env sets losses to None. Map to +inf so it never gets selected.
        if base_loss is None or reg_loss is None:
            base_loss = float("inf")
            reg_loss = float("inf")

        return {
            "base_loss": float(base_loss),  # minimize
            "reg_loss": float(reg_loss),    # minimize
        }

def main():
    env = SixHumpCamel(horizon=200)
    train_dataset = dict(inputs=[None], infos=[None])

    instruction = env.reset()
    initial_input = instruction.split("\n")[0].strip()
    param = trace.node(initial_input, description="Input x into the hidden function to get y.", trainable=True)

    guide = RewardGuide(env)
    logger = TensorboardLogger(log_dir="./logs/basicsearch_multiobjective_on_loss_landscape")

    # We want high reward, but penalize invalid actions and overly long outputs.
    objective_config = ObjectiveConfig(
        mode="weighted",
        weights={"base_loss": 1.0, "reg_loss": 1.0},
        minimize=frozenset({"base_loss", "reg_loss"}),
        seed=0,
    )

    trainer.train(
        model=param,
        algorithm=SearchAlgorithm,
        train_dataset=train_dataset,
        logger=logger,
        score_range=[-10, 10],
        num_epochs=1,
        num_steps=5,
        batch_size=1,
        num_batches=2,
        verbose=False,
        guide=guide,
        objective_config=objective_config,
        # basic search knobs (keep small for smoke test)
        num_candidates=4,
        num_proposals=4,
        optimizer_kwargs={
            "objective": "You have a task of guessing two numbers. Output x=[x1,x2] and minimize y.",
            "memory_size": 10,
        },
    )


if __name__ == "__main__":
    main()
