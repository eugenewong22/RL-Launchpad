"""Dynamics randomization for the Fetch tasks (declared env modification, R5).

Resamples the manipulated block's mass and sliding friction at every reset.
The point is not a better FetchPush number — it is measuring whether the
policy survives dynamics it never trained on, which is the honest sim-only
proxy for a sim-to-real gap. We have no Fetch arm, so we do not claim
sim-to-real; we claim sim-to-sim transfer and name it that way.

The experiment this enables:
  train nominal + eval nominal      <- the reported agent, already have it
  train nominal + eval shifted      <- how badly does dynamics shift hurt?
  train randomized + eval shifted   <- does DR recover it?
Evaluation ranges are deliberately OUTSIDE the training ranges, so the
third row is extrapolation rather than interpolation.

R5 note: this modifies the simulator's physical parameters only. Reward,
observation space, action space and success criterion are untouched, and
the reported FetchPush/FetchPickAndPlace results use the stock env with
this wrapper absent.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

BODY = "object0"


@dataclass(frozen=True)
class DomainRandomizationSpec:
    """Multiplicative ranges applied to the block's nominal parameters.

    Multiplicative rather than absolute so the same spec is meaningful
    across tasks whose blocks differ, and so "1.0 = stock" is always the
    centre of the distribution.
    """

    mass: tuple[float, float] = (0.5, 2.0)
    friction: tuple[float, float] = (0.5, 2.0)

    def contains(self, mass_mult: float, friction_mult: float) -> bool:
        return (
            self.mass[0] <= mass_mult <= self.mass[1]
            and self.friction[0] <= friction_mult <= self.friction[1]
        )


class DynamicsRandomization(gym.Wrapper):
    """Resample block mass and sliding friction on every reset.

    Nominal values are captured once at construction, before anything is
    perturbed, and every reset scales *those* — not the previous episode's
    values, which would random-walk the dynamics away from stock over a
    training run instead of sampling a fixed distribution around it.
    """

    def __init__(self, env, spec: DomainRandomizationSpec | None = None, seed: int = 0):
        super().__init__(env)
        # NOT `self.spec`: gym.Wrapper already defines that as the read-only
        # EnvSpec, and assigning it raises at construction.
        self.dr_spec = spec or DomainRandomizationSpec()
        self._rng = np.random.default_rng(seed)

        import mujoco

        model = self.env.unwrapped.model
        self._body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY)
        if self._body_id < 0:
            raise ValueError(f"no body named {BODY!r} in this env — DR needs one")
        self._geom_ids = [
            i for i in range(model.ngeom) if model.geom_bodyid[i] == self._body_id
        ]
        self._nominal_mass = float(model.body_mass[self._body_id])
        self._nominal_friction = model.geom_friction[self._geom_ids].copy()
        self.last_sample: dict[str, float] = {}

    def _apply(self, mass_mult: float, friction_mult: float) -> None:
        model = self.env.unwrapped.model
        model.body_mass[self._body_id] = self._nominal_mass * mass_mult
        # Only the sliding coefficient (column 0); torsional and rolling are
        # left at stock, since sliding is what a push actually fights.
        friction = self._nominal_friction.copy()
        friction[:, 0] *= friction_mult
        model.geom_friction[self._geom_ids] = friction
        self.last_sample = {"mass_mult": mass_mult, "friction_mult": friction_mult}

    def reset(self, **kwargs):
        m = self._rng.uniform(*self.dr_spec.mass)
        f = self._rng.uniform(*self.dr_spec.friction)
        self._apply(m, f)
        return self.env.reset(**kwargs)


class FixedDynamics(gym.Wrapper):
    """Pin the block to one (mass, friction) multiplier — for evaluation.

    Held-out evaluation needs *specified* dynamics, not sampled ones, so a
    number can be attributed to a known shift rather than to whatever the
    RNG happened to draw.
    """

    def __init__(self, env, mass_mult: float = 1.0, friction_mult: float = 1.0):
        super().__init__(env)
        self._inner = DynamicsRandomization(env)
        self.mass_mult, self.friction_mult = mass_mult, friction_mult

    def reset(self, **kwargs):
        self._inner._apply(self.mass_mult, self.friction_mult)
        return self.env.reset(**kwargs)
