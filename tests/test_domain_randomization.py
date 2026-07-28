"""Dynamics randomization must actually change the physics.

The failure mode worth guarding is silent: if the env reloads its model on
reset, or MuJoCo caches derived quantities, the writes land in the struct
and the simulation ignores them. Training would then run to completion,
"with DR", having randomized nothing — and the resulting robustness claim
would be fabricated. So the tests assert on trajectories, not on the model
fields we just set.
"""

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (registers Fetch env IDs)
import numpy as np
import pytest

from src.agent.domain_randomization import (
    DomainRandomizationSpec,
    DynamicsRandomization,
    FixedDynamics,
)

ENV_ID = "FetchPush-v4"


def rollout(env, seed, n=25):
    """Deterministic scripted push; returns the block's final position."""
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(0)  # same action sequence every call
    actions = rng.uniform(-1, 1, size=(n, env.action_space.shape[0]))
    for a in actions:
        obs, *_ = env.step(a)
    return np.asarray(obs["achieved_goal"], dtype=float)


def test_friction_change_alters_the_trajectory():
    """Identical seed and actions, different friction -> different outcome."""
    low = rollout(FixedDynamics(gym.make(ENV_ID), friction_mult=0.2), seed=7)
    high = rollout(FixedDynamics(gym.make(ENV_ID), friction_mult=5.0), seed=7)
    assert not np.allclose(low, high, atol=1e-4), (
        f"friction had no effect on the physics: {low} vs {high} — the writes "
        "are being ignored or undone on reset"
    )


def test_mass_change_alters_the_trajectory():
    light = rollout(FixedDynamics(gym.make(ENV_ID), mass_mult=0.2), seed=7)
    heavy = rollout(FixedDynamics(gym.make(ENV_ID), mass_mult=10.0), seed=7)
    assert not np.allclose(light, heavy, atol=1e-4), (
        f"mass had no effect on the physics: {light} vs {heavy}"
    )


def test_identical_multipliers_reproduce_exactly():
    """Same dynamics + same seed must be deterministic, or nothing above
    distinguishes a real effect from simulator noise."""
    a = rollout(FixedDynamics(gym.make(ENV_ID), mass_mult=1.3, friction_mult=0.7), seed=11)
    b = rollout(FixedDynamics(gym.make(ENV_ID), mass_mult=1.3, friction_mult=0.7), seed=11)
    assert np.allclose(a, b, atol=1e-9), f"non-deterministic at fixed dynamics: {a} vs {b}"


def test_wrapper_samples_within_spec_and_varies():
    spec = DomainRandomizationSpec(mass=(0.5, 2.0), friction=(0.5, 2.0))
    env = DynamicsRandomization(gym.make(ENV_ID), spec=spec, seed=3)
    seen = []
    for _ in range(30):
        env.reset(seed=0)
        s = env.last_sample
        assert spec.contains(s["mass_mult"], s["friction_mult"]), s
        seen.append((s["mass_mult"], s["friction_mult"]))
    assert len({tuple(np.round(x, 6)) for x in seen}) > 20, "samples are not varying"


def test_perturbations_are_relative_to_stock_not_cumulative():
    """Each reset scales the NOMINAL value.

    Scaling the previous episode's value instead would random-walk the
    dynamics away from stock over a training run — the distribution would
    drift instead of staying centred on the real robot's parameters.
    """
    env = DynamicsRandomization(gym.make(ENV_ID), seed=1)
    model = env.unwrapped.model
    nominal = env._nominal_mass

    for _ in range(40):
        env.reset(seed=0)
    mass = float(model.body_mass[env._body_id])
    lo, hi = env.dr_spec.mass
    assert nominal * lo <= mass <= nominal * hi, (
        f"mass {mass} escaped [{nominal*lo}, {nominal*hi}] after 40 resets — "
        "perturbations are compounding"
    )


def test_stock_env_is_untouched_without_the_wrapper():
    """R5: the reported results use the unmodified env."""
    plain = gym.make(ENV_ID)
    wrapped = DynamicsRandomization(gym.make(ENV_ID), seed=0)
    for _ in range(5):
        wrapped.reset(seed=0)
    assert float(plain.unwrapped.model.body_mass[wrapped._body_id]) == pytest.approx(
        wrapped._nominal_mass
    ), "wrapping one env perturbed another — the model is being shared"
