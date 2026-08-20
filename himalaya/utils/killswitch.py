"""Early-kill monitor: abort runs that are clearly dead, keep the ones that aren't.

Asymmetry that drives the design: killing a recoverable run costs one restart;
letting a dead run finish costs ~14 GPU-hours. But a NEW reward config has no
baseline curve to compare against, and velocity tracking can sit flat for a
while and still recover. So the aggressive checks are opt-in (`strict: true`)
and meant for configs you've already seen work once.

Wire into the training loop:

    ks = KillSwitch.from_config(cfg, strict=args.strict_kill)
    ...
    verdict = ks.update(elapsed_min, {"steps_per_sec": sps, ...})
    if verdict.should_kill:
        log.error(verdict.reason)
        break
"""
from dataclasses import dataclass, field


@dataclass
class Verdict:
    should_kill: bool = False
    reason: str = ""


@dataclass
class Check:
    at_min: float
    metric: str
    min: float
    strict_only: bool = False
    fired: bool = False


@dataclass
class KillSwitch:
    checks: list
    strict: bool = False
    enabled: bool = True
    grace_runs: int = 0
    history: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg, strict=None):
        k = cfg.get("killswitch", {})
        checks = [Check(**{kk: vv for kk, vv in c.items()}) for c in k.get("checks", [])]
        return cls(
            checks=checks,
            strict=k.get("strict", False) if strict is None else strict,
            enabled=k.get("enabled", True),
        )

    def update(self, elapsed_min, metrics):
        """Call once per logging interval. Returns a Verdict."""
        for name, value in metrics.items():
            self.history.setdefault(name, []).append((elapsed_min, value))

        if not self.enabled:
            return Verdict()

        for c in self.checks:
            if c.fired or elapsed_min < c.at_min:
                continue
            if c.strict_only and not self.strict:
                continue
            if c.metric not in metrics:
                continue

            c.fired = True
            value = metrics[c.metric]
            if value < c.min:
                return Verdict(
                    True,
                    f"KILL at {elapsed_min:.0f}min: {c.metric}={value:.4g} "
                    f"below threshold {c.min:.4g}. {self._diagnosis(c.metric)}",
                )
        return Verdict()

    @staticmethod
    def _diagnosis(metric):
        return {
            "steps_per_sec": (
                "Throughput problem, not a learning problem -- check num_envs, "
                "collision geom count, and physics substeps."
            ),
            "episode_length": (
                "Robot is falling almost immediately. Check PD gains, action_scale, "
                "and initial pose."
            ),
            "track_lin_vel": (
                "Primary objective is not improving. Run "
                "`preflight.py --test reward-audit` -- a competing term is likely "
                "swamping velocity tracking."
            ),
        }.get(metric, "")

    def plateaued(self, metric, window=20, rel_tol=0.01):
        """Soft signal for sweep pruning -- NOT auto-kill.

        Report it, let a human decide. Plateau detection is exactly where
        auto-kill throws away recoverable runs.
        """
        series = [v for _, v in self.history.get(metric, [])]
        if len(series) < 2 * window:
            return False
        prev = sum(series[-2 * window:-window]) / window
        curr = sum(series[-window:]) / window
        if abs(prev) < 1e-9:
            return False
        return (curr - prev) / abs(prev) < rel_tol
