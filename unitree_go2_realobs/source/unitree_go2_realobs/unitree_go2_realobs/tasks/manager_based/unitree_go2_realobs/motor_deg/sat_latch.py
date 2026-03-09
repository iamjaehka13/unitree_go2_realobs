from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SatLatchCfg:
    """Windowed saturation-ratio trigger config."""

    sat_thr: float = 0.99
    window_steps: int = 15
    trigger: float = 0.95


class SatRatioLatch:
    """
    Compute sat_any_over_thr_ratio over a fixed control-step window.

    - sat_any_t: any joint saturation >= sat_thr at control step t
    - ratio_t: mean of sat_any over recent window
    - over_trigger_t: ratio_t >= trigger
    """

    def __init__(self, num_envs: int, device: torch.device, cfg: SatLatchCfg):
        if cfg.window_steps <= 0:
            raise ValueError(f"window_steps must be > 0, got {cfg.window_steps}")
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = device
        self._window = int(cfg.window_steps)

        self._hist = torch.zeros((self.num_envs, self._window), dtype=torch.uint8, device=self.device)
        self._sum = torch.zeros((self.num_envs,), dtype=torch.int16, device=self.device)
        self._valid_steps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._idx = 0

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._hist.zero_()
            self._sum.zero_()
            self._valid_steps.zero_()
            self._idx = 0
            return
        ids = env_ids.to(device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        self._hist[ids] = 0
        self._sum[ids] = 0
        self._valid_steps[ids] = 0

    @torch.no_grad()
    def update(
        self,
        torque_saturation: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        sat_any_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torque_saturation.ndim != 2 or torque_saturation.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected torque_saturation shape=({self.num_envs}, J), got {tuple(torque_saturation.shape)}"
            )

        if sat_any_override is None:
            sat_any = (torque_saturation >= float(self.cfg.sat_thr)).any(dim=-1)
        else:
            if sat_any_override.ndim != 1 or sat_any_override.shape[0] != self.num_envs:
                raise ValueError(
                    f"Expected sat_any_override shape=({self.num_envs},), got {tuple(sat_any_override.shape)}"
                )
            sat_any = sat_any_override.to(dtype=torch.bool, device=self.device)

        if valid_mask is None:
            valid = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        else:
            if valid_mask.ndim != 1 or valid_mask.shape[0] != self.num_envs:
                raise ValueError(
                    f"Expected valid_mask shape=({self.num_envs},), got {tuple(valid_mask.shape)}"
                )
            valid = valid_mask.to(dtype=torch.bool, device=self.device)
            sat_any = sat_any & valid

        sat_u8 = sat_any.to(torch.uint8)

        old = self._hist[:, self._idx]
        new_col = torch.where(valid, sat_u8, old)
        self._hist[:, self._idx] = new_col
        self._sum += new_col.to(self._sum.dtype) - old.to(self._sum.dtype)
        self._valid_steps = torch.clamp(self._valid_steps + valid.to(self._valid_steps.dtype), max=self._window)
        self._idx = (self._idx + 1) % self._window

        # Fixed denominator W (conservative at early steps).
        ratio = self._sum.to(torch.float32) / float(self._window)
        over_trigger = ratio >= float(self.cfg.trigger)
        return sat_any, ratio, over_trigger

    @property
    def valid_steps(self) -> torch.Tensor:
        return self._valid_steps

    @property
    def ratio(self) -> torch.Tensor:
        """Current window ratio per environment."""
        return self._sum.to(torch.float32) / float(self._window)
