# graph_engine.py
"""
Reward-aware cohort state machine for full observed tier lines.

The model unit is a rank cell / player cohort, not an observed cutoff line.
Observed tiers are emissions from the cohort score field.  The public class names
are kept stable so the Streamlit experiment panel can continue to call this
module while the internals are replaced.
"""
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from math_models import SeasonalityHandler
from tier_surface import align_observed_tier_surface, build_tier_surface


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SPEED_PROFILE_PATH = PROJECT_ROOT / "event_data" / "tier_speed_profiles_last100.json"
DEFAULT_CAPACITY_SHAPE_PATH = PROJECT_ROOT / "event_data" / "song_rank_shape_event311.json"
DEFAULT_BEHAVIOR_PRIOR_PATH = PROJECT_ROOT / "event_data" / "graph_behavior_prior_rollout_multitier_baseline.json"
BEHAVIOR_COEFF_NAMES = ['cruise', 'profile', 'follow', 'gap', 'lag', 'panic']


class BehaviorMode(str, Enum):
    IDLE = "idle"
    CRUISING = "cruising"
    WATCHING = "watching"
    CHASING = "chasing"
    DEFENDING = "defending"
    PANIC_RUSHING = "panic_rushing"
    DROPPED = "dropped"


MODE_ORDER = [
    BehaviorMode.IDLE,
    BehaviorMode.CRUISING,
    BehaviorMode.WATCHING,
    BehaviorMode.CHASING,
    BehaviorMode.DEFENDING,
    BehaviorMode.PANIC_RUSHING,
    BehaviorMode.DROPPED,
]


@dataclass
class TierNode:
    """Observable tier emission used by the existing graph plot."""

    tier: int
    score: float = 0.0
    speed: float = 0.0              # pt/min
    base_norm_speed: float = 0.0
    pressure: float = 0.0
    target_affinity: float = 0.0
    volatility: float = 0.0
    is_reward_line: bool = False
    mode_mix: Dict[str, float] = field(default_factory=dict)


@dataclass
class CohortCell:
    """Hidden player-group state for a contiguous rank cell."""

    rank_start: int
    rank_end: int
    center_rank: float
    score: float
    speed: float                 # pt/min
    base_norm_speed: float
    capacity_norm_speed: float
    pressure: float
    target_affinity: float
    mode_probs: Dict[BehaviorMode, float]
    target_rank: Optional[int] = None
    target_score: float = 0.0
    target_gap_norm: float = 0.0
    target_profile_norm: float = 0.0
    target_follow_norm: float = 0.0
    target_plan_lag_norm: float = 0.0
    behavior_lazy: float = 0.0
    follow_rank: Optional[int] = None
    target_surplus_norm: float = 0.0
    target_boundary_proximity: float = 0.0
    target_risk_level: float = 0.5
    target_pressure: float = 0.0
    neighbor_pressure: float = 0.0
    density_pressure: float = 0.0
    target_importance: float = 0.0
    speed_cruise_norm: float = 0.0
    speed_target_norm: float = 0.0
    speed_profile_norm: float = 0.0
    speed_follow_norm: float = 0.0
    speed_lag_norm: float = 0.0
    speed_panic_feature_norm: float = 0.0
    speed_committed_norm: float = 0.0
    speed_defend_norm: float = 0.0
    speed_boundary_drive: float = 0.0
    speed_preseason_norm: float = 0.0
    speed_season_effect: float = 1.0
    speed_desired_norm: float = 0.0
    behavior_coeffs: Dict[str, float] = field(default_factory=dict)
    speed_limit_reason: str = "unknown"
    pressure_source: str = "unknown"
    target_probs: Dict[int, float] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return max(int(self.rank_end - self.rank_start + 1), 1)


@dataclass
class GraphModelConfig:
    dt_hours: float = 1.0

    # Rank-cell sizing: around T100 use cells of 10 players, around T1000 use 100.
    min_cell_size: int = 1
    max_cell_size: int = 500

    # Cell refinement around important target lines.
    reward_anchor_radius: float = 0.08

    # Base and caps, normalized by T10 pt/min scale.
    base_norm_floor: float = 0.001
    speed_profile_path: Optional[str] = None
    capacity_shape_path: Optional[str] = None
    behavior_prior_path: Optional[str] = None
    behavior_prior_sample_weight: float = 16.0


class GraphStateSpaceEngine:
    def __init__(
        self,
        seasonality: SeasonalityHandler,
        config: Optional[GraphModelConfig] = None,
    ):
        self.seasonality = seasonality
        self.config = config or GraphModelConfig()
        self._start_ts: int = 0
        self._surface = None
        self._observed_tiers: List[int] = []
        self._reward_tiers: List[int] = []
        self._behavior_target_tiers: List[int] = []
        self._emission_tiers: List[int] = []
        self._cells: List[CohortCell] = []
        self._last_hour: float = 0.0
        self._scale: float = 1.0
        self._total_hours: Optional[float] = None
        self._filter_metrics: Dict[str, float] = {}
        self._projection_cache: Dict[Tuple[int, int, float, float], float] = {}
        self._target_component_cache: Dict[Tuple[int, int, float, float, float], Tuple[float, float]] = {}
        self._season_mean_cache: Dict[Tuple[float, float], float] = {}
        self._target_distribution_cache: Dict[float, Dict[int, float]] = {}
        self._target_risk_cache: Dict[Tuple[float, int], float] = {}
        self._progress_fraction_cache: Dict[Tuple[int, float, float], Optional[float]] = {}
        self._progress_speed_factor_cache: Dict[Tuple[int, float, float], Optional[float]] = {}
        self._target_importance_by_tier: Dict[int, float] = {}
        self._target_importance_norm_cache: Dict[int, float] = {}
        self._behavior_lazy_by_rank: Dict[int, float] = {}
        self._behavior_lazy_global: float = 0.0
        self._behavior_coeffs_global: Dict[str, float] = {}
        self._behavior_coeffs_by_rank: Dict[int, Dict[str, float]] = {}
        self._guide_rank_cache: Dict[int, Optional[int]] = {}
        self._cell_array_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._row_score_cache: Dict[Tuple[int, object], Tuple[np.ndarray, np.ndarray]] = {}
        self._speed_profile = self._load_speed_profile(self.config.speed_profile_path)
        self._capacity_shape = self._load_capacity_shape(self.config.capacity_shape_path)
        self._behavior_prior = self._load_behavior_prior(self.config.behavior_prior_path)
        self._speed_profile_adapted: Dict[str, np.ndarray] = {}

    def init_nodes(
        self,
        tier_data: dict,
        meta: dict,
        reward_tiers: List[int],
        scale: Optional[float] = None,
        align_freq_hours: Optional[float] = None,
        observed_until_hour: Optional[float] = None,
    ) -> Dict[int, TierNode]:
        """Initialize hidden rank cells, then emit observable tier states."""
        self._prepare_context(tier_data, meta, reward_tiers, scale, align_freq_hours, observed_until_hour)
        if self._surface is None or self._surface.empty:
            return {}

        current_hour = float(self._surface.index[-1])
        self._last_hour = current_hour
        self._cells = self._build_cells_from_surface(self._surface, current_hour)
        self._hydrate_cells(current_hour)
        return self._emit_tier_nodes(self._cells, current_hour)

    def replay_observed_states(
        self,
        tier_data: dict,
        meta: dict,
        reward_tiers: List[int],
        scale: float,
        observed_until_hour: Optional[float] = None,
        align_freq_hours: Optional[float] = None,
    ) -> Tuple[Dict[int, List[TierNode]], np.ndarray]:
        """Replay the hidden cohort state through history with observation updates."""
        self._prepare_context(tier_data, meta, reward_tiers, scale, align_freq_hours, observed_until_hour)
        if self._surface is None or self._surface.empty:
            return {}, np.array([])

        histories: Dict[int, List[TierNode]] = {tier: [] for tier in self._emission_tiers}
        hours = np.asarray(self._surface.index, dtype=float)

        self._cells = self._build_cells_from_surface(self._surface.iloc[:1], float(hours[0]))
        self._hydrate_cells(float(hours[0]))
        first_nodes = self._emit_tier_nodes(self._cells, float(hours[0]))
        for tier, node in first_nodes.items():
            histories.setdefault(tier, []).append(TierNode(**node.__dict__))

        prior_mae_values = []
        post_mae_values = []
        prior_error_by_tier = {int(tier): [] for tier in self._observed_tiers}
        post_error_by_tier = {int(tier): [] for tier in self._observed_tiers}
        last_prior_error_by_tier = {}
        last_post_error_by_tier = {}
        last_prior_mae = 0.0
        last_post_mae = 0.0

        for pos in range(1, len(hours)):
            prev_hour = float(hours[pos - 1])
            hour = float(hours[pos])
            dt = max(hour - prev_hour, 1e-6)
            remaining = max((self._total_hours or hour) - prev_hour, dt)
            previous_row = self._surface.iloc[pos - 1]
            current_row = self._surface.iloc[pos]

            prior_cells = self._step_cells(self._cells, prev_hour, remaining, dt)
            prior_error_map = self._emission_error_map(prior_cells, current_row)
            prior_errors = list(prior_error_map.values())
            self._cells = self._correct_cells_to_observation(
                prior_cells,
                current_row,
                previous_row,
                hour,
                dt,
            )
            self._hydrate_cells(hour)
            post_error_map = self._emission_error_map(self._cells, current_row)
            post_errors = list(post_error_map.values())

            if prior_errors:
                last_prior_mae = float(np.mean(np.abs(prior_errors)))
                prior_mae_values.append(last_prior_mae)
                last_prior_error_by_tier = {int(k): float(v) for k, v in prior_error_map.items()}
                for tier, err in prior_error_map.items():
                    prior_error_by_tier.setdefault(int(tier), []).append(float(err))
            if post_errors:
                last_post_mae = float(np.mean(np.abs(post_errors)))
                post_mae_values.append(last_post_mae)
                last_post_error_by_tier = {int(k): float(v) for k, v in post_error_map.items()}
                for tier, err in post_error_map.items():
                    post_error_by_tier.setdefault(int(tier), []).append(float(err))

            emitted = self._emit_tier_nodes(self._cells, hour)
            for tier, node in emitted.items():
                histories.setdefault(tier, []).append(TierNode(**node.__dict__))

        self._last_hour = float(hours[-1])
        self._filter_metrics = {
            'steps': float(max(len(hours) - 1, 0)),
            'prior_mae': float(np.mean(prior_mae_values)) if prior_mae_values else 0.0,
            'post_mae': float(np.mean(post_mae_values)) if post_mae_values else 0.0,
            'last_prior_mae': float(last_prior_mae),
            'last_post_mae': float(last_post_mae),
            'prior_signed_by_tier': {
                int(tier): float(np.mean(values))
                for tier, values in prior_error_by_tier.items()
                if values
            },
            'prior_mae_by_tier': {
                int(tier): float(np.mean(np.abs(values)))
                for tier, values in prior_error_by_tier.items()
                if values
            },
            'last_prior_error_by_tier': last_prior_error_by_tier,
            'last_post_error_by_tier': last_post_error_by_tier,
        }
        return histories, hours

    def current_nodes(self) -> Dict[int, TierNode]:
        if not self._cells:
            return {}
        return self._emit_tier_nodes(self._cells, self._last_hour)

    def filter_metrics(self) -> Dict[str, float]:
        return dict(self._filter_metrics)

    def step(
        self,
        nodes: Dict[int, TierNode],
        hour: float,
        scale: float,
        remaining_hours: float,
        reward_tiers: List[int],
        dt_hours: Optional[float] = None,
    ) -> Dict[int, TierNode]:
        """Advance hidden cohort cells and emit observed tiers."""
        if not self._cells:
            return nodes or {}

        dt = float(dt_hours if dt_hours is not None else self.config.dt_hours)
        self._scale = max(float(scale), 1e-6)
        self._reward_tiers = sorted({int(t) for t in reward_tiers or []})
        self._cells = self._step_cells(self._cells, float(hour), max(float(remaining_hours), dt), dt)
        self._last_hour = float(hour) + dt
        return self._emit_tier_nodes(self._cells, self._last_hour)

    def rollout(
        self,
        nodes: Dict[int, TierNode],
        start_hour: float,
        hours_forward: float,
        scale: float,
        total_hours: float,
        reward_tiers: List[int],
        steps: Optional[int] = None,
    ) -> Dict[int, List[TierNode]]:
        """Roll cohorts forward and return observable tier trajectories."""
        if not self._cells:
            return {tier: [TierNode(**node.__dict__)] for tier, node in (nodes or {}).items()}

        c = self.config
        if steps is None:
            steps = max(int(np.ceil(hours_forward / c.dt_hours)), 1)
        dt = float(hours_forward) / max(int(steps), 1)
        self._scale = max(float(scale), 1e-6)
        self._total_hours = float(total_hours)
        self._reward_tiers = sorted({int(t) for t in reward_tiers or []})

        current_nodes = self._emit_tier_nodes(self._cells, float(start_hour))
        trajectories = {tier: [TierNode(**node.__dict__)] for tier, node in current_nodes.items()}

        for i in range(int(steps)):
            hour = float(start_hour) + i * dt
            remaining = max(float(total_hours) - hour, dt)
            current_nodes = self.step(current_nodes, hour, self._scale, remaining, self._reward_tiers, dt_hours=dt)
            for tier, node in current_nodes.items():
                trajectories.setdefault(tier, []).append(TierNode(**node.__dict__))
        return trajectories

    def cell_rollout_snapshots(
        self,
        start_hour: float,
        hours_forward: float,
        scale: float,
        total_hours: float,
        reward_tiers: List[int],
        steps: Optional[int] = None,
    ) -> List[dict]:
        """Return hidden cell snapshots across future rollout without mutating state."""
        if not self._cells:
            return []

        saved_cells = self._clone_cells(self._cells)
        saved_last_hour = self._last_hour
        saved_scale = self._scale
        saved_total_hours = self._total_hours
        saved_reward_tiers = list(self._reward_tiers)
        saved_projection_cache = dict(self._projection_cache)
        saved_target_component_cache = dict(self._target_component_cache)
        saved_cell_array_cache = dict(self._cell_array_cache)

        try:
            c = self.config
            if steps is None:
                steps = max(int(np.ceil(hours_forward / c.dt_hours)), 1)
            dt = float(hours_forward) / max(int(steps), 1) if hours_forward > 0 else c.dt_hours
            self._scale = max(float(scale), 1e-6)
            self._total_hours = float(total_hours)
            self._reward_tiers = sorted({int(t) for t in reward_tiers or []})

            current_nodes = self._emit_tier_nodes(self._cells, float(start_hour))
            frames = [{
                'hour': float(start_hour),
                'rows': self.cell_snapshot(),
                'nodes': {tier: TierNode(**node.__dict__) for tier, node in current_nodes.items()},
            }]
            for i in range(int(steps)):
                hour = float(start_hour) + i * dt
                remaining = max(float(total_hours) - hour, dt)
                current_nodes = self.step(
                    current_nodes,
                    hour,
                    self._scale,
                    remaining,
                    self._reward_tiers,
                    dt_hours=dt,
                )
                emitted_nodes = {
                    tier: TierNode(**node.__dict__)
                    for tier, node in current_nodes.items()
                }
                frames.append({
                    'hour': float(hour + dt),
                    'rows': self.cell_snapshot(),
                    'nodes': emitted_nodes,
                })
            return frames
        finally:
            self._cells = saved_cells
            self._last_hour = saved_last_hour
            self._scale = saved_scale
            self._total_hours = saved_total_hours
            self._reward_tiers = saved_reward_tiers
            self._projection_cache = saved_projection_cache
            self._target_component_cache = saved_target_component_cache
            self._cell_array_cache = saved_cell_array_cache

    def cell_snapshot(self) -> List[dict]:
        """Return the current hidden cohort states for inspection/visualization."""
        rows = []
        observed_set = {int(t) for t in self._observed_tiers}
        reward_set = {int(t) for t in self._reward_tiers}
        for idx, cell in enumerate(self._cells):
            mode_probs = {mode.value: float(cell.mode_probs.get(mode, 0.0)) for mode in MODE_ORDER}
            dominant_mode = max(mode_probs.items(), key=lambda item: item[1])[0] if mode_probs else "unknown"
            nearest_reward, reward_distance = self._nearest_reward(cell.center_rank)
            strategic_target = self._strategic_target_for_rank(cell.center_rank)
            target_distance = (
                float(abs(cell.center_rank - strategic_target))
                if strategic_target is not None else float("inf")
            )
            target_probs = {
                f"T{int(tier)}": float(prob)
                for tier, prob in sorted(cell.target_probs.items())
            }
            rows.append({
                'cell_id': idx,
                'rank_start': int(cell.rank_start),
                'rank_end': int(cell.rank_end),
                'rank_center': float(cell.center_rank),
                'cell_size': int(cell.size),
                'score': float(cell.score),
                'speed': float(cell.speed),
                'speed_norm': float(cell.speed / max(self._scale, 1e-6)),
                'base_norm_speed': float(cell.base_norm_speed),
                'capacity_norm_speed': float(cell.capacity_norm_speed),
                'pressure': float(cell.pressure),
                'pressure_source': str(cell.pressure_source),
                'target_rank': None if cell.target_rank is None else int(cell.target_rank),
                'target_score': float(cell.target_score),
                'target_gap_norm': float(cell.target_gap_norm),
                'target_profile_norm': float(cell.target_profile_norm),
                'target_follow_norm': float(cell.target_follow_norm),
                'target_plan_lag_norm': float(cell.target_plan_lag_norm),
                'behavior_lazy': float(cell.behavior_lazy),
                'follow_rank': None if cell.follow_rank is None else int(cell.follow_rank),
                'target_surplus_norm': float(cell.target_surplus_norm),
                'target_boundary_proximity': float(cell.target_boundary_proximity),
                'target_risk_level': float(cell.target_risk_level),
                'target_pressure': float(cell.target_pressure),
                'neighbor_pressure': float(cell.neighbor_pressure),
                'density_pressure': float(cell.density_pressure),
                'target_importance': float(cell.target_importance),
                'speed_cruise_norm': float(cell.speed_cruise_norm),
                'speed_target_norm': float(cell.speed_target_norm),
                'speed_profile_norm': float(cell.speed_profile_norm),
                'speed_follow_norm': float(cell.speed_follow_norm),
                'speed_lag_norm': float(cell.speed_lag_norm),
                'speed_panic_feature_norm': float(cell.speed_panic_feature_norm),
                'speed_committed_norm': float(cell.speed_committed_norm),
                'speed_defend_norm': float(cell.speed_defend_norm),
                'speed_boundary_drive': float(cell.speed_boundary_drive),
                'speed_preseason_norm': float(cell.speed_preseason_norm),
                'speed_season_effect': float(cell.speed_season_effect),
                'speed_desired_norm': float(cell.speed_desired_norm),
                'behavior_coeffs': dict(cell.behavior_coeffs),
                'behavior_coeffs_label': ", ".join(
                    f"{name}:{value:.2f}"
                    for name, value in cell.behavior_coeffs.items()
                    if abs(float(value)) > 1e-3
                ),
                'speed_limit_reason': str(cell.speed_limit_reason),
                'target_probs': target_probs,
                'target_probs_label': ", ".join(
                    f"T{tier}:{prob:.2f}"
                    for tier, prob in sorted(cell.target_probs.items(), key=lambda item: item[1], reverse=True)[:4]
                ),
                'target_affinity': float(cell.target_affinity),
                'dominant_mode': dominant_mode,
                'nearest_reward_tier': None if nearest_reward is None else int(nearest_reward),
                'reward_rank_distance': float(reward_distance),
                'nearest_target_tier': None if strategic_target is None else int(strategic_target),
                'target_rank_distance': float(target_distance),
                'is_observed_anchor': int(cell.rank_start) == int(cell.rank_end) and int(cell.rank_start) in observed_set,
                'is_reward_anchor': int(cell.rank_start) == int(cell.rank_end) and int(cell.rank_start) in reward_set,
                **{f'mode_{name}': prob for name, prob in mode_probs.items()},
            })
        return rows

    @staticmethod
    def _clone_cells(cells: List[CohortCell]) -> List[CohortCell]:
        clones = []
        for cell in cells:
            data = dict(cell.__dict__)
            data['mode_probs'] = dict(cell.mode_probs)
            data['target_probs'] = dict(cell.target_probs)
            clones.append(CohortCell(**data))
        return clones

    def _prepare_context(
        self,
        tier_data: dict,
        meta: dict,
        reward_tiers: List[int],
        scale: Optional[float],
        align_freq_hours: Optional[float],
        observed_until_hour: Optional[float],
    ) -> None:
        surface = build_tier_surface(tier_data, meta)
        if observed_until_hour is not None:
            surface = surface[surface.index <= float(observed_until_hour)]
        surface, observed_tiers, _ = align_observed_tier_surface(surface, freq_hours=align_freq_hours)
        self._surface = surface
        self._observed_tiers = [int(t) for t in observed_tiers]
        self._reward_tiers = sorted({int(t) for t in reward_tiers or []})
        self._behavior_target_tiers = sorted(set(self._observed_tiers) | set(self._reward_tiers))
        self._emission_tiers = sorted(set(self._observed_tiers) | set(self._reward_tiers))
        self._start_ts = int(meta.get('start_at', 0) or 0)
        self._scale = max(float(scale or 1.0), 1e-6)
        self._total_hours = self._total_hours_from_meta(meta)
        self._speed_profile_adapted = self._adapt_speed_profile(surface)
        self._target_importance_by_tier = self._infer_target_importance(surface)
        self._guide_rank_cache = {}
        self._target_distribution_cache = {}
        self._target_risk_cache = {}
        self._progress_fraction_cache = {}
        self._progress_speed_factor_cache = {}
        self._target_importance_norm_cache = {}
        self._row_score_cache = {}
        self._target_component_cache = {}
        self._behavior_lazy_by_rank, self._behavior_lazy_global = self._fit_behavior_lazy(surface)
        self._behavior_coeffs_global, self._behavior_coeffs_by_rank = self._fit_behavior_coefficients(surface)
        self._target_distribution_cache = {}
        self._target_risk_cache = {}
        self._progress_fraction_cache = {}
        self._progress_speed_factor_cache = {}
        self._target_importance_norm_cache = {}
        self._row_score_cache = {}
        self._target_component_cache = {}

    def _infer_target_importance(self, surface) -> Dict[int, float]:
        if surface is None or surface.empty:
            return {}
        row = surface.iloc[-1]
        ranks = np.asarray(sorted({int(t) for t in self._observed_tiers if int(t) > 0 and int(t) in row.index}), dtype=float)
        if len(ranks) < 3:
            return {int(t): 1.0 if int(t) in set(self._reward_tiers) else 0.0 for t in self._observed_tiers}

        scores = np.asarray([float(row[int(rank)]) for rank in ranks], dtype=float)
        valid = np.isfinite(scores)
        ranks = ranks[valid]
        scores = scores[valid]
        if len(ranks) < 3:
            return {}

        left = ranks[:-1]
        right = ranks[1:]
        log_span = np.log(np.maximum(right, 1.0) / np.maximum(left, 1.0))
        score_gap = np.maximum(scores[:-1] - scores[1:], 0.0)
        valid_span = log_span > 1e-9
        if not np.any(valid_span):
            return {}

        slopes = score_gap[valid_span] / log_span[valid_span]
        left = left[valid_span]
        right = right[valid_span]
        log_span = log_span[valid_span]
        rank_floor = float(np.quantile(ranks, 0.65))
        tail_slopes = slopes[left >= rank_floor]
        if len(tail_slopes) < 2:
            tail_slopes = slopes[-min(3, len(slopes)):]
        natural = float(np.quantile(tail_slopes, 0.35)) if len(tail_slopes) else 0.0
        spread_pool = np.abs(slopes - natural)
        spread = float(np.quantile(spread_pool, 0.75)) if len(spread_pool) else 0.0
        spread = max(spread, max(abs(natural), 1.0) * 0.10, 1.0)

        span_scale = max(float(np.median(log_span)), 1e-6)
        local_evidence = np.maximum((slopes - natural) / spread, 0.0) * (log_span / span_scale)
        cumulative = np.zeros_like(local_evidence)
        running = 0.0
        for idx in range(len(local_evidence) - 1, -1, -1):
            running += float(local_evidence[idx])
            cumulative[idx] = running

        local_norm = local_evidence / max(float(np.quantile(local_evidence, 0.90)), 1e-6)
        cumulative_norm = cumulative / max(float(np.quantile(cumulative, 0.90)), 1e-6)
        reward_set = {int(t) for t in self._reward_tiers}
        raw_by_tier = {}
        for idx, rank in enumerate(left.astype(int)):
            evidence = 0.65 * local_norm[idx] + 0.35 * cumulative_norm[idx]
            reward_prior = 0.18 if int(rank) in reward_set else 0.0
            raw_by_tier[int(rank)] = float(max(evidence, 0.0) + reward_prior)

        if int(ranks[-1]) in reward_set:
            raw_by_tier[int(ranks[-1])] = max(raw_by_tier.get(int(ranks[-1]), 0.0), 0.18)
        else:
            raw_by_tier.setdefault(int(ranks[-1]), 0.0)
        for tier in self._reward_tiers:
            if int(tier) not in raw_by_tier and int(tier) > 0:
                raw_by_tier[int(tier)] = 0.18

        if not raw_by_tier:
            return {}
        raw_values = np.asarray(list(raw_by_tier.values()), dtype=float)
        max_raw = float(np.quantile(raw_values[raw_values > 0.0], 0.85)) if np.any(raw_values > 0.0) else 1.0
        max_raw = max(max_raw, 1e-6)
        return {
            int(tier): float(np.clip((max(value, 0.0) / max_raw) ** 0.75, 0.0, 1.0))
            for tier, value in raw_by_tier.items()
        }

    def _build_cells_from_surface(self, surface, current_hour: float) -> List[CohortCell]:
        max_rank = max(self._emission_tiers) if self._emission_tiers else 0
        if max_rank <= 0 or surface.empty:
            return []

        boundaries = set(self._emission_tiers)
        cells = []
        rank = max(min(self._observed_tiers or [1]), 1)
        while rank <= max_rank:
            if rank in boundaries:
                # Observed cutoff ranks are emission anchors; keep them as exact cells
                # so rollout starts from the cutoff score rather than a cell center.
                rank_end = rank
            else:
                group_size = self._rank_cell_size(rank)
                group_size = min(group_size, self._target_aware_cell_size(rank))
                rank_end = min(rank + group_size - 1, max_rank)
                next_boundary = min([b for b in boundaries if b > rank], default=None)
                if next_boundary is not None and next_boundary <= rank_end:
                    rank_end = next_boundary - 1
            center = (rank + rank_end) / 2.0
            score = self._score_at_rank(surface.iloc[-1], center)
            speed = self._observed_speed_at_rank(surface, center)
            base_speed = self._base_speed_at_rank(surface, center, current_hour)
            season_now = max(self._season_factor(current_hour), 1e-6)
            base_norm = float(np.clip(
                (base_speed / self._scale) / season_now,
                self.config.base_norm_floor,
                self._base_norm_upper_for_rank(center),
            ))
            mode_probs = self._initial_mode_probs(speed, base_speed, center)
            cells.append(CohortCell(
                rank_start=int(rank),
                rank_end=int(rank_end),
                center_rank=float(center),
                score=float(score),
                speed=float(max(speed, 0.0)),
                base_norm_speed=base_norm,
                capacity_norm_speed=self._capacity_norm_for_rank(center),
                pressure=0.0,
                target_affinity=0.0,
                mode_probs=mode_probs,
            ))
            rank = rank_end + 1
        return cells

    def _step_cells(self, cells: List[CohortCell], hour: float, remaining_hours: float, dt: float) -> List[CohortCell]:
        season = self._season_factor(hour)
        urgency = self._urgency(remaining_hours)
        self._projection_cache = {}
        self._target_component_cache = {}
        self._season_mean_cache = {}
        self._cell_array_cache = {}
        next_cells = []
        for idx, cell in enumerate(cells):
            target_state = self._target_pressure_state(cell, cells, hour, remaining_hours)
            neighbor_pressure = self._neighbor_pressure(cells, idx, remaining_hours)
            density = self._local_density(cells, idx, remaining_hours)
            pressure, pressure_source = self._compose_pressure(
                target_state['pressure'],
                neighbor_pressure,
                density,
                target_state['gap_norm'],
                target_state['surplus_norm'],
                target_state['boundary_proximity'],
            )

            mode_probs = self._transition_modes(
                cell,
                pressure,
                urgency,
                target_state['reachable'],
                target_state['gap_norm'],
                target_state['profile_norm'],
                target_state['follow_norm'],
                target_state['plan_lag_norm'],
                self._behavior_lazy_for_rank(cell.center_rank),
                target_state['surplus_norm'],
                target_state['boundary_proximity'],
            )
            desired_speed, speed_parts = self._desired_speed(
                cell,
                mode_probs,
                pressure,
                season,
                urgency,
                target_state['gap_norm'],
                target_state['profile_norm'],
                target_state['follow_norm'],
                target_state['boundary_proximity'],
            )

            cap = self._scale * max(cell.capacity_norm_speed, self.config.base_norm_floor)
            next_speed = float(np.clip(desired_speed, 0.0, cap))
            desired_norm = float(desired_speed / max(self._scale, 1e-6))
            cap_norm = float(cap / max(self._scale, 1e-6))
            if next_speed >= cap - 1e-9 and desired_norm >= cap_norm - 1e-9:
                speed_parts['speed_limit_reason'] = "capacity_cap"
            else:
                speed_parts['speed_limit_reason'] = "behavior_direct"
            next_score = cell.score + next_speed * dt * 60.0

            next_cells.append(CohortCell(
                rank_start=cell.rank_start,
                rank_end=cell.rank_end,
                center_rank=cell.center_rank,
                score=float(next_score),
                speed=next_speed,
                base_norm_speed=cell.base_norm_speed,
                capacity_norm_speed=cell.capacity_norm_speed,
                pressure=float(pressure),
                target_affinity=float(target_state['affinity']),
                target_rank=target_state['target_rank'],
                target_score=float(target_state['target_score']),
                target_gap_norm=float(target_state['gap_norm']),
                target_profile_norm=float(target_state['profile_norm']),
                target_follow_norm=float(target_state['follow_norm']),
                target_plan_lag_norm=float(target_state['plan_lag_norm']),
                behavior_lazy=float(speed_parts.get('behavior_lazy', 0.0)),
                follow_rank=target_state.get('follow_rank'),
                target_surplus_norm=float(target_state['surplus_norm']),
                target_boundary_proximity=float(target_state['boundary_proximity']),
                target_risk_level=float(target_state.get('risk_level', 0.5)),
                target_pressure=float(target_state['pressure']),
                neighbor_pressure=float(neighbor_pressure),
                density_pressure=float(density),
                target_importance=float(speed_parts.get('target_importance', 0.0)),
                speed_cruise_norm=float(speed_parts.get('speed_cruise_norm', 0.0)),
                speed_target_norm=float(speed_parts.get('speed_target_norm', 0.0)),
                speed_profile_norm=float(speed_parts.get('speed_profile_norm', 0.0)),
                speed_follow_norm=float(speed_parts.get('speed_follow_norm', 0.0)),
                speed_lag_norm=float(speed_parts.get('speed_lag_norm', 0.0)),
                speed_panic_feature_norm=float(speed_parts.get('speed_panic_feature_norm', 0.0)),
                speed_committed_norm=float(speed_parts.get('speed_committed_norm', 0.0)),
                speed_defend_norm=float(speed_parts.get('speed_defend_norm', 0.0)),
                speed_boundary_drive=float(speed_parts.get('speed_boundary_drive', 0.0)),
                speed_preseason_norm=float(speed_parts.get('speed_preseason_norm', 0.0)),
                speed_season_effect=float(speed_parts.get('speed_season_effect', 1.0)),
                speed_desired_norm=float(speed_parts.get('speed_desired_norm', 0.0)),
                behavior_coeffs=dict(speed_parts.get('behavior_coeffs', {})),
                speed_limit_reason=str(speed_parts.get('speed_limit_reason', 'unknown')),
                target_probs=dict(target_state['target_probs']),
                pressure_source=pressure_source,
                mode_probs=mode_probs,
            ))
        return self._enforce_rank_monotonicity(next_cells)

    def _desired_speed(
        self,
        cell: CohortCell,
        mode_probs: Dict[BehaviorMode, float],
        pressure: float,
        season: float,
        urgency: float,
        target_gap_norm: float,
        target_profile_norm: float,
        target_follow_norm: float,
        boundary_proximity: float,
    ) -> Tuple[float, Dict[str, float]]:
        c = self.config
        importance_norm = self._target_importance_norm(int(cell.target_rank)) if cell.target_rank is not None else 0.0
        cruise_norm = float(min(max(cell.base_norm_speed, c.base_norm_floor), cell.capacity_norm_speed))
        capacity_norm = float(max(cell.capacity_norm_speed, cruise_norm, c.base_norm_floor))

        required_norm = float(max(target_gap_norm, 0.0))
        target_norm = float(min(max(required_norm, cruise_norm), capacity_norm))

        profile_norm = float(min(max(target_profile_norm, 0.0), capacity_norm))
        follow_norm = float(min(max(target_follow_norm, 0.0), capacity_norm))
        coeffs = self._behavior_coeffs_for_rank(cell.center_rank)
        lag_norm = max(float(cell.target_plan_lag_norm), 0.0)
        panic_feature = capacity_norm * min(max(lag_norm / max(capacity_norm, 1e-6), 0.0), 1.0)
        planned_raw = (
            coeffs.get('cruise', 0.0) * cruise_norm
            + coeffs.get('profile', 0.0) * profile_norm
            + coeffs.get('follow', 0.0) * follow_norm
            + coeffs.get('gap', 0.0) * target_norm
            + coeffs.get('lag', 0.0) * lag_norm
            + coeffs.get('panic', 0.0) * panic_feature
        )
        planned_norm = float(min(max(max(cruise_norm, planned_raw), cruise_norm), capacity_norm))
        lazy = float(min(max(
            coeffs.get('follow', 0.0) / max(coeffs.get('profile', 0.0) + coeffs.get('follow', 0.0), 1e-6),
            0.0,
        ), 1.0))

        defend_need_raw = max(boundary_proximity, max(pressure, 0.0) / (1.0 + max(pressure, 0.0)))
        defend_need = float(min(max(defend_need_raw, 0.0), 1.0))
        defend_norm = float(cruise_norm + defend_need * (capacity_norm - cruise_norm))

        panic_norm = capacity_norm
        dropped_norm = cruise_norm

        idle_norm = cruise_norm
        watching_norm = cruise_norm
        preseason_norm = float(
            mode_probs.get(BehaviorMode.IDLE, 0.0) * idle_norm
            + mode_probs.get(BehaviorMode.CRUISING, 0.0) * cruise_norm
            + mode_probs.get(BehaviorMode.WATCHING, 0.0) * watching_norm
            + mode_probs.get(BehaviorMode.CHASING, 0.0) * planned_norm
            + mode_probs.get(BehaviorMode.DEFENDING, 0.0) * defend_norm
            + mode_probs.get(BehaviorMode.PANIC_RUSHING, 0.0) * panic_norm
            + mode_probs.get(BehaviorMode.DROPPED, 0.0) * dropped_norm
        )
        preseason_norm = float(min(max(preseason_norm, 0.0), capacity_norm))

        season_effect = max(float(season), 0.05)
        desired_norm = float(preseason_norm * season_effect)
        return float(self._scale * desired_norm), {
            'target_importance': float(importance_norm),
            'speed_cruise_norm': float(cruise_norm),
            'speed_target_norm': float(target_norm),
            'speed_profile_norm': float(profile_norm),
            'speed_follow_norm': float(follow_norm),
            'speed_lag_norm': float(lag_norm),
            'speed_panic_feature_norm': float(panic_feature),
            'behavior_lazy': float(lazy),
            'speed_committed_norm': float(planned_norm),
            'behavior_coeffs': dict(coeffs),
            'speed_defend_norm': float(defend_norm),
            'speed_boundary_drive': float(defend_need),
            'speed_preseason_norm': float(preseason_norm),
            'speed_season_effect': float(season_effect),
            'speed_desired_norm': float(desired_norm),
        }

    def _hydrate_cells(self, hour: float, cells: Optional[List[CohortCell]] = None) -> None:
        cells = self._cells if cells is None else cells
        self._cell_array_cache = {}
        total = self._total_hours or hour + self.config.dt_hours
        remaining = max(float(total) - float(hour), self.config.dt_hours)
        urgency = self._urgency(remaining)
        self._projection_cache = {}
        self._target_component_cache = {}
        for idx, cell in enumerate(cells):
            target_state = self._target_pressure_state(cell, cells, hour, remaining)
            npv = self._neighbor_pressure(cells, idx, remaining)
            density = self._local_density(cells, idx, remaining)
            pressure, pressure_source = self._compose_pressure(
                target_state['pressure'],
                npv,
                density,
                target_state['gap_norm'],
                target_state['surplus_norm'],
                target_state['boundary_proximity'],
            )
            cell.pressure = float(pressure)
            cell.target_affinity = float(target_state['affinity'])
            cell.target_rank = target_state['target_rank']
            cell.target_score = float(target_state['target_score'])
            cell.target_gap_norm = float(target_state['gap_norm'])
            cell.target_profile_norm = float(target_state['profile_norm'])
            cell.target_follow_norm = float(target_state['follow_norm'])
            cell.target_plan_lag_norm = float(target_state['plan_lag_norm'])
            cell.behavior_lazy = float(self._behavior_lazy_for_rank(cell.center_rank))
            cell.follow_rank = target_state.get('follow_rank')
            cell.target_surplus_norm = float(target_state['surplus_norm'])
            cell.target_boundary_proximity = float(target_state['boundary_proximity'])
            cell.target_risk_level = float(target_state.get('risk_level', 0.5))
            cell.target_pressure = float(target_state['pressure'])
            cell.neighbor_pressure = float(npv)
            cell.density_pressure = float(density)
            cell.target_importance = self._target_importance_norm(int(cell.target_rank)) if cell.target_rank is not None else 0.0
            cell.target_probs = dict(target_state['target_probs'])
            cell.pressure_source = pressure_source
            cell.mode_probs = self._transition_modes(
                cell,
                pressure,
                urgency,
                target_state['reachable'],
                target_state['gap_norm'],
                target_state['profile_norm'],
                target_state['follow_norm'],
                target_state['plan_lag_norm'],
                self._behavior_lazy_for_rank(cell.center_rank),
                target_state['surplus_norm'],
                target_state['boundary_proximity'],
            )
            _, speed_parts = self._desired_speed(
                cell,
                cell.mode_probs,
                pressure,
                self._season_factor(hour),
                urgency,
                target_state['gap_norm'],
                target_state['profile_norm'],
                target_state['follow_norm'],
                target_state['boundary_proximity'],
            )
            cell.target_importance = float(speed_parts.get('target_importance', cell.target_importance))
            cell.speed_cruise_norm = float(speed_parts.get('speed_cruise_norm', 0.0))
            cell.speed_target_norm = float(speed_parts.get('speed_target_norm', 0.0))
            cell.speed_profile_norm = float(speed_parts.get('speed_profile_norm', 0.0))
            cell.speed_follow_norm = float(speed_parts.get('speed_follow_norm', 0.0))
            cell.speed_lag_norm = float(speed_parts.get('speed_lag_norm', 0.0))
            cell.speed_panic_feature_norm = float(speed_parts.get('speed_panic_feature_norm', 0.0))
            cell.speed_committed_norm = float(speed_parts.get('speed_committed_norm', 0.0))
            cell.speed_defend_norm = float(speed_parts.get('speed_defend_norm', 0.0))
            cell.speed_boundary_drive = float(speed_parts.get('speed_boundary_drive', 0.0))
            cell.speed_preseason_norm = float(speed_parts.get('speed_preseason_norm', 0.0))
            cell.speed_season_effect = float(speed_parts.get('speed_season_effect', 1.0))
            cell.speed_desired_norm = float(speed_parts.get('speed_desired_norm', 0.0))
            cell.behavior_coeffs = dict(speed_parts.get('behavior_coeffs', {}))

    def _emit_tier_nodes(self, cells: List[CohortCell], hour: float) -> Dict[int, TierNode]:
        emitted = {}
        for tier in self._emission_tiers:
            cell = self._cell_for_rank(cells, tier)
            if cell is None:
                continue
            anchor_score = self._observed_anchor_score(tier, hour)
            emitted[int(tier)] = TierNode(
                tier=int(tier),
                score=float(anchor_score if anchor_score is not None else self._score_from_cells(cells, tier)),
                speed=float(cell.speed),
                base_norm_speed=float(cell.base_norm_speed),
                pressure=float(cell.pressure),
                target_affinity=float(cell.target_affinity),
                volatility=0.0,
                is_reward_line=int(tier) in set(self._reward_tiers),
                mode_mix={m.value: float(cell.mode_probs.get(m, 0.0)) for m in MODE_ORDER},
            )
        return emitted

    def _observed_anchor_score(self, tier: int, hour: float) -> Optional[float]:
        """Return exact observed score at an emitted tier when replay/init is on an observed row."""
        if self._surface is None or self._surface.empty or int(tier) not in self._surface.columns:
            return None

        index_values = self._surface.index.to_numpy(dtype=float)
        if len(index_values) == 0:
            return None

        pos = int(np.argmin(np.abs(index_values - float(hour))))
        if abs(index_values[pos] - float(hour)) > 1e-6:
            return None

        value = self._surface.iloc[pos][int(tier)]
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except Exception:
            return None

    def _rank_cell_size(self, rank: int) -> int:
        if rank <= 0:
            return self.config.min_cell_size
        power = max(int(np.floor(np.log10(rank))) - 1, 0)
        size = 10 ** power
        return int(np.clip(size, self.config.min_cell_size, self.config.max_cell_size))

    def _target_aware_cell_size(self, rank: int) -> int:
        if not self._behavior_target_tiers:
            return self.config.max_cell_size

        rank = max(int(rank), 1)
        nearest, _ = self._nearest_behavior_target(float(rank))
        if nearest is None or nearest <= 0:
            return self.config.max_cell_size

        radius = max(int(np.ceil(nearest * self.config.reward_anchor_radius)), 1)
        if abs(rank - nearest) > radius:
            return self.config.max_cell_size

        target_base = self._rank_cell_size(int(nearest))
        refined = max(target_base // 10, self.config.min_cell_size)
        return int(np.clip(refined, self.config.min_cell_size, self.config.max_cell_size))

    def _score_at_rank(self, row, rank: float) -> float:
        cache_key = (id(row.index), getattr(row, "name", None))
        cached = self._row_score_cache.get(cache_key)
        if cached is None:
            valid = row.dropna()
            ranks = np.asarray(sorted(valid.index), dtype=float)
            if len(ranks) == 0:
                scores = np.asarray([], dtype=float)
            else:
                scores = np.asarray([valid[int(r)] for r in ranks], dtype=float)
            cached = (ranks, scores)
            self._row_score_cache[cache_key] = cached
        ranks, scores = cached
        if len(ranks) == 0:
            return 0.0
        if rank <= ranks[0]:
            return float(scores[0])
        if rank >= ranks[-1]:
            return float(scores[-1])
        hi_idx = int(np.searchsorted(ranks, rank, side="right") - 1)
        lo_idx = hi_idx + 1
        r_hi, r_lo = ranks[hi_idx], ranks[lo_idx]
        s_hi, s_lo = scores[hi_idx], scores[lo_idx]
        x = (rank - r_hi) / max(r_lo - r_hi, 1e-6)
        shaped_x = self._reward_aware_interval_position(r_hi, r_lo, x)
        return float(s_hi * (1.0 - shaped_x) + s_lo * shaped_x)

    def _reward_aware_interval_position(self, r_hi: float, r_lo: float, x: float) -> float:
        reward_set = set(self._reward_tiers)
        if int(r_hi) in reward_set:
            # Just outside an upper reward line, many players stop chasing that
            # specific line.  This still applies when the lower anchor is another
            # reward tier, e.g. T1200 is usually much closer to T2000 than T1000.
            return float(np.clip(x, 0.0, 1.0))
        if int(r_lo) in reward_set and int(r_hi) not in reward_set:
            # Approaching a lower reward boundary, attraction increases near the boundary.
            return float(np.clip(x, 0.0, 1.0))
        return float(np.clip(x, 0.0, 1.0))

    def _observed_speed_at_rank(self, surface, rank: float) -> float:
        if len(surface) < 2:
            return 0.0
        s_now = self._score_at_rank(surface.iloc[-1], rank)
        s_prev = self._score_at_rank(surface.iloc[-2], rank)
        dt = max(float(surface.index[-1] - surface.index[-2]), 1e-6)
        return max((s_now - s_prev) / (dt * 60.0), 0.0)

    def _observed_or_interpolated_score(self, row, tier: int) -> float:
        if int(tier) in row.index:
            try:
                value = float(row[int(tier)])
                if np.isfinite(value):
                    return value
            except Exception:
                pass
        return self._score_at_rank(row, float(tier))

    def _observed_speed_series_by_tier(self, surface, tiers: List[int]) -> Dict[int, np.ndarray]:
        hours = np.asarray(surface.index, dtype=float)
        result = {}
        for tier in tiers:
            tier = int(tier)
            scores = np.asarray([
                self._observed_or_interpolated_score(row, tier)
                for _, row in surface.iterrows()
            ], dtype=float)
            speeds = np.zeros_like(scores)
            if len(scores) > 1:
                dt = np.maximum(np.diff(hours), 1e-6)
                speeds[1:] = np.maximum(np.diff(scores) / (dt * 60.0), 0.0)
                speeds[0] = speeds[1]
            result[tier] = speeds
        return result

    def _observed_pressure_for_tier(
        self,
        tier: int,
        score: float,
        score_by_tier: Dict[int, np.ndarray],
        pos: int,
        remaining_hours: float,
    ) -> Tuple[float, float, float]:
        reward, _ = self._nearest_reward(float(tier))
        if reward is None:
            return 0.0, 0.0, 1.0

        affinity = self._target_importance_norm(int(reward))

        reward_scores = score_by_tier.get(int(reward))
        if reward_scores is None or pos >= len(reward_scores):
            return 0.0, affinity, 1.0

        target_score = float(reward_scores[pos])
        score_gap = max(target_score - float(score), 0.0)
        required_norm = score_gap / max(remaining_hours * 60.0 * self._scale, 1e-6)
        capacity_norm = max(self._capacity_norm_for_rank(float(tier)), self.config.base_norm_floor)
        reachable = float(required_norm <= capacity_norm)
        pressure = float(affinity * required_norm / max(capacity_norm, 1e-6))
        return float(pressure), affinity, reachable

    def _observed_mode_probs(
        self,
        speed: float,
        base_speed: float,
        pressure: float,
        reachable: float,
    ) -> Dict[BehaviorMode, float]:
        if reachable <= 0.0:
            return {mode: (1.0 if mode == BehaviorMode.DROPPED else 0.0) for mode in MODE_ORDER}
        if pressure > 0.0:
            panic = float(np.clip(pressure, 0.0, 1.0))
            return {
                mode: (
                    1.0 - panic if mode == BehaviorMode.CHASING
                    else panic if mode == BehaviorMode.PANIC_RUSHING
                    else 0.0
                )
                for mode in MODE_ORDER
            }
        return {mode: (1.0 if mode == BehaviorMode.CRUISING else 0.0) for mode in MODE_ORDER}

    def _fit_behavior_lazy(self, surface) -> Tuple[Dict[int, float], float]:
        if surface is None or surface.empty or len(surface) < 3:
            return {}, 0.0
        hours = np.asarray(surface.index, dtype=float)
        if len(hours) < 3:
            return {}, 0.0

        latest_row = surface.iloc[-1]
        latest_progress = float(hours[-1]) / max(float(self._total_hours or hours[-1] or 1.0), 1e-6)
        min_progress = min(max(0.10, latest_progress * 0.25), latest_progress)

        by_rank: Dict[int, float] = {}
        all_samples = []
        for rank in self._observed_tiers:
            rank = int(rank)
            target = self._strategic_target_for_rank(float(rank))
            if target is None:
                continue
            guide = self._guide_rank_for_target(int(target))
            if guide is None:
                continue

            samples = []
            for pos in range(1, len(hours)):
                hour = float(hours[pos])
                progress = hour / max(float(self._total_hours or hour or 1.0), 1e-6)
                if progress < min_progress:
                    continue
                dt = max(float(hours[pos] - hours[pos - 1]), 1e-6)
                season = max(self._season_factor(hour), 0.05)
                current_score = self._observed_or_interpolated_score(surface.iloc[pos], rank)
                previous_score = self._observed_or_interpolated_score(surface.iloc[pos - 1], rank)
                observed_norm = max((current_score - previous_score) / (dt * 60.0), 0.0)
                observed_norm /= max(self._scale * season, 1e-6)

                guide_score = self._observed_or_interpolated_score(surface.iloc[pos], int(guide))
                guide_prev = self._observed_or_interpolated_score(surface.iloc[pos - 1], int(guide))
                follow_norm = max((guide_score - guide_prev) / (dt * 60.0), 0.0)
                follow_norm /= max(self._scale * season, 1e-6)

                target_score_now = self._observed_or_interpolated_score(surface.iloc[pos], int(target))
                fraction = self._score_progress_fraction(int(target), progress, 0.0)
                if fraction is None or fraction <= 0.0:
                    continue
                target_final = max(target_score_now / max(fraction, 1e-6), target_score_now)
                profile_norm = self._target_profile_norm(int(target), hour, target_final, 0.0)
                if profile_norm is None:
                    continue

                denom = follow_norm - profile_norm
                if abs(denom) < 1e-4:
                    continue
                lazy = (observed_norm - profile_norm) / denom
                if np.isfinite(lazy):
                    samples.append(float(min(max(lazy, 0.0), 1.0)))

            if samples:
                weights = np.linspace(0.5, 1.0, len(samples))
                order = np.argsort(samples)
                sorted_samples = np.asarray(samples, dtype=float)[order]
                sorted_weights = weights[order]
                cutoff = 0.5 * float(sorted_weights.sum())
                value = float(sorted_samples[np.searchsorted(np.cumsum(sorted_weights), cutoff)])
                by_rank[rank] = value
                all_samples.extend(samples)

        global_value = float(np.median(all_samples)) if all_samples else 0.0
        return by_rank, global_value

    def _fit_behavior_coefficients(self, surface) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
        fallback = {'cruise': 0.20, 'profile': 0.45, 'follow': 0.20, 'gap': 0.10, 'lag': 0.05, 'panic': 0.0}
        if surface is None or surface.empty or len(surface) < 4:
            prior_fallback = self._behavior_prior_coeffs_for_rank(None, fallback)
            return prior_fallback, {}

        rows, y, weights, by_rank_rows, by_rank_y, by_rank_weights = self._collect_behavior_fit_samples(surface)
        global_prior = self._behavior_prior_stats_for_rank(None)
        global_coeffs = self._solve_behavior_coefficients(
            rows,
            y,
            BEHAVIOR_COEFF_NAMES,
            self._behavior_prior_coeffs_for_rank(None, fallback),
            prior_stats=global_prior,
            sample_weights=weights,
            prior_sample_weight=float(self.config.behavior_prior_sample_weight),
        )
        by_rank = {}
        for rank, rank_rows in by_rank_rows.items():
            if len(rank_rows) >= len(BEHAVIOR_COEFF_NAMES) + 2:
                by_rank[rank] = self._solve_behavior_coefficients(
                    rank_rows,
                    by_rank_y[rank],
                    BEHAVIOR_COEFF_NAMES,
                    self._behavior_prior_coeffs_for_rank(rank, global_coeffs),
                    prior_stats=self._behavior_prior_stats_for_rank(rank),
                    sample_weights=by_rank_weights.get(rank),
                    prior_sample_weight=float(self.config.behavior_prior_sample_weight),
                )
        return global_coeffs, by_rank

    def collect_behavior_fit_normal_equations(
        self,
        tier_data: dict,
        meta: dict,
        reward_tiers: List[int],
        scale: float,
        align_freq_hours: Optional[float] = 1.0,
        sample_weight_mode: str = "uniform",
    ) -> dict:
        """Collect graph behavior fitting statistics for offline training."""
        self._prepare_context(tier_data, meta, reward_tiers, scale, align_freq_hours, None)
        rows, y, weights, by_rank_rows, by_rank_y, by_rank_weights = self._collect_behavior_fit_samples(
            self._surface,
            sample_weight_mode=sample_weight_mode,
        )
        payload = self._behavior_normal_equations_payload(rows, y, weights, by_rank_rows, by_rank_y, by_rank_weights)
        payload["sample_weight_mode"] = str(sample_weight_mode)
        return payload

    def _collect_behavior_fit_samples(self, surface, sample_weight_mode: str = "uniform"):
        rows = []
        y = []
        weights = []
        by_rank_rows: Dict[int, List[List[float]]] = {}
        by_rank_y: Dict[int, List[float]] = {}
        by_rank_weights: Dict[int, List[float]] = {}
        hours = np.asarray(surface.index, dtype=float)
        for pos in range(1, len(hours)):
            hour = float(hours[pos])
            previous_hour = float(hours[pos - 1])
            dt = max(hour - previous_hour, 1e-6)
            remaining = max((self._total_hours or hour) - hour, dt)
            season = max(self._season_factor(hour), 0.05)
            progress = hour / max(float(self._total_hours or hour or 1.0), 1e-6)
            current_row = surface.iloc[pos]
            previous_row = surface.iloc[pos - 1]
            for rank in self._observed_tiers:
                rank = int(rank)
                target = self._strategic_target_for_rank(float(rank))
                if target is None:
                    continue
                current_score = self._observed_or_interpolated_score(current_row, rank)
                previous_score = self._observed_or_interpolated_score(previous_row, rank)
                observed_norm = max((current_score - previous_score) / (dt * 60.0), 0.0)
                observed_norm /= max(self._scale * season, 1e-6)

                cruise = self._speed_profile_value("cruise", float(rank)) or self.config.base_norm_floor
                cruise = min(max(float(cruise), self.config.base_norm_floor), self._capacity_norm_for_rank(float(rank)))
                capacity = max(self._capacity_norm_for_rank(float(rank)), self.config.base_norm_floor)

                target_score_now = self._observed_or_interpolated_score(current_row, int(target))
                fraction = self._score_progress_fraction(int(target), progress, 0.0)
                if fraction is None or fraction <= 0.0:
                    continue
                target_final = max(target_score_now / max(fraction, 1e-6), target_score_now)
                profile = self._target_profile_norm(int(target), hour, target_final, 0.0)
                if profile is None:
                    continue
                guide = self._guide_rank_for_target(int(target))
                follow = 0.0
                if guide is not None:
                    guide_score = self._observed_or_interpolated_score(current_row, int(guide))
                    guide_previous = self._observed_or_interpolated_score(previous_row, int(guide))
                    follow = max((guide_score - guide_previous) / (dt * 60.0), 0.0)
                    follow /= max(self._scale * season, 1e-6)

                season_hours = remaining * self._remaining_season_mean(hour, remaining)
                gap = max((target_final - current_score) / max(season_hours * 60.0 * self._scale, 1e-6), 0.0)
                plan_score = target_final * float(fraction)
                lag = max(plan_score - current_score, 0.0) / max(remaining * 60.0 * self._scale, 1e-6)
                panic = capacity * min(max(lag / max(capacity, 1e-6), 0.0), 1.0)

                feature = [
                    min(max(cruise, 0.0), capacity),
                    min(max(float(profile), 0.0), capacity),
                    min(max(float(follow), 0.0), capacity),
                    min(max(float(gap), 0.0), capacity),
                    min(max(float(lag), 0.0), capacity),
                    min(max(float(panic), 0.0), capacity),
                ]
                target_y = min(max(float(observed_norm), 0.0), capacity)
                sample_weight = self._behavior_fit_sample_weight(
                    progress=progress,
                    remaining_hours=remaining,
                    total_hours=float(self._total_hours or hour or 1.0),
                    observed_norm=target_y,
                    capacity_norm=capacity,
                    mode=sample_weight_mode,
                )
                rows.append(feature)
                y.append(target_y)
                weights.append(sample_weight)
                by_rank_rows.setdefault(rank, []).append(feature)
                by_rank_y.setdefault(rank, []).append(target_y)
                by_rank_weights.setdefault(rank, []).append(sample_weight)
        return rows, y, weights, by_rank_rows, by_rank_y, by_rank_weights

    @staticmethod
    def _behavior_fit_sample_weight(
        *,
        progress: float,
        remaining_hours: float,
        total_hours: float,
        observed_norm: float,
        capacity_norm: float,
        mode: str,
    ) -> float:
        """Weight one-hour fitting samples for different offline objectives."""
        progress = float(min(max(progress, 0.0), 1.0))
        remaining_fraction = float(min(max(remaining_hours / max(total_hours, 1e-6), 0.0), 1.0))
        effort = float(min(max(observed_norm / max(capacity_norm, 1e-6), 0.0), 1.0))
        mode = str(mode or "uniform").lower()
        if mode in ("uniform", "plain"):
            value = 1.0
        elif mode in ("rollout_path", "path", "remaining"):
            # A speed error early affects more future curve checkpoints.
            value = 0.35 + 1.65 * remaining_fraction
        elif mode in ("mid_curve", "mid"):
            # Emphasize the part that dominates post-freeze path shape.
            value = 0.50 + 2.00 * (progress * (1.0 - progress) * 4.0)
        elif mode in ("late_final", "late"):
            # Final-score oriented objective.
            value = 0.35 + 1.65 * (progress ** 2)
        elif mode in ("effort", "burst"):
            value = 0.50 + 1.50 * effort
        else:
            value = 1.0
        return float(min(max(value, 0.05), 4.0))

    @staticmethod
    def _behavior_normal_equations_payload(
        rows: List[List[float]],
        y: List[float],
        weights: List[float],
        by_rank_rows: Dict[int, List[List[float]]],
        by_rank_y: Dict[int, List[float]],
        by_rank_weights: Dict[int, List[float]],
    ) -> dict:
        def build_stats(sample_rows, sample_y, sample_weights):
            x = np.asarray(sample_rows, dtype=float)
            target = np.asarray(sample_y, dtype=float)
            weight = np.asarray(sample_weights, dtype=float)
            if len(weight) != len(target):
                weight = np.ones(len(target), dtype=float)
            valid = np.isfinite(target) & np.all(np.isfinite(x), axis=1) if len(target) and len(x) else np.asarray([], dtype=bool)
            x = x[valid] if len(valid) else np.empty((0, len(BEHAVIOR_COEFF_NAMES)))
            target = target[valid] if len(valid) else np.asarray([], dtype=float)
            weight = weight[valid] if len(valid) else np.asarray([], dtype=float)
            weight = np.where(np.isfinite(weight), np.maximum(weight, 0.0), 0.0)
            if len(target) == 0:
                return {
                    "sample_count": 0,
                    "weight_sum": 0.0,
                    "xtx": np.zeros((len(BEHAVIOR_COEFF_NAMES), len(BEHAVIOR_COEFF_NAMES))).tolist(),
                    "xty": np.zeros(len(BEHAVIOR_COEFF_NAMES)).tolist(),
                    "y_sum": 0.0,
                    "y2_sum": 0.0,
                }
            weighted_x = x * weight[:, None]
            return {
                "sample_count": int(len(target)),
                "weight_sum": float(np.sum(weight)),
                "xtx": (x.T @ weighted_x).tolist(),
                "xty": (x.T @ (target * weight)).tolist(),
                "y_sum": float(np.sum(target * weight)),
                "y2_sum": float(np.sum(np.square(target) * weight)),
            }

        return {
            "coeff_names": list(BEHAVIOR_COEFF_NAMES),
            "global": build_stats(rows, y, weights),
            "by_rank": {
                str(int(rank)): build_stats(rank_rows, by_rank_y[int(rank)], by_rank_weights.get(int(rank), []))
                for rank, rank_rows in sorted(by_rank_rows.items())
            },
        }

    @staticmethod
    def _solve_behavior_coefficients(
        rows: List[List[float]],
        y: List[float],
        names: List[str],
        fallback: Dict[str, float],
        prior_stats: Optional[dict] = None,
        sample_weights: Optional[List[float]] = None,
        prior_sample_weight: float = 1.0,
    ) -> Dict[str, float]:
        if len(y) != len(rows):
            return dict(fallback)
        x = np.asarray(rows, dtype=float)
        target = np.asarray(y, dtype=float)
        weights = np.asarray(sample_weights, dtype=float) if sample_weights is not None else np.ones(len(target), dtype=float)
        if len(weights) != len(target):
            weights = np.ones(len(target), dtype=float)
        valid = np.isfinite(target) & np.all(np.isfinite(x), axis=1)
        x = x[valid]
        target = target[valid]
        weights = weights[valid]
        weights = np.where(np.isfinite(weights), np.maximum(weights, 0.0), 0.0)
        if len(target) < len(names) and not prior_stats:
            return dict(fallback)
        if len(target):
            xtx = x.T @ (x * weights[:, None])
            xty = x.T @ (target * weights)
        else:
            xtx = np.zeros((len(names), len(names)), dtype=float)
            xty = np.zeros(len(names), dtype=float)

        if prior_stats:
            prior_count = max(int(prior_stats.get("sample_count", 0) or 0), 0)
            if prior_count > 0:
                prior_xtx = np.asarray(prior_stats.get("xtx") or [], dtype=float)
                prior_xty = np.asarray(prior_stats.get("xty") or [], dtype=float)
                if prior_xtx.shape == xtx.shape and prior_xty.shape == xty.shape:
                    # Historical prior is normalized then given the same sample
                    # mass as the current observed segment. This regularizes
                    # missing early features without letting old events dominate.
                    prior_mass = max(float(prior_stats.get("weight_sum", prior_count) or prior_count), 1.0)
                    current_mass = max(float(np.sum(weights)) if len(weights) else 0.0, float(len(names)))
                    prior_weight = max(float(prior_sample_weight), 0.0)
                    xtx = xtx + prior_xtx / prior_mass * current_mass * prior_weight
                    xty = xty + prior_xty / prior_mass * current_mass * prior_weight
        try:
            coeff = np.linalg.solve(xtx + np.eye(len(names)) * 1e-9, xty)
        except np.linalg.LinAlgError:
            try:
                coeff, *_ = np.linalg.lstsq(xtx + np.eye(len(names)) * 1e-9, xty, rcond=None)
            except np.linalg.LinAlgError:
                return dict(fallback)
        coeff = np.maximum(np.asarray(coeff, dtype=float), 0.0)
        if not np.any(coeff > 0.0):
            return dict(fallback)
        if len(target):
            pred = x @ coeff
            scale = float(np.median(target) / max(float(np.median(pred)), 1e-6)) if len(pred) else 1.0
            coeff *= min(max(scale, 0.1), 10.0)
        coeff = np.clip(coeff, 0.0, 3.0)
        return {name: float(value) for name, value in zip(names, coeff)}

    def _behavior_lazy_for_rank(self, rank: float) -> float:
        if not self._behavior_lazy_by_rank:
            return float(self._behavior_lazy_global)
        ranks = np.asarray(sorted(self._behavior_lazy_by_rank), dtype=float)
        values = np.asarray([self._behavior_lazy_by_rank[int(r)] for r in ranks], dtype=float)
        value = float(np.interp(
            np.log(max(float(rank), 1.0)),
            np.log(np.maximum(ranks, 1.0)),
            values,
            left=values[0],
            right=values[-1],
        ))
        return float(min(max(value, 0.0), 1.0))

    def _behavior_coeffs_for_rank(self, rank: float) -> Dict[str, float]:
        if not self._behavior_coeffs_by_rank:
            return dict(self._behavior_coeffs_global)
        ranks = np.asarray(sorted(self._behavior_coeffs_by_rank), dtype=float)
        log_rank = float(np.log(max(float(rank), 1.0)))
        result = {}
        names = sorted(set().union(*[set(v) for v in self._behavior_coeffs_by_rank.values()]))
        for name in names:
            values = np.asarray([
                self._behavior_coeffs_by_rank[int(r)].get(name, self._behavior_coeffs_global.get(name, 0.0))
                for r in ranks
            ], dtype=float)
            result[name] = float(np.interp(
                log_rank,
                np.log(np.maximum(ranks, 1.0)),
                values,
                left=values[0],
                right=values[-1],
            ))
        for name, value in self._behavior_coeffs_global.items():
            result.setdefault(name, float(value))
        return result

    def _guide_rank_for_target(self, target: int) -> Optional[int]:
        target = int(target)
        if target in self._guide_rank_cache:
            return self._guide_rank_cache[target]
        inner = [int(tier) for tier in self._observed_tiers if 0 < int(tier) < target]
        guide = max(inner) if inner else target
        self._guide_rank_cache[target] = guide
        return guide

    def _follow_line_norm(self, target: int, hour: float, cells: List[CohortCell]) -> float:
        guide = self._guide_rank_for_target(int(target))
        if guide is None:
            return 0.0
        guide_cell = self._cell_for_rank(cells, int(guide))
        if guide_cell is None:
            return 0.0
        season = max(self._season_factor(hour), 0.05)
        return float(max(guide_cell.speed, 0.0) / max(self._scale * season, 1e-6))

    def _correct_cells_to_observation(
        self,
        cells: List[CohortCell],
        current_row,
        previous_row,
        hour: float,
        dt: float,
    ) -> List[CohortCell]:
        corrected = []
        observed_set = {int(t) for t in self._observed_tiers}
        season_now = max(self._season_factor(hour), 1e-6)
        for cell in cells:
            measurement_score = self._score_at_rank(current_row, cell.center_rank)
            previous_measurement_score = self._score_at_rank(previous_row, cell.center_rank)
            measurement_speed = max((measurement_score - previous_measurement_score) / (dt * 60.0), 0.0)

            is_exact_anchor = (
                int(cell.rank_start) == int(cell.rank_end)
                and int(cell.rank_start) in observed_set
            )
            score = measurement_score
            speed = measurement_speed
            observed_base_norm = (measurement_speed / max(self._scale, 1e-6)) / season_now
            observed_base_norm = float(np.clip(
                observed_base_norm,
                self.config.base_norm_floor,
                self._base_norm_upper_for_rank(cell.center_rank),
            ))
            base_norm = observed_base_norm

            corrected.append(CohortCell(
                rank_start=cell.rank_start,
                rank_end=cell.rank_end,
                center_rank=cell.center_rank,
                score=float(score),
                speed=float(max(speed, 0.0)),
                base_norm_speed=base_norm,
                capacity_norm_speed=cell.capacity_norm_speed,
                pressure=float(cell.pressure),
                target_affinity=float(cell.target_affinity),
                target_rank=cell.target_rank,
                target_score=float(cell.target_score),
                target_gap_norm=float(cell.target_gap_norm),
                target_profile_norm=float(cell.target_profile_norm),
                target_follow_norm=float(cell.target_follow_norm),
                target_plan_lag_norm=float(cell.target_plan_lag_norm),
                behavior_lazy=float(cell.behavior_lazy),
                follow_rank=cell.follow_rank,
                target_surplus_norm=float(cell.target_surplus_norm),
                target_boundary_proximity=float(cell.target_boundary_proximity),
                target_risk_level=float(cell.target_risk_level),
                target_pressure=float(cell.target_pressure),
                neighbor_pressure=float(cell.neighbor_pressure),
                density_pressure=float(cell.density_pressure),
                target_importance=float(cell.target_importance),
                speed_cruise_norm=float(cell.speed_cruise_norm),
                speed_target_norm=float(cell.speed_target_norm),
                speed_committed_norm=float(cell.speed_committed_norm),
                speed_defend_norm=float(cell.speed_defend_norm),
                speed_boundary_drive=float(cell.speed_boundary_drive),
                speed_preseason_norm=float(cell.speed_preseason_norm),
                speed_season_effect=float(cell.speed_season_effect),
                speed_desired_norm=float(cell.speed_desired_norm),
                behavior_coeffs=dict(cell.behavior_coeffs),
                speed_limit_reason=str(cell.speed_limit_reason),
                pressure_source=cell.pressure_source,
                target_probs=dict(cell.target_probs),
                mode_probs=dict(cell.mode_probs),
            ))
        return self._enforce_rank_monotonicity(corrected)

    def _emission_errors(self, cells: List[CohortCell], row) -> List[float]:
        return list(self._emission_error_map(cells, row).values())

    def _emission_error_map(self, cells: List[CohortCell], row) -> Dict[int, float]:
        errors = []
        result = {}
        for tier in self._observed_tiers:
            if int(tier) not in row.index:
                continue
            try:
                observed = float(row[int(tier)])
            except Exception:
                continue
            if not np.isfinite(observed):
                continue
            predicted = self._score_from_cells(cells, int(tier))
            result[int(tier)] = float(predicted - observed)
        return result

    def _base_speed_at_rank(self, surface, rank: float, current_hour: float) -> float:
        window_start = float(current_hour) - 24.0
        recent = surface[surface.index >= window_start]
        if len(recent) < 2:
            recent = surface
        speeds = []
        for i in range(1, len(recent)):
            dt = max(float(recent.index[i] - recent.index[i - 1]), 1e-6)
            ds = self._score_at_rank(recent.iloc[i], rank) - self._score_at_rank(recent.iloc[i - 1], rank)
            speeds.append(max(ds / (dt * 60.0), 0.0))
        if not speeds:
            return 0.0
        positive = np.asarray([speed for speed in speeds if speed > 1e-6], dtype=float)
        if len(positive) == 0:
            return 0.0
        observed_active_idle = float(np.quantile(positive, 0.25))
        profile_cruise = self._speed_profile_value("cruise", rank, use_adapted=False)
        if profile_cruise is not None:
            return float(min(observed_active_idle, profile_cruise * self._scale))
        return observed_active_idle

    @staticmethod
    def _load_speed_profile(path: Optional[str]) -> Dict[str, dict]:
        profile_path = Path(path) if path else DEFAULT_SPEED_PROFILE_PATH
        if not profile_path.is_absolute():
            profile_path = PROJECT_ROOT / profile_path
        try:
            raw = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}

        summary = raw.get("profile_summary", {}) if isinstance(raw, dict) else {}
        grid = np.asarray(summary.get("rank_grid") or [], dtype=float)
        if grid.size == 0:
            return {}

        result = {"rank_grid": grid}
        curve_specs = {
            "capacity": ("capacity_curve_distribution", "q50"),
            "cruise": ("cruise_curve_distribution", "q50"),
            "cruise_high": ("cruise_curve_distribution", "q75"),
            "idle": ("idle_curve_distribution", "q50"),
        }
        for name, (section_name, quantile_name) in curve_specs.items():
            section = summary.get(section_name, {})
            values = np.asarray(section.get(quantile_name) or [], dtype=float)
            if values.size == grid.size:
                result[name] = values
        progress_summary = raw.get("score_progress_summary", {}) if isinstance(raw, dict) else {}
        progress_grid = np.asarray(progress_summary.get("progress_grid") or [], dtype=float)
        tier_summaries = progress_summary.get("tier_summaries", {})
        progress_tiers = []
        progress_curves_q10 = []
        progress_curves_q25 = []
        progress_curves_q50 = []
        if progress_grid.size:
            for tier_raw, tier_summary in sorted(
                tier_summaries.items(),
                key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
            ):
                try:
                    tier = int(tier_raw)
                except (TypeError, ValueError):
                    continue
                quantiles = tier_summary.get("progress_quantiles", {}) if isinstance(tier_summary, dict) else {}
                curve_q10 = np.asarray(quantiles.get("q10") or quantiles.get("q25") or quantiles.get("q50") or [], dtype=float)
                curve_q25 = np.asarray(quantiles.get("q25") or quantiles.get("q50") or [], dtype=float)
                curve_q50 = np.asarray(quantiles.get("q50") or quantiles.get("q25") or [], dtype=float)
                if curve_q10.size == progress_grid.size and curve_q25.size == progress_grid.size and curve_q50.size == progress_grid.size:
                    progress_tiers.append(float(tier))
                    progress_curves_q10.append(curve_q10)
                    progress_curves_q25.append(curve_q25)
                    progress_curves_q50.append(curve_q50)
        if progress_tiers:
            result["progress_grid"] = progress_grid
            result["progress_tiers"] = np.asarray(progress_tiers, dtype=float)
            result["progress_q10"] = np.asarray(progress_curves_q10, dtype=float)
            result["progress_q25"] = np.asarray(progress_curves_q25, dtype=float)
            result["progress_q50"] = np.asarray(progress_curves_q50, dtype=float)
            grid_arr = np.asarray(progress_grid, dtype=float)
            result["progress_dq10"] = np.asarray(
                [np.gradient(curve, grid_arr) for curve in progress_curves_q10],
                dtype=float,
            )
            result["progress_dq25"] = np.asarray(
                [np.gradient(curve, grid_arr) for curve in progress_curves_q25],
                dtype=float,
            )
            result["progress_dq50"] = np.asarray(
                [np.gradient(curve, grid_arr) for curve in progress_curves_q50],
                dtype=float,
            )
        return result

    @staticmethod
    def _load_capacity_shape(path: Optional[str]) -> Dict[str, np.ndarray]:
        shape_path = Path(path) if path else DEFAULT_CAPACITY_SHAPE_PATH
        if not shape_path.is_absolute():
            shape_path = PROJECT_ROOT / shape_path
        try:
            raw = json.loads(shape_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}

        shape_raw = raw.get("shape_by_tier_normalized_to_best_available", {}) if isinstance(raw, dict) else {}
        ranks = []
        values = []
        for tier_raw, stats in sorted(
            shape_raw.items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
        ):
            try:
                tier = float(tier_raw)
                value = float((stats or {}).get("q50"))
            except (TypeError, ValueError):
                continue
            if tier > 0 and np.isfinite(value) and value > 0:
                ranks.append(tier)
                values.append(value)

        if not ranks:
            return {}

        ranks_arr = np.asarray(ranks, dtype=float)
        values_arr = np.asarray(values, dtype=float)
        values_arr = np.clip(values_arr, 0.0, 1.0)
        values_arr = GraphStateSpaceEngine._monotone_curve(values_arr)
        return {"rank_grid": ranks_arr, "shape_q50": values_arr}

    @staticmethod
    def _load_behavior_prior(path: Optional[str]) -> dict:
        if path == "":
            return {}
        prior_path = Path(path) if path else DEFAULT_BEHAVIOR_PRIOR_PATH
        if not prior_path.is_absolute():
            prior_path = PROJECT_ROOT / prior_path
        try:
            raw = json.loads(prior_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        names = raw.get("coeff_names") or []
        if list(names) != list(BEHAVIOR_COEFF_NAMES):
            return {}
        return raw

    def _behavior_prior_stats_for_rank(self, rank: Optional[int]) -> Optional[dict]:
        if not self._behavior_prior:
            return None
        if rank is None:
            stats = self._behavior_prior.get("global")
            return stats if isinstance(stats, dict) else None
        by_rank = self._behavior_prior.get("by_rank")
        if not isinstance(by_rank, dict) or not by_rank:
            stats = self._behavior_prior.get("global")
            return stats if isinstance(stats, dict) else None
        if str(int(rank)) in by_rank and isinstance(by_rank[str(int(rank))], dict):
            return by_rank[str(int(rank))]

        ranks = sorted(int(item) for item in by_rank.keys() if str(item).isdigit())
        if not ranks:
            stats = self._behavior_prior.get("global")
            return stats if isinstance(stats, dict) else None
        nearest = min(ranks, key=lambda item: abs(np.log(max(item, 1.0)) - np.log(max(float(rank), 1.0))))
        stats = by_rank.get(str(nearest))
        return stats if isinstance(stats, dict) else None

    def _behavior_prior_coeffs_for_rank(self, rank: Optional[int], fallback: Dict[str, float]) -> Dict[str, float]:
        stats = self._behavior_prior_stats_for_rank(rank)
        if not stats:
            return dict(fallback)
        coeffs = stats.get("coeffs")
        if not isinstance(coeffs, dict):
            return dict(fallback)
        result = dict(fallback)
        for name in BEHAVIOR_COEFF_NAMES:
            if name in coeffs:
                try:
                    result[name] = float(coeffs[name])
                except (TypeError, ValueError):
                    pass
        return result

    def _adapt_speed_profile(self, surface) -> Dict[str, np.ndarray]:
        grid = self._speed_profile.get("rank_grid")
        if grid is None or len(grid) == 0 or surface is None or surface.empty:
            return {}

        adapted = {}
        for curve_name, sampler_name in (
            ("capacity", "max"),
            ("cruise", "median"),
            ("cruise_high", "upper"),
            ("idle", "low"),
        ):
            prior = self._speed_profile.get(curve_name)
            if prior is None or len(prior) != len(grid):
                continue

            anchor_ranks = []
            anchor_ratios = []
            capacity_lower = {}
            for tier in self._observed_tiers:
                speeds = self._norm_speed_samples_at_rank(surface, float(tier))
                if len(speeds) == 0:
                    continue
                observed = self._profile_observed_stat(speeds, sampler_name)
                prior_value = self._speed_profile_value(curve_name, float(tier), use_adapted=False)
                if prior_value is None or prior_value <= 0:
                    continue
                anchor_ranks.append(float(tier))
                if curve_name == "capacity":
                    # Capacity is an ability upper bound. Observed burst speed is
                    # lower-bound evidence only; not seeing max effort early must
                    # not shrink the player group's possible ceiling.
                    anchor_ratios.append(max(max(float(observed), 0.0) / max(float(prior_value), 1e-9), 1.0))
                    capacity_lower[float(tier)] = max(float(observed), 0.0)
                else:
                    anchor_ratios.append(max(float(observed), 0.0) / max(float(prior_value), 1e-9))

            if not anchor_ratios:
                adapted[curve_name] = np.asarray(prior, dtype=float)
                continue

            log_ratios = np.log(np.maximum(np.asarray(anchor_ratios, dtype=float), 1e-6))
            global_ratio = float(np.exp(np.median(log_ratios)))
            grid_arr = np.asarray(grid, dtype=float)
            prior_arr = np.asarray(prior, dtype=float)

            if len(anchor_ranks) >= 2:
                order = np.argsort(anchor_ranks)
                anchor_logs = np.log(np.maximum(np.asarray(anchor_ranks, dtype=float)[order], 1.0))
                ratio_logs = log_ratios[order]
                local_ratio = np.exp(np.interp(
                    np.log(np.maximum(grid_arr, 1.0)),
                    anchor_logs,
                    ratio_logs,
                    left=ratio_logs[0],
                    right=ratio_logs[-1],
                ))
                adjusted = prior_arr * np.sqrt(np.maximum(local_ratio, 1e-6) * max(global_ratio, 1e-6))
            else:
                adjusted = prior_arr * global_ratio

            if curve_name == "capacity" and capacity_lower:
                lower_ranks = np.asarray(sorted(capacity_lower), dtype=float)
                lower_values = np.asarray([capacity_lower[r] for r in lower_ranks], dtype=float)
                lower_curve = np.interp(
                    np.log(np.maximum(grid_arr, 1.0)),
                    np.log(np.maximum(lower_ranks, 1.0)),
                    lower_values,
                    left=lower_values[0],
                    right=lower_values[-1],
                )
                adjusted = np.maximum(adjusted, lower_curve)

            adjusted = self._monotone_curve(np.asarray(adjusted, dtype=float))
            adapted[curve_name] = np.maximum(adjusted, self.config.base_norm_floor)

        return adapted

    def _norm_speed_samples_at_rank(self, surface, rank: float) -> np.ndarray:
        if surface is None or surface.empty or len(surface) < 2:
            return np.asarray([], dtype=float)
        speeds = []
        for i in range(1, len(surface)):
            dt = max(float(surface.index[i] - surface.index[i - 1]), 1e-6)
            ds = self._score_at_rank(surface.iloc[i], rank) - self._score_at_rank(surface.iloc[i - 1], rank)
            norm = max(ds / (dt * 60.0), 0.0) / max(self._scale, 1e-6)
            if np.isfinite(norm):
                speeds.append(norm)
        return np.asarray(speeds, dtype=float)

    @staticmethod
    def _profile_observed_stat(speeds: np.ndarray, sampler_name: str) -> float:
        if len(speeds) == 0:
            return 0.0
        if sampler_name == "max":
            return float(np.max(speeds))
        if sampler_name == "upper":
            return float(np.quantile(speeds, 0.75))
        if sampler_name == "low":
            return float(np.quantile(speeds, 0.10))
        return float(np.quantile(speeds, 0.50))

    @staticmethod
    def _monotone_curve(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values
        out = np.asarray(values, dtype=float).copy()
        for idx in range(len(out) - 2, -1, -1):
            out[idx] = max(out[idx], out[idx + 1])
        return out

    def _speed_profile_value(self, curve_name: str, rank: float, use_adapted: bool = True) -> Optional[float]:
        grid = self._speed_profile.get("rank_grid")
        values = self._speed_profile_adapted.get(curve_name) if use_adapted else None
        if values is None:
            values = self._speed_profile.get(curve_name)
        if grid is None or values is None or len(grid) == 0 or len(values) != len(grid):
            return None
        log_grid = np.log(np.maximum(np.asarray(grid, dtype=float), 1.0))
        log_rank = float(np.log(max(float(rank), 1.0)))
        value = float(np.interp(log_rank, log_grid, np.asarray(values, dtype=float)))
        return value if np.isfinite(value) else None

    def _score_progress_fraction(
        self,
        rank: int,
        event_progress: float,
        risk_level: float = 0.5,
    ) -> Optional[float]:
        cache_key = (int(rank), round(float(event_progress), 4), round(float(risk_level), 3))
        if cache_key in self._progress_fraction_cache:
            return self._progress_fraction_cache[cache_key]
        progress_grid = self._speed_profile.get("progress_grid")
        progress_tiers = self._speed_profile.get("progress_tiers")
        progress_q10 = self._speed_profile.get("progress_q10")
        progress_q25 = self._speed_profile.get("progress_q25")
        progress_q50 = self._speed_profile.get("progress_q50")
        if (
            progress_grid is None
            or progress_tiers is None
            or progress_q50 is None
            or len(progress_grid) == 0
            or len(progress_tiers) == 0
        ):
            self._progress_fraction_cache[cache_key] = None
            return None

        risk_level = float(min(max(risk_level, 0.0), 1.0))
        if progress_q10 is not None and progress_q25 is not None and risk_level > 0.5:
            high = np.asarray(progress_q10, dtype=float)
            mid = np.asarray(progress_q25, dtype=float)
            w = (risk_level - 0.5) / 0.5
            progress_matrix = (1.0 - w) * mid + w * high
        elif progress_q25 is not None:
            mid = np.asarray(progress_q25, dtype=float)
            low = np.asarray(progress_q50, dtype=float)
            w = risk_level / 0.5
            progress_matrix = (1.0 - w) * low + w * mid
        else:
            progress_matrix = np.asarray(progress_q50, dtype=float)

        event_progress = float(min(max(event_progress, 0.0), 1.0))
        tier_values = []
        for curve in np.asarray(progress_matrix, dtype=float):
            tier_values.append(float(np.interp(event_progress, progress_grid, curve)))
        tier_values = np.asarray(tier_values, dtype=float)
        log_tiers = np.log(np.maximum(np.asarray(progress_tiers, dtype=float), 1.0))
        log_rank = float(np.log(max(float(rank), 1.0)))
        value = float(np.interp(log_rank, log_tiers, tier_values, left=tier_values[0], right=tier_values[-1]))
        if not np.isfinite(value) or value <= 0.0:
            self._progress_fraction_cache[cache_key] = None
            return None
        result = float(np.clip(value, 0.02, 1.0))
        self._progress_fraction_cache[cache_key] = result
        return result

    def _target_profile_norm(
        self,
        rank: int,
        hour: float,
        target_score: float,
        risk_level: float = 0.5,
    ) -> Optional[float]:
        progress_grid = self._speed_profile.get("progress_grid")
        progress_tiers = self._speed_profile.get("progress_tiers")
        progress_q10 = self._speed_profile.get("progress_q10")
        progress_q25 = self._speed_profile.get("progress_q25")
        progress_q50 = self._speed_profile.get("progress_q50")
        progress_dq10 = self._speed_profile.get("progress_dq10")
        progress_dq25 = self._speed_profile.get("progress_dq25")
        progress_dq50 = self._speed_profile.get("progress_dq50")
        if (
            progress_grid is None
            or progress_tiers is None
            or progress_q50 is None
            or progress_dq50 is None
            or len(progress_grid) < 2
            or len(progress_tiers) == 0
            or not self._total_hours
            or self._total_hours <= 0
        ):
            return None

        risk_level = float(min(max(risk_level, 0.0), 1.0))
        cache_key = (int(rank), round(float(hour), 4), round(risk_level, 3))
        if cache_key in self._progress_speed_factor_cache:
            factor = self._progress_speed_factor_cache[cache_key]
            if factor is None:
                return None
            return float(max(float(target_score), 0.0) * factor)

        if progress_dq10 is not None and progress_dq25 is not None and risk_level > 0.5:
            high = np.asarray(progress_dq10, dtype=float)
            mid = np.asarray(progress_dq25, dtype=float)
            w = (risk_level - 0.5) / 0.5
            derivative_matrix = (1.0 - w) * mid + w * high
        elif progress_dq25 is not None:
            mid = np.asarray(progress_dq25, dtype=float)
            low = np.asarray(progress_dq50, dtype=float)
            w = risk_level / 0.5
            derivative_matrix = (1.0 - w) * low + w * mid
        else:
            derivative_matrix = np.asarray(progress_dq50, dtype=float)

        event_progress_raw = float(hour) / max(float(self._total_hours), 1e-6)
        event_progress = float(min(max(event_progress_raw, 0.0), 1.0))
        progress_grid_arr = np.asarray(progress_grid, dtype=float)
        tier_factors = []
        season = max(self._season_factor(hour), 0.05)
        denominator = max(float(self._total_hours) * 60.0 * self._scale * season, 1e-6)
        for derivative in np.asarray(derivative_matrix, dtype=float):
            dfrac_dprogress = float(np.interp(event_progress, progress_grid_arr, derivative))
            tier_factors.append(max(dfrac_dprogress, 0.0) / denominator)

        log_tiers = np.log(np.maximum(np.asarray(progress_tiers, dtype=float), 1.0))
        log_rank = float(np.log(max(float(rank), 1.0)))
        factor = float(np.interp(log_rank, log_tiers, np.asarray(tier_factors, dtype=float), left=tier_factors[0], right=tier_factors[-1]))
        if not np.isfinite(factor):
            self._progress_speed_factor_cache[cache_key] = None
            return None
        self._progress_speed_factor_cache[cache_key] = float(max(factor, 0.0))
        return float(max(float(target_score), 0.0) * max(factor, 0.0))

    def _base_norm_upper_for_rank(self, rank: float) -> float:
        cruise = self._speed_profile_value("cruise", rank, use_adapted=False)
        if cruise is not None:
            capacity = self._capacity_norm_for_rank(rank)
            upper = cruise if capacity is None else min(cruise, capacity)
            return float(min(max(upper, self.config.base_norm_floor), 1.0))
        if float(rank) <= 100.0:
            return 1.0
        return 1.0

    def _capacity_norm_for_rank(self, rank: float) -> float:
        shape_value = self._capacity_shape_value(rank)
        if shape_value is not None:
            # Use song-rank shape as the theoretical upper envelope.  Squaring
            # applies the efficiency loss by rank distance without introducing
            # a separate global linear loss coefficient here.
            shape_capacity = shape_value ** 2
            observed_lower = self._speed_profile_value("capacity", rank)
            if observed_lower is not None:
                shape_capacity = max(shape_capacity, float(observed_lower))
            return float(np.clip(shape_capacity, self.config.base_norm_floor, 1.0))

        empirical = self._speed_profile_value("capacity", rank)
        if empirical is not None:
            return float(max(empirical, self.config.base_norm_floor))

        ranks = [int(t) for t in self._emission_tiers if int(t) > 0]
        if not ranks:
            return 1.0

        lo = max(min(ranks), 1)
        hi = max(max(ranks), lo)
        rank = float(np.clip(float(rank), lo, hi))
        if hi <= lo:
            log_pos = 0.0
        else:
            log_pos = np.log(rank / lo) / max(np.log(hi / lo), 1e-6)
        log_pos = float(np.clip(log_pos, 0.0, 1.0))

        capacity = 1.0 - log_pos
        return float(np.clip(capacity, self.config.base_norm_floor, 1.0))

    def _capacity_shape_value(self, rank: float) -> Optional[float]:
        grid = self._capacity_shape.get("rank_grid")
        values = self._capacity_shape.get("shape_q50")
        if grid is None or values is None or len(grid) == 0 or len(grid) != len(values):
            return None

        log_grid = np.log(np.maximum(np.asarray(grid, dtype=float), 1.0))
        log_rank = float(np.log(max(float(rank), 1.0)))
        value = float(np.interp(
            log_rank,
            log_grid,
            np.asarray(values, dtype=float),
            left=float(values[0]),
            right=float(values[-1]),
        ))
        return value if np.isfinite(value) else None

    def _initial_mode_probs(self, speed: float, base_speed: float, rank: float) -> Dict[BehaviorMode, float]:
        mode = BehaviorMode.CHASING if self._nearest_behavior_target(rank)[0] is not None else BehaviorMode.CRUISING
        return {candidate: (1.0 if candidate == mode else 0.0) for candidate in MODE_ORDER}

    def _transition_modes(
        self,
        cell: CohortCell,
        pressure: float,
        urgency: float,
        reachable: float,
        gap_norm: float,
        profile_norm: float,
        follow_norm: float,
        plan_lag_norm: float,
        behavior_lazy: float,
        surplus_norm: float,
        boundary_proximity: float,
    ) -> Dict[BehaviorMode, float]:
        target_open = float(gap_norm > 1e-6)
        capacity_norm = max(float(cell.capacity_norm_speed), self.config.base_norm_floor)
        effort_ratio = max(float(gap_norm), 0.0) / max(capacity_norm, 1e-6)
        if target_open > 0.0:
            if reachable <= 0.0:
                return {mode: (1.0 if mode == BehaviorMode.DROPPED else 0.0) for mode in MODE_ORDER}
            coeffs = self._behavior_coeffs_for_rank(cell.center_rank)
            cruise_norm = float(min(max(cell.base_norm_speed, self.config.base_norm_floor), capacity_norm))
            normal_chase = (
                coeffs.get('cruise', 0.0) * cruise_norm
                + coeffs.get('profile', 0.0) * max(float(profile_norm), 0.0)
                + coeffs.get('follow', 0.0) * max(float(follow_norm), 0.0)
                + coeffs.get('gap', 0.0) * max(float(gap_norm), 0.0)
                + coeffs.get('lag', 0.0) * max(float(plan_lag_norm), 0.0)
            )
            panic_excess = max(max(float(gap_norm), 0.0) - max(normal_chase, 0.0), 0.0)
            panic_raw = max(panic_excess, max(float(plan_lag_norm), 0.0)) / max(capacity_norm, 1e-6)
            panic = float(min(max(panic_raw, 0.0), 1.0))
            return {
                mode: (
                    1.0 - panic if mode == BehaviorMode.CHASING
                    else panic if mode == BehaviorMode.PANIC_RUSHING
                    else 0.0
                )
                for mode in MODE_ORDER
            }

        defend = float(min(max(boundary_proximity, 0.0), 1.0))
        return {
            mode: (
                defend if mode == BehaviorMode.DEFENDING
                else 1.0 - defend if mode == BehaviorMode.CRUISING
                else 0.0
            )
            for mode in MODE_ORDER
        }

    def _target_pressure_state(
        self,
        cell: CohortCell,
        cells: List[CohortCell],
        hour: float,
        remaining_hours: float,
    ) -> Dict[str, float]:
        target_probs = self._target_distribution_for_rank(cell.center_rank)
        if not target_probs:
            return {
                'target_rank': None,
                'target_score': float(cell.score),
                'pressure': 0.0,
                'affinity': 0.0,
                'reachable': 1.0,
                'gap_norm': 0.0,
                'profile_norm': 0.0,
                'follow_norm': 0.0,
                'plan_lag_norm': 0.0,
                'surplus_norm': 0.0,
                'follow_rank': None,
                'boundary_proximity': 0.0,
                'risk_level': 0.5,
                'target_probs': {},
            }

        weighted_pressure = 0.0
        weighted_affinity = 0.0
        weighted_reachable = 0.0
        weighted_gap = 0.0
        weighted_profile = 0.0
        weighted_follow = 0.0
        weighted_plan_lag = 0.0
        weighted_surplus = 0.0
        weighted_boundary = 0.0
        weighted_risk = 0.0
        target_scores = {}
        for reward, prob in target_probs.items():
            importance = self._target_importance_norm(reward)
            rank_log_dist = abs(np.log(max(cell.center_rank, 1.0) / max(reward, 1.0)))
            affinity = importance / (1.0 + rank_log_dist)
            risk_level = self._target_risk_level(cell.center_rank, reward)
            component_key = (
                id(cells),
                int(reward),
                round(float(hour), 6),
                round(float(remaining_hours), 6),
                round(float(risk_level), 3),
            )
            component = self._target_component_cache.get(component_key)
            if component is None:
                target_score = self._projected_target_score(reward, hour, remaining_hours, cells, risk_level)
                profile_norm = self._target_profile_norm(reward, hour, target_score, risk_level)
                component = (
                    float(target_score),
                    float(profile_norm) if profile_norm is not None else float("nan"),
                )
                self._target_component_cache[component_key] = component
            target_score, profile_norm = component
            target_scores[int(reward)] = float(target_score)
            season_hours = remaining_hours * self._remaining_season_mean(hour, remaining_hours)
            signed_required_norm = (target_score - cell.score) / max(season_hours * 60.0 * self._scale, 1e-6)
            required_norm = max(signed_required_norm, 0.0)
            if not np.isfinite(profile_norm):
                profile_norm = required_norm
            follow_norm = self._follow_line_norm(reward, hour, cells)
            progress = float(hour) / max(float(self._total_hours or hour or 1.0), 1e-6)
            plan_fraction = self._score_progress_fraction(int(reward), progress, risk_level)
            plan_score = float(target_score) * float(plan_fraction) if plan_fraction is not None else float(cell.score)
            plan_lag_norm = max(plan_score - cell.score, 0.0) / max(remaining_hours * 60.0 * self._scale, 1e-6)
            surplus_norm = max(-signed_required_norm, 0.0)
            capacity_norm = max(float(cell.capacity_norm_speed), self.config.base_norm_floor)
            effort_ratio = required_norm / max(capacity_norm, 1e-6)
            reachable = float(effort_ratio <= 1.0)
            pressure = affinity * effort_ratio
            boundary = np.exp(
                -rank_log_dist / max(self.config.reward_anchor_radius, 1e-6)
            )

            weighted_pressure += prob * pressure
            weighted_affinity += prob * affinity
            weighted_reachable += prob * reachable
            weighted_gap += prob * required_norm
            weighted_profile += prob * profile_norm
            weighted_follow += prob * follow_norm
            weighted_plan_lag += prob * plan_lag_norm
            weighted_surplus += prob * surplus_norm
            weighted_boundary += prob * boundary
            weighted_risk += prob * risk_level

        dominant_target = max(target_probs.items(), key=lambda item: item[1])[0]
        return {
            'target_rank': int(dominant_target),
            'target_score': float(target_scores.get(int(dominant_target), cell.score)),
            'pressure': float(weighted_pressure),
            'affinity': float(weighted_affinity),
            'reachable': float(weighted_reachable),
            'gap_norm': float(weighted_gap),
            'profile_norm': float(weighted_profile),
            'follow_norm': float(weighted_follow),
            'plan_lag_norm': float(weighted_plan_lag),
            'surplus_norm': float(weighted_surplus),
            'follow_rank': self._guide_rank_for_target(int(dominant_target)),
            'boundary_proximity': float(min(max(weighted_boundary, 0.0), 1.0)),
            'risk_level': float(min(max(weighted_risk, 0.0), 1.0)),
            'target_probs': {int(k): float(v) for k, v in target_probs.items()},
        }

    def _compose_pressure(
        self,
        target_pressure: float,
        neighbor_pressure: float,
        density_pressure: float,
        target_gap_norm: float,
        surplus_norm: float,
        boundary_proximity: float,
    ) -> Tuple[float, str]:
        if target_gap_norm > 1e-6:
            return float(target_pressure + neighbor_pressure), "target"
        if boundary_proximity > 0.0 and neighbor_pressure + density_pressure > 0.0:
            return float(boundary_proximity * (neighbor_pressure + density_pressure)), "defend"
        return 0.0, "satisfied"

    def _strategic_target_for_rank(self, rank: float) -> Optional[int]:
        """Map a player cohort to the reward line they are likely thinking about."""
        probs = self._target_distribution_for_rank(rank)
        if not probs:
            return None
        return int(max(probs.items(), key=lambda item: item[1])[0])

    def _target_risk_level(self, rank: float, target: int) -> float:
        if self._surface is None or self._surface.empty or not self._total_hours:
            return 0.0
        target = int(target)
        observed_hour = float(self._surface.index[-1])
        cache_key = (target, round(observed_hour, 4))
        if cache_key in self._target_risk_cache:
            return float(self._target_risk_cache[cache_key])

        progress = observed_hour / max(float(self._total_hours), 1e-6)
        if progress <= 0.0:
            self._target_risk_cache[cache_key] = 0.0
            return 0.0

        current_score = self._observed_or_interpolated_score(self._surface.iloc[-1], target)
        hours = np.asarray(self._surface.index, dtype=float)
        start_hour = max(observed_hour - 24.0, float(hours[0]))
        start_candidates = np.where(hours >= start_hour - 1e-6)[0]
        start_pos = int(start_candidates[0]) if len(start_candidates) else 0
        if start_pos >= len(self._surface) - 1 and len(self._surface) > 1:
            start_pos = len(self._surface) - 2
        start_score = self._observed_or_interpolated_score(self._surface.iloc[start_pos], target)
        dt_hours = max(observed_hour - float(hours[start_pos]), 1e-6)
        recent_speed = max((current_score - start_score) / (dt_hours * 60.0), 0.0)
        remaining = max(float(self._total_hours) - observed_hour, 0.0)
        observed_projection = current_score + recent_speed * remaining * 60.0

        anchors = []
        for risk in (0.0, 0.5, 1.0):
            fraction = self._score_progress_fraction(target, progress, risk)
            if fraction is None or fraction <= 0.0:
                continue
            projection = current_score / max(float(fraction), 1e-6)
            if np.isfinite(projection):
                anchors.append((float(projection), float(risk)))
        if len(anchors) < 2:
            self._target_risk_cache[cache_key] = 0.0
            return 0.0

        anchors.sort(key=lambda item: item[0])
        projections = np.asarray([item[0] for item in anchors], dtype=float)
        risks = np.asarray([item[1] for item in anchors], dtype=float)
        # Historical progress curves can be close; collapse duplicate projection
        # anchors so interpolation remains stable.
        unique_projection = []
        unique_risk = []
        for projection, risk_value in zip(projections, risks):
            if unique_projection and abs(projection - unique_projection[-1]) <= max(abs(projection), 1.0) * 1e-6:
                unique_risk[-1] = max(unique_risk[-1], float(risk_value))
            else:
                unique_projection.append(float(projection))
                unique_risk.append(float(risk_value))
        if len(unique_projection) < 2:
            result = 0.0
        else:
            result = float(np.interp(
                observed_projection,
                np.asarray(unique_projection, dtype=float),
                np.asarray(unique_risk, dtype=float),
                left=unique_risk[0],
                right=unique_risk[-1],
            ))
        result = float(min(max(result, 0.0), 1.0))
        self._target_risk_cache[cache_key] = result
        return result

    def _target_distribution_for_rank(self, rank: float) -> Dict[int, float]:
        """Probabilistic target assignment around posterior-important rank lines."""
        if not self._behavior_target_tiers:
            return {}

        rank = float(max(rank, 1.0))
        cache_key = round(rank, 6)
        if cache_key in self._target_distribution_cache:
            return dict(self._target_distribution_cache[cache_key])
        targets = sorted({int(t) for t in self._behavior_target_tiers if int(t) > 0})

        candidate_weights = {}
        better_targets = [t for t in targets if t < rank]
        worse_targets = [t for t in targets if t >= rank]
        candidate_targets = []
        candidate_targets.extend(better_targets[-3:])
        candidate_targets.extend(worse_targets[:3])
        if targets:
            candidate_targets.append(min(targets, key=lambda t: abs(np.log(rank / max(float(t), 1.0)))))

        for target in sorted(set(candidate_targets)):
            log_dist = abs(np.log(rank / max(float(target), 1.0)))
            importance = self._target_importance_norm(target)
            structural_weight = self._target_structural_weight(target)
            candidate_weights[int(target)] = float(structural_weight / max(log_dist, 1e-6))

        total = sum(candidate_weights.values())
        if total <= 0:
            return {}
        result = {target: weight / total for target, weight in candidate_weights.items() if weight > 0}
        self._target_distribution_cache[cache_key] = dict(result)
        return result

    def _target_structural_weight(self, target: int) -> float:
        target = int(target)
        posterior = self._target_importance_norm(target)
        return float(np.clip(posterior, 0.0, 1.0))


    def _projected_target_score(
        self,
        target_rank: int,
        hour: float,
        remaining_hours: float,
        cells: List[CohortCell],
        risk_level: float = 0.5,
    ) -> float:
        """Player-facing final-line estimate using recent observed trend as fallback."""
        target_rank = int(target_rank)
        cache_key = (
            id(cells),
            target_rank,
            round(float(hour), 6),
            round(float(remaining_hours), 6),
            round(float(risk_level), 3),
        )
        if cache_key in self._projection_cache:
            return self._projection_cache[cache_key]

        model_score = self._score_from_cells(cells, target_rank) if cells else 0.0
        model_cell = self._cell_for_rank(cells, target_rank) if cells else None
        model_speed = max(float(model_cell.speed), 0.0) if model_cell is not None else 0.0

        projected = model_score + model_speed * max(remaining_hours, 0.0) * 60.0
        if self._surface is None or self._surface.empty:
            self._projection_cache[cache_key] = float(projected)
            return float(projected)

        hours = np.asarray(self._surface.index, dtype=float)
        observed_horizon = float(hours[-1]) if len(hours) else float(hour)
        if float(hour) > observed_horizon + 1e-6:
            # During rollout the relevant target line is the target rank's
            # current simulated line plus the last known progress-based final
            # floor.  Re-dividing the simulated score by the current progress
            # fraction every hour creates an artificial positive feedback loop.
            prior_projection_floor = 0.0
            if self._total_hours and self._total_hours > 0 and len(hours):
                horizon_progress = observed_horizon / max(float(self._total_hours), 1e-6)
                horizon_fraction = self._score_progress_fraction(target_rank, horizon_progress, risk_level)
                if horizon_fraction is not None:
                    horizon_score = self._observed_or_interpolated_score(self._surface.iloc[-1], target_rank)
                    prior_projection_floor = float(horizon_score / max(horizon_fraction, 1e-6))

            progress = float(hour) / max(float(self._total_hours or hour or 1.0), 1e-6)
            prior_projection = prior_projection_floor
            prior_weight = self._target_prior_weight(target_rank, progress)
            if prior_projection > 0.0:
                result = float(max(model_score, (1.0 - prior_weight) * projected + prior_weight * prior_projection))
            else:
                result = float(max(model_score, projected))
            self._projection_cache[cache_key] = result
            return result

        usable = np.where(hours <= float(hour) + 1e-6)[0]
        if len(usable) == 0:
            self._projection_cache[cache_key] = float(projected)
            return float(projected)

        end_pos = int(usable[-1])
        start_hour = float(hours[end_pos]) - 24.0
        start_candidates = np.where(hours[:end_pos + 1] >= start_hour)[0]
        start_pos = int(start_candidates[0]) if len(start_candidates) else 0
        if start_pos == end_pos and end_pos > 0:
            start_pos = end_pos - 1

        current_score = self._observed_or_interpolated_score(self._surface.iloc[end_pos], target_rank)
        if start_pos < end_pos:
            previous_score = self._observed_or_interpolated_score(self._surface.iloc[start_pos], target_rank)
            dt_hours = max(float(hours[end_pos] - hours[start_pos]), 1e-6)
            observed_speed = max((current_score - previous_score) / (dt_hours * 60.0), 0.0)
        else:
            observed_speed = model_speed

        observed_projection = current_score + observed_speed * max(remaining_hours, 0.0) * 60.0
        profile_projection = 0.0
        if self._total_hours and self._total_hours > 0:
            progress = float(hours[end_pos]) / max(float(self._total_hours), 1e-6)
            progress_fraction = self._score_progress_fraction(target_rank, progress, risk_level)
            if progress_fraction is not None:
                profile_projection = float(current_score / max(progress_fraction, 1e-6))

        current_projection = max(projected, observed_projection)
        progress = float(hours[end_pos]) / max(float(self._total_hours or hours[end_pos] or 1.0), 1e-6)
        prior_weight = self._target_prior_weight(target_rank, progress)
        if profile_projection > 0.0:
            result = float(max(model_score, (1.0 - prior_weight) * current_projection + prior_weight * profile_projection))
        else:
            result = float(max(model_score, current_projection))
        self._projection_cache[cache_key] = result
        return result

    def _target_prior_weight(self, target_rank: int, progress: float) -> float:
        return float(np.clip(1.0 - float(progress), 0.0, 1.0))

    def _target_importance_norm(self, target_rank: int) -> float:
        target_rank = int(target_rank)
        if target_rank in self._target_importance_norm_cache:
            return self._target_importance_norm_cache[target_rank]
        if target_rank in self._target_importance_by_tier:
            value = float(np.clip(self._target_importance_by_tier[target_rank], 0.0, 1.0))
            self._target_importance_norm_cache[target_rank] = value
            return value
        if not self._target_importance_by_tier:
            return 0.0
        ranks = np.asarray(sorted(self._target_importance_by_tier), dtype=float)
        values = np.asarray([self._target_importance_by_tier[int(rank)] for rank in ranks], dtype=float)
        value = float(np.clip(np.interp(np.log(max(target_rank, 1)), np.log(np.maximum(ranks, 1.0)), values), 0.0, 1.0))
        self._target_importance_norm_cache[target_rank] = value
        return value

    def _neighbor_pressure(self, cells: List[CohortCell], idx: int, remaining_hours: float) -> float:
        pressure = 0.0
        cell = cells[idx]
        for ni in (idx - 1, idx + 1):
            if ni < 0 or ni >= len(cells):
                continue
            other = cells[ni]
            score_gap = abs(other.score - cell.score)
            required_norm = score_gap / max(remaining_hours * 60.0 * self._scale, 1e-6)
            speed_threat = max(other.speed - cell.speed, 0.0) / max(self._scale, 1e-6)
            chasing_threat = 0.0
            future_threat = 0.0
            if ni > idx and other.target_rank is not None:
                target_between = cell.center_rank <= float(other.target_rank) <= other.center_rank
                if target_between or other.pressure_source == "target":
                    chasing_threat = max(other.pressure, 0.0)

                cell_future = max(
                    cell.score + max(cell.speed, 0.0) * remaining_hours * 60.0,
                    cell.target_score if cell.target_rank is not None else 0.0,
                )
                other_future = max(
                    other.score + max(other.speed, 0.0) * remaining_hours * 60.0,
                    other.target_score if other.target_rank is not None else 0.0,
                )
                future_gap_norm = (cell_future - other_future) / max(remaining_hours * 60.0 * self._scale, 1e-6)
                future_compact = 1.0 / (1.0 + max(future_gap_norm, 0.0))
                shared_target = 0.0
                if cell.target_probs and other.target_probs:
                    shared_target = sum(
                        min(float(cell.target_probs.get(target, 0.0)), float(other.target_probs.get(target, 0.0)))
                        for target in set(cell.target_probs) | set(other.target_probs)
                    )
                future_threat = future_compact * max(speed_threat, chasing_threat, shared_target)

            pressure += max(speed_threat + chasing_threat / (1.0 + required_norm), future_threat)
        return float(pressure)

    def _local_density(self, cells: List[CohortCell], idx: int, remaining_hours: float) -> float:
        if idx <= 0 or idx >= len(cells) - 1:
            return 0.0
        upper = cells[idx - 1]
        lower = cells[idx + 1]
        gap = max(upper.score - lower.score, 0.0)
        norm_gap = gap / max(remaining_hours * 60.0 * self._scale, 1e-6)
        return float(1.0 / (1.0 + norm_gap))

    def _nearest_reward(self, rank: float) -> Tuple[Optional[int], float]:
        if not self._reward_tiers:
            return None, float("inf")
        reward = min(self._reward_tiers, key=lambda r: abs(np.log(max(rank, 1.0) / max(r, 1.0))))
        return int(reward), abs(float(rank) - float(reward))

    def _nearest_behavior_target(self, rank: float) -> Tuple[Optional[int], float]:
        if not self._behavior_target_tiers:
            return None, float("inf")
        target = min(self._behavior_target_tiers, key=lambda r: abs(np.log(max(rank, 1.0) / max(r, 1.0))))
        return int(target), abs(float(rank) - float(target))

    def _score_from_cells(self, cells: List[CohortCell], rank: int) -> float:
        if not cells:
            return 0.0
        cache_key = id(cells)
        cached = self._cell_array_cache.get(cache_key)
        if cached is None:
            centers = np.asarray([c.center_rank for c in cells], dtype=float)
            scores = np.asarray([c.score for c in cells], dtype=float)
            self._cell_array_cache[cache_key] = (centers, scores)
        else:
            centers, scores = cached
        if rank <= centers[0]:
            return float(scores[0])
        if rank >= centers[-1]:
            return float(scores[-1])
        return float(np.interp(rank, centers, scores))

    @staticmethod
    def _cell_for_rank(cells: List[CohortCell], rank: int) -> Optional[CohortCell]:
        for cell in cells:
            if cell.rank_start <= rank <= cell.rank_end:
                return cell
        if not cells:
            return None
        return min(cells, key=lambda c: abs(c.center_rank - rank))

    def _urgency(self, remaining_hours: float) -> float:
        return 1.0

    def _enforce_rank_monotonicity(self, cells: List[CohortCell]) -> List[CohortCell]:
        if not cells:
            return cells
        adjusted = []
        previous_score = float("inf")
        for cell in cells:
            clone = CohortCell(**cell.__dict__)
            if clone.score > previous_score:
                clone.score = previous_score
            previous_score = clone.score
            adjusted.append(clone)
        return adjusted

    def _season_factor(self, hour: float) -> float:
        ts_ms = self._start_ts + int(float(hour) * 3600000)
        try:
            return max(float(self.seasonality.get_factor(ts_ms)), 0.0)
        except Exception:
            return 1.0

    def _remaining_season_mean(self, hour: float, remaining_hours: float) -> float:
        remaining_hours = max(float(remaining_hours), self.config.dt_hours)
        cache_key = (round(float(hour), 3), round(float(remaining_hours), 3))
        if cache_key in self._season_mean_cache:
            return self._season_mean_cache[cache_key]
        step = max(min(self.config.dt_hours, 3.0), 1.0)
        sample_count = int(np.clip(np.ceil(remaining_hours / step), 1, 32))
        samples = [
            self._season_factor(float(hour) + (idx + 0.5) * remaining_hours / sample_count)
            for idx in range(sample_count)
        ]
        if not samples:
            return 1.0
        value = float(max(np.mean(samples), 0.05))
        self._season_mean_cache[cache_key] = value
        return value

    @staticmethod
    def _total_hours_from_meta(meta: dict) -> Optional[float]:
        try:
            start_at = float(meta.get('start_at', 0) or 0)
            end_at = float(meta.get('end_at', 0) or 0)
            if end_at > start_at:
                return (end_at - start_at) / 3600000.0
        except Exception:
            pass
        return None
