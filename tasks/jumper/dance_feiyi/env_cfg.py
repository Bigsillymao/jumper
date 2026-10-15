"""Environment config for jumper.dance_feiyi: the folk-dance piece,
`media/feiyi.npz`.

The environment is `tasks/jumper/common/dance/env.py`; this file holds every
number. Where a number follows a rule `jumper.dance` wrote down for its own
clip, the rule is applied to this clip and the result is quoted -- read
`tasks/jumper/dance/env_cfg.py` for the reasoning behind each rule.

## This clip, measured

Authored at 100 BPM, 50 Hz, 35.6 s (`tools/author_feiyi_dance.py` in the
submission repository). The whip slides are the fastest content: a cubic
ease-out carries the body sideways in ~0.3 beat, and the fastest joint reaches
11.9 rad/s against the servo's 30.7 corner. The four support feet stay planted
by per-frame IK (0.16 mm out of plane, no height offset). The tremble section
is a 7 Hz whole-body shiver, which the IK turns into planted feet on a
trembling body.

The reward weights are starting points, as they are for `jumper.dance_waist`:
no policy has been trained on this clip in this repository yet.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..common.dance.env import dance_env_cfg

#: This task's material, written by `tools/author_dazhaxie_dance.py`.
MEDIA = Path(__file__).resolve().parent / "media"

#: Episode length, seconds. The clip is 35.6 s, so an episode is a window into
#: it and the command's adaptive sampler chooses where it starts. 10 s at
#: 100 BPM is about 17 beats, more than two full 闪身步 cycles or a drum phrase.
EPISODE_S = 10.0


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config. See `jumper.dance`'s for the arguments."""
    return dance_env_cfg(
        media=MEDIA,
        asset=asset,
        play=play,
        episode_s=EPISODE_S,
        # ── Reference-state initialisation ────────────────────────────────
        # jumper.dance's: fractions of this robot, not of the clip, so they
        # carry over unchanged.
        rsi_pose_range={
            "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005),
            "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05),
        },
        rsi_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
        rsi_joint_range=(-0.05, 0.05),
        # ── Observation noise ─────────────────────────────────────────────
        # The tracking skeleton's, as jumper.dance has them; not measured.
        joint_pos_noise=0.01,
        joint_vel_noise=0.5,
        # ── Rewards ───────────────────────────────────────────────────────
        # Positive budget 6.5 a step, as jumper.dance: mjlab's six tracking
        # terms unchanged (5.0) plus the two joint-space terms below (1.5).
        #
        # Joint position: `jumper.dance`'s rule is a std at 0.44 of the clip's
        # RMS amplitude. This clip's amplitude is ~0.35 rad at the arms and
        # ~0.02 rad at the planted legs (it is arm-driven), so the second term
        # is widened from the waist clip's 0.10 to 0.16 to keep "stand still
        # through the clip" at the same ~0.006-0.008 it costs there.
        #
        # 0.16 -> 0.35 after a measured plateau (2026-10-13, iteration 3649):
        # the both-claws-overhead phrases sat at ||q_err|| ~= 1.4 rad for the
        # last ~1500 iterations while everything else tracked, i.e. the policy
        # had not found that region of the action space (reach = ~11 action
        # units per arm at scale 0.25) and the term's gradient at err 1.4 is
        # exp(-8.75) * 8.75 -- effectively zero, so training could not pull it
        # in either. At std 0.35 the same error's gradient is ~13x larger
        # (exp(-4) * 4), trading some precision at small errors for reach at
        # large ones; the near-zero-error terms (vel, smoothness) are
        # untouched. The standing-still cost rises from 0.008 to 0.03 of the
        # budget, which is the trade taken here on purpose.
        joint_pos_std=0.35,
        joint_pos_weight=1.0,
        # Joint velocity: two thirds of the clip's velocity RMS. The clip's
        # peak is 5.1 rad/s and its RMS is far below the recorded dances'
        # 1.5 rad/s; 1.0 keeps the term meaningful against the smoother
        # reference, and half the position weight, as jumper.dance.
        joint_vel_std=1.0,
        joint_vel_weight=0.5,
        # Smoothness: -0.1 is calibrated at this action scale (0.25) and the
        # second difference at half of it -- properties of the robot and the
        # action space, not of the clip.
        action_rate_weight=-0.1,
        action_acc_weight=-0.05,
        # Power: 22 joints at ~0.25 N*m and this clip's sub-1 rad/s is under
        # 6 W, so -0.02 is about -0.12 a step, under 2% of the budget.
        power_weight=-0.02,
        # Torque above the continuous rating: anchored to the servo, not the
        # clip -- one joint at the plateau costs 0.6, 9% of the budget.
        torque_headroom_weight=-2.0,
        # Joint limits. The reference stays inside them (worst slack 80 mrad),
        # so this term charges a faithful policy nothing.
        joint_limit_weight=-10.0,
        # Leg-on-leg contact above 1.0 N, ~8% of the robot's worst-case push.
        self_collision_weight=-1.0,
        self_collision_force=1.0,
        # ── Terminations ──────────────────────────────────────────────────
        # Root height: this clip's base stays between STAND_Z and STAND_Z+15 mm
        # (bounce and bow), so 40 mm exceeds all of it and cannot fire on a
        # policy merely tracking badly -- jumper.dance's rule, and its number.
        anchor_height_error=0.04,
        # 1 - cos(tilt) = 0.3 is 45.6 degrees, "has fallen over". This dance
        # tilts at most ~13 degrees (yaw sway with the roll lean on top).
        anchor_tilt_error=0.3,
        # A support foot 50 mm out vertically is half the standing height.
        support_foot_error=0.05,
        # ── Disturbance ───────────────────────────────────────────────────
        # jumper.dance's: gentler and rarer than mjlab's, sized to a 2 kg robot.
        push_interval_s=(3.0, 8.0),
        push_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
    )
