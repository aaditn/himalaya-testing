# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Vendored from MuJoCo Playground (google-deepmind/mujoco_playground),
# _src/gait.py, Apache-2.0. See LICENSE.playground.
#
# Only get_rz is vendored -- it is the one gait function the joystick task
# calls, and it shapes REWARD BEHAVIOUR rather than plumbing: it defines the
# swing-foot height trajectory that _reward_feet_phase scores against. Leaving
# it in the library would mean the gait shape stays uneditable while every
# other reward term is local. draw_joystick_command (visualization) is not
# vendored.
# ==============================================================================
"""Swing-foot trajectory shaping for the feet-phase reward."""


import jax
import jax.numpy as jp


def get_rz(
    phi: jax.Array | float, swing_height: jax.Array | float = 0.08
) -> jax.Array:
  """Target foot height at gait phase `phi`, in [-pi, pi].

  Two cubic Bezier segments: the foot rises from 0 to swing_height over the
  first half of the cycle and comes back down over the second. The reward term
  penalizes distance from this curve, so raising swing_height asks for a
  higher step and changing the interpolation changes the shape of the stride.
  """

  def cubic_bezier_interpolation(y_start, y_end, x):
    y_diff = y_end - y_start
    bezier = x**3 + 3 * (x**2 * (1 - x))
    return y_start + y_diff * bezier

  x = (phi + jp.pi) / (2 * jp.pi)
  stance = cubic_bezier_interpolation(0, swing_height, 2 * x)
  swing = cubic_bezier_interpolation(swing_height, 0, 2 * x - 1)
  return jp.where(x <= 0.5, stance, swing)
