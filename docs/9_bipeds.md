# 9. Bipeds in MuJoCo

Bipeds are the only robots simulated in MuJoCo; multirotors and wheeled
robots stay in Gazebo, so no vehicle is modelled twice. A biped run shares
the sensor bus, the IMU error model, the health-verdict shape, and the
output conventions with the Gazebo robots, but has its own settings
(`simulation/mujoco/config.py`) and its own step-based loop
(`experiments/biped_loop.py`). Contact, friction, and servo behaviour come
from MuJoCo and the model file: results are not comparable with Gazebo
physics, and the servos are MuJoCo position actuators (an approximation of
the real ones, not a calibrated model).

**Status:** the backend core, the joint-level controller contract, the
ONNX policy runner (Pollen's Microduck policies), keyboard walking, and a
Rerun view are implemented. More sensors and an RL environment are planned
([6 Roadmap](6_roadmap.md#bipeds)).

## Running

```bash
pip install -e '.[dev,mujoco]'
robotlab --robot biped --duration 10 --output outputs/biped        # built-in test biped
robotlab --robot biped --generate-only --output outputs/biped-scene # writes scene.xml
# Pollen Robotics' Microduck: clone microduck_rl beside this repository
git clone https://github.com/pollen-robotics/microduck_rl ../microduck_rl
robotlab --config configs/bipeds/microduck.json --duration 10 --output outputs/duck
# Walk it with Pollen's velstand policy (Apache-2.0) and the keyboard
pip install -e '.[policies,gazebo]'   # onnxruntime; Rerun
curl -L --create-dirs -o ../microduck-policies/velstand.onnx \
  https://huggingface.co/pollen-robotics/microduck-policies/resolve/main/velstand.onnx
robotlab --config configs/bipeds/microduck_walk.json --keyboard --output outputs/duck-walk
```

The Microduck model is not copied here: its meshes are CC BY-NC-SA 4.0
(the MJCF is Apache-2.0), so `biped.model_path` points at your checkout and
`summary.json.robot.model` records its path, commit, and local changes.
Holding the STAND pose without a balance policy, the Microduck tips over in
about a second (verdict `failed`, `fallen`); the built-in test biped
(`builtin:test_biped`, wide feet, statically stable) stands. Gazebo-only
options (`--world`, `--edit`, `--ekf`, `--render-sensors`, `--scene`,
`--target`, `--control-mode`) are rejected.

`--keyboard` opens a key window (↑/↓ walk, ←/→ turn, A/D step sideways,
Esc ends; maxima and ramp in `biped.keyboard`) and paces the run to real
time; `--keyboard` or `--rerun` shows the robot's bodies and trunk/command
plots in Rerun and records `biped.rrd`. MuJoCo's own viewer is not used:
its passive mode crashed or hung on shutdown in about half of the tries
here (MuJoCo 3.17, X11).

## Model, settings, and timing

| Key | Meaning |
| --- | --- |
| `biped.model_path` | MJCF file, or `builtin:test_biped` |
| `biped.initial_keyframe`, `biped.imu_site` | start keyframe (qpos, ctrl) and the IMU site |
| `biped.add_floor` | add a z = 0 plane when the world has none |
| `biped.step_s`, `control_rate_hz`, `record_rate_hz` | physics step (s) and task rates (Hz) |
| `sensors.imu` | `enabled`, `rate_hz`, `error_model` (as the drone IMU) |
| `sensors.joint_encoders` | `enabled`, `rate_hz`, `counts_per_revolution`, `velocity_resolution_rad_s`, `position_noise_std_rad` (per sample), `seed` |
| `sensor_timing` | `imu`, `joint_encoders` latency/jitter/offset, `reorder_window_s`, `seed` |
| `biped.keyboard` | `max_forward_mps`, `max_left_mps`, `max_yaw_rate_rad_s`, `ramp_per_s` |
| `algorithms` | `controller`, `controller_options`, `control_state_source` (`truth` or `sensors`) |
| `health` | `fall_tilt_deg`, `fall_height_fraction` |

The model needs one free joint (the trunk), the IMU site, the keyframe, and
only position actuators on hinge or slide joints. RobotLab sets the step
and gravity (0, 0, -9.81) m/s², reporting the model's own values in
`summary.json.robot.model.overrides`. Every rate must be a whole number of
physics steps (default 2 ms: control 50 Hz, IMU 250 Hz, encoders 50 Hz), so
timestamps are exact integer nanoseconds. Joint encoders are read before
the control update of their step; an IMU sample at time t depends on the
command applied at t, so it reaches the bus one step later.

## Controllers

A controller (`control/legged.py`) is built as `cls(context, options)`;
`context.legged` holds the joint names (MJCF actuator order), target
bounds, servo gains, initial targets, and control rate, and
`context.sensors` the IMU mount. Every control period the mission calls
`update(state, request, dt_s)` and applies the returned `JointTargets`
(rad per joint). `state` has the newest `joint_encoders` and `imu` bus
samples and, with `algorithms.control_state_source: "truth"` (default), the
trunk's true pose (`trunk`, a stand-in for an onboard attitude filter);
`"sensors"` leaves it out. `request` (`WalkCommand`) comes from the
keyboard, else None. Targets outside the bounds are clipped and counted per
joint (`summary.json.control.limited_commands`); a wrong length or
non-finite target fails the run.

| `algorithms.controller` | Behaviour |
| --- | --- |
| empty or `hold` | hold the start keyframe's targets |
| `template` (`control/legged_template.py`) | hold plus forward-lean feedback on the joints in `lean_gains` (rad/rad; not a balance controller) |
| `onnx_policy` | an exported RL policy (see below) |
| `package.module:Class` | your controller; test it with `robotlab.testing.plugins.check_legged_controller` |

## Pollen's policies (`onnx_policy`)

`algorithms.controller: onnx_policy` (`control/onnx_policy.py`) runs an
exported Microduck policy at 50 Hz: observation (61 for 14 joints) = gyro,
projected gravity, joint angles minus the default pose, joint velocities,
previous action, and a 13-value command (walking twist, then head and body
pose held at zero); target = default + scale · action. The model's
metadata must list the robot's joints in the same order and this layout;
its `default_joint_pos` and `action_scale` are used. LSTM exports
(`h_in`/`c_in`) keep their state between periods. Options: `policy_path`,
`action_scale` (override), `twist` and `twist_after_s` (a constant
command when nobody drives).

Verified on 2026-10-25 (microduck_rl `b02fdea`, `velstand.onnx` from the
policy set at `665ae4e`): with trunk truth standing in for the robot's
IMU fusion, the Microduck stands still and, commanded 0.3 m/s, walks at
about 0.1 m/s without falling (0.69 m in 7 s); at 0.15 m/s it keeps
standing. With `control_state_source: "sensors"` gravity comes from the
accelerometer direction, which includes the trunk's accelerations, and it
falls after about 2.7 s. Pollen trains with BAM servo models (back-EMF,
battery-voltage limits, fitted friction); these MuJoCo position actuators
have a fixed torque cap at every speed, the likely reason for the weak
walking. A Python XL330 model is a possible next step.

## Outputs

| File | Contents |
| --- | --- |
| `summary.json` | `completed`, `health` (`fallen`, `non_finite`, `simulator_warning`, `incomplete`, `interrupted`), `robot` (model provenance, joint names), `control` (controller, state source, ticks, limited commands, its `summary()`), `config`, `timing` (real-time factor), `sensors`, `bus` |
| `telemetry.jsonl` | trunk truth (position, quaternion xyzw, world velocity, body rates) and joint angles/velocities at `record_rate_hz`, plus the initial and final states |
| `commands.jsonl` | per control period: applied `joint_target_rad`, `requested_rad`, `limited` joint indices, `walk_command` when there is a request |
| `scene.xml` | `--generate-only`: the edited model, loadable on its own |
| `biped.rrd` | `--keyboard` or `--rerun`: Rerun recording |

The `joint_encoders` bus stream comes only from MuJoCo runs; the synthetic
estimator scenarios have no joints. Tests: `tests/unit/*/test_*biped*`,
`test_mujoco_*`, and the opt-in `tests/integration/test_microduck.py`
(`ROBOTLAB_MICRODUCK_MODEL=<microduck_rl>/src/mjlab_microduck/robot/microduck/scene.xml`;
the walking test also `ROBOTLAB_MICRODUCK_POLICY=<path>/velstand.onnx`);
`ROBOTLAB_TEST_GUI=1` runs the paced keyboard test.
