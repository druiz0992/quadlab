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

**Status:** the backend core and the joint-level controller contract are
implemented. Pollen's policies, more sensors, and an RL environment are
planned ([6 Roadmap](6_roadmap.md#bipeds)).

## Running

```bash
pip install -e '.[dev,mujoco]'
robotlab --robot biped --duration 10 --output outputs/biped        # built-in test biped
robotlab --robot biped --generate-only --output outputs/biped-scene # writes scene.xml
# Pollen Robotics' Microduck: clone microduck_rl beside this repository
git clone https://github.com/pollen-robotics/microduck_rl ../microduck_rl
robotlab --config configs/bipeds/microduck.json --duration 10 --output outputs/duck
```

The Microduck model is not copied here: its meshes are CC BY-NC-SA 4.0
(the MJCF is Apache-2.0), so `biped.model_path` points at your checkout and
`summary.json.robot.model` records its path, commit, and local changes.
Holding the STAND pose without a balance policy, the Microduck tips over in
about a second (verdict `failed`, `fallen`); the built-in test biped
(`builtin:test_biped`, wide feet, statically stable) stands. Gazebo-only
options (`--world`, `--edit`, `--ekf`, `--keyboard`, `--rerun`,
`--render-sensors`, `--scene`, `--target`, `--control-mode`) are rejected.

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
`"sensors"` leaves it out. `request` (`WalkCommand`) is None until the
keyboard arrives. Targets outside the bounds are clipped and counted per
joint (`summary.json.control.limited_commands`); a wrong length or
non-finite target fails the run.

| `algorithms.controller` | Behaviour |
| --- | --- |
| empty or `hold` | hold the start keyframe's targets |
| `template` (`control/legged_template.py`) | hold plus forward-lean feedback on the joints in `lean_gains` (rad/rad; not a balance controller) |
| `package.module:Class` | your controller; test it with `robotlab.testing.plugins.check_legged_controller` |

## Outputs

| File | Contents |
| --- | --- |
| `summary.json` | `completed`, `health` (`fallen`, `non_finite`, `simulator_warning`, `incomplete`, `interrupted`), `robot` (model provenance, joint names), `control` (controller, state source, ticks, limited commands, its `summary()`), `config`, `timing` (real-time factor), `sensors`, `bus` |
| `telemetry.jsonl` | trunk truth (position, quaternion xyzw, world velocity, body rates) and joint angles/velocities at `record_rate_hz`, plus the initial and final states |
| `commands.jsonl` | per control period: applied `joint_target_rad`, `requested_rad`, `limited` joint indices |
| `scene.xml` | `--generate-only`: the edited model, loadable on its own |

The `joint_encoders` bus stream comes only from MuJoCo runs; the synthetic
estimator scenarios have no joints. Tests: `tests/unit/*/test_*biped*`,
`test_mujoco_*`, and the opt-in `tests/integration/test_microduck.py`
(`ROBOTLAB_MICRODUCK_MODEL=<microduck_rl>/src/mjlab_microduck/robot/microduck/scene.xml`).
