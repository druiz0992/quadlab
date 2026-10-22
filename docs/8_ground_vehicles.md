# 8. Ground vehicles: rover, car, tractor

RobotLab simulates three wheeled robots next to the multirotors. They share
the Gazebo runtime, lockstep stepping, sensors and their error models, the
sensor bus and estimator plug-ins, evaluation, and the health verdict
([2 Simulation](2_simulation.md), [7 Experiments](7_experiments.md)). This
document covers what is specific to wheels. The vehicles are textbook-plausible
models with constant coefficients, not calibrated machines.

| `--robot` | Vehicle | Steering | Drive |
| --- | --- | --- | --- |
| `rover` | 50 kg research rover, 0.9 × 0.5 m body | skid (left/right torque) | 4 wheels, 30 N·m, 150 W each, ~1.1 m/s |
| `car` | 700 kg single-seater race car, slicks, wings | Ackermann (front) | rear wheels, 105 kW each, drag-limited ~72 m/s |
| `tractor` | 3500 kg electric tractor modelled on published Voltrac Thor figures | four-wheel (front, coordinated, crab) | 4 wheel motors, 28 kW each, 40 km/h |

The tractor uses the published mass, length, wheelbase, width, ground
clearance, tyre size (380/85R28 → 0.68 m radius), motor power, and top
speed; the track width (1.32 m), centre of mass, inertia, torque, and the
tyre and suspension constants are assumptions.

## Running

```bash
# Drive with the keyboard (Rerun opens; arrows drive, Esc ends)
robotlab --robot car --keyboard --headless --scene worlds/airfield.json --output outputs/car
robotlab --robot rover --keyboard --headless --scene worlds/hills.json --output outputs/rover
robotlab --robot tractor --keyboard --headless --output outputs/tractor

# Your controller (see "Controllers"); here the template at a 10 m/s cruise
robotlab --robot car --headless --config my_controller.json --duration 20 --output outputs/car-auto
```

`--robot` loads the vehicle's preset (`simulation/wheeled/config.py`) under
all `--config` layers; a layer may instead set `robot.kind`, and the two must
agree. Without a controller and keyboard the vehicle stays parked. Wheeled
robots always run in lockstep with a 1 ms physics step and 500 Hz odometry,
because Python computes the wheel forces every period; `real_time_factor`
1 (the default) paces keyboard driving in real time, 0 runs as fast as
possible. Not supported yet: `--world` (saved worlds), `--ekf` (use
`algorithms.estimator: gps_imu_ekf`), mission references, randomization,
and the multirotor-only features (payload, wind, battery, fixed wing, ground
effect, optical flow, airspeed); the configuration rejects them.

### Keyboard

| Key | Action |
| --- | --- |
| ↑ | Throttle (ramps up while held) |
| ↓ | Brake while moving forward faster than 0.5 m/s, else reverse |
| ←/→ | Steer (rover: turn on the spot); returns to centre on release |
| Space | Full brake |
| 1 / 2 / 3 | Tractor steering: front / coordinated (front and rear opposite) / crab |
| H | Tractor implement down / up (recorded; implements are not modelled yet) |
| Esc | End the run |

Ramp rates are `ground_vehicle.keyboard.throttle_rate_per_s` and
`steer_rate_per_s`; `max_speed_mps` (0: off) fades the throttle out above
that speed. The built-in keyboard driver (`control/wheeled_keyboard.py`) is
open loop: throttle × torque limit on every driven wheel (equal torque, like
an open differential), brake × brake limit, and steering geometry
(Ackermann about the rear axle line, or about mid-wheelbase for coordinated
four-wheel steering).

## Vehicle model

Gazebo simulates one rigid chassis link carrying the whole vehicle's mass
and inertia, with gravity and collision boxes. Wheels are not Gazebo joints
or contacts: Python computes their forces every odometry period
(`simulation/wheeled/model.py`):

1. **Suspension (ray cast).** Each strut casts a ray along body −z from its
   mount against an analytic copy of the scene ground (flat plane or seeded
   terrain, plus the tops of static boxes; `ground.py`). Compression x gives
   the normal load N = max(0, k x + c ẋ + k_bump max(0, x − travel)).
2. **Tyres.** Simplified Magic Formula (Pacejka 2012, §4.3.2) for the
   longitudinal and lateral directions, combined by normalized slip
   (§4.3.3), with transient (relaxation-length) slip (§7.2.1) and low-speed
   damping; the total is capped to the friction ellipse μ N. At standstill
   the carcass deflection acts as a spring, so a braked vehicle holds on a
   slope (`tyre.py`).
3. **Wheel spin.** I ω̇ = T_drive − R F_x, with brakes and rolling resistance
   as Coulomb torques that stop the wheel without reversing it; tyre and
   spin states are integrated in `substeps` per period.
4. **Actuators.** Drive torque limited by torque, power (constant-power
   curve), and a taper to zero at top wheel speed; brake limit; steering
   angle limit and slew rate (`actuators.py`).
5. **Aerodynamics.** Drag ½ρC_dA|v|v at the centre of mass, downforce
   ½ρC_LA v_x² split over the axles.

The chassis receives each wheel's ground force at the hub, the part of its
moment the bearing transmits, and the reaction of the drive and brake
torques; exact for wheels with spin inertia and no translational mass. The
sum is applied as one instantaneous Gazebo wrench per period carrying the
period's impulse (`BurstImpulseWrench`): persistent wrenches accumulate in
Gazebo and made each step slower as a run went on.

Per wheel, a collision cylinder at full suspension compression acts as a
bump stop and lets wheels hit walls; the body box `body` is watched by the
contact sensor (rollover, bottoming out, obstacles).

**Limitations.** Wheels see the analytic ground, Gazebo's chassis sees the
terrain mesh (interpolation error at the mesh resolution). A box side is a
wall, not a step a wheel can climb; driving onto a box needs the strut to
start above its top. Moving obstacles do not support wheels. Gazebo's wheel
visuals neither spin nor steer; Rerun draws the live wheels.

## Configuration

| Key | Meaning |
| --- | --- |
| `robot.kind` | `multirotor` (default), `rover`, `car`, `tractor` |
| `ground_vehicle.model_name` | Gazebo model name; "" uses the kind |
| `ground_vehicle.steering_layout` | `skid`, `ackermann`, `four_wheel` (keyboard mapping and validation) |
| `ground_vehicle.chassis` | `mass_kg`, `inertia_kg_m2`, `collision_size_m`, `collision_offset_m` (box centre from the centre of mass) |
| `ground_vehicle.wheels` | list of `name`, `position_m` (hub at full droop, body axes from the centre of mass), `steered`, `driven`, `braked`; the order is the command order |
| `ground_vehicle.wheel` | `width_m`, `inertia_kg_m2` (spin) |
| `ground_vehicle.tyre` | `radius_m`; `longitudinal`/`lateral`: `friction`, `b`, `c`, `e`, `relaxation_length_m`; `rolling_resistance`, `low_speed_mps`, `low_speed_damping_n_s_per_m` |
| `ground_vehicle.suspension` | `stiffness_n_per_m`, `damping_n_s_per_m`, `travel_m`, `bump_stiffness_n_per_m` (per wheel) |
| `ground_vehicle.drive` | per wheel: `max_torque_nm`, `max_power_w`, `max_wheel_speed_rad_s`, `max_brake_torque_nm` |
| `ground_vehicle.steering` | `max_angle_rad`, `max_rate_rad_s` |
| `ground_vehicle.aero` | `drag_area_m2`, `downforce_area_m2`, `downforce_front_fraction` |
| `ground_vehicle.spawn_yaw_deg`, `spawn_index` | start heading without scene spawn points; grid slot with them |
| `ground_vehicle.substeps` | tyre and wheel-spin integration substeps per period |
| `ground_vehicle.keyboard` | keyboard ramps and speed limit (above) |
| `sensors.wheel_encoders` | `enabled`, `rate_hz`, `counts_per_revolution`, `steering_noise_std_rad`, `seed` |

Encoders count floor(θ · cpr / 2π) of the true wheel angle and report the
count difference over the time since the previous reading (quantized,
interval-averaged speed); steering-angle sensors add seeded white noise.

Scene keys for wheeled robots (ignored by multirotors):

- `spawn_points`: `[{"position_m": [x, y], "yaw_deg": ψ}, ...]`, e.g. a race
  grid; the ego vehicle takes `ground_vehicle.spawn_index`;
- `ground_surface`: `{"name", "friction_scale", "rolling_resistance_scale"}`
  outside every zone (default 1, 1);
- `surfaces`: zones, later ones override earlier ones; `"shape":
  "rectangle"` with `center_m`, `size_m`, `yaw_deg`, or `"shape": "strip"`
  with `points_m`, `width_m`, `closed` (a road or track centreline). Named
  surfaces (`asphalt`, `gravel`, `grass`, `mud`, `soil`, `sand`, `ice`,
  `snow`, `kerb`) get matching colours in Gazebo (flat ground only) and
  Rerun.

## Controllers

A wheeled controller is `cls(context, options)` with `update(state,
request, dt_s) -> WheelCommands` (`control/wheeled.py`), selected by
`algorithms.controller` (`package.module:Class`, or the aliases `keyboard`
and `template`) with `algorithms.controller_options`, and called at
`control.control_rate_hz` (100 Hz in the presets):

- `context.wheeled` is the nominal vehicle (`core/wheeled_vehicle.py`): wheel
  order and positions, tyre radius, actuator limits;
- `state` (`WheeledControlState`) is the chassis state from truth or the
  estimate (`algorithms.control_state_source`) plus the latest wheel-encoder
  speeds and measured steering angles;
- `request` is the keyboard's `DriverCommand` (throttle, brake, steer,
  steering mode, implement) or None; with `--keyboard` and a controller set,
  the controller receives the driver input (driver assists);
- `WheelCommands` (`core/wheeled.py`) has per-wheel drive torque, brake
  torque, and steering angle. Structural errors (wrong length, non-finite
  values, negative brake, a command on a wheel without that actuator) raise
  ValueError; physical limits are applied and reported per channel in
  `commands.jsonl` (`limited`) and `summary.json.limited_commands`.

Start from `control/wheeled_template.py` (speed and yaw-rate P control) and
check a controller without Gazebo:

```python
from robotlab.testing.plugins import check_wheeled_controller
report = check_wheeled_controller(MyController, {"gain": 1.0}, kind="tractor")
```

## Outputs

Besides the shared artifacts ([2 Simulation](2_simulation.md#run-artifacts)):

| File | Contents |
| --- | --- |
| `commands.jsonl` | per control tick: accepted drive/brake/steer, `limited` wheels, requested values when they differ, driver input |
| `wheels.jsonl` | 50 Hz: per wheel contact, compression, normal load, longitudinal/lateral force, slip ratio and angle, speed, steering angle, delivered torque, friction scale; the chassis wrench |
| `summary.json` | `robot`, `control_ticks`, `limited_commands`, `wrench_messages`, `max_speed_mps`, `wheel_encoders`, and the shared entries |

Rerun shows the live wheels, surface zones, and the tabs Vehicle, Wheels,
Driver input, and Wheel commands. The health verdict treats the vehicle's
own collisions as its own, and body-on-ground contact as
`chassis_ground_contact` (warning; bottoming out or rollover).

## Verified behaviour

`tests/integration/test_gazebo_wheeled.py` (opt-in, `ROBOTLAB_TEST_GAZEBO=1`)
pins, verified on 2026-10-22: the car on the template controller at 10 m/s
covers 48.53 m in 6 s and agrees with the same model on the Python
reference integrator (`rigid_body.py`) within 0.1 m; parked ride heights
0.219 m (rover), 0.358 m (car), 1.023 m (tractor); repeated rover runs are
identical. A 10 s car run takes about 14 s wall time on the reference
desktop (Python load 0.69 per simulated second).
