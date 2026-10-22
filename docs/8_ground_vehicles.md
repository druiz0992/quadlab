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

# Race: you drive the car on the circuit against three opponents
robotlab --robot car --keyboard --headless --scene worlds/race_circuit.json \
  --config configs/scenarios/race_opponents.json --output outputs/race

# Tractor on the farm: plough (H lowers/raises it, RTK GPS) or towing a trailer
robotlab --keyboard --headless --scene worlds/farm_field.json \
  --config configs/scenarios/field_plough.json --output outputs/plough
robotlab --keyboard --headless --scene worlds/farm_field.json \
  --config configs/scenarios/tractor_trailer.json --output outputs/trailer

# Rover off-road (bumps, slopes, sand, gravel and mud patches)
robotlab --robot rover --keyboard --headless --scene worlds/rover_terrain.json --output outputs/rover-terrain
```

`--robot` loads the vehicle's preset (`simulation/wheeled/config.py`) under
all `--config` layers; a layer may instead set `robot.kind`, and the two must
agree. Without a controller and keyboard the vehicle stays parked. Wheeled
robots always run in lockstep, because Python computes the wheel forces
every period: 250 Hz odometry (wheel-force updates) with two 2 ms physics
steps per period and four 1 ms tyre substeps, a 125 Hz controller. Against
500 Hz / 1 ms steps a follower lap differed by 0.01 % (56.736 s vs
56.730 s). Gazebo runs the bursts unthrottled and Python paces the run:
`simulation.real_time_factor` 1 (the default) is real time (keyboard
driving), 0 as fast as possible; `summary.json.real_time` reports the
achieved factor (a slow period is not caught up). Not supported yet: `--world` (saved worlds), `--ekf` (use
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
| H | Tractor implement down / up |
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
| `ground_vehicle.randomization` | `enabled`, `seed`, `mass_fraction_std`, `inertia_fraction_std`, `center_of_mass_std_m`, `tyre_friction_fraction_std`, `drive_torque_fraction_std` (below) |
| `ground_vehicle.route` | `waypoints_m`, `width_m` (below) |
| `sensors.wheel_encoders` | `enabled`, `rate_hz`, `counts_per_revolution`, `steering_noise_std_rad`, `seed` |

Encoders count floor(θ · cpr / 2π) of the true wheel angle and report the
count difference over the time since the previous reading (quantized,
interval-averaged speed); steering-angle sensors add seeded white noise.
Readings are taken from the ego's wheels on the first odometry period at
or after each multiple of 1/`rate_hz`, so the rate holds even when it does
not divide the odometry rate (the controller is scheduled the same way;
before 2026-10-22 a 100 Hz encoder on 4 ms periods read every 12 ms). The controller gets
the latest reading in `WheeledControlState`. When the sensor bus exists
(an estimator is configured), every reading is also a `wheel_encoders`
sample (`EncoderReading`, frame `vehicle/wheels`, timing
`sensor_timing.wheel_encoders`), so an estimator can declare it as an
input. Disabled encoders pass truth to the controller and never enter
the bus.

Scene keys for wheeled robots (ignored by multirotors):

- `spawn_points`: `[{"position_m": [x, y], "yaw_deg": ψ}, ...]`, e.g. a race
  grid; the ego vehicle takes `ground_vehicle.spawn_index`;
- `ground_surface`: `{"name", "friction_scale", "rolling_resistance_scale"}`
  outside every zone (default 1, 1);
- `surfaces`: zones, later ones override earlier ones; `"shape":
  "rectangle"` with `center_m`, `size_m`, `yaw_deg`, or `"shape": "strip"`
  with `points_m`, `width_m`, `closed` (a road or track centreline). Named
  surfaces (`asphalt`, `gravel`, `grass`, `mud`, `soil`, `sand`, `ice`,
  `snow`, `kerb`) get matching colours in Gazebo (one flat mesh per zone;
  flat ground only) and Rerun. Zones are rasterized at 0.25 m for the wheel
  lookups, so a boundary is resolved to within 0.18 m;
- `track`: `{"centreline_m": [[x, y], ...], "width_m": w, "closed": true}`,
  a race track (`core/track.py`): lap timing, track limits, and map
  knowledge for controllers (`context.track`).

## Worlds

| Scene | For | Contents |
| --- | --- | --- |
| `worlds/race_circuit.json` | car | 1.23 km circuit, kerbs, gravel traps, grass, grid, `track` (below) |
| `worlds/farm_field.json` | tractor | concrete yard, gravel farm track, 160 × 100 m soil field with crop rows on its south half, grass headlands, barn, shed, silo, trees; flat |
| `worlds/rover_terrain.json` | rover | seeded height field (140 overlapping bumps and slopes up to about 2.5 m; median slope 6°, 95th percentile 13°), sand, gravel and mud patches, boulders seated on the terrain; flat start disc |

The terrain height field has a step down to flat ground at the edge of its
60 m square; stay inside it. Its heights are summed over all hills at once
(bit-identical to the sequential sum), so wheel ray casts on terrain cost
about 30 µs.

## Tractor implement and trailer

The tractor carries either a rear three-point-hitch implement or a towed
trailer (one rear hitch; the configuration rejects both).

**Implement** (`ground_vehicle.implement`, `simulation/wheeled/implement.py`).
Soil draft follows the ASABE D497 form D = F_i (A + B S + C S²) W T (S in
km/h, W width in m, T depth in cm; ASABE D497.7, 2015). The built-in
`moldboard_plough` uses A = 652, B = 0, C = 5.1 and F_i = 1.0 / 0.70 / 0.45
for fine / medium / coarse soil; `type: custom` takes `draft_a`,
`draft_b`, `draft_c`, and `texture_factor`. The draft acts at the tool point
(bottom centre of the implement) against its horizontal velocity, faded in
from standstill (no draft without motion); vertical soil forces are not
modelled. The lift moves at `lift_rate_mps` and, lowered, controls the
working depth (`working_depth_m`) within `lift_travel_m`. In Gazebo the
implement is a visual-only link with its mass on a prismatic joint that
follows the Python lift. Keys: `enabled`, `type`, `texture`, `width_m`,
`working_depth_m`, `mass_kg`, `size_m`, `hitch_position_m` (box centre when
raised, body axes from the centre of mass), `lift_travel_m`,
`lift_rate_mps`, `start_lowered`. The keyboard's H toggles it; a controller
may define `implement_lowered() -> bool` (called each control tick) to
command it instead. `summary.json.implement` reports worked area (width ×
distance at ≥ half the working depth), draft energy, and peak draft;
`wheels.jsonl` and Rerun (Implement tab) the lift, depth, and draft.
Verified: lowered at 5 s from 40 km/h with 4000 N·m per wheel, the tractor
ploughs at 0.20 m depth and 2.15 m/s against 21.5 kN, with 16 % front and
6 % rear wheel slip; wheel loads sum to tractor plus implement weight.

**Trailer** (`ground_vehicle.trailer`, `experiments/trailer.py`). A link in
the tractor's model on revolute yaw and pitch joints at the drawbar hitch
(roll follows the tractor: a simplification). Gazebo publishes the joint
angles and rates every physics step; Python computes the trailer's state by
forward kinematics, runs its own unpowered wheel model (tyres and
suspension derived from the tractor's, scaled so the static sag matches),
and applies its wrench. Its brakes follow the tractor's commanded brake
fraction. Keys: `enabled`, `mass_kg`, `inertia_kg_m2`, `size_m`,
`hitch_position_m`, `drawbar_m`, `cg_below_hitch_m`, `wheel_radius_m`,
`wheels` (trailer axes, hub at full droop; `braked`), `max_brake_torque_nm`,
`yaw_limit_deg`, `pitch_limit_deg`. A hitch yaw beyond 90 % of its limit is
a `jackknife` warning. `summary.json.trailer` reports the peak articulation
and jackknife time; `wheels.jsonl` the hitch angles and trailer wheel
loads. Verified: parked, the trailer axle carries 13.5 kN (expected 13.4 kN
for the axle 0.3 m behind the 1500 kg trailer's centre of mass) and the
loads of both vehicles sum to their weight; a full-lock left turn
articulates to −71°.

## Race circuit and opponents

`worlds/race_circuit.json` is a 1.23 km closed circuit: 12 m asphalt
(friction scale 1), kerbs on the corners, three gravel traps (0.45, high
rolling resistance), grass run-off (0.5), a pit building and grandstand,
and a six-slot grid behind the start/finish line (centreline point 0).

`ground_vehicle.opponents` adds up to seven cars of the ego's kind on the
next grid slots:

| Key | Meaning |
| --- | --- |
| `count` | number of opponents (0: none) |
| `mode` | `kinematic` (default): cheap moving obstacles; `dynamic`: full wheel model |
| `update_rate_hz` | kinematic: odometry and command rate (default 50; must divide `control.odometry_rate_hz`) |
| `controller` | their controller (`package.module:Class` or an alias); default `track_follower` (the only one in kinematic mode) |
| `controller_options` | options for every opponent |
| `per_opponent_options` | list of option dicts merged per opponent (first opponent first) |

`configs/scenarios/race_opponents.json` sets three followers with
different grip fractions and lines. The ego controller receives their
ground-truth states in `state.opponents` (labelled truth: a stand-in for
perception). Contact with an opponent is a `vehicle_contact` warning, not
a failed run, unless it is a high-speed impact.

**Kinematic opponents** (the default) are moving obstacles, not vehicle
simulations. They have no wheel model or wrench. Gazebo moves each one
with its VelocityControl system, with gravity off and the body box as the
only collision. At `update_rate_hz` Python turns the track follower's
guidance into a body twist: the speed follows the target speed at the
follower's grip-limited acceleration, and the yaw rate is v κ. A vertical
term holds the start-up height above the ground, and roll and pitch rates
level the body after a knock. Gazebo still resolves collisions with the
ego. A velocity-controlled body keeps its commanded velocity, so it acts
as a very heavy obstacle. The ego sees their states up to one update
period old (20 ms by default). Their speeds are not limited by tyre
physics.

**Dynamic opponents** (`mode: "dynamic"`) run any `WheeledController` on
their own truth. The batched wheel model steps all cars together every
period, and each car publishes its own wrench. This mode costs about
0.3 ms of Python per car per period, plus Gazebo's work for the full
vehicle model.

Migration (2026-10-22): before this change all opponents were dynamic.
Set `"mode": "dynamic"` to keep the old behaviour. `summary.json.opponents[]`
gained a `mode` entry.

**Track follower** (`control/track_follower.py`, alias `track_follower`)
is the traffic driver: pure pursuit on the (optionally offset) centreline,
a curvature-limited speed target (`grip_fraction` × μ g), and acceleration
and braking capped by grip so a rear-drive car does not spin its wheels.
Options: `lateral_offset_m`, `lookahead_s`, `min_lookahead_m`, `preview_m`,
`grip_fraction`, `max_speed_mps`, `speed_gain_per_s`. It is a baseline,
not a racing line; it can also drive the ego (`algorithms.controller:
track_follower`).

Lap timing (`evaluation/laps.py`) counts forward crossings of the
start/finish line (the first one starts timing; reversing over the line
undoes it) and track limits (every wheel contact beyond the track edge).
`summary.json.track` and each `summary.json.opponents[].track` report lap
times, best lap, progress along the track, and off-track episodes and
time; the health verdict warns with `off_track`.

## Routes, randomization, and batch runs

**Routes.** `ground_vehicle.route.waypoints_m` (`[[x, y], ...]`, world m)
is a reference path. It replaces the scene track as `context.track`, so
`track_follower` and your own controllers can follow it. Lap timing and
track limits are evaluated on it, with a corridor `width_m` wide. Straight
legs are resampled at most 2 m apart. A last point equal to the first
makes a closed loop. On an open route `summary.json.track` reports
`progress_m` from the start and `finished_s`, the first time within 1 m of
the end. Every route and track reports `max_lateral_offset_m` and
`rms_lateral_offset_m`. The multirotor `mission` section is rejected for
wheeled robots, with a pointer to the route. Example:
`configs/scenarios/route_airfield.json` (a 313 m loop on the airfield).

**Randomization.** `ground_vehicle.randomization` draws a true vehicle
that differs from the nominal one, seeded and truncated at 3σ (see
`simulation/randomization.py`). The draws cover mass, the mass
distribution (second moments, which keeps the inertia physically valid),
centre-of-mass shift, tyre friction, and drive torque/power. Gazebo and
the Python wheel model simulate the true vehicle. Controllers and
estimators get the nominal one in `context.wheeled`. Opponents are built
from the true vehicle too. `summary.json.randomization` lists the draws,
and `robot.mass_kg` is the true mass. Example layer:
`configs/vehicles/randomized_wheeled.json`.

**Batch runs.** An experiment file with `"robot": "car"` (or `rover`,
`tractor`) runs wheeled robots in `robotlab-batch`
([7 Experiments](7_experiments.md#batch-experiments-and-baselines)).
`summary.track.*` and `summary.randomization.*` work as metrics.
`configs/experiments/wheeled_route_randomized.json` drives the airfield
loop with `track_follower` on four randomized cars for 90 s each. Result
(2026-10-22): best lap 42.78 ± 0.18 s, lateral offset RMS 0.35 ± 0.10 m,
maximum 1.5 ± 0.7 m, no off-track episodes.

## Controllers

A wheeled controller is `cls(context, options)` with `update(state,
request, dt_s) -> WheelCommands` (`control/wheeled.py`), selected by
`algorithms.controller` (`package.module:Class`, or the aliases `keyboard`,
`template`, and `track_follower`) with `algorithms.controller_options`, and called at
`control.control_rate_hz` (100 Hz in the presets):

- `context.wheeled` is the nominal vehicle (`core/wheeled_vehicle.py`): wheel
  order and positions, tyre radius, actuator limits;
- `state` (`WheeledControlState`) is the chassis state from truth or the
  estimate (`algorithms.control_state_source`) plus the latest wheel-encoder
  speeds and measured steering angles;
- `request` is the keyboard's `DriverCommand` (throttle, brake, steer,
  steering mode, implement) or None; with `--keyboard` and a controller set,
  the controller receives the driver input (driver assists);
- `context.track` is the scene's track (or None) and `state.opponents` the
  other cars' ground-truth states;
- `WheelCommands` (`core/wheeled.py`) has per-wheel drive torque, brake
  torque, and steering angle. Structural errors (wrong length, non-finite
  values, negative brake, a command on a wheel without that actuator) raise
  ValueError; physical limits are applied and reported per channel in
  `commands.jsonl` (`limited`) and `summary.json.limited_commands`.

Optional: `implement_lowered() -> bool` commands the tractor implement.

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
covers 48.59 m in 6 s and agrees with the same model on the Python
reference integrator (`rigid_body.py`) within 0.1 m; parked ride heights
0.219 m (rover), 0.358 m (car), 1.023 m (tractor); repeated rover runs are
identical; a race with two opponents is deterministic in both opponent
modes (identical ego telemetry and opponent summaries); scripted keys
through the keyboard path drive at real time (achieved factor 1.0 ± 0.08).
Speed on the reference desktop, running as fast as possible
(`real_time_factor: 0`) for 20 s on the circuit: one car alone reaches
1.32× real time; with three opponents it reaches 1.06× (kinematic) versus
0.85× (dynamic); with five kinematic opponents it reaches 0.98×. The
remaining cost is mostly the ego's own Python loop, about 1.7 ms per 4 ms
period, of which the wheel model takes 0.7 ms.
