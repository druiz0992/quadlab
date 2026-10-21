# 6. Roadmap and proposed APIs

Everything in this document is **proposed, not implemented**, unless marked
otherwise. Code examples here do not run against the current repository.
When a part is implemented, move its description into the relevant numbered
document and update the status below.

## Status summary

| Item | Status |
| --- | --- |
| Immutable command/receipt types, exclusive thrust handle and worker | Implemented ([4 Control](4_control.md)) |
| Validated drone JSON configuration | Implemented ([3 Configuration](3_configuration.md)) |
| Layered config files (repeat `--drone`, `extends`, provenance), topic folders | Implemented ([3 Configuration](3_configuration.md#combining-files)) |
| Command-line overrides (`--set key=value`) | Future, if needed |
| `Sample`, IMU/GPS/state payloads, basic `Channel`/`Subscription`/`Publisher` | Implemented ([5 Estimation](5_estimation.md)) |
| Live EKF observer, GPS dropouts, wind, accelerometer vibration | Implemented |
| Seeded IMU Allan-variance errors and correlated GPS error | Implemented ([3 Configuration](3_configuration.md#imu-and-gps-error-models)) |
| Barometer sensor with seeded offset/drift/noise | Implemented ([3 Configuration](3_configuration.md#barometer-sensorsbaro)) |
| Barometer fusion in the EKF (altitude update plus baro-offset state) | Future, with EKF robustness work |
| Magnetometer, downward rangefinder, synthetic optical flow, multi-channel 3D lidar | Implemented ([3 Configuration](3_configuration.md#magnetometer-sensorsmagnetometer)) |
| EKF fusion of magnetometer (heading), rangefinder and optical flow (GPS-denied) | Future, with EKF robustness work |
| Magnetometer calibration (hard/soft iron, current/throttle compensation; automated procedure; calibration for filters) | Implemented ([3 Configuration](3_configuration.md#magnetometer-calibration-sensorsmagnetometercalibration)) |
| Online (in-filter) magnetometer offset estimation to follow drift | Future, with EKF robustness work |
| IMU calibration (six-position plus thermal, startup gyro bias, bypass layer, staleness fingerprints) | Implemented ([3 Configuration](3_configuration.md#imu-calibration-sensorsimucalibration)) |
| Gyro scale/misalignment calibration on a simulated rate table | Future |
| Vehicle presets (micro, racing 5-inch, camera 10-inch, heavy-lift), FPV camera layer | Implemented ([3 Configuration](3_configuration.md#vehicle-presets)) |
| Battery monitor sensor; Rerun sensor markers and status indicators | Implemented ([3 Configuration](3_configuration.md#battery-monitor-sensorsbattery_monitor), [2 Simulation](2_simulation.md#viewing)) |
| Acro (rate-mode) controller for racing and freestyle handling (flips, rolls) | Future |
| Freestyle/acrobatic 5-inch preset | Implemented ([3 Configuration](3_configuration.md#vehicle-presets)) |
| Reversible (3D) motors with negative thrust | Future (schema change) |
| Hexa/octocopter layouts | Implemented: general `vehicle.rotors`, hexa preset |
| Motor/propeller failure (stop, partial thrust, speed limit) | Implemented ([3 Configuration](3_configuration.md#motor-faults-motorsfaults)) |
| Fault-tolerant control (survive a lost motor by giving up yaw); integral action for persistent disturbances | Future (control algorithms) |
| Keyboard toggles for effects during keyboard flight (rotor failure, wind, GPS dropout, multipath, IMU faults, …) | Implemented ([4 Control](4_control.md#keyboard-effect-toggles)). Original notes: Notes: each fault model gets a live on/off override next to its schedule; the Tk window shows the key map and current state; toggles are logged as events and in Rerun's Sensor status tab so runs stay analysable; keyboard flight is real time, so these runs are not repeatable (scheduled faults remain the repeatable, lockstep option). Wind needs a configured wind section to toggle. |
| Take-off from the ground (removes the airborne-spawn sag) | Implemented ([2 Simulation](2_simulation.md#ground-start-and-take-off)) |
| Landing sequence (automatic descent, touchdown detection, disarm) | Implemented for the low-battery failsafe ([3 Configuration](3_configuration.md#low-battery-failsafe-failsafelow_battery)) |
| Landing-gear contact model; pilot-commanded landing key | Future |
| Quadratic body drag, translational lift, vortex ring state (blade flapping via Gazebo rotor drag) | Implemented ([3 Configuration](3_configuration.md#aerodynamics-aerodynamics)) |
| Non-flat terrain; moving ground and flying obstacles; keyboard-spawned flying object | Implemented ([2 Simulation](2_simulation.md#terrain-terrain-in-a-scene)) |
| Terrain-aware obstacles for flow/ground effect; obstacle shapes other than boxes | Future |
| Camera lens distortion, fisheye lenses, motion blur, rolling shutter, video latency | Implemented ([3 Configuration](3_configuration.md#camera-realism-sensorscamera)) |
| Other camera types: depth/RGB-D, stereo pair, thermal, segmentation; multiple cameras; equirectangular 360° | Future, with perception work |
| Lidar range noise, dropouts, motion distortion; Livox Mid-360-like layer | Implemented ([3 Configuration](3_configuration.md#lidar-realism-sensorslidareffects)) |
| Lidar rain/dust/fog clutter and attenuation, beam-spread edge ghosts, surface reflectivity | Implemented ([3 Configuration](3_configuration.md#weather-environmentweather)) |
| Lidar multiple returns (first/last/strongest/dual), per-point reflectivity output | Implemented ([3 Configuration](3_configuration.md#lidar-realism-sensorslidareffects)) |
| Lidar returns through vegetation or semi-transparent objects | Future (needs such scene objects) |
| Obstacle avoidance on a sensor-independent local 3D map (lidar now, depth camera later) | Future, with planning work |
| Payload pickup/drop (mass and centre-of-gravity changes) | Implemented: rigid hook, and tethered payload on a winch with swing ([3 Configuration](3_configuration.md#tethered-payload-on-a-winch-payloadtether)) |
| Slack-capable cable (segmented rope), gripper attach conditions | Future |
| Pickup/delivery sensors for autonomy: winch cable length (encoder), cable tension (force/torque sensor on the winch joint), hook latch state, AprilTag on the payload seen by a downward camera; automatic pickup/release logic using only these (today's latch check uses simulator truth, which is fine as latch physics but not as controller input) | Future, with autonomy work |
| Slung-load swing damping (controller) | Future, with controller work |
| Quadplane autopilot polish: transition and return overshoot, energy-based (TECS) altitude/airspeed control, coordinated turns | Future, with controller work |
| Keyboard battery drain/refill; infinite battery option | Implemented ([4 Control](4_control.md#keyboard-effect-toggles)) |
| Battery and IMU temperature effects (with [ / ] ambient keys) | Implemented ([3 Configuration](3_configuration.md#temperature-effects)) |
| Fixed-wing VTOL (quadplane) inspired by the IMS Kite/Falziot (2.99 m span, 17 kg empty, <25 kg MTOW, 60 km/h stall, 65-80 km/h cruise): general rotor list (4 lift + pusher), wing/tail/control-surface aerodynamics, airspeed sensor, transition and fixed-wing baseline control, large world. Public specs omit rotor layout, motors, propellers, battery, wing area, and inertias, so it is a plausible model, not a calibrated one | Implemented 2026-10-21 (steps below) |
| VTOL step 1: general N-rotor vehicle schema (per-rotor position, axis, direction); quad becomes one rotor list; enables hexa/octo | Implemented ([3 Configuration](3_configuration.md#vehicle-and-motors)) |
| VTOL step 2: pitot airspeed sensor; wing/tail/control-surface aerodynamics and servos | Implemented in Python (Beard & McLain coefficient model via ApplyLinkWrench, which follows DroneLab's wind) instead of Gazebo LiftDrag ([3 Configuration](3_configuration.md#fixed-wing-and-quadplanes-fixed_wing)) |
| VTOL step 3: Kite-like quadplane preset (derived parameters) and a kilometre-scale world | Implemented (`kite_quadplane.json`, `worlds/airfield.json`) |
| VTOL step 4: minimal baseline hover / transition / cruise control, replaceable through the algorithm plug-in API | Implemented ([4 Control](4_control.md#quadplane-baseline-autopilot)) |
| VTOL step 5: Gazebo verification of stall speed, cruise power, and endurance against the published figures | Implemented ([3 Configuration](3_configuration.md#kite-like-quadplane-preset)): 62 vs 60 km/h minimum speed, 672 W cruise, ≈183 vs 180 min |
| Radio-link delay/loss and failsafes; GNSS jamming/spoofing | Out of scope (user decision, 2026-10-20) |
| Barometer propeller-wash and airspeed errors | Implemented ([3 Configuration](3_configuration.md#barometer-sensorsbaro)) |
| GPS blockout zones, multipath jumps, scheduled IMU faults | Implemented ([3 Configuration](3_configuration.md#fault-injection)) |
| EKF robustness: GPS gate lock-out recovery, fault detection, IMU sample-gap handling | Future (exposed by fault runs, see [5 Estimation](5_estimation.md#live-operation)) |
| Geometry-based (ray-traced) multipath | Not planned |
| Fixed controller rate, separate position-loop rate, state and command latency | Implemented ([4 Control](4_control.md#controller-timing-and-latency)) |
| Sensor delivery delay to the EKF | Implemented with the sensor bus; the built-in EKF rejects GPS older than its state (no re-propagation) |
| Lockstep (deterministic) simulation stepping | Implemented ([2 Simulation](2_simulation.md#lockstep-simulation)); use it for timing-sensitive work (gain tuning, latency margins, estimator-driven control) |
| Lockstep keyboard flight (real-time pacing cap with interactive input) | Future |
| Measure Python pipeline delay in real-time mode (queue wait, processing time, publish lag behind Gazebo) | Future (lockstep removes the need for most experiments) |
| Battery voltage sag, body drag without wind, plausible motor/prop preset | Implemented ([3 Configuration](3_configuration.md#battery-and-powertrain-battery)) |
| Dryden-style turbulence, ground effect | Implemented ([3 Configuration](3_configuration.md#turbulence-windturbulence)) |
| Obstacle-aware ground effect | Future |
| Low-battery failsafe: warn, land, return home and land | Implemented ([3 Configuration](3_configuration.md#low-battery-failsafe-failsafelow_battery)) |
| IMU bias instability, scale/misalignment, g-sensitivity, saturation, sample loss and jitter | Implemented ([3 Configuration](3_configuration.md#imu-imperfections-sensorsimuimperfections)) |
| GPS velocity errors, receiver status, RTK, dual-antenna heading, latency | Implemented ([3 Configuration](3_configuration.md#gps-receiver-velocity-status-rtk-heading)) |
| Magnetometer time-varying interference and structure anomalies | Implemented ([3 Configuration](3_configuration.md#magnetometer-sensorsmagnetometer)) |
| ESC update rate, command resolution, RPM telemetry | Implemented ([3 Configuration](3_configuration.md#esc-command-path-and-telemetry-motorsesc)) |
| Per-run vehicle randomization, centre-of-mass offset, per-rotor spread | Implemented ([3 Configuration](3_configuration.md#model-mismatch-and-randomization)) |
| `GazeboSession` (launch/readiness/clock/pause/cleanup, `sensors`/`truth` namespaces) | Proposed |
| Sensor descriptions, stream status, byte budgets, camera/lidar/contact/truth channels | Proposed |
| Estimator/controller plug-ins on a typed sensor bus (serial dispatch on the mission thread) | Implemented ([7 Experiments](7_experiments.md#algorithm-plug-ins-algorithms)) |
| `AlgorithmRunner` with per-algorithm worker threads and declared outputs | Proposed (the serial runner covers current needs and is deterministic in lockstep) |
| Velocity mode routed through exclusive control; `rotor_speed` mode | Proposed |
| Experiment configuration files (`configs/experiments/`): batch sweeps, aggregation, baselines | Implemented ([7 Experiments](7_experiments.md#batch-experiments-and-baselines)) |
| Estimator-driven control | Implemented ([7 Experiments](7_experiments.md#algorithm-plug-ins-algorithms)) |
| Estimation (NEES/NIS) and control metrics against truth | Implemented ([7 Experiments](7_experiments.md#metrics-metricsjson)) |
| Run health verdict (flip, hard impact, obstacle contact, below ground, left ground area, non-finite values, saturation, drops); batch runs exclude failed runs and `--check` fails on them | Implemented ([7 Experiments](7_experiments.md#run-health-verdict-health)) |
| Opt-in Gazebo regression suite pinning verified numbers (lockstep) | Implemented (`tests/integration/test_gazebo_regression.py`, see the README) |
| Split the Gazebo mission loop (`experiments/gazebo.py`) into pipeline stages under the regression suite | Implemented ([2 Simulation](2_simulation.md#mission-lifecycle)); verified byte-identical lockstep outputs |
| Determinism gaps found while verifying the split: Python-published wind commands can land on either side of a lockstep step under CPU load; ground-start, ESC-telemetry, rendered-sensor, and magnetometer/sensor-error runs vary between identical runs; a ground-start run once never took off yet scored `ok` | Future (simulation robustness item 6) |
| Audit remaining Gazebo defaults (atmosphere, magnetic table, solver and contact settings) | Future (simulation robustness) |
| Lockstep performance on large worlds (pose-message volume) | Future (simulation robustness) |
| Gazebo process cleanup and hang detection | Future (simulation robustness) |
| Sensor delivery latency, jitter, clock offsets, cross-stream reordering | Implemented ([7 Experiments](7_experiments.md#sensor-bus-and-delivery-timing-sensor_timing)) |
| Waypoint-sequence missions / circuit laps; step, circle, figure-eight, chirp, doublet references | Implemented ([7 Experiments](7_experiments.md#reference-missions-mission)) |
| Allocation helpers exposing achieved wrench and residuals | Future |
| Perception, planning (behavior → global → local), obstacle avoidance, RL | Future (placeholder packages) |
| Spatial wind and calibrated aerodynamics | Future |

## Gazebo session

```text
Experiment / mission / user controller
    ├── session.sensors: typed measurement channels
    ├── session.control: exclusive handle for the configured mode
    └── session.truth:   explicit optional evaluation/control input
                     │
                GazeboSession
    ├── transport subscriptions and publishers, bounded buffers, decoding
    ├── clock, readiness, pause, lifecycle, command arbitration
    └── observers → recorder / Rerun
                     │
                   Gazebo (physics, contacts, sensors, motors)
```

| Proposed module | Ownership |
| --- | --- |
| `simulation/gazebo/transport.py` | Native subscriptions, ingress times/sequences, dispatch |
| `simulation/gazebo/session.py` | Launch, readiness, clock, cleanup, sensor/truth namespaces |
| `experiments/algorithms.py` | Runner, serial workers, connections, health, outputs |
| `control/allocation.py` | Explicit body-wrench → rotor-thrust allocation |

```python
with GazeboSession.launch(experiment="configs/experiments/circuit.json",
                          control_mode="rotor_thrust") as sim:
    sim.wait_ready(required=["imu", "gps"], timeout_wall_s=30)
    with sim.sensors.imu.subscribe(capacity=2048, overflow="error") as imu:
        with sim.control.acquire() as control:
            for sample in imu:
                control.send(RotorThrusts(controller.update(sample)), lifetime_sim_s=0.05)
```

**Lifecycle:** validate config → generate SDF → subscribe/advertise → launch
paused → transport/clock ready → subscribe algorithms → unpause → await
required sensors → run. Pause must not look like transport failure. Cleanup
is idempotent and releases readers, workers, publishers, recorder, and the
owned process even after partial startup. Keep the private transport
partition. First release: launch and close only; no attach (needs a
simulator-side command timeout, since a killed Python process cannot run its
watchdog), reset, or deterministic `step(n)`. Time reversal faults the
session; epochs leave room for a later reset API.

Saved `--world` files: never silently replace drone properties; validate the
required model, topics, and mode and record an extracted manifest.

## Sensor channels

Every sample carries stream ID, sequence (from 1, counted before decode or
queue loss so gaps are visible), epoch, integer source simulation time,
monotonic receipt time (diagnostics only), frame ID, immutable payload with
read-only arrays, and status. Never substitute receipt time for missing
simulation time; never invent status, covariance, or health Gazebo lacks.

| Channel | Payload |
| --- | --- |
| `sensors.imu` | `ImuReading`: specific force, angular rate in IMU axes |
| `sensors.gps` | `GpsReading`: lat/lon degrees, geodetic altitude m, optional velocity/status/covariance |
| `sensors.camera` | `CameraFrame`: owned RGB uint8 H×W×3, calibration ID, optical frame |
| `sensors.lidar` | `LidarScan`: raw ranges, angular/range limits, sensor frame |
| `sensors.contacts` | `ContactBatch`: collision pairs at the source time |
| `truth.odometry` | `TruthState`: world position/velocity, xyzw attitude, body rate |

- `SensorDescription` per stream: frame, payload type, nominal rate, mount
  position and quaternion, calibration (camera intrinsics plus explicit
  sensor-to-optical rotation). Frame IDs such as `drone/imu`, `drone/body`,
  `drone/camera_optical`, `world`.
- Raw measurements are preserved; body/world conversions are explicit named
  helpers. An off-centre accelerometer is not made body-origin by rotation.
  Verify Gazebo frame and gravity semantics at ingress with rotated mounts.
- Configured noise/bias describe the simulator, never an automatically known
  correction or estimator covariance.

Planned additions to the implemented `Channel`/`Subscription`:
`ChannelStatus`/`SubscriptionStatus`, `SensorDisabled` and `UnknownStream`
errors, lost-sequence ranges in `StreamOverflow`, ingress-gap failure for
`error` readers, per-stream ingress buffers with image byte budgets, decode on
internal workers, and a `snapshot()` of latest readings with individual ages
(never presented as synchronized). Time may repeat within a stream but never
decrease. An async-iterator adapter can follow the blocking API.

## Algorithm runner

```python
with AlgorithmRunner() as runner:
    fusion = runner.add("state_estimator", required=True)
    estimates = runner.output("state", EstimatedState, producer=fusion)
    fusion.connect("imu", session.sensors.imu, on_imu, payload_type=ImuReading, capacity=2048)
    fusion.connect("gps", session.sensors.gps, on_gps, payload_type=GpsReading, capacity=128)
    runner.start()
    # mission loop calls runner.check_health() while owning the simulator
```

- Each worker owns one subscription per port and invokes handlers serially,
  round-robin over nonempty queues: arrival processing, not timestamp sorting
  or implicit fusion. Different algorithms may run concurrently. No user code
  runs on transport callbacks.
- Unique names; duplicate ports, payload-type mismatches, disabled sensors,
  and changes after start fail. Handlers return `None`; publishing is explicit.
- Outputs are declared before start under `algorithms/<algorithm>/<output>`,
  single-writer, cannot shadow `sensors`/`truth`, and no worker cycles.
  `publish_from(sample, payload, frame_id=...)` keeps the input's time and
  epoch and records provenance; fusion outputs use explicit time and parents.
  Do not stamp an estimate with a delayed GPS time.
- A required worker's exception, overflow, or contract violation makes
  `check_health()` raise `AlgorithmFailed`; the owner stops thrust and
  terminates Gazebo. Optional observers detach on failure. Handlers must
  cooperate with shutdown; a timeout is reported, never claimed as killed.
- Record gaps, queue high-water marks, counts, decode failures, handler
  duration, dispatch age, failures, and provenance. Recording/Rerun schema
  changes need an explicit migration.

**Stages:** envelopes/decoders → channels with fake transport → session
extraction → worker connections → sensor-driven estimation experiment →
estimator-driven control. Preserve CLI behavior and timestamp contracts.
Do not build a universal Gazebo/RotorPy interface.

## Control modes

| Mode | Command | Inner loop |
| --- | --- | --- |
| `rotor_thrust` (primary, implemented) | `RotorThrusts(thrust_n)` | User algorithm |
| `velocity` (implemented without exclusive handle) | `BodyVelocity(linear_mps, yaw_rate_rps)` | Gazebo native controller |
| `rotor_speed` (advanced) | `RotorSpeeds(rad_s)` | User algorithm via the same actuator adapter |

Mode is fixed per session; exactly one writer. The waypoint runner should use
the same exclusive mechanism. A world-velocity helper needs an explicit,
fresh attitude source rather than silently reading truth. Rotor-state
telemetry should be a separate `actuators.rotor_state` channel with explicit
provenance. Estimator-driven control: the experiment acquires the thrust
handle and gives the controller a narrow command adapter, replacing the truth
controller; stale estimates must stop command submission even while
heartbeats continue.

## Experiment configuration

Three files with distinct ownership: `worlds/circuit.json` (geometry),
`configs/vehicles/quad_x.json` (vehicle, implemented), and
`configs/experiments/circuit.json` (scene, spawn, environment, mode, mission,
recording, timing; references the drone file). Resolve file references
relative to the containing file; apply defaults → files → CLI overrides; save
the resolved result and a hash per run. Possible later schema extensions:
arbitrary rotor lists with per-rotor axis/direction, full inertia matrices,
sensor quaternions, spawn attitude (reject nonzero initial velocity until
tested). Expose typed overrides, never raw SDF strings.

## Planning and autonomy

Waypoint sequences for circuit laps come first, then local obstacle
avoidance using perception output. Intended flow: behavior selection →
global route → local trajectory → controller, with perception supplying
observations and estimation supplying state. Validate each step in Gazebo
before claiming autonomous laps or realistic wind response.
