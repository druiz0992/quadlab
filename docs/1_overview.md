# 1. Overview and architecture

RobotLab provides inspectable robot experiments for learning state
estimation and control: multirotors and a quadplane, wheeled robots
(rover, race car, tractor; [8 Ground vehicles](8_ground_vehicles.md)), and
bipeds in MuJoCo ([9 Bipeds](9_bipeds.md)). This document covers purpose, current status, code
organization, and shared conventions. Following documents:
[2 Simulation](2_simulation.md) · [3 Configuration](3_configuration.md) ·
[4 Control](4_control.md) · [5 Estimation](5_estimation.md) ·
[6 Roadmap](6_roadmap.md) · [7 Experiments](7_experiments.md) ·
[8 Ground vehicles](8_ground_vehicles.md) · [9 Bipeds](9_bipeds.md).

## Status

| Area | Implemented today |
| --- | --- |
| Simulation | Gazebo Harmonic physics, motors, IMU/GPS/barometer/magnetometer/rangefinder/camera/2D or 3D lidar/contacts, synthetic optical flow, private transport partition per run |
| Flight control | Native Gazebo waypoint velocity control; Python per-rotor thrust controller (keyboard or hover) with configurable loop rates and state/command latency |
| Vehicle configuration | Validated JSON: mass/inertia, motors, gains, sensor mounts/rates/noise/bias |
| Sensor errors | Gazebo native bias/noise (IMU, GPS, barometer, camera, lidar); seeded IMU noise density, bias random walk, turn-on bias, bias instability, scale/misalignment, g-sensitivity, saturation, sample loss and jitter; time-correlated GPS position and velocity error, receiver status, RTK, dual-antenna heading; magnetometer interference, drift, and structure anomalies; ESC telemetry; lidar weather, reflectivity, beam spread, and multiple returns |
| Vehicle physics | Gazebo motor lag/limits; ESC update rate and resolution; Python battery voltage sag and depletion, rotor ground effect; still-air body drag; centre-of-mass offset, per-rotor spread, per-run randomization of the true vehicle |
| Disturbances and faults | GPS dropouts/outages/blockout zones, GPS multipath jumps, scheduled IMU faults (bias step, stuck, spikes, scale, noise), native wind with Dryden-style turbulence, accelerometer vibration/clipping |
| Estimation | GPS/IMU error-state EKF as an estimator plug-in (observer or in the loop); reusable filter estimator for your own filters, driven by the IMU or wheel encoders (write only predict/update/state) |
| Algorithm harness | Estimator/controller plug-ins on a typed sensor bus with modelled delivery delay and clock offsets; control on truth or on the estimate; reference missions (step, waypoints/laps, circle, figure-eight, chirp, doublet); estimation and control metrics; truth-based run health verdict (crash, flip, impact, contact, no lift-off); offline estimator replay of recorded runs; plug-in debug channel; batch sweeps with regression baselines ([7 Experiments](7_experiments.md)) |
| Viewing and recording | Rerun world, sensor views, thrust plots, JSONL logs, `.rrd` replay |
| Worlds | JSON scene generation, paused Gazebo editing, saved SDF reuse; surface friction zones and spawn points for wheeled robots |
| Ground vehicles | `--robot rover/car/tractor`: ray-cast suspension, transient Magic Formula tyres, drive/brake/steering actuators, aerodynamics in Python; wheel encoders; keyboard driver; wheeled controller plug-ins ([8 Ground vehicles](8_ground_vehicles.md)) |
| Bipeds | `--robot biped` in MuJoCo: MJCF models (built-in test biped, or Pollen Robotics' Microduck from a local checkout), IMU and joint encoders on the sensor bus, joint-level controller plug-ins (hold, example, ONNX policies such as Pollen's), keyboard walking, Rerun view, fall health check ([9 Bipeds](9_bipeds.md)) |

**Not implemented:** the full Gazebo session API (algorithms run serially
on the mission thread, not in worker threads), obstacle avoidance,
perception, planning, and RL. `perception`, the three `planning` packages,
and `rl` contain only package docstrings. See [6 Roadmap](6_roadmap.md).

Flight controllers use **explicit simulator truth** as feedback unless
`algorithms.control_state_source` is `estimate`. Estimators never receive
truth; it is used only to synthesize measurements and to score runs.

## Code organization

```text
src/robotlab/
├── cli/                    # Argument parsing: `robotlab` (gazebo.py), `robotlab-batch`
│                           #   (batch.py), `robotlab-replay` (replay.py), `robotlab-magcal`
│                           #   (magcal.py), `robotlab-imucal` (imucal.py); shared exit
│                           #   statuses and SIGTERM handling (common.py)
├── core/                   # Small shared types: frames, gravity, vehicle properties,
│                           #   measurement samples and every bus payload type
│                           #   (measurements.py), WGS84 ↔ local ENU (geodesy.py), rotor
│                           #   commands, JSON codec (serialization.py), wheel commands
│                           #   (wheeled.py), wheeled vehicle properties
│                           #   (wheeled_vehicle.py), joint targets and legged properties
│                           #   (legged.py)
├── sensors/                # Sensor models, by sensor; Gazebo decoding (gazebo.py)
│   ├── bus/                # Sensor bus (delivery delay, clock offsets) and streams
│   ├── imu/                # Errors, imperfections (saturation, timing), temperature,
│   │                       #   faults, vibration (each an ImuPerturbation, applied as
│   │                       #   one ordered chain), applying a calibration
│   ├── gnss/               # Position error, receiver (velocity, status, RTK, heading),
│   │                       #   multipath, dropout, fault zones
│   ├── magnetometer/       # Errors, magnetic anomalies, applying a calibration
│   ├── imaging/            # Camera and lidar effects Gazebo does not render
│   └── barometer.py, airspeed.py, optical_flow.py, battery_monitor.py,
│                           #   esc_telemetry.py, wheel_encoders.py, joint_encoders.py
├── estimation/
│   ├── interface.py        # StateEstimator plug-in contract, Innovation
│   ├── examples/           # Example estimators, one per contract: complementary_estimator
│   │                       #   (full plug-in), imu_gps_filter, wheel_odometry_filter
│   ├── magnetometer_calibration.py  # Hard/soft iron and interference fits
│   ├── imu_calibration.py  # Static IMU calibration fit
│   ├── filter/             # Reusable for any predict/update filter: node (timing, held
│   │                       #   driving readings), measurement conversion, the `filter`
│   │                       #   plug-in
│   └── gps_imu_ekf/
│       └── ekf.py          # The EKF: only a RecursiveFilter (run by filter/)
├── evaluation/             # Scoring against truth: estimation (NEES/NIS), control,
│                           #   truth interpolation, effective IMU bias, run health
│                           #   verdict (health.py; bipeds: biped_health.py)
├── control/
│   ├── interface.py        # RotorController plug-in contract, ControlState
│   ├── examples/           # Example controllers, one per contract: cascaded_pd_rotor_,
│   │                       #   speed_yaw_wheeled_, pose_hold_legged_controller
│   ├── references.py       # Mission reference trajectories (analytic derivatives)
│   ├── waypoint.py         # Waypoint → body-velocity feedback (native mode)
│   ├── manual.py           # Position and attitude loops → four rotor thrusts
│   ├── timing.py           # Fixed-rate ticks, state/command latency buffers
│   ├── drone.py            # Exclusive thrust handle, receipts, publication worker
│   ├── legged.py           # LeggedController plug-in contract, state, WalkCommand
│   ├── legged_hold.py      # Built-in "hold": keep the start keyframe's targets
│   ├── legged_keyboard.py  # Held keys → ramped WalkCommand
│   ├── onnx_policy.py      # Exported RL policy runner (Microduck contract)
│   ├── wheeled.py          # WheeledController plug-in contract, state, DriverCommand
│   └── wheeled_keyboard.py # Keyboard driver and steering geometry
├── perception/             # Placeholder: images and point clouds
├── planning/               # Placeholder: global_planning, local_planning, behavior
├── rl/                     # Placeholder: environments, rewards, training, policies
├── simulation/
│   ├── battery.py          # LiPo pack voltage, current, state of charge
│   ├── aerodynamics.py     # Ground effect, Dryden-style turbulence
│   ├── powertrain.py       # Battery speed limit and ground effect on motor speeds
│   ├── esc.py              # ESC update rate and command resolution
│   ├── randomization.py    # Per-run true vehicle vs nominal (model mismatch)
│   ├── mujoco/             # Bipeds: settings (config.py), MJCF scene loading and edits
│   │                       #   (scene.py), truth and IMU readout (state.py), test biped
│   ├── wheeled/            # Wheeled robots: presets and validation (config.py), tyres,
│                           #   ground ray casts and surfaces, actuators, the vehicle model,
│                           #   spawn poses, reference rigid-body integrator
│   └── gazebo/
│       ├── config.py       # Drone JSON loading, defaults, validation
│       ├── properties.py   # Resolved vehicle properties
│       ├── world.py        # Scene loading and world composition
│       ├── vehicle.py      # Drone, motor, controller, and sensor SDF
│       ├── wheeled_vehicle.py  # Wheeled-robot chassis SDF
│       ├── actuators.py    # Thrust → motor-speed transport adapter
│       ├── wind.py         # Seeded wind targets and native wind commands
│       ├── sdf.py          # XML-building helpers
│       ├── environment.py  # Child-process library environment (Snap fix)
│       ├── process.py      # Owned gz sim process: parent-death signal, complete stop
│       ├── ingress.py      # Topic subscriptions into one bounded, counted queue; gz command
│       ├── stepping.py     # Lockstep stepping, start-up subscriber barrier
│       ├── plugins.py      # RobotLab Gazebo plugins: location, build, barrier choice
│       └── messages.py     # Gazebo timestamp/vector adapters
├── experiments/            # Orchestration (see experiments/__init__.py for the map)
│   ├── run.py              # Entry point: run settings (GazeboConfig), dispatch by robot kind
│   ├── harness/            # What plug-in authors use: algorithms.py (loading, context,
│   │                       #   runner, estimate-driven state), algorithm_debug.py,
│   │                       #   run_evaluation.py (metrics.json), replay.py and
│   │                       #   replay_log.py (robotlab-replay), comparison.py,
│   │                       #   batch.py (robotlab-batch)
│   ├── common/             # Shared by several robots: run_setup.py (option checks, world),
│   │                       #   sensor_pipeline.py (error/fault models → recording → bus),
│   │                       #   gazebo_helpers.py, mission.py (reference clock),
│   │                       #   schedule.py, loop_timing.py, interrupts.py, exit_reason.py
│   ├── multirotor/         # Drone missions: mission_run.py (composition), mission_loop.py,
│   │                       #   rotor_mission.py, rotor_loop.py, takeoff.py, failsafe.py,
│   │                       #   vehicle_forces.py, run_summary.py, keyboard_flight.py,
│   │                       #   effect_toggles.py, flying_object.py
│   ├── wheeled/            # Rover/car/tractor: wheeled_run.py, wheeled_mission.py,
│   │                       #   wheeled_summary.py, opponents.py, trailer.py
│   ├── biped/              # MuJoCo bipeds: biped_run.py, biped_loop.py, biped_keyboard.py
│   └── calibration/        # robotlab-imucal / robotlab-magcal procedures and their session
├── testing/
│   ├── plugins.py          # Plug-in contract test kit (synthetic data, no Gazebo),
│                           #   including check_wheeled_controller, check_legged_controller
│   ├── scenarios.py        # Estimator scenario suite: scenarios, sensor selection, scores
│   ├── synthetic_scene.py  # Ray-cast ground + boxes for synthetic ray sensors
│   ├── synthetic_sensors.py  # Truth → bus payloads for every sensor stream
│   └── trajectories.py     # Truth trajectories (hover, straight, circle)
└── visualization/
    ├── gazebo_rerun.py     # Gazebo-message-to-Rerun recording adapter
    ├── wheeled_rerun.py    # Wheeled-robot Rerun layout (wheels, surfaces, plots)
    ├── biped_rerun.py      # Biped Rerun view (MuJoCo geoms, trunk/command plots)
    ├── driving_keyboard.py # Tk driving window
    └── keyboard.py         # Tk keyboard window (key/focus state only)

gazebo_plugins/             # C++ gz-sim system plugins (lockstep barrier); built into
                            #   build/gazebo_plugins by robotlab-build-plugins
configs/                    # Layerable drone JSON configurations, by topic:
├── vehicles/               #   airframes: micro, racing_5in, 10in_4s, heavy_lift, battery_low
│                           #   (quad_x.json lists every default)
├── sensors/                #   sensor errors, barometer, vibration
├── faults/                 #   GPS dropout/blockout/multipath, IMU faults
├── environment/            #   wind, turbulence
├── control/                #   controller timing and latency
├── scenarios/              #   named combinations using "extends"
├── missions/               #   reference missions (step, square, circuit lap, chirp, ...)
├── algorithms/             #   estimator/controller plug-in setups
└── experiments/            #   batch experiment files (robotlab-batch)
baselines/                  # Stored batch metric baselines for regression checks
worlds/                     # Authored JSON scenes and saved SDF worlds
docs/                       # These numbered documents
tests/unit/<package>/       # Unit tests grouped by package
tests/integration/          # Multi-component and opt-in Gazebo/GUI tests, including
                            #   the Gazebo regression suite (test_gazebo_regression.py)
outputs/                    # Generated worlds, recordings, telemetry (not committed)
```

## Responsibilities and dependencies

| Package | Responsibility |
| --- | --- |
| cli | Parse arguments and invoke experiments; nothing else |
| core | Frame conventions, gravity, shared immutable types |
| sensors | Describe, decode, or perturb measurements; never estimate |
| estimation / control / perception / planning / rl | Numerical algorithms, independent of Gazebo, Rerun, GUI, CLI, and processes |
| simulation | Simulator-specific state, configuration, SDF, and transport adapters |
| experiments | Connect simulator, algorithms, and recording; may depend on all layers |
| visualization | Observe and record data; never decide commands or run algorithms |

- Gazebo SDF includes native plugins (motors, velocity controller, sensors).
  These are simulator features, not RobotLab Python algorithms.
- The `gazebo_rerun` adapter deliberately understands Gazebo messages but
  does not control, plan, or estimate.
- There is no shared simulator interface for Gazebo and MuJoCo: Gazebo is
  asynchronous transport, MuJoCo explicit stepping. Bipeds run
  only in MuJoCo, every other robot only in Gazebo. Add shared types to
  `core` only when real integrations require them.
- Optional imports (Gazebo bindings, Rerun, Tk) are deferred, so world
  generation and numerical tests work without them.

## Adding an algorithm

- **Predict/update estimator** (driven by the IMU, wheel encoders, or any
  other stream): write only the filter, a class with one `predict_<kind>`,
  one `update_<kind>` per fused sensor, and `state`
  ([5 Estimation](5_estimation.md#writing-your-own-filter));
  `estimation/examples/imu_gps_filter.py` (IMU) and `wheel_odometry_filter.py` (wheel
  encoders) are optional starting points. Select it with
  `algorithms.estimator: filter` and
  `estimator_options.filter: "package.module:Class"`; the built-in
  `gps_imu_ekf` works this way.
- **Other estimators** (e.g. batch or learned, or needing several IMUs or
  rewinding for late measurements) and **controllers:** write a full
  plug-in following the contracts in
  [7 Experiments](7_experiments.md#algorithm-plug-ins-algorithms), starting
  from `estimation/examples/complementary_estimator.py` or `control/examples/cascaded_pd_rotor_controller.py`, and select it
  with `algorithms.estimator`/`algorithms.controller` as
  `package.module:Class`.

Put the code in `estimation/` or `control/` (or any importable package)
with tests in `tests/unit/<package>/` that call
`robotlab.testing.plugins.check_estimator` (with `FilterEstimator` and
its options for a filter) or `check_controller`, and add estimators to the
[scenario suite](7_experiments.md#estimator-scenario-suite). Develop estimators against recorded runs with
`robotlab-replay` (several at once for a comparison table), then confirm
in Gazebo and with batches.
At the algorithm boundary use the bus samples (decoded measurements with
explicit timestamps and frames) and the `AlgorithmContext`; never simulator
messages, viewer state, or hidden truth.

Intended future composition: behavior selection → global route → local
trajectory → controller. Perception supplies observations to estimation and
planning; estimation supplies the state used by planning and control. RL may
implement or train a policy at any of these levels. This is a guide, not an
implemented pipeline. Choosing estimated state instead of truth for control
must be an explicit, recorded experiment decision.

## Conventions

- **Units and frames:** SI units; right-handed z-up world; gravity
  (0, 0, -9.81) m/s², set explicitly in generated worlds (Gazebo's own
  default is 9.8). Body frame: +X forward, +Y left, +Z up.
- **Attitude:** quaternions (x, y, z, w) rotate body vectors into world.
- **IMU:** gyros report body/sensor angular velocity; accelerometers measure
  specific force Rᵀ(a − g), so a level hover reads about +9.81 m/s² on z.
  Measurements keep their sensor axes; mount rotations are explicit.
- **Time:** integer simulation nanoseconds from the source message, never
  wall-clock receipt time. IMU readings are held from interval starts; the
  next sample ends the interval.
- **GPS:** WGS84 latitude/longitude/altitude with an ENU reference near
  Madrid at 650 m elevation. GPS altitude is not local world z.
- **Randomness:** local seeded generators; units and per-sample versus
  density semantics are documented with each model.

## Dependencies

Base: NumPy, SciPy, and Matplotlib (reads mesh textures for the Rerun
view). The `gazebo` extra adds pinned `rerun-sdk==0.24.1` and protobuf. The `gz` executable and `gz.transport13`/`gz.msgs10` bindings are
system prerequisites (ROS Jazzy, Python 3.12 here). The `mujoco` extra adds
pinned `mujoco==3.17.0` (CPU wheel) for bipeds; `policies` adds pinned
`onnxruntime==1.31.0` for the ONNX policy runner. The `dev` extra adds
pytest and black.

