# 1. Overview and architecture

RobotLab provides inspectable robot experiments for learning state
estimation and control: multirotors and a quadplane, and wheeled robots
(rover, race car, tractor; [8 Ground vehicles](8_ground_vehicles.md)). This document covers purpose, current status, code
organization, and shared conventions. Following documents:
[2 Simulation](2_simulation.md) · [3 Configuration](3_configuration.md) ·
[4 Control](4_control.md) · [5 Estimation](5_estimation.md) ·
[6 Roadmap](6_roadmap.md) · [7 Experiments](7_experiments.md) ·
[8 Ground vehicles](8_ground_vehicles.md).

*Migration (2026-10-22):* the project was renamed from DroneLab to RobotLab
with no compatibility alias, in preparation for ground vehicles. The Python
package is `robotlab`, the commands are `robotlab`, `robotlab-batch`,
`robotlab-replay`, … , environment variables start with `ROBOTLAB_`, the
Gazebo world is named `robotlab`, and the lockstep plugin is
`RobotlabLockstep` (`librobotlab-lockstep.so`; rebuild with
`robotlab-build-plugins`). Recordings now carry the replay format tag
`robotlab-replay`; replace `dronelab-replay` in the first line of an older
`replay.jsonl` to replay it. Reinstall with `pip uninstall dronelab && pip
install -e .`.

## Status

| Area | Implemented today |
| --- | --- |
| Simulation | Gazebo Harmonic physics, motors, IMU/GPS/barometer/magnetometer/rangefinder/camera/2D or 3D lidar/contacts, synthetic optical flow, private transport partition per run |
| Flight control | Native Gazebo waypoint velocity control; Python per-rotor thrust controller (keyboard or hover) with configurable loop rates and state/command latency |
| Vehicle configuration | Validated JSON: mass/inertia, motors, gains, sensor mounts/rates/noise/bias |
| Sensor errors | Gazebo native bias/noise (IMU, GPS, barometer, camera, lidar); seeded IMU noise density, bias random walk, turn-on bias, bias instability, scale/misalignment, g-sensitivity, saturation, sample loss and jitter; time-correlated GPS position and velocity error, receiver status, RTK, dual-antenna heading; magnetometer interference, drift, and structure anomalies; ESC telemetry; lidar weather, reflectivity, beam spread, and multiple returns |
| Vehicle physics | Gazebo motor lag/limits; ESC update rate and resolution; Python battery voltage sag and depletion, rotor ground effect; still-air body drag; centre-of-mass offset, per-rotor spread, per-run randomization of the true vehicle |
| Disturbances and faults | GPS dropouts/outages/blockout zones, GPS multipath jumps, scheduled IMU faults (bias step, stuck, spikes, scale, noise), native wind with Dryden-style turbulence, accelerometer vibration/clipping |
| Estimation | GPS/IMU error-state EKF, as a live observer (`--ekf`) or an estimator plug-in |
| Algorithm harness | Estimator/controller plug-ins on a typed sensor bus with modelled delivery delay and clock offsets; control on truth or on the estimate; reference missions (step, waypoints/laps, circle, figure-eight, chirp, doublet); estimation and control metrics; truth-based run health verdict (crash, flip, impact, contact, no lift-off); offline estimator replay of recorded runs; plug-in debug channel; batch sweeps with regression baselines ([7 Experiments](7_experiments.md)) |
| Viewing and recording | Rerun world, sensor views, thrust plots, JSONL logs, `.rrd` replay |
| Worlds | JSON scene generation, paused Gazebo editing, saved SDF reuse; surface friction zones and spawn points for wheeled robots |
| Ground vehicles | `--robot rover/car/tractor`: ray-cast suspension, transient Magic Formula tyres, drive/brake/steering actuators, aerodynamics in Python; wheel encoders; keyboard driver; wheeled controller plug-ins ([8 Ground vehicles](8_ground_vehicles.md)) |

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
│                           #   measurement samples, rotor commands, JSON codec
│                           #   (serialization.py), wheel commands (wheeled.py),
│                           #   wheeled vehicle properties (wheeled_vehicle.py)
├── sensors/                # Synthetic models, Gazebo decoding, geodesy, in-process
│                           #   streams, GPS dropout, accelerometer vibration,
│                           #   IMU/GPS error models (imu_errors.py, gps_errors.py),
│                           #   faults (imu_faults.py, multipath.py, zones.py),
│                           #   barometer conversion and errors (barometer.py),
│                           #   magnetometer errors, synthetic optical flow,
│                           #   sensor bus (bus.py), IMU imperfections, GPS receiver
│                           #   (velocity, status, RTK, heading), ESC telemetry,
│                           #   magnetic anomalies, lidar returns, wheel encoders
├── estimation/
│   ├── interface.py        # StateEstimator plug-in contract, Innovation
│   ├── template.py         # Copyable template estimator (complementary filter)
│   ├── magnetometer_calibration.py  # Hard/soft iron and interference fits
│   ├── imu_calibration.py  # Static IMU fit and calibration driver
│   ├── dead_reckoning.py   # Retained IMU dead-reckoning baseline
│   └── gps_imu_ekf/
│       ├── ekf.py          # Numerical prediction and correction
│       ├── node.py         # Timestamp/frame adaptation and EKF state channel
│       └── plugin.py       # The EKF as an estimator plug-in
├── evaluation/             # Scoring against truth: estimation (NEES/NIS), control,
│                           #   truth interpolation, effective IMU bias, run health
│                           #   verdict (health.py)
├── control/
│   ├── interface.py        # RotorController plug-in contract, ControlState
│   ├── template.py         # Copyable template controller (cascaded PD)
│   ├── references.py       # Mission reference trajectories (analytic derivatives)
│   ├── waypoint.py         # Waypoint → body-velocity feedback (native mode)
│   ├── manual.py           # Position and attitude loops → four rotor thrusts
│   ├── timing.py           # Fixed-rate ticks, state/command latency buffers
│   ├── drone.py            # Exclusive thrust handle, receipts, publication worker
│   ├── wheeled.py          # WheeledController plug-in contract, state, DriverCommand
│   ├── wheeled_keyboard.py # Keyboard driver and steering geometry
│   └── wheeled_template.py # Copyable template wheeled controller
├── perception/             # Placeholder: images and point clouds
├── planning/               # Placeholder: global_planning, local_planning, behavior
├── rl/                     # Placeholder: environments, rewards, training, policies
├── simulation/
│   ├── state.py            # Initial state for retained backend helpers
│   ├── battery.py          # LiPo pack voltage, current, state of charge
│   ├── aerodynamics.py     # Ground effect, Dryden-style turbulence
│   ├── powertrain.py       # Battery speed limit and ground effect on motor speeds
│   ├── esc.py              # ESC update rate and command resolution
│   ├── randomization.py    # Per-run true vehicle vs nominal (model mismatch)
│   ├── backends/           # Retained RotorPy/reference library adapters (no CLI)
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
├── experiments/
│   ├── gazebo.py           # Compose a Gazebo mission (run entry point, wiring)
│   ├── run_setup.py        # Option checks, true vehicle draw, world and resolved config
│   ├── mission_loop.py     # The running mission: message loop, truth step, shutdown
│   ├── sensor_pipeline.py  # Error/fault models → recording → sensor bus; synthetic sensors
│   ├── rotor_mission.py    # Rotor-thrust tick: powertrain, state source, failsafe, pilot
│   ├── vehicle_forces.py   # Python body drag and fixed-wing wrench
│   ├── keyboard_flight.py  # Keyboard effect toggles wiring and log
│   ├── run_summary.py      # summary.json assembly
│   ├── loop_timing.py      # Mission-loop wall cost and queue age (loop_timing)
│   ├── replay_log.py       # Record bus deliveries and truth (replay.jsonl)
│   ├── replay.py           # Offline estimator replay and scoring (robotlab-replay)
│   ├── algorithm_debug.py  # Plug-in debug channel (algorithm-debug.jsonl, Rerun)
│   ├── comparison.py       # Side-by-side metric tables (comparison.md)
│   ├── rotor_loop.py       # Fixed-rate rotor-thrust control loop with latency
│   ├── algorithms.py       # Plug-in loading, context, runner, estimate-driven state
│   ├── mission.py          # Mission clock and reference sampling
│   ├── run_evaluation.py   # Per-run metrics (metrics.json)
│   ├── batch.py            # Seed/parameter sweeps, aggregation, baselines
│   ├── interrupts.py       # SIGTERM as Ctrl-C; cleanup shielded from repeated signals
│   ├── magnetometer_calibration.py  # Automated calibration run (robotlab-magcal)
│   ├── imu_calibration.py  # Automated IMU calibration run (robotlab-imucal)
│   ├── calibration_session.py  # Shared Gazebo pose/sensor session for calibrations
│   ├── live_ekf.py         # Bounded live IMU/GPS ordering and EKF composition
│   ├── wheeled_run.py      # Compose a wheeled-robot mission
│   ├── wheeled_mission.py  # The running wheeled mission (lockstep loop)
│   └── wheeled_summary.py  # Wheeled summary.json assembly
├── testing/
│   └── plugins.py          # Plug-in contract test kit (synthetic data, no Gazebo),
│                           #   including check_wheeled_controller
└── visualization/
    ├── gazebo_rerun.py     # Gazebo-message-to-Rerun recording adapter
    ├── wheeled_rerun.py    # Wheeled-robot Rerun layout (wheels, surfaces, plots)
    ├── driving_keyboard.py # Tk driving window
    ├── keyboard.py         # Tk keyboard window (key/focus state only)
    ├── live.py             # Matplotlib live drone view (retained)
    └── plotting.py         # Static flight plots (retained)

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
- There is no shared simulator interface for Gazebo and RotorPy: one is
  asynchronous transport, the other explicit stepping. Add shared types to
  `core` only when real integrations require them.
- Optional imports (Gazebo bindings, Rerun, Tk) are deferred, so world
  generation and numerical tests work without them.

## Adding an algorithm

Copy `estimation/template.py` or `control/template.py`, following the
plug-in contracts in
[7 Experiments](7_experiments.md#algorithm-plug-ins-algorithms); put it in
`estimation/<name>.py` or `control/<name>.py` (or any importable package)
with tests in `tests/unit/<package>/` that call
`robotlab.testing.plugins.check_estimator`/`check_controller`, and select
it with `algorithms.estimator`/`algorithms.controller` as
`package.module:Class`. Develop estimators against recorded runs with
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

Base: NumPy, SciPy, Matplotlib, and pinned RotorPy (used only by retained
backend library code). The `gazebo` extra adds pinned `rerun-sdk==0.24.1` and
protobuf. The `gz` executable and `gz.transport13`/`gz.msgs10` bindings are
system prerequisites (ROS Jazzy, Python 3.12 here). The `dev` extra adds
pytest and black.

