# 1. Overview and architecture

DroneLab provides inspectable drone experiments (multirotors and a quadplane) for learning state
estimation and control. This document covers purpose, current status, code
organization, and shared conventions. Following documents:
[2 Simulation](2_simulation.md) · [3 Configuration](3_configuration.md) ·
[4 Control](4_control.md) · [5 Estimation](5_estimation.md) ·
[6 Roadmap](6_roadmap.md) · [7 Experiments](7_experiments.md).

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
| Algorithm harness | Estimator/controller plug-ins on a typed sensor bus with modelled delivery delay and clock offsets; control on truth or on the estimate; reference missions (step, waypoints/laps, circle, figure-eight, chirp, doublet); estimation and control metrics; batch sweeps with regression baselines ([7 Experiments](7_experiments.md)) |
| Viewing and recording | Rerun world, sensor views, thrust plots, JSONL logs, `.rrd` replay |
| Worlds | JSON scene generation, paused Gazebo editing, saved SDF reuse |

**Not implemented:** the full Gazebo session API (algorithms run serially
on the mission thread, not in worker threads), obstacle avoidance,
perception, planning, and RL. `perception`, the three `planning` packages,
and `rl` contain only package docstrings. See [6 Roadmap](6_roadmap.md).

Flight controllers use **explicit simulator truth** as feedback unless
`algorithms.control_state_source` is `estimate`. Estimators never receive
truth; it is used only to synthesize measurements and to score runs.

## Code organization

```text
src/dronelab/
├── cli/                    # Argument parsing: `dronelab` (gazebo.py), `dronelab-batch`
│                           #   (batch.py), `dronelab-magcal` (magcal.py)
├── core/                   # Small shared types: frames, gravity, vehicle properties,
│                           #   measurement samples, rotor commands
├── sensors/                # Synthetic models, Gazebo decoding, geodesy, in-process
│                           #   streams, GPS dropout, accelerometer vibration,
│                           #   IMU/GPS error models (imu_errors.py, gps_errors.py),
│                           #   faults (imu_faults.py, multipath.py, zones.py),
│                           #   barometer conversion and errors (barometer.py),
│                           #   magnetometer errors, synthetic optical flow,
│                           #   sensor bus (bus.py), IMU imperfections, GPS receiver
│                           #   (velocity, status, RTK, heading), ESC telemetry,
│                           #   magnetic anomalies, lidar returns
├── estimation/
│   ├── interface.py        # StateEstimator plug-in contract, Innovation
│   ├── magnetometer_calibration.py  # Hard/soft iron and interference fits
│   ├── dead_reckoning.py   # Retained IMU dead-reckoning baseline
│   └── gps_imu_ekf/
│       ├── ekf.py          # Numerical prediction and correction
│       ├── node.py         # Timestamp/frame adaptation and EKF state channel
│       └── plugin.py       # The EKF as an estimator plug-in
├── evaluation/             # Scoring against truth: estimation (NEES/NIS), control,
│                           #   truth interpolation
├── control/
│   ├── interface.py        # RotorController plug-in contract, ControlState
│   ├── references.py       # Mission reference trajectories (analytic derivatives)
│   ├── waypoint.py         # Waypoint → body-velocity feedback (native mode)
│   ├── manual.py           # Position and attitude loops → four rotor thrusts
│   ├── timing.py           # Fixed-rate ticks, state/command latency buffers
│   └── drone.py            # Exclusive thrust handle, receipts, publication worker
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
│   └── gazebo/
│       ├── config.py       # Drone JSON loading, defaults, validation
│       ├── properties.py   # Resolved vehicle properties
│       ├── world.py        # Scene loading and world composition
│       ├── vehicle.py      # Drone, motor, controller, and sensor SDF
│       ├── actuators.py    # Thrust → motor-speed transport adapter
│       ├── wind.py         # Seeded wind targets and native wind commands
│       ├── sdf.py          # XML-building helpers
│       ├── environment.py  # Child-process library environment (Snap fix)
│       └── messages.py     # Gazebo timestamp/vector adapters
├── experiments/
│   ├── gazebo.py           # Compose and run a Gazebo mission
│   ├── rotor_loop.py       # Fixed-rate rotor-thrust control loop with latency
│   ├── algorithms.py       # Plug-in loading, context, runner, estimate-driven state
│   ├── mission.py          # Mission clock and reference sampling
│   ├── run_evaluation.py   # Per-run metrics (metrics.json)
│   ├── batch.py            # Seed/parameter sweeps, aggregation, baselines
│   ├── magnetometer_calibration.py  # Automated calibration run (dronelab-magcal)
│   └── live_ekf.py         # Bounded live IMU/GPS ordering and EKF composition
└── visualization/
    ├── gazebo_rerun.py     # Gazebo-message-to-Rerun recording adapter
    ├── keyboard.py         # Tk keyboard window (key/focus state only)
    ├── live.py             # Matplotlib live drone view (retained)
    └── plotting.py         # Static flight plots (retained)

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
└── experiments/            #   batch experiment files (dronelab-batch)
baselines/                  # Stored batch metric baselines for regression checks
worlds/                     # Authored JSON scenes and saved SDF worlds
docs/                       # These numbered documents
tests/unit/<package>/       # Unit tests grouped by package
tests/integration/          # Multi-component and opt-in Gazebo/GUI tests
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
  These are simulator features, not DroneLab Python algorithms.
- The `gazebo_rerun` adapter deliberately understands Gazebo messages but
  does not control, plan, or estimate.
- There is no shared simulator interface for Gazebo and RotorPy: one is
  asynchronous transport, the other explicit stepping. Add shared types to
  `core` only when real integrations require them.
- Optional imports (Gazebo bindings, Rerun, Tk) are deferred, so world
  generation and numerical tests work without them.

## Adding an algorithm

Write an estimator or controller class following the plug-in contracts in
[7 Experiments](7_experiments.md#algorithm-plug-ins-algorithms), put it in
`estimation/<name>.py` or `control/<name>.py` (or any importable package)
with tests in `tests/unit/<package>/`, and select it with
`algorithms.estimator`/`algorithms.controller` as `package.module:Class`.
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
  (0, 0, -9.81) m/s². Body frame: +X forward, +Y left, +Z up.
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

## History

`dronelab`, `dronelab-gazebo`, and `python -m dronelab` all invoke `cli.gazebo`.
The former fixed-step flight experiment (`experiments.flight`,
`experiments.artifacts`, `cli.flight`, `--backend`), its NPZ/CSV artifacts,
and the synthetic EKF demo were retired. Retained backend, plotting,
synthetic-sensor, and dead-reckoning modules remain as tested library code
without a maintained CLI.
