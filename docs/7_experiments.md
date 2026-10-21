# 7. Algorithms, missions, metrics, and batch experiments

This document covers the test harness for your own estimation and control
algorithms: how they plug in, what they receive, what they fly, how they are
scored, and how to run sweeps with regression baselines. Simulator models
are in [2 Simulation](2_simulation.md) and [3 Configuration](3_configuration.md);
the built-in controller and EKF are in [4 Control](4_control.md) and
[5 Estimation](5_estimation.md).

```text
Gazebo sensors ─▶ error/fault models ─▶ sensor bus ─▶ estimator ─┐
                                        (delay, offsets)          │ estimate
Gazebo odometry (truth) ─┬──────────────────────────────▶ control state ─▶ controller ─▶ rotor thrusts
                         │                  (truth or estimate)        ▲
                         └─▶ evaluation (metrics.json)    mission reference
```

Truth is used in three places only: to synthesize measurements (inside the
sensor models), to drive simulator physics, and to score runs afterwards
(`evaluation/`). It never reaches the bus or an algorithm, and the
controller only receives it when `control_state_source` is `truth`.

## Algorithm plug-ins (`algorithms`)

| Key | Default | Meaning |
| --- | --- | --- |
| `estimator` | "" | "" (none), `gps_imu_ekf` (built-in), or `package.module:Class` |
| `estimator_options` | {} | Passed to the estimator; built-in default: the `ekf` section |
| `controller` | "" | "" or `manual` (built-in), or `package.module:Class` |
| `controller_options` | {} | Passed to the controller; built-in default: the `controller` section |
| `control_state_source` | truth | `truth` or `estimate` (needs an estimator) |
| `max_estimate_age_s` | 0.1 | Older estimate at a control tick stops the mission (`StaleEstimate`) |
| `evaluation_rate_hz` | 50 | Rate at which estimates are scored |
| `evaluation_start_s` | 2 | Estimates in the first seconds are not scored (convergence) |
| `record_bus` | off | Record bus deliveries and truth for offline replay: `off`, `light` (all streams but camera and lidar), `all` ([Offline replay](#offline-replay)) |

Plug-ins are constructed as `cls(context, options)`.
`context` is an `AlgorithmContext` (`core/algorithms.py`):
- the local navigation origin (`gps_reference`),
- the **nominal** vehicle properties (`vehicle`), as designed, not the
  randomized truth,
- one `SensorDescription` per enabled stream: frame ID, nominal rate, and
  mount position and quaternion, as from a calibration file.

It deliberately excludes simulator truth and the configured error models.

`context.log(name, sim_time_ns, values, labels=None)` is the plug-ins'
**debug channel** for their own time series (innovations, internal states,
gains). Entries go to `algorithm-debug.jsonl` (`name`, `sim_time_ns`,
`values`, `labels`) and to Rerun under `algorithms/<name>` in Gazebo runs,
and to the replay output offline. Stamp each entry with the sample time it
describes. Names use letters, digits, `_ . / -`; values must be finite;
invalid entries raise `ValueError` at the call. The file is only created
when something is logged; `summary.json.algorithm_debug_entries` counts
entries. The built-in EKF logs `ekf/gps` (NIS, accepted) and `ekf/bias`
(accelerometer and gyro bias estimates) at each GPS update.
Option objects are passed through without validation, so plug-ins must
check them.

**Estimator** (`estimation/interface.py`, `StateEstimator`):

```python
class MyEstimator:
    inputs = ("imu", "gps", "baro")              # streams to receive

    def __init__(self, context, options): ...
    def on_sample(self, kind, sample): ...        # one bus Sample, in delivery order
    def latest(self): ...                         # Sample[EstimatedState] or None
    def drain_innovations(self): return []        # Innovation(name, nis, dof, accepted)
    def summary(self): return {}
```

Samples arrive serially on the mission thread, which is deterministic in
lockstep. One stream is always in time order; different streams can
interleave out of time order when their latencies differ (see the sensor
bus below). `EstimatedState` holds position, velocity, attitude, both
biases, and a 15×15 covariance in the error order position, velocity,
attitude (right, body), accelerometer bias, gyro bias. Metrics use
covariance blocks in that order.

**Controller** (`control/interface.py`, `RotorController`):

```python
class MyController:
    saturated = False
    def __init__(self, context, options): ...
    @property
    def setpoint_m(self): ...                     # position setpoint or None
    def update_position(self, state, request, dt_s): ...   # position-loop rate
    def update_attitude(self, state): ...         # control rate → rotor thrusts (N)
    def allow_landing(self): ...
```

- `state` is a `ControlState`: position, world velocity, xyzw attitude,
  body rates, and `source` ("truth" or "estimate").
- `request` is a `FlightCommand` (keyboard sticks, take-off climb, failsafe)
  or a `ReferencePoint` (mission).
- Rates and latencies come from the `control` section
  ([4 Control](4_control.md#controller-timing-and-latency)).
- `ControlState.airspeed_mps` is the true body-x airspeed (truth control) or
  the latest delivered pitot reading (estimate control).
- Controllers for vehicles with `fixed_wing` may define `surface_commands()`
  returning aileron, elevator, and rudder deflections in radians. It is read
  after each attitude update and sent to the servos. The quadplane baseline
  autopilot ([4 Control](4_control.md#quadplane-baseline-autopilot)) is the
  example.

The built-ins (`ManualRotorController`, `GpsImuEkfEstimator`) follow
exactly these contracts and are the reference examples.

**Estimate-driven control.** With `control_state_source: estimate`, the
controller state comes from the estimator:
- **Position, velocity and attitude:** from the latest estimate.
- **Body rates:** the latest delivered gyro sample, rotated by the declared
  IMU mount, minus the estimated gyro bias.

Before the first estimate, ticks are skipped (at most
`max_estimate_age_s`). After that, an estimate older than the limit at a
control tick ends the run and stops the motors, with
`summary.json.algorithms.max_estimate_age_ms` recording the worst age.
The mission start position and the low-battery failsafe also use the
estimate in this mode, as on a real vehicle.
`configs/algorithms/ekf_observer.json` scores the EKF while flying on truth;
`ekf_in_the_loop.json` flies on it. Both declare the launch point as the
filter's prior.

`--ekf` keeps the older live-EKF path ([5 Estimation](5_estimation.md)); it
cannot be combined with `algorithms.estimator`.

### Writing a plug-in: templates and the contract test kit

Start from a template; both are short and commented, and pass the kit:

| Template | Selects with | What it does |
| --- | --- | --- |
| `estimation/template.py` | `dronelab.estimation.template:TemplateEstimator` | Complementary filter: gyro-integrated attitude pulled towards the accelerometer's gravity direction, GPS-blended position and velocity; fixed reported covariance (NEES not meaningful); logs `template/gps_error` |
| `control/template.py` | `dronelab.control.template:TemplateController` | Cascaded PD: position → tilt-limited acceleration → desired attitude (geometric attitude error, Lee et al. 2010) → torque → pseudo-inverse allocation over the lift rotors |

Both reject unknown option keys. Verified in lockstep on the training
square (two laps, 45 s): the template controller tracked with 0.148 m
RMSE (built-in 0.101 m), peak tilt 8.7°, health `ok`; the template
estimator, as observer on the same flight, had 1.4 cm position and 1.4°
attitude RMSE.

`dronelab.testing.plugins` checks a plug-in's contract without Gazebo, in
a unit test:

```python
from dronelab.testing.plugins import check_controller, check_estimator

def test_my_filter():
    report = check_estimator(MyFilter, {"gain": 2.0})
    assert report.final_position_error_m < 0.5

def test_my_controller():
    check_controller(MyController, {})
```

- **`check_estimator`** feeds a seeded synthetic hover (IMU 200 Hz, GPS
  10 Hz, small white noise) and checks: known `inputs`; estimates that are
  `Sample[EstimatedState]`, never ahead of the newest sample and never
  going back in time; finite values, unit quaternion, symmetric positive
  semi-definite covariance; valid innovations and `context.log` entries;
  a JSON-serializable `summary()`; identical results on a second run.
- **`check_controller`** runs the hover reference on the nominal vehicle
  and checks: one finite, non-negative thrust per rotor, within the limit
  unless `saturated` is set; total thrust within 10 % of the weight;
  a valid `setpoint_m`; `allow_landing()`; identical results on a rerun.

Failures raise `PluginContractError` (an `AssertionError`) with the reason;
exceptions from the plug-in itself propagate unchanged. Passing means the
plug-in fits the harness, not that it is good; judge that with Gazebo runs,
`dronelab-replay`, and batches.

## Sensor bus and delivery timing (`sensor_timing`)

`sensors/bus.py` turns each processed measurement into an immutable
`Sample` on `sensors/<kind>`:

| Stream | Payload | Frame ID |
| --- | --- | --- |
| `imu` | `ImuReading` (specific force, angular rate) | `drone/imu` |
| `gps` | `GpsReading` (lat, lon, alt, ENU velocity, optional `GpsStatus`) | `wgs84` |
| `gps_heading` | `GpsHeadingReading` | `drone/gps_heading` |
| `baro` | `BaroReading` (pressure, ISA altitude) | `drone/baro` |
| `magnetometer` | `MagnetometerReading` (µT) | `drone/magnetometer` |
| `rangefinder` | `RangeReading` (NaN when out of range) | `drone/rangefinder` |
| `optical_flow` | `OpticalFlowReading` | `drone/optical_flow` |
| `battery_monitor` | `BatteryReading` | `drone/battery` |
| `esc` | `EscTelemetry` (rotor speeds) | `drone/esc` |
| `lidar` | `LidarPoints` (body-axis points, reflectivity) | `drone/body` |
| `camera` | `CameraFrame` (RGB) | `drone/camera` |

Measurements enter after the error and fault models; a dropped GPS fix never
enters. The bus is created when an estimator is configured.

Per stream, `sensor_timing.<kind>` has:

| Key | Default | Meaning |
| --- | --- | --- |
| `timestamp_offset_s` | 0 | Sensor clock error: `sim_time_ns` = measurement time + offset |
| `latency_s` | 0 | Delivery delay after measurement |
| `jitter_s` | 0 | Extra half-normal delay, std of the underlying normal |

Two more keys sit on `sensor_timing` itself: `seed` and `reorder_window_s`
(default 0.004 s). A stream stays first-in first-out. Different latencies
make streams arrive out of order relative to each other; `late_samples` in
`summary.json.sensor_bus` counts them. Gazebo delivers topics on separate
transport threads, so the window holds samples briefly and releases them by
delivery time, then timestamp, then a fixed priority (IMU first). The window
delays delivery only, never timestamps. Camera frames carry their capture
time and arrive after `sensors.camera.latency_s`.

## Reference missions (`mission`)

`control/references.py` gives analytic position, velocity, acceleration,
yaw, and yaw rate. The built-in controller uses the acceleration as
feed-forward: `a = Kp(p_ref − p) + Kv(v_ref − v) + a_ref`, clipped as
before. Missions need rotor-thrust mode without keyboard. Mission time
starts at the first state, or once take-off completes after a ground start.

| Type | Keys used |
| --- | --- |
| `hold` | `center_m`, `yaw_deg` |
| `step` | `center_m`, `step_m`, `step_time_s` |
| `waypoints` | `waypoints_m`, `speed_mps`, `dwell_s`, `laps`, `yaw_mode` (`fixed`/`path`), `yaw_rate_dps` |
| `circle` | `center_m`, `radius_m`, `speed_mps`, `yaw_mode` |
| `figure_eight` | `center_m`, `radius_m` (half-width), `speed_mps` (peak) |
| `chirp` | `center_m`, `axis`, `amplitude_m`, `frequency_start_hz`, `frequency_end_hz`, `duration_s` |
| `doublet` | `center_m`, `axis`, `amplitude_m`, `pulse_s`, `step_time_s` |

How each type behaves:
- **Waypoints** fly stop-and-go minimum-jerk segments from the mission
  start position. Each segment lasts `1.875·d / speed_mps`, so the peak
  speed equals `speed_mps`. In `path` yaw mode the vehicle first turns to
  face each segment.
- **Steps and doublets** are discontinuous on purpose, to measure step
  responses.
- **Chirps** sweep frequency linearly, for closed-loop frequency-response
  and system-identification runs.
- **When a mission ends**, the vehicle holds the final reference.

Example layers are in `configs/missions/`: `step_x`, `square` (two laps in
the training scene), `circle`, `figure_eight`, `chirp_x`, `doublet_z`, and
`circuit_lap` (for `worlds/circuit.json`; the lap takes 107 s and was flown in lockstep without contacts, 0.31 m tracking RMSE).

## Metrics (`metrics.json`)

Every run writes `metrics.json`, also copied to `summary.json.metrics`.

**Estimation** (`evaluation/estimation.py`). Truth is interpolated to each
estimate's timestamp (linear for position and velocity, slerp for attitude)
before comparing:
- **Errors:** RMSE, mean, 95th percentile, maximum, and final value of the
  position (also horizontal and vertical), velocity, attitude, and yaw
  errors, plus bias errors against the effective bias of the IMU readings
  the estimator receives (`evaluation/imu_bias.py`). That bias is the
  delivered reading minus the error-free Gazebo reading, in body axes, so
  raw or calibrated depending on the active calibration, averaged over
  1 s. It covers every error source that does not average out (offsets,
  drift, temperature, scale and mount errors at the current motion), and it
  stays correct whether or not a calibration is applied.
- **NEES consistency:** NEES = eᵀP⁻¹e over the 15 states, with mean and
  fraction inside the two-sided 95 % χ² interval. A consistent filter has a
  mean equal to the degrees of freedom and about 95 % inside.
- **NIS:** the same statistics for each estimator-reported innovation
  (GPS for the built-in EKF).

**Control** (`evaluation/control.py`):
- **Tracking error** against the reference at the same time: position,
  horizontal, vertical, velocity, and yaw.
- **Maximum tilt.**
- **Effort:** mean thrust-to-weight, RMS thrust deviation from weight, and
  mean rotor-thrust slew in N/s.
- **Saturated fraction** of commands.
- **Step response** for `step` and `doublet` missions: 10–90 % rise time,
  overshoot, settling time (to within the larger of 2 % and 0.05 m), and
  steady-state error over the last 20 % of the window.

Rerun shows the per-sample errors under `evaluation/estimation` and
`evaluation/tracking`.

Verified in lockstep on the training square (two laps at 1.5 m/s, built-in
controller and EKF, ideal sensors): tracking RMSE was 0.102 m flying on
truth and 0.103 m flying on the estimate, with 4 mm estimation RMSE.

With `configs/sensors/sensor_errors.json` the estimate drifts. That layer
has a 1 m horizontal and 2 m vertical GPS error correlated over 60 s, while
the EKF assumes white GPS noise. Results:
- **Estimation:** 2.0 m position RMSE and NEES 147, an inconsistent filter.
- **Control:** flying on the estimate gave 2.1 m true tracking error, and
  the vehicle touched the ground at 2 s while following a high altitude
  estimate.

That is the estimator's limitation, which this harness is meant to expose.

## Run health verdict (`health`)

Every run also gets a truth-based health verdict in `summary.json.health`
(`evaluation/health.py`), printed at the end of the run. It answers "did
the simulated flight go physically wrong?", so that a crash is never read
as a slightly worse metric. Like the metrics it is evaluation only and
never affects the flight.

`verdict` is `failed` if any finding failed, `warning` if any warned, else
`ok`. Each entry in `findings` has `kind`, `severity`, `time_s` (first
occurrence, seconds since the first odometry sample; null for run-level
findings), `count` (occurrences, e.g. contact samples), `value` (the worst
value seen), `subject` (e.g. the obstacle's model), and a `message`.

| Kind | Severity | Condition |
| --- | --- | --- |
| `non_finite` | failed | NaN or infinity in truth, a thrust command, or an estimate |
| `flipped` | failed | tilt above `flip_tilt_deg` for at least `flip_duration_s` |
| `hard_impact` | failed | a contact while the true speed in the preceding `impact_window_s` exceeded `hard_impact_speed_mps` |
| `obstacle_contact` | failed | contact with any model other than `ground` or `terrain` (the attached payload is ignored) |
| `below_ground` | failed | body centre more than `below_ground_margin_m` below the ground or terrain surface |
| `no_liftoff` | failed | commanded thrust ≥ `no_liftoff_thrust_ratio` × weight for `no_liftoff_s` while the vehicle neither rose 5 cm nor moved faster than 0.2 m/s (commands not reaching the motors, a pinned vehicle) |
| `incomplete` | failed | the run ended on an error before its duration (e.g. a stale estimate) |
| `interrupted` | warning | Ctrl-C (closing the keyboard window is a normal end) |
| `ground_contact` | warning | a touchdown after at least `touchdown_gap_s` without contact; resting on the ground from the start is not one |
| `tracking_lost` | warning | mission position error above `tracking_lost_m` for `tracking_lost_s`; a warning on purpose, so poor tracking stays in batch aggregates as a result |
| `left_ground_area` | warning | flew past the ground plane's edge (also `left_ground_area_s`) |
| `saturated` | warning | more than `saturated_fraction` of non-warm-up commands saturated |
| `dropped_messages` | warning | more than `max_dropped_messages` messages dropped (ingress queues plus command events) |
| `ordering_timeouts` | warning | lockstep periods processed before every scheduled sample arrived (not deterministic; [2 Simulation](2_simulation.md#lockstep-simulation)) |

The thresholds are the drone config's `health` section, recorded in the
verdict:

| Key | Default | Units |
| --- | --- | --- |
| `flip_tilt_deg` | 90 | deg, in (0, 180] |
| `flip_duration_s` | 0.5 | s |
| `hard_impact_speed_mps` | 2.0 | m/s |
| `impact_window_s` | 0.1 | s |
| `below_ground_margin_m` | 0.1 | m |
| `touchdown_gap_s` | 0.5 | s |
| `saturated_fraction` | 0.5 | fraction, in (0, 1] |
| `max_dropped_messages` | 0 | count |
| `no_liftoff_s` | 1.0 | s |
| `no_liftoff_thrust_ratio` | 1.2 | × true weight, > 1 |
| `tracking_lost_m` | 2.0 | m |
| `tracking_lost_s` | 3.0 | s |

Contacts come from the drone body's contact sensor, so they cover the
body box only. The impact speed looks back over a window because Gazebo can
deliver a contact after the post-impact odometry. A deliberate landing
gives a `ground_contact` warning; aerobatic experiments should raise
`flip_tilt_deg` or `flip_duration_s`.

Verified in lockstep: a hover was `ok`; with the battery started at 1 %
charge from 2 m, the vehicle fell and the run `failed` with a 6.0 m/s
ground impact at 3.5 s.

## Offline replay

Estimators can be developed and compared without Gazebo. A run with
`algorithms.record_bus` set to `light` (`configs/algorithms/record_replay.json`)
or `all` writes `replay.jsonl` (`experiments/replay_log.py`):

- a header with the resolved settings, the true weight, and the
  `AlgorithmContext` the plug-ins were given;
- every sample the sensor bus delivered, in delivery order, interleaved
  with the truth samples (and the effective IMU bias the metrics score
  against) exactly as the live run processed them.

Recording creates the sensor bus even without an estimator, so a flight
can be recorded once and used for any number of estimators. `light` keeps
logs small (about 190 kB per simulated second with IMU at 200 Hz and GPS at
10 Hz); `all` adds camera frames and lidar points.

```bash
dronelab ... --drone configs/algorithms/record_replay.json --output outputs/rec
dronelab-replay outputs/rec --output outputs/rec-ekf                      # estimator as flown
dronelab-replay outputs/rec --output outputs/rec-mine --estimator mypkg.filters:MyFilter \
  --set 'algorithms.estimator_options={"gain": 2}'
dronelab-replay outputs/rec --output outputs/rec-tuned --set 'ekf.gps_std_m=[1,1,2]'
```

Repeat `--estimator` to compare several on the same recording: each
replays into `<output>/<class name>`, and `comparison.md` lists their
estimation errors (lowest marked) and NEES. `--set` applies to all of
them, so give estimator-specific options only when comparing one at a
time.

`dronelab-replay` (`experiments/replay.py`) builds the estimator from the
recorded settings plus `--set` overrides (dotted keys, JSON values,
validated like any config layer) and the recorded context, feeds it the
samples in order, and scores it against the recorded truth with the same
metrics as a live run. It writes `estimates.jsonl`, `metrics.json`
(`estimation` section), `algorithm-debug.jsonl`, `replay-settings.json`,
and `replay-summary.json` into a fresh directory. The estimator never sees
truth.

Replay is **open loop**: the recorded flight does not change, so it suits
estimators, not controllers, and flying on a different estimate still
needs the simulator. The context is as recorded (sensor overrides do not
change it). The `--ekf` live observer does not use the bus and is not
recorded. Verified in lockstep: replaying the built-in EKF on a 20 s
square with sensor errors reproduced the live `estimates.jsonl`,
`algorithm-debug.jsonl`, and estimation metrics exactly, in 3.4 s.

## Batch experiments and baselines

```bash
dronelab-batch configs/experiments/step_wind_seeds.json --output outputs/batch --dry-run
dronelab-batch configs/experiments/step_wind_seeds.json --output outputs/batch --save-baseline
dronelab-batch configs/experiments/step_wind_seeds.json --output outputs/batch-2 --check   # exit 1 on regression or failed run
```

An experiment file (`experiments/batch.py` documents every key) names:
- the scene, `drone` layers, control mode, and duration,
- `seeds`, and `parameters` (dotted config keys with lists of values),
- fixed `set` overrides,
- the `metrics` to collect (dotted paths into `metrics.json`, or
  `summary.<path>`),
- the `baseline` file and `tolerances`.

How runs are built and executed:
- Runs are every parameter combination times every seed.
- Each run's overrides become an extra config layer, so they are validated
  like any configuration.
- A seed sets every `seed`/`random_seed` in the resolved configuration to
  seed + CRC32(key path), so the random streams stay independent and adding
  new seed keys never changes existing ones (stable baselines).
- Runs are separate `dronelab` processes, `parallel` at a time, each with a
  fresh output directory (non-empty ones are refused) and a private Gazebo
  transport partition.

`comparison.md` (also printed) puts the groups side by side, one row per
metric with `mean ± std (n)`, marking the best group where the metric's
tolerance `direction` is `increase` (lowest best) or `decrease` (highest
best; ties are not marked), and listing excluded runs. Sweeping
`algorithms.estimator` or `algorithms.controller` as a parameter therefore
compares algorithms: `configs/experiments/controllers_square.json` flies
the built-in and template controllers on three randomized vehicles each
(52 s wall). Result: tracking RMSE 0.67 ± 0.08 m (built-in) against
1.30 ± 0.20 m (template), peak tilt 8.5° against 9.8°, while the template
commands smoother thrust (rotor slew 0.56 against 0.86 N/s).

`batch-results.json` holds every run, with its health verdict and
findings, and, per parameter group, the count, mean, standard deviation,
minimum, maximum, and values of each metric.

Runs whose health verdict is `failed` are left out of the aggregates and
listed in their group's `excluded_runs`: a crashed run's numbers describe
the crash, not the algorithm. Warnings are kept. Runs from before the
verdict existed have none and are included.

A baseline stores the group means. `--check` flags any metric whose mean
moved more than `absolute + relative·|baseline|`. The `direction` can be
`both` (default; deliberate improvements then need a new baseline),
`increase` (higher is worse), or `decrease`. Missing metrics count as
regressions, and `--check` also fails when any run's health verdict is
`failed` (listed as `health_failed_runs`).

Verified with `step_wind_seeds` (2 wind conditions × 3 seeds, 3 in parallel,
21 s wall time):
- **Repeatability:** a repeated batch passed the check exactly, since
  lockstep is deterministic.
- **Detection:** assuming 3× the GPS noise in the EKF failed the check on
  position error, attitude error, NEES, and NIS (exit code 1).

The same batch shows that a steady 3 m/s wind leaves the built-in
controller (no integral action) a 0.58 m steady-state error.
