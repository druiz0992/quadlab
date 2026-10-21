# 5. State estimation

DroneLab's estimator is a GPS/IMU error-state EKF. It runs live from Gazebo
sensor messages with `--ekf`, as an **observer only**: flight control keeps
using truth. The estimator is initialized from declared priors and never
receives simulator truth, true biases, wind, or fault-model information.

```text
Gazebo IMU/GPS → decode → (dropout, vibration) → typed channels
      → live_ekf.py (bounded timestamp ordering) → GpsImuEkfNode → GpsImuEkf
                                                        ↓
                                              state channel → Rerun / estimates.jsonl
```

| Module | Responsibility |
| --- | --- |
| `estimation/gps_imu_ekf/ekf.py` | Numerical prediction and correction (`GpsImuEkf`) |
| `estimation/gps_imu_ekf/node.py` | Frame/timestamp adaptation, WGS84 reference, state output (`GpsImuEkfNode`) |
| `experiments/live_ekf.py` | Gazebo decoding to channels, bounded reordering, watchdogs (`LiveEkf`) |
| `core/measurements.py` | Immutable `Sample`, `ImuReading`, `GpsReading`, `EstimatedState` |
| `sensors/streams.py` | Bounded in-process `Publisher`/`Channel`/`Subscription` fanout |
| `sensors/geodesy.py` | WGS84 ↔ ECEF ↔ local ENU conversion |
| `estimation/dead_reckoning.py` | Retained IMU dead-reckoning baseline (truth-initialized, explicit `dt`; not used live) |

None of the numerical modules depend on Gazebo transport or Rerun.

## Numerical filter

15-dimensional error state:

| State | Convention |
| --- | --- |
| Position, velocity | Body-origin, in a declared local ENU frame |
| Attitude | xyzw quaternion rotating body vectors into local ENU |
| Accelerometer bias | Body axes, m/s² |
| Gyroscope bias | Body axes, rad/s |
| Error covariance | 15×15, order: position, velocity, small right attitude error, accel bias, gyro bias |

- **Prediction:** specific force plus world gravity (0, 0, −9.81), integrated
  with body angular rate. IMU readings are held from the interval start; the
  next IMU timestamp ends the interval. dt comes from explicit timestamps.
- **Correction:** GPS gives local ENU antenna position with an explicit 3×3
  covariance and receiver lever arm. Uses a Mahalanobis innovation gate,
  Joseph-form covariance update, and attitude-error reset.
- **Noise semantics:** `accel_std_mps2` and `gyro_std_rad_s` are isotropic
  standard deviations **per IMU sample**. `accel_bias_walk` is in
  (m/s²)/√s and `gyro_bias_walk` in (rad/s)/√s; bias-walk increments have
  variance `walk² · dt` at interval ends.
- **Initialization:** position, velocity, normalized attitude, bias priors,
  and a positive-definite covariance, all supplied explicitly. By default
  (`initialize_position_from_gps`) the first time-aligned GPS fix then
  replaces the position prior: position = antenna fix minus the lever arm,
  position covariance = that fix's covariance, cross-covariances cleared.
  It counts as an initialization, not a correction (`gps_updates` and the
  innovations start with the second fix).
- **GPS covariance:** by default (`use_receiver_gps_accuracy`) each fix uses
  the receiver's own 1-σ estimates from its `GpsStatus` (horizontal for
  east/north, vertical for up, uncorrelated); fixes without valid ones fall
  back to `gps_std_m`. The receiver's reports follow the configured GPS
  error ([3 Configuration](3_configuration.md#gps-receiver-velocity-status-rtk-heading)),
  e.g. 1.04 m horizontal and 2.06 m vertical for `sensor_errors.json`. Its
  correlated part (60 s) still violates the filter's white-noise
  assumption: on `step_wind_seeds` mean NEES is about 150 (consistent
  ≈ 15), attitude RMSE 2.36°, GPS NIS 0.25.

The right attitude error follows the error-state approach of
[Solà, Quaternion kinematics for the error-state Kalman filter](https://arxiv.org/abs/1711.02508).
This is an educational, simple discrete held-input model, not a reproduction
of every model in that reference or a calibrated INS.

**Limits:** stationary single-antenna GPS plus IMU does not fully constrain
heading and all bias/attitude combinations. Delayed-update rewind and
repropagation are not implemented.

## Node and frames

- GPS latitude/longitude/altitude convert through WGS84 ECEF to local ENU.
  The reference origin and altitude datum are explicit; altitude 650 m is not
  local z = 650. The ENU frame must be aligned with the simulation world.
- The node rotates an explicitly mounted IMU into body axes. Nonzero IMU
  position offsets are rejected (no rotational lever-arm compensation).
  GPS receiver offsets are in the measurement model. Camera, lidar,
  barometer, magnetometer, rangefinder, and optical flow are not inputs yet.
- The first IMU sample sets the prior's time. IMU intervals must be positive
  and at most 0.1 s. The node validates frames, epochs, stream identity, and
  sequence continuity, serializes handlers, and starts no threads.
- Output: `node.outputs["state"]`, also `node.estimates` and
  `LiveEkf.estimates`, a read-only channel of `Sample[EstimatedState]`.
  Consumers subscribe before dispatch, e.g.
  `node.estimates.subscribe(capacity=128)`. Channels have bounded queues,
  overflow policies (`error`, `drop_oldest`), timeouts, and closure.
  Prediction and correction outputs may share timestamps with distinct
  sequence numbers.

## Live operation

`experiments/live_ekf.py` decodes Gazebo measurements on the mission thread
into typed channels. A bounded adapter waits until both streams reach a
timestamp, then dispatches buffered samples in source-time order (IMU before
GPS at equal times). This tolerates asynchronous arrival and different rates
without altering timestamps, at the cost of roughly one GPS interval of
latency. No EKF math runs on a transport callback.

A GPS time inside an IMU interval is handled by **segmented propagation**:
propagate with the held reading to the GPS time, correct, then continue to
the next IMU sample. Process noise is applied per segment, without modelling
correlated held-sample noise across the split; an educational approximation.
GPS fixes that cannot be placed are counted as `rejected_gps_timing`; spatial
outliers are gated separately.

Failures are visible and run owned-simulator cleanup: duplicate or reversed
per-source timestamps, known IMU/GPS ingress drops, buffer overflow,
excessive lag, IMU intervals over 0.1 s, and stalled streams. Flight
watchdogs stay independent, so a pause can end the mission. At cleanup the
unmatched tail after the last common timestamp is discarded rather than
extrapolated, so estimates may end about one GPS interval before truth. A
duration-completed mission without an accepted GPS correction is an error.

**Requirements:** enabled IMU (≥ 20 Hz) and GPS, body-origin IMU (rotated
mounts allowed), generated ENU/WGS84 world with zero heading. The GPS origin
is read from the generated SDF's spherical coordinates, never from truth or
the first fix. `--ekf` is rejected with `--world` and `--edit`.

GPS dropouts still advance the ordering watermark through raw arrival timing,
without exposing coordinates, so IMU prediction continues and corrections
resume on recovery with contiguous sequences. Accelerometer vibration and the
seeded IMU/GPS error models reach the EKF only as the resulting readings. See
[3 Configuration](3_configuration.md#gps-dropouts-and-wind) and
[IMU and GPS error models](3_configuration.md#imu-and-gps-error-models).

**Example behaviour with sensor errors.** A 30 s headless rotor-thrust hover
with `configs/sensors/sensor_errors.json` showed:

- The position estimate follows the correlated GPS error: about 1 m north and
  −1.2 m vertical offset, which a white-noise GPS model cannot remove.
- Roll/pitch gyro biases were recovered to within about 2e-4 rad/s.
- Yaw gyro bias and horizontal accelerometer biases stayed near their priors
  with large reported uncertainty, because a level hover cannot reveal them.

This is one seeded run, not a statistical evaluation.

**Example behaviour with faults.** Waypoint flight through circuit gate G1
(`--scene worlds/circuit.json --target 11 0 1.8 --duration 28`). A fault-free
baseline accepted all 281 fixes with error ≤ 1 cm.

- `configs/faults/imu_faults.json`: spikes and the stuck axis raised position error
  to at most 10 cm, with every GPS fix accepted. The 0.02 rad/s yaw gyro bias
  step went **undetected**: heading error reached 3° at 18 s and 17° at 24 s
  while the drone hovered, because a hover cannot reveal heading.
- `configs/faults/gps_faults.json`: the filter **diverged** (hundreds of metres).
  The gate first rejected a 4.6 m multipath jump; position uncertainty then
  grew until the jump was accepted, tilting the attitude estimate about 9° to
  explain it. When the episode ended, true fixes were rejected as outliers
  for the rest of the run (188 rejected, 64 accepted) and the tilt error
  integrated into drift. This "gate lock-out" is a known EKF failure mode.

These runs expose missing robustness rather than test a recovery path: the
filter has no fault detection and no re-initialization after repeated GPS
rejections.

## Configuration

The `ekf` section of the drone JSON (all defaults listed in
`configs/vehicles/quad_x.json`) is independent of simulated sensor errors:

```json
{
  "ekf": {
    "initial_position_m": [0, 0, 0],
    "initial_velocity_mps": [0, 0, 0],
    "initial_rotation_deg": [0, 0, 0],
    "initial_position_std_m": 5.0,
    "gps_std_m": [0.5, 0.5, 0.8],
    "accel_std_mps2": 0.08,
    "gyro_std_rad_s": 0.003
  }
}
```

- Position/velocity priors are local ENU at the world's geodetic origin;
  attitude is body-to-ENU roll/pitch/yaw in degrees; bias priors are body
  axes. The default position prior is zero, not the true spawn; with
  `initialize_position_from_gps` (default) the first fix replaces it.
- Initial uncertainties are scalar standard deviations per axis (attitude in
  degrees), squared into the covariance.
- `gps_std_m` gives three positive ENU standard deviations (used for fixes
  without valid receiver accuracies, or for all fixes with
  `use_receiver_gps_accuracy: false`); `gps_gate_squared` is the squared
  Mahalanobis threshold.
- *Migration (2026-10-22):* set `initialize_position_from_gps` and
  `use_receiver_gps_accuracy` to false to reproduce earlier EKF results.
  On `step_wind_seeds` the change lowered attitude RMSE from 2.23° to
  2.05° and mean NEES from 137 to 110; GPS NIS fell from 0.27 to 0.18
  (more conservative GPS weighting); the batch baseline was re-saved.
- `reorder_capacity` bounds pending samples (default 4096);
  `max_sensor_lag_sim_s` limits source-time separation (default 1 s, keep it
  above a GPS period); `sensor_watchdog_wall_s` catches stalled streams
  (default 5 s).

## Outputs

`estimates.jsonl` records integer simulation time, sequence/epoch, ENU frame,
position, velocity, attitude, biases, full 15×15 covariance, and input
ancestry; it grows quickly at high IMU rates. `summary.json` adds EKF input
and dispatch counts, GPS corrections and outliers, timing rejections,
`gps_modeled_drops`, queue high-water mark, pending samples, and
initialization/control-feedback provenance. In Rerun, the magenta vehicle and
path are the estimate and the **EKF vs truth** tab compares positions only at
matching timestamps. Truth is used there for evaluation, never by the EKF.

## Verification status

Unit tests cover hover, yaw integration, GPS drift reduction, lever arms,
outlier gates, covariance, prediction Jacobians, GPS conversion, rotated IMU
input, timing rejection, reordering, intermediate GPS corrections, watchdogs,
bounds, mount restrictions, output fanout, and closure. Opt-in native tests
(`DRONELAB_TEST_GAZEBO=1`, `tests/integration/test_live_gazebo_ekf.py`) fly a
moving drone with unaligned IMU/GPS rates, plus noisy/rotated sensors.
Rerun appearance needs a separate visual check.
