# 5. State estimation

RobotLab's estimator is a GPS/IMU error-state EKF, run as an estimator
plug-in on the [sensor bus](7_experiments.md#sensor-bus-and-delivery-timing-sensor_timing):
as an observer (`configs/algorithms/ekf_observer.json`, control flies on
truth) or in the loop (`ekf_in_the_loop.json`). It is initialized from
declared priors and never receives simulator truth, true biases, wind, or
fault-model information. Your own filters run the same way
([Writing your own](#writing-your-own-filter)).

```text
sensor bus samples (imu, gps)
      → FilterNode (timing, mounts, GPS → ENU) → GpsImuEkf.predict_imu / update_gps
                                                        ↓
                                              state channel → Rerun / estimates.jsonl
```

| Module | Responsibility |
| --- | --- |
| `estimation/gps_imu_ekf/ekf.py` | The filter only (`GpsImuEkf`, an IMU-driven `RecursiveFilter`); `from_options` builds it from the `ekf` section |
| `experiments/harness/algorithms.py` | `gps_imu_ekf`: the `filter` plug-in running `GpsImuEkf` (`gps_imu_ekf_options`, calibrated bias priors) |
| `estimation/filter/` | Reusable for any recursive filter: `FilterNode` (sample checks, held driving readings, timing, publication), measurement conversion (mounts, ENU, wheel geometry), the `filter` plug-in, and the shared math ([building blocks](#shared-building-blocks)) |
| `estimation/lie.py` | SO(3) helpers: `skew`, `right_jacobian`, `right_jacobian_inverse` |
| `core/measurements.py` | Immutable `Sample`, `ImuReading`, `GpsReading`, `EstimatedState` |
| `sensors/bus/streams.py` | Bounded in-process `Publisher`/`Channel`/`Subscription` fanout |
| `core/geodesy.py` | WGS84 ↔ ECEF ↔ local ENU conversion |

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
  It counts as an initialization, not a correction (`updates.gps` and the
  innovations start with the second fix; its time is `initialized_s.gps`).
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

The node is `FilterNode` (`estimation/filter/node.py`), shared by every
filter; for the EKF (driven by the IMU):

- GPS latitude/longitude/altitude convert through WGS84 ECEF to local ENU.
  The reference origin and altitude datum are explicit; altitude 650 m is not
  local z = 650. The ENU frame must be aligned with the simulation world.
- The node rotates an explicitly mounted IMU into body axes. Nonzero IMU
  position offsets are rejected (no rotational lever-arm compensation).
  GPS receiver offsets are in the measurement model. This EKF fuses GPS
  position only; the shared node also converts GPS velocity, magnetometer,
  and barometer measurements for filters that fuse them.
- The first IMU sample sets the prior's time. IMU intervals must be positive
  and at most `max_driver_interval_s` (0.1 s). The node validates frames, epochs, stream identity, and
  sequence continuity, serializes handlers, and starts no threads.
- Output: `node.outputs["state"]`, also `node.estimates`, a read-only
  channel of `Sample[EstimatedState]`.
  Consumers subscribe before dispatch, e.g.
  `node.estimates.subscribe(capacity=128)`. Channels have bounded queues,
  overflow policies (`error`, `drop_oldest`), timeouts, and closure.
  Prediction and correction outputs may share timestamps with distinct
  sequence numbers.

## Writing your own filter

Write only the mathematics; `estimation/filter/` does the rest. A filter
is a class built as `cls(filter_options)` (or a factory named as
`"pkg.mod:Class.from_options"`, as the built-in EKF does) with exactly
one `predict_<kind>`, the stream that **drives** it, plus any `update_<kind>`
methods and `state()`:

| Method | Called with |
| --- | --- |
| `predict_imu(ImuInput, dt_s)` | one IMU reading (specific force, angular rate) in body axes, held over `dt_s` |
| `predict_wheel_encoders(WheelMotion, dt_s)` | per wheel: hub position and rolling direction in body x/y, rolling speed (m/s), held over `dt_s` |
| `update_gps(GpsFix)` | ENU antenna position, 3×3 covariance, lever arm, optional ENU velocity and variance |
| `update_magnetometer(MagneticField)` | field in body axes (µT) and covariance |
| `update_baro(BaroAltitude)` | altitude above the GPS reference (m) and variance |
| `update_wheel_encoders(WheelMotion)` | as above, as a correction (e.g. an IMU-driven ground-vehicle filter) |
| `predict_<kind>` / `update_<kind>` for any other bus stream | its bus payload, unchanged |
| `state()` | returns `EstimatedState` (covariance with RobotLab's right attitude error) |

Each `update_*` returns an `Innovation` (or None for an initialization),
and the estimator fuses every enabled stream its filter has a method for;
the driving stream must be enabled and cannot also be an update. Between
driving samples the node predicts up to each measurement's time, so the
streams may run at any rates; driving samples more than
`max_driver_interval_s` apart (option, default 0.1 s) are rejected. With
an IMU and wheel encoders, let the IMU drive and fuse the wheels as an
update: prediction needs one continuous input, everything else is
information to correct it with. An optional `summary()` returning a dict
appears as `filter_summary` in `summary.json`. Select it with:

```json
{"algorithms": {"estimator": "filter",
                "estimator_options": {"filter": "mypkg.my_ekf:MyEkf",
                                      "filter_options": {"gyro_std_rad_s": 0.003},
                                      "gps_std_m": [0.5, 0.5, 0.8],
                                      "magnetometer_std_ut": 1.0, "baro_std_m": 0.5}}}
```

Optional starting points: `estimation/examples/imu_gps_filter.py` (IMU-driven:
6-state position/velocity Kalman filter, gyro-integrated attitude, GPS and
barometer updates) and `estimation/examples/wheel_odometry_filter.py` (driven by
wheel encoders: least-squares planar body velocity from the wheels'
rolling speeds, position and yaw corrected by GPS). Check a filter with
`check_estimator(FilterEstimator, options)`, the scenario suite, and
`robotlab-replay ... --estimator filter --set
'algorithms.estimator_options={...}'` (one estimator per replay when it
needs its own options); compare filters on the synthetic scenarios with
`robotlab-eval-estimators` and live, beside the EKF, as
[observers](7_experiments.md#observer-estimators). On the 10 s lockstep square with sensor errors the
IMU example replayed with 2.0 m position and 0.85° attitude RMSE (EKF: 1.6 m,
1.3°); it estimates no biases, so it drifts on longer runs. On the
`car_circle` scenario the wheel-odometry example scored 2 cm position
and 0.09° heading RMSE with mean NIS 0.96 per degree of freedom; its
sideways speed is observable only through the steering angle, so quantized
encoder speeds make its velocity noisy (0.6 m/s RMS).

### Shared building blocks

A filter need not write its own Kalman algebra. `estimation/filter/` holds
the pieces the EKF and both examples are built from:

- `StateLayout` (`state_layout.py`): named blocks of the error state, e.g.
  `StateLayout.of("position", "velocity", "attitude", ("baro_offset", 1))`.
  Standard names fix the conventions (`attitude`: right body-axes rotation
  vector; `yaw`: heading error about world up; biases in body axes);
  `index(name)` gives a block's slice, and `estimated_state(P, ...)` builds
  the `EstimatedState` with the covariance marginalized to the reported
  blocks.
- Measurement models (`models.py`): `gps_position`, `gps_velocity`,
  `baro_altitude`, `magnetometer_field`. Each takes the layout, the
  converted measurement, and the nominal values, and returns a
  `LinearizedMeasurement` (residual, Jacobian placed by block name, noise),
  so one model serves any state that has the blocks it needs.
- `kalman_update(P, measurement, gate_squared)` (`kalman.py`): NIS gate,
  gain, and Joseph-form posterior; returns the correction δx (None when
  gated) for the filter to apply to its own state (⊕ for rotations), and
  the `Innovation` to return. `checked_covariance` validates priors.

Adding a sensor to a filter is then one method, e.g.

```python
def update_baro(self, baro: BaroAltitude) -> Innovation:
    z = baro_altitude(LAYOUT, baro, position_m=self.position)
    correction = kalman_update(self.P, z, self.baro_gate_squared)
    if correction.delta is not None:
        self._inject(correction.delta, correction.covariance)
    return correction.innovation
```

Jacobians are checked against finite differences in
`tests/unit/estimation/test_measurement_models.py`.

## Running it

```bash
robotlab --headless --control-mode rotor-thrust --drone configs/missions/square.json \
  --drone configs/algorithms/ekf_observer.json --drone configs/control/lockstep.json \
  --duration 45 --output outputs/ekf
```

The sensor bus delivers the IMU and GPS samples in delivery order, after
the error models and any modelled latency
([7 Experiments](7_experiments.md#sensor-bus-and-delivery-timing-sensor_timing)).
A GPS time inside an IMU interval is handled by **segmented propagation**:
propagate with the held reading to the GPS time, correct, then continue to
the next IMU sample. Process noise is applied per segment, without modelling
correlated held-sample noise across the split; an educational approximation.
A fix older than the filter time (e.g. after GPS latency) is counted in
`rejected_timing.gps`, since there is no rewind; spatial outliers are gated
separately. Lost fixes (dropouts, outages) never reach the bus, so IMU
prediction simply continues and corrections resume on recovery.

**Requirements:** enabled IMU and GPS, body-origin IMU (rotated mounts
allowed), generated ENU/WGS84 world with zero heading: the GPS origin is
read from the generated SDF's spherical coordinates
(`simulation/gazebo/world.py: reference_from_world`), never from truth or
the first fix. Accelerometer vibration and the seeded IMU/GPS error models
reach the EKF only as the resulting readings. See
[3 Configuration](3_configuration.md#gps-dropouts-and-wind) and
[IMU and GPS error models](3_configuration.md#imu-and-gps-error-models).

The examples below were measured with the former `--ekf` observer (removed
2026-10-25), which ran the same filter on the same readings.

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
- *Migration (2026-10-25):* `reorder_capacity`, `max_sensor_lag_sim_s`,
  and `sensor_watchdog_wall_s` belonged to the removed `--ekf` observer and
  are no longer accepted (recordings that carry them still replay).

## Outputs

`estimates.jsonl` records integer simulation time, sequence/epoch, ENU frame,
position, velocity, attitude, biases, full 15×15 covariance, and input
ancestry; it grows quickly at high IMU rates. (Plug-in estimators may
publish partial estimates, with `null` for absent blocks; see
[7 Experiments](7_experiments.md#algorithm-plug-ins-algorithms).) `summary.json` adds EKF input
and dispatch counts, GPS corrections and outliers (`updates`, `outliers`),
timing rejections (`rejected_timing`), the filter's initialization and bias
priors (`filter_summary`),
`gps_modeled_drops`, queue high-water mark, pending samples, and
initialization/control-feedback provenance. In Rerun, the magenta vehicle and
path are the estimate and the **EKF vs truth** tab compares positions only at
matching timestamps. Truth is used there for evaluation, never by the EKF.

## Verification status

Unit tests cover hover, yaw integration, GPS drift reduction, lever arms,
outlier gates, covariance, prediction Jacobians, GPS conversion, rotated IMU
input, timing rejection, intermediate GPS corrections, calibrated IMU
values, measurement conversion, the generic estimator and example filter,
bounds, mount restrictions, output fanout, and closure. Opt-in native tests
(`ROBOTLAB_TEST_GAZEBO=1`, `tests/integration/test_gazebo_ekf.py`) fly a
moving drone with unaligned IMU/GPS rates, plus noisy/rotated sensors, and
wind with a modelled GPS outage.
Rerun appearance needs a separate visual check.
