# 3. Drone configuration

JSON files configure the Gazebo vehicle, motors, sensors, controller gains,
pilot speeds, timing, EKF assumptions, faults, and environment. Pass one or
more with `--drone` (alias `--config`); paths are relative to the working
directory. These are simplified simulation parameters, not calibrated
hardware models.

## Combining files

Files are small building blocks, organized by topic under `configs/`. Repeat
`--drone` to layer them; later files override earlier ones:

```bash
dronelab --headless --rerun --ekf --control-mode rotor-thrust \
  --drone configs/vehicles/10in_4s.json \
  --drone configs/sensors/sensor_errors.json \
  --drone configs/environment/turbulence.json \
  --duration 30 --output outputs/combined
```

To save a combination you reuse, write a scenario file that `extends` its
building blocks (one path or a list, relative to the scenario file) and adds
its own overrides:

```json
{
  "extends": ["../vehicles/10in_4s.json", "../sensors/sensor_errors.json"],
  "wind": {"enabled": true, "velocity_enu_mps": [3, 0, 0]}
}
```

**Merge rules:**

- Built-in defaults come first, then each `--drone` file in order. A file's
  `extends` parents apply just before the file itself, in listed order and
  recursively. Cycles are rejected.
- Objects merge key by key. **Lists and single values replace the earlier
  value whole**: two files that each set `sensors.gps.dropout.zones` do not
  combine their zones; the later list wins.
- Only the final merged result is validated. A layer may be incomplete on its
  own (for example turbulence in one file and `wind.enabled` in another).
- `drone-resolved.json` records `sources` (every file in merge order, parents
  included) and `provenance` (for each setting a file supplied, the file
  that last set it). Settings not listed there are built-in defaults. The
  configuration hash depends only on the merged settings, so the same
  settings give the same hash however they were split across files.

| Folder | Example files | Purpose |
| --- | --- | --- |
| `vehicles/` | `quad_x.json` | Complete file listing every default (reference; equals the built-ins) |
| | `10in_4s.json` | Camera-class 10-inch/4S preset (1.54 kg) |
| | `hexa_x.json` | 2.4 kg hexacopter (general rotor layout) |
| | `randomized.json` | Per-run randomized true vehicle (model mismatch) |
| | `micro.json` | Micro research class (Crazyflie-like, 30 g) |
| | `racing_5in.json` | Racing/FPV 5-inch on 6S (0.65 kg) |
| | `freestyle_5in.json` | Freestyle/acrobatic 5-inch on 6S with an action camera (0.76 kg) |
| | `heavy_lift.json` | Heavy-lift 15-inch on 6S (6 kg) |
| | `battery_low.json` | Small 4S pack at 15 % charge; runs out in about 49 s of hover |
| `sensors/` | `noisy.json` | Gazebo native sensor noise and bias |
| | `sensor_errors.json` | IMU Allan-variance errors, correlated GPS error, matching EKF assumptions |
| | `imu_vibration.json` | Accelerometer vibration and clipping |
| | `baro.json` | Barometer with offset, slow drift, and noise |
| | `magnetometer.json` | Magnetometer with hard/soft iron, motor interference, and noise (uncalibrated) |
| | `downward.json` | Downward rangefinder and optical flow with noise |
| | `lidar_3d.json` | 16-channel 3D lidar, ±15°, 30 m |
| | `lidar_livox_mid360.json` | Livox Mid-360-like 3D lidar: 360° × (−7° to 52°), 40 m, noise, dropouts, motion distortion, beam spread, reflectivity detection |
| | `battery_monitor.json` | Power-module voltage/current with noise and a 0.3 A current offset (layer on a vehicle with a battery) |
| | `fpv_camera.json` | FPV camera: 150° fisheye, 30° up-tilt, 640×480 at 30 Hz, 8 ms exposure, 20 ms rolling shutter, 25 ms latency |
| `faults/` | `gps_dropout.json` | Random GPS losses and a 10–15 s outage |
| | `gps_faults.json` | GPS blockout zone under circuit gate G1 and multipath around it |
| | `imu_faults.json` | Gyro bias step, accelerometer spikes, stuck accelerometer axis |
| | `motor_degraded.json` | Rotor 1 at 70 % thrust (damaged propeller) from 5 s |
| | `motor_failure.json` | Rotor 0 stops at 5 s |
| `environment/` | `gusty_wind.json` | Wind with random and scheduled changes and periodic gusts |
| | `aerodynamics.json` | Quadratic body drag, translational lift, and vortex ring state |
| | `turbulence.json` | 3 m/s wind with Dryden-style turbulence |
| | `cold_weather.json` | 0 °C with battery thermal and IMU temperature drift on |
| | `fog.json`, `rain.json`, `dust.json` | Lidar attenuation and near-range clutter (enable lidar detection) |
| `control/` | `control_timing.json` | 250 Hz attitude / 50 Hz position loop with state and command latency |
| | `lockstep.json` | Deterministic lockstep simulation, as fast as possible |
| | `failsafe_rth.json` | Low-battery return-to-home and landing (layer on a vehicle with a battery) |
| `scenarios/` | `disturbances.json` | `extends` gusty wind and GPS dropout |
| | `tethered_delivery.json` | Light payload on a winch cable for pickup/delivery practice |
| | `racing_fpv.json` | Racing airframe + FPV camera, no GPS |
| | `freestyle_fpv.json` | Freestyle airframe + FPV camera, no GPS |
| | `ground_effect_hover.json` | Hover 0.12 m above ground with ground effect |
| | `ground_start.json` | Start resting on the ground and take off automatically |

When adding a file, keep it focused on one topic so it combines cleanly, and
put combinations in `scenarios/`.

## Loading rules

`simulation/gazebo/config.py` loads versioned JSON (`schema_version: 1`),
merges the layers over built-in defaults, and validates the result. No file
is required. Unknown keys (reported with the file that supplied them),
unsupported versions, non-finite numbers, wrong types, and invalid
physical/timing combinations fail before Gazebo starts. JSON has no
comments.

The same resolved settings feed SDF generation, thrust allocation, command
validation, motor plugins, sensor subscriptions, and Rerun. Generated runs,
including `--generate-only`, save `drone-resolved.json` with every value,
derived total mass, weight, effective rotor thrust limit, and a SHA-256 hash
also recorded in the summary. A configuration cannot be applied to a saved
`--world`; regenerate from a scene instead.

```json
{
  "schema_version": 1,
  "vehicle": {"body_mass_kg": 1.8},
  "motors": {"max_thrust_n": 7.0},
  "pilot": {"climb_mps": 0.4},
  "sensors": {
    "imu": {
      "rate_hz": 100,
      "acceleration_mps2": {"bias": [0.02, -0.01, 0.03], "noise_stddev": [0.08, 0.08, 0.08]}
    },
    "gps": {"horizontal_position_m": {"noise_stddev": 0.5}}
  }
}
```

## Sections

| Section | Properties |
| --- | --- |
| `vehicle` | Body and per-rotor mass (kg); diagonal body and rotor inertia (kg·m²); body box size (m); rotor arm x/y offset (m); centre-of-mass offset (m) |
| `motors` | Thrust coefficient (N·s²); reaction moment ratio (m); max motor speed (rad/s); per-rotor max thrust (N); rise/fall time constants (s); rotor drag and rolling moment coefficients; per-rotor thrust and moment scale factors |
| `randomization` | Per-run differences between the simulated and the nominal vehicle ([Model mismatch](#model-mismatch-and-randomization)) |
| `algorithms`, `sensor_timing`, `mission` | Plug-ins, sensor delivery timing, reference missions ([7 Experiments](7_experiments.md)) |
| `sensors` | IMU, GPS, barometer, magnetometer, rangefinder, optical flow, battery monitor, contacts, camera, lidar (2D or 3D): enablement, rate (Hz), mount (except contacts), errors; camera resolution/FOV/clipping; lidar samples/ranges; GPS dropout and error model; IMU vibration and error model; barometer native noise and error model |
| `controller` | Python thrust-mode position, velocity, attitude, angular-rate gains; acceleration limit; max position-setpoint error; min commanded altitude |
| `velocity_controller` | Native Gazebo velocity-mode gains and acceleration limits |
| `ekf` | Estimator priors, uncertainties, assumed noise, gate, buffer/watchdog settings (used with `--ekf`) |
| `control` | Publication rate, command lifetime (sim s), watchdog (wall s), truth-odometry rate; controller tick and position-loop rates, state/command latency, publish-on-send (see [4 Control](4_control.md#controller-timing-and-latency)) |
| `pilot` | Forward/back speed (m/s), yaw rate (rad/s), climb/descent speed (m/s); automatic take-off altitude (m) and climb rate (m/s) |
| `environment` | `ambient_temperature_c` (°C) for the battery and IMU temperature models ([Temperature effects](#temperature-effects)) |
| `keyboard_effects` | What keys 1–9, 0, [ and ] do: stopped and damaged rotor, damaged thrust fraction, multipath ENU jump (m), gyro bias step (rad/s), flying object size/radius/speed, battery drain step, ambient temperature step (°C) ([4 Control](4_control.md#keyboard-effect-toggles)) |
| `wind` | Native wind velocity, force scale (drag coefficient), smoothing, gusts, random and scheduled changes, still-air drag, turbulence |
| `battery` | LiPo pack: cells, capacity, internal resistance, initial charge, powertrain efficiency, avionics power (rotor-thrust mode) |
| `aerodynamics` | Air density; ground effect, translational lift, vortex ring state (rotor-thrust mode); quadratic body drag (all modes) |
| `simulation` | Physics step (s), real-time factor (0 = unlimited, lockstep only), lockstep, initial world position (m), start on ground |

### Vehicle and motors

- `body_mass_kg` excludes the rotors. Total mass is
  `body_mass_kg + rotor count × rotor_mass_kg` (default quad: 1.54 kg),
  weight is total × 9.81 N, about 3.777 N per rotor at level hover.
- `max_thrust_n` is a common per-rotor limit. `null` means
  `thrust_coefficient_n_s2 * max_speed_rad_s²` (default 8.55 N at 1000 rad/s);
  an explicit value may only reduce it. Configurations unable to hover are
  rejected. Zero thrust stays available for stop/expiry.
- Gains are explicit and are not retuned when mass, inertia, geometry, or
  motor dynamics change.
- **Default layout:** an empty `vehicle.rotors` gives four vertical rotors
  in an X (0/1 CCW, 2/3 CW) at ±`arm_offset_m` and `rotor_height_m`. The
  reaction moment ratio maps thrust to reaction torque, hence metres.
- **General layouts:** `vehicle.rotors` is a list of
  `{"position_m": [x, y, z], "direction": "ccw"|"cw", "axis": [ax, ay, az]}`
  in body axes, with unit thrust axes.
  - **Pushers:** each rotor link is rotated so its +z points along the
    axis. Gazebo's motor model pushes along that axis, which was checked
    with a pitched rotor in zero gravity, so tilted rotors such as a pusher
    work.
  - **Lift rotors:** at least four must have vertical axes. The built-in
    controller allocates over them with a minimum-norm solution, which is
    exact for four and spreads the load for more. Other rotors get zero
    thrust from it, for a plug-in to command.
  - **Rotor-dependent models:** ground effect, translational lift, and
    vortex ring state apply to lift rotors only. Per-rotor scales, motor
    faults, and keyboard rotor indices follow the rotor count.
  - **Velocity mode** requires every rotor to be vertical.
- `configs/vehicles/hexa_x.json` is a 2.4 kg hexacopter: six rotors on a
  0.35 m radius, alternating spin. In lockstep it flew the training square
  with 0.12 m tracking RMSE (quad: 0.10 m), and the quad batch baseline was
  unchanged by the general allocation.

### Model mismatch and randomization

The vehicle Gazebo simulates (the "true" vehicle) can differ from the
nominal one the algorithms know. The nominal vehicle is what goes into the
`AlgorithmContext`, the built-in controller's allocation, and the
thrust-to-motor-speed conversion. Two deterministic knobs:

- `vehicle.center_of_mass_offset_m`: the mass centre relative to the body
  origin, written as the base link's inertial pose. Odometry stays at the
  body origin, and the controller assumes the mass is there. Limited to
  half the arm offset.
- `motors.rotor_thrust_scale` and `motors.rotor_moment_scale` (one per
  rotor): true per-rotor thrust coefficient and reaction moment ratio as
  multiples of the nominal values, as from manufacturing spread or worn
  propellers. They are hidden truth: thrust-to-speed conversion and
  the power model keep the nominal coefficient. Do not use them to describe
  a deliberately different (known) motor, or commanded and real thrust
  will disagree.

`randomization` (`simulation/randomization.py`) draws a new true vehicle
per run from `seed`. Each draw is N(0, σ²) truncated at ±3σ, and fraction
σ must stay below 1/3:

| Key | Units | Perturbs |
| --- | --- | --- |
| `mass_fraction_std` | – | body mass × (1 + δ) |
| `inertia_fraction_std` | – | each principal inertia × (1 + δ) |
| `center_of_mass_std_m` | m, per axis | centre-of-mass offset + δ |
| `rotor_thrust_fraction_std` | – | each rotor's thrust coefficient × (1 + δ) |
| `rotor_moment_fraction_std` | – | each rotor's moment ratio × (1 + δ) |
| `motor_time_constant_fraction_std` | – | motor rise and fall time constants × (1 + δ) |
| `imu_mount_rotation_std_deg` | deg, per axis | true IMU mount rotation + δ (the context keeps the nominal mount) |

The draws and true values go into `drone-resolved.json` (`randomized_truth`)
and `summary.json` (`randomization`). Evaluation uses the true mass and IMU
mount. `configs/vehicles/randomized.json` is a moderate example, and
`configs/experiments/randomized_square.json` sweeps it over six seeds, since
batch seeds also reseed `randomization.seed`.

Verified in lockstep:
- **Centre-of-mass offset:** a 1 cm offset along x makes the
  integral-free built-in controller hold a 4.8° attitude error
  (m·g·c / K_att) and settle at 0.67 m along x after 20 s, against 0.69 m
  predicted (g·tan(e) / K_pos).
- **Randomized square:** six randomized vehicles flew the square with
  0.48 ± 0.15 m tracking RMSE, against 0.10 m for the nominal vehicle.

### Timing limits

Odometry rate must be at least 20 Hz and at most the physics rate; sensor
rates cannot exceed the physics rate. Command lifetime must exceed one
odometry interval and one publication interval and be at most 1 s. Physics
steps above 0.01 s are rejected. Rate changes affect flight quality and
recording load.

## Sensor mounts

IMU, GPS, camera, and lidar accept an optional `mount` relative to
`base_link`: **+X forward, +Y left, +Z up**, position in metres and
roll/pitch/yaw in degrees.

```json
{"sensors": {"camera": {"enabled": true,
  "mount": {"position_m": [0.3, 0.0, 0.1], "rotation_deg": [0, 20, 0]}}}}
```

This places the camera 30 cm forward, 10 cm up, pitched 20° down. Angles are
right-handed rotations about fixed body axes, giving sensor-to-body rotation
`Rz(yaw) Ry(pitch) Rx(roll)`; positive yaw turns left, positive pitch points
the forward axis down. SDF stores radians.

| Sensor | Default `position_m` | Default `rotation_deg` |
| --- | --- | --- |
| IMU | `[0, 0, 0]` | `[0, 0, 0]` |
| GPS | `[0, 0, 0]` | `[0, 0, 0]` |
| Camera | `[0.3, 0, 0.1]` | `[0, 0, 0]` |
| Lidar | `[0, 0, 0.2]` | `[0, 0, 0]` |

Partial mounts inherit missing fields; a mount does not enable a sensor.
`[0, 0, 0]` is inside the body: fine for the IMU, but place cameras and lidars
outside and pointing outward, since there is no occlusion correction.
Gazebo simulates measurements at the mounted pose: IMU readings stay in sensor
axes, an off-centre IMU also senses rotational acceleration, and GPS measures
the receiver location. Nothing is silently converted to body-origin values.
Rerun uses actual SDF sensor poses and transforms lidar points into body axes.

## Sensor noise and bias

There are two independent ways to add measurement errors:

- **Gazebo native noise** (this section): constant bias plus white noise,
  generated inside Gazebo, not seeded.
- **Python sensor-error models** ([next section](#imu-and-gps-error-models)):
  seeded IMU noise densities, bias random walk, turn-on bias, and
  time-correlated GPS error. Use these for realistic estimator studies.

Both can be combined; their errors add. Gazebo native errors default to zero
and use Gazebo's Gaussian model: `bias` is the signed mean (a constant
offset) and `noise_stddev` the per-measurement standard deviation. These are
**not** noise densities.

| Key | Units and shape |
| --- | --- |
| `imu.acceleration_mps2` | Three x/y/z values, m/s² |
| `imu.angular_rate_rad_s` | Three x/y/z values, rad/s |
| `gps.horizontal_position_m`, `gps.vertical_position_m` | Scalars, m |
| `gps.horizontal_velocity_mps`, `gps.vertical_velocity_mps` | Scalars, m/s |
| `lidar.range_m` | Scalars, m |
| `camera.intensity` | Scalars, normalized intensity 0–1; pixels are clipped |

Enabled camera/lidar require rendering (EGL when headless).
`--render-sensors` enables both regardless of the file, keeping their rates
and errors. Disabled sensors get no SDF sensor, no subscription, and no
required-stream check. Truth odometry is always enabled for the controllers,
which are unaffected by sensor noise.

**GPS horizontal noise mapping.** Gazebo Sensors 8 applies horizontal
position noise directly to latitude/longitude in degrees
([source](https://github.com/gazebosim/gz-sensors/blob/gz-sensors8/src/NavSatSensor.cc)).
DroneLab converts configured metres to degrees with Earth radius 6378137 m, a
north-equivalent scale; east noise is therefore smaller by about
cos(latitude). This is not isotropic ENU noise. Vertical values keep metre
units. Regenerate older worlds to use this mapping. The Python GPS error
model below is isotropic in metres and avoids this distortion.

## IMU and GPS error models

Seeded Python models applied at the mission boundary, before Rerun and the
EKF, like GPS dropout and vibration. Raw Gazebo topics, timestamps, and
vehicle physics are unchanged; opening an SDF directly does not apply them.
Both are off by default. `configs/sensors/sensor_errors.json` enables
both with values of the same order as published MEMS IMU and consumer GPS
characteristics, and declares matching EKF assumptions. These are plausible
stress parameters, **not** a calibrated sensor.

### IMU (`sensors.imu.error_model`)

`sensors/imu_errors.py` adds, per axis in IMU sensor axes, for the
accelerometer and the gyro:
`measured = input + turn_on_bias + bias_walk(t) + white(t)`.

| Key (accel / gyro) | Units | Meaning |
| --- | --- | --- |
| `accel_noise_density_mps2_per_sqrt_hz` / `gyro_noise_density_rad_s_per_sqrt_hz` | (m/s²)/√Hz, (rad/s)/√Hz | White-noise density N (velocity / angle random walk) |
| `accel_bias_random_walk_mps2_per_sqrt_s` / `gyro_bias_random_walk_rad_s_per_sqrt_s` | (m/s²)/√s, (rad/s)/√s | Bias random walk K |
| `accel_turn_on_bias_std_mps2` / `gyro_turn_on_bias_std_rad_s` | m/s², rad/s | Std of the constant bias drawn once per run |
| `seed` | integer | Local RNG seed |

All are three-element per-axis lists, default zero (term disabled).

- **Discretization.** Per-sample white-noise std is `N · sqrt(rate_hz)` at
  the configured IMU rate (the Allan-variance/Kalibr convention). Bias-walk
  increments have variance `K² · dt` over the actual sample interval, the
  same convention as the EKF's `accel_bias_walk`/`gyro_bias_walk`. To match
  the filter to the simulated sensor, set `ekf.accel_std_mps2 = N·√rate`,
  `ekf.accel_bias_walk = K` (likewise for the gyro), as the example does.
- **Ordering.** IMU errors are applied before vibration, so the vibration
  clip limit applies to the final reading.
- Bias instability, scale factor, misalignment, g-sensitivity, saturation,
  sample loss, and timestamp jitter are in `sensors.imu.imperfections`
  (below). Temperature drift is a separate model, see
  [Temperature effects](#temperature-effects).

### IMU imperfections (`sensors.imu.imperfections`)

`sensors/imu_imperfections.py`; off by default. The IMU pipeline runs, in
order:
1. imperfections, on the true signal,
2. Allan-variance errors,
3. temperature drift,
4. vibration and its clip,
5. saturation,
6. scheduled faults,
7. timing.

| Key | Units | Default | Meaning |
| --- | --- | --- | --- |
| `accel_scale_factor`, `gyro_scale_factor` | fraction | 0 | Per-axis scale-factor error s |
| `accel_cross_coupling`, `gyro_cross_coupling` | rad (3×3) | 0 | Off-diagonal misalignment C (diagonal must be 0); y = (I + diag(s) + C)·x |
| `gyro_g_sensitivity_rad_s_per_mps2` | (rad/s)/(m/s²), 3×3 | 0 | Gyro output per specific force (typical 0.01–0.1 °/s/g = 2e-5–2e-4) |
| `accel_bias_instability_mps2`, `gyro_bias_instability_rad_s` | per axis | 0 | Stationary σ of a first-order Gauss–Markov bias (flicker floor) |
| `bias_instability_correlation_s` | s | 100 | Its correlation time τ |
| `accel_range_mps2`, `gyro_range_rad_s` | – | null | Full-scale range; each axis is clipped |
| `sample_drop_probability` | – | 0 | Each sample lost with this probability |
| `outages_s` | s | [] | `[start, end]` intervals with no samples |
| `timestamp_jitter_s` | s | 0 | Reported stamp moved by N(0, σ²), clipped to ±0.45 sample periods |

How the effects behave:
- **Lost samples** are never recorded, delivered, or given to `--ekf`;
  `summary.json.imu_imperfections` counts them.
- **Jitter** changes only the reported timestamp, never the physics.
- **Bias truth:** the Gauss–Markov bias is part of the true bias that
  estimation metrics compare against.

`configs/sensors/imu_consumer_mems.json` holds consumer-grade values:
- about 0.5 % scale error and 0.2° cross-coupling,
- 0.005 °/s/g of g-sensitivity,
- bias instability of 50 µg and 10 °/h,
- ±16 g and ±2000 °/s ranges,
- 0.1 % sample loss and 0.5 ms jitter.

On a 30 s lockstep figure-eight with the EKF scored, it doubled the
attitude error (0.18° → 0.36° RMSE) and raised NEES from 11 to 281. The
EKF does not model these errors, so it becomes inconsistent. 7 of
6,004 samples were lost.

### IMU calibration (`sensors.imu.calibration`)

The IMU counterpart of the magnetometer calibration. On real drones it
follows the same pattern:
- **Factory or six-position calibration:** done once, for accelerometer
  offset, scale, and misalignment.
- **Optional thermal calibration:** for temperature drift.
- **Gyro bias at every power-up:** averaged while the vehicle is still,
  because the turn-on bias changes each time.
- **In flight:** the EKF estimates what is left.

**1. Procedure: `dronelab-imucal`** (`experiments/imu_calibration.py`):

```bash
dronelab-imucal --drone configs/sensors/imu_consumer_mems.json --startup-gyro-s 2 --output outputs/imucal
```

The configured vehicle is made static, as if held by a fixture, with
gravity on. It is set into the six axis-aligned orientations plus
`--extra-poses` (12) seeded tilted ones, and `--samples` (40) readings are
averaged per pose. With `sensors.imu.temperature` enabled, the whole set is
repeated at 5, 25, and 45 °C ambient, a fridge-to-warm-room sweep, using the
die temperature the IMU reports.

Readings go through the mission's IMU models (imperfections, Allan errors,
temperature, saturation). Motors are off, so vibration and scheduled faults
are left out. The known specific force per pose uses the nominal mount, as
an operator knows it.

Outputs: `imu-calibration.json`, a layer for `--drone`, and
`calibration-report.json` with fit residuals, before/after static errors,
and the injected truth for comparison. Rate-table quantities, meaning gyro
scale and misalignment, need a turntable and are not calibrated.

**2. Fitting** (`estimation/imu_calibration.py`): per axis, a linear least
squares on static samples:
- accelerometer: a = A·f + b + k·(T − T_ref),
- gyro: ω = G·f + b + k·(T − T_ref).

The temperature slopes are only identifiable when temperatures vary;
otherwise they are zero.

**3. Use.** f̂ = A⁻¹(a − b_a − k_a ΔT) and
ω̂ = ω − b_g − k_g ΔT − G f̂ − b_startup:

| Key | Units | Meaning |
| --- | --- | --- |
| `enabled` | | Apply the calibration |
| `accel_offset_mps2`, `accel_correction` | m/s², 3×3 | b_a and A⁻¹ |
| `accel_temperature_mps2_per_c`, `gyro_temperature_rad_s_per_c` | per °C | Temperature slopes k |
| `gyro_offset_rad_s`, `gyro_g_sensitivity` | rad/s, (rad/s)/(m/s²) | b_g and G |
| `reference_temperature_c` | °C | T_ref |
| `startup_gyro_calibration_s` | s | Average the corrected gyro this long after power-up (0: off). Needs `simulation.start_on_ground`; take-off waits, with motors off |
| `fingerprint` | | Settings the calibration was made for (below) |

When enabled, the autopilot side applies it like a driver:
- **Inputs:** it uses the die temperature the IMU reports (when the
  temperature model is on).
- **Sensor bus:** `ImuReading` then carries
  `calibrated_specific_force_mps2`, `calibrated_angular_rate_rad_s`, and
  `temperature_c` next to the raw values. `best_specific_force_mps2` and
  `best_angular_rate_rad_s` return the calibrated values when present.
- **Consumers:** the built-in EKF and estimate-driven body rates use the
  calibrated values.
- **Rerun:** logs `imu/calibrated/*`; `summary.json` records the measured
  startup gyro bias.

Verified with the consumer-MEMS, Allan-error, and cold-weather layers. The
procedure took 17 s:
- **Accelerometer:** scale and misalignment recovered to within 0.014 %
  (correction × distortion = identity); temperature slopes 0.0064/−0.0038/
  0.0080 m/s²/°C against 0.006/−0.004/0.008.
- **Gyro:** temperature slopes and offsets recovered to within a few
  percent; g-sensitivity 4.3–5.7e-5 against 5e-5.
- **Static errors:** accelerometer 0.21 → 0.009 m/s², gyro
  0.0051 → 0.0007 rad/s.
- **In a 30 s lockstep flight from the ground, with the EKF scored:**
  - attitude RMSE 8.7° bypassed → 2.35° calibrated,
  - gyro bias error 0.0074 → 0.0021 rad/s,
  - accelerometer bias error 0.24 → 0.034 m/s².

  Position stayed about 1.8 m, dominated by the correlated GPS error of
  `sensor_errors.json`.

### Using and bypassing calibrations

Calibrations are made once and reused. Pass the produced layer with
`--drone` on every run; the simulated errors are deterministic for a given
configuration, so the calibration stays valid.
- **Bypass:** `configs/sensors/no_calibration.json` switches both IMU and
  magnetometer calibration off, so sensors stay raw and filters receive no
  calibration. Put it last on the command line, or set `calibration.enabled`
  to false, or leave the calibration layer out.
- **Staleness:** each calibration stores a `fingerprint` of the sensor
  settings it was made for: mount, native noise, error model,
  imperfections, and temperature for the IMU; mount, native noise, and error
  model for the magnetometer. Seeds are excluded, because turn-on biases
  change at every power-up anyway. When a run's settings differ, it prints a
  warning and lists it in `summary.json.calibration_warnings`, so
  recalibration is needed only when the hardware model really changed.

### GPS (`sensors.gps.error_model`)

`sensors/gps_errors.py` adds a local east/north/up position error to each
fix: `error = correlated(t) + white(t)`. Real GPS error is dominated by
slowly varying components, so consecutive fixes are not independent.

| Key | Units | Default | Meaning |
| --- | --- | --- | --- |
| `horizontal_std_m`, `vertical_std_m` | m | 0 | Stationary std of the correlated error (east and north share the horizontal value) |
| `correlation_time_s` | s, > 0 | 60 | Gauss-Markov correlation time |
| `white_horizontal_std_m`, `white_vertical_std_m` | m | 0 | Independent per-fix noise |
| `seed` | integer | 0 | Local RNG seed |

- The correlated part is a first-order Gauss-Markov process, updated exactly
  over the elapsed simulation time and started in its stationary
  distribution (the receiver was already running).
- The offset is converted to latitude/longitude/altitude with WGS84 radii of
  curvature (µm-level accuracy for metre-level offsets). GPS velocity fields
  are unchanged.
- The model sees every fix, including those later removed by dropout, so
  enabling dropouts does not change the error sequence.
- An EKF that treats GPS as white noise cannot remove correlated error: its
  position estimate follows the GPS drift. That is the realistic behaviour.

`summary.json` fields `imu_errors` and `gps_errors` record the injected
turn-on and final biases and the GPS error state. They are for evaluation
only and are never given to the EKF.

### GPS receiver: velocity, status, RTK, heading

`sensors/gps_receiver.py` models four more receiver outputs. The GPS fix on
the sensor bus carries the ENU velocity and a `GpsStatus`.

- **Velocity** (`velocity_error`, off by default): Doppler velocity error
  per ENU axis, made of a Gauss–Markov part (`correlated_std_mps`,
  `correlation_time_s`) plus white noise (`white_horizontal_std_mps`,
  `white_vertical_std_mps`).
- **Status** (`receiver`):
  - **Satellites:** `satellites` ± a bounded random walk of
    `satellite_variation`, minus `multipath_satellite_loss` while multipath
    is active.
  - **HDOP:** `hdop`·√(sats₀/sats).
  - **Fix type:** `3d`, `rtk_float`, or `rtk_fixed`.
  - **Reported accuracies:** horizontal is HDOP·`uere_m` (or the RTK σ),
    vertical is 1.6× that, and speed is `speed_accuracy_mps`. These are the
    receiver's own estimates, not the true error, so a filter that trusts
    them can be tested for consistency.
  - **Logging:** Rerun logs `gps/status`.
- **RTK** (`rtk`, exclusive with `error_model`): replaces the position
  error.
  - **Float:** Gauss–Markov error of `float_std_m` (τ 10 s) until
    `convergence_s` after the first fix.
  - **Fixed:** then white error of `fixed_std_m` horizontally and
    `fixed_vertical_std_m` vertically.
  - **Multipath:** drops the solution back to float, and it re-fixes
    `convergence_s`/3 after multipath ends.
- **Dual-antenna heading** (`heading`): with each delivered fix, the true
  yaw + `mount_yaw_deg` + N(0, `noise_std_deg`²), published on the bus as
  `gps_heading` and logged as `gps/heading_rad`. Typical accuracy is about
  0.2° per metre of baseline.

**Latency** is part of the sensor bus: `sensor_timing.gps.latency_s`
delivers fixes late with their true time-of-validity stamps. A receiver
that stamps fixes on output is modelled with `timestamp_offset_s`
([7 Experiments](7_experiments.md#sensor-bus-and-delivery-timing-sensor_timing)).

Example layers: `configs/sensors/gps_rtk_dual_antenna.json` (RTK fixed after
10 s, 0.2° heading, 3–5 cm/s velocity noise) and
`configs/sensors/gps_latency.json` (150 ms ± 10 ms). Verified in lockstep
on a 20 s circle:
- **RTK:** the first 10 s were float and the remaining 100 fixes fixed,
  with heading delivered alongside all 201 fixes.
- **Latency:** all 199 delayed fixes arrived after newer IMU samples (up to
  182 ms). The built-in EKF, which cannot apply past measurements, rejected
  every one (`rejected_gps_timing`) and dead-reckoned to 1.1 m error, which
  is a known EKF limitation.

## Fault injection

Faults model sensor malfunctions and environment effects, as opposed to the
normal error statistics above. All are seeded, off by default, applied at the
Python mission boundary before Rerun and the EKF, and reported in
`summary.json` for evaluation only.

### Fault zones

GPS blockout and multipath zones are axis-aligned world boxes (`min_m` <
`max_m` on every axis, metres, z up; the boundary counts as inside).
Membership uses the latest **true** vehicle position from odometry (50 Hz, so
up to one odometry interval old). Truth only decides *whether* a fault
applies, like simulation time for scheduled outages; it never reaches the
estimator.

### GPS multipath (`sensors.gps.multipath`)

`sensors/multipath.py` adds sudden position jumps that persist and then
clear, approximating reflections near structures. It runs after the GPS
error model and before dropout, on every fix.

| Key | Units | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | bool | false | Turn the model on |
| `onset_rate_hz` | 1/s | 0 | Poisson rate of new episodes (probability `1 − exp(−rate·dt)` per fix) |
| `duration_s` | s, > 0 | 3 | Episode length |
| `horizontal_std_m`, `vertical_std_m` | m | 0 | Std of each episode's ENU offset, drawn once per episode |
| `zones` | boxes | `[]` | If non-empty, episodes start only inside a zone and end on leaving all zones |
| `seed` | integer | 0 | Local RNG seed |

One episode is active at a time; a new one may start on the fix where the
previous expires. GPS velocity fields are unchanged. `gps_multipath` in the
summary reports `episodes`, `affected_fixes`, and the first 1000 episodes'
start times and offsets. This is a statistical stand-in, not geometry-based
(ray-traced) multipath.

### IMU faults (`sensors.imu.faults`)

`sensors/imu_faults.py` applies scheduled malfunctions after IMU errors and
vibration/clipping, as digital output faults (spikes can exceed the clip
limit). Format: `{"seed": 0, "events": [...]}`. Each event has `type`,
`instrument` (`accel` in m/s² or `gyro` in rad/s, sensor axes), `start_s`,
and `end_s` (absolute simulation seconds, `[start, end)`; `null` = until the
end).

| `type` | Extra keys | Effect while active |
| --- | --- | --- |
| `bias_step` | `value` [3] | Add a constant offset |
| `scale` | `value` [3] | Multiply by a per-axis factor (1 = unchanged) |
| `noise` | `value` [3] ≥ 0 | Add white noise with this per-sample std |
| `spikes` | `value` [3] ≥ 0, `probability` in [0, 1] | Per sample, with this probability, add N(0, value²) |
| `stuck` | `axes` [3 booleans] | Selected axes hold their value from the first faulted sample |

```json
{"sensors": {"imu": {"faults": {"seed": 22, "events": [
  {"type": "bias_step", "instrument": "gyro", "start_s": 15, "end_s": null, "value": [0, 0, 0.02]},
  {"type": "spikes", "instrument": "accel", "start_s": 18, "end_s": 21, "value": [5, 5, 5], "probability": 0.02},
  {"type": "stuck", "instrument": "accel", "start_s": 23, "end_s": 24, "axes": [true, false, false]}
]}}}}
```

Events apply in listed order. Samples are never removed, so the EKF's 0.1 s
IMU-interval check is unaffected. `imu_faults` in the summary reports, per
event, affected samples, spike count, and first/last fault time. IMU sample
dropout is not modelled; the current EKF would treat it as a fatal timing
error.

### Motor faults (`motors.faults`)

`simulation/motor_faults.py` schedules motor and propeller failures in the
rotor-thrust powertrain (rotor-thrust and keyboard modes only; Gazebo's
motor plugin has no failure hook). `{"events": [...]}`; each event has
`type`, `rotor` (0–3, actuator order), `start_s`, `end_s` (`null` = until the
end), and for two types a `value`:

| `type` | `value` | Effect |
| --- | --- | --- |
| `stop` | none | Motor/ESC failure: zero speed, thrust, and power |
| `partial_thrust` | remaining thrust fraction, [0, 1) | Damaged propeller: same speed, less thrust |
| `speed_limit` | speed fraction of maximum, (0, 1) | Weak or overheated motor: speed capped |

Overlapping events on one rotor multiply thrust fractions and take the
lowest speed cap. Reduced thrust is produced by lowering the published speed
(Gazebo's thrust coefficient is fixed). The controller is not informed.
Rerun's **Sensor status** tab plots `motor_<i>_thrust_available`
(thrust fraction × speed-cap², 1 = healthy); `summary.json` reports each
event's first and last active time under `powertrain.motor_faults`.

Observed in lockstep with the default vehicle hovering at 1 m:

- `motor_failure.json` (rotor 0 stops at 5 s): flipped and hit the ground
  0.5 s later, ending inverted. A quadrotor needs fault-tolerant control
  (giving up yaw) to survive a lost motor; the built-in controller has none.
- `motor_degraded.json` (rotor 1 at 70 % from 5 s): no flip or saturation,
  but it sank 0.86 m and drifted 5.2 m. The controller has no integral term,
  and its anti-windup keeps the setpoint within 1.5 m of the vehicle, so a
  persistent thrust deficit moves the whole vehicle instead of being trimmed.

## ESC command path and telemetry (`motors.esc`)

Rotor-thrust mode only. `simulation/esc.py` applies the ESC's command path
to the motor-speed targets before Gazebo's motor model:

| Key | Default | Meaning |
| --- | --- | --- |
| `update_rate_hz` | null (ideal) | Changed targets take effect at most this often; meanwhile the previous target holds |
| `resolution_steps` | 0 (continuous) | Quantize targets to N levels of maximum speed (DShot: 2000) |
| `telemetry.enabled` | false | Measured rotor speeds (bidirectional DShot style) |
| `telemetry.rate_hz` | 100 | Report rate |
| `telemetry.noise_std_rad_s` | 0 | White noise per report, rad/s |
| `telemetry.seed` | 0 | Local RNG |

ESC ticks are evaluated when commands are sent, so the hold is aligned to
the control loop. Unchanged targets (the thrust service's heartbeat
republishing) pass without using a tick, so a held change goes out with the
next send after the tick.

Telemetry is real simulator output. A Gazebo `JointStatePublisher` reports
the rotor joint velocities: Gazebo's motor model spins the joints at the
actual lagged and limited rotor speed divided by
`rotorVelocitySlowdownSim` (10), and this was checked empirically. Python
rescales them to rad/s, decimates, and adds noise. Reports are logged to
Rerun (`esc/rotor_speed_rad_s`) and published on the sensor bus as `esc`.
`summary.json.esc` counts forwarded and held commands and telemetry samples.

`configs/control/esc_dshot.json` (100 Hz, 2000 steps, 100 Hz telemetry
with 3 rad/s noise) was verified in lockstep on the step mission:
- **Commands:** no change was held at 50 Hz control, and the step response
  matched the ideal ESC.
- **Telemetry:** the mean in hover matched the commanded speeds to within
  0.3 rad/s, with the configured 3 rad/s spread.
- **Slow ESC:** a 20 Hz ESC held 247 changes and raised peak tilt from
  6.3° to 7.2°.

## Battery and powertrain (`battery`)

`simulation/battery.py` and `simulation/powertrain.py`. Rotor-thrust and
keyboard modes only (Gazebo's native velocity controller drives the motors
directly, so other modes reject it). Off by default.

| Key | Units | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | bool | false | Turn the battery model on |
| `cells` | integer | 4 | Series LiPo cells (4S: 16.8 V full) |
| `capacity_mah` | mAh | 5000 | Pack capacity |
| `internal_resistance_ohm` | Ω | 0.02 | Pack resistance, causes voltage sag under load |
| `initial_soc` | (0, 1] | 1.0 | Starting state of charge |
| `infinite` | bool | false | Charge never drops (voltage still sags under load); for long experiments |
| `powertrain_efficiency` | (0, 1] | 0.7 | Shaft power / electrical power (motors and ESCs) |
| `avionics_power_w` | W | 5 | Constant extra electrical load |

- **Voltage.** Open-circuit voltage follows a typical LiPo curve (4.2 V per
  cell full, 3.27 V empty); the loaded terminal voltage is
  `V = (OCV + sqrt(OCV² − 4PR)) / 2`, capped at the maximum power point.
- **Power.** Rotor shaft power `Σ c_m · k_t · ω³` (reaction torque × speed)
  from the last commanded speeds, divided by the efficiency, plus avionics.
  Hover with the default vehicle draws about 235 W (about 19 min on a full
  5000 mAh 4S pack).
- **Motor limit.** Achievable speed scales with loaded voltage:
  `max_speed_rad_s · V / V_full` (`max_speed_rad_s` applies to a full pack).
  Faster requests are clipped (`limited_commands`), so authority shrinks as
  the pack drains. The controller is not told; it keeps its nominal limits.
- **Empty pack.** At 0 % charge motor speeds go to zero and the vehicle
  falls, unless the `failsafe.low_battery` section lands it first.
- **Temperature.** Optional, see [Temperature effects](#temperature-effects).
- Not modelled: ageing, RC relaxation, ESC behaviour.

Rerun's **Battery** tab shows terminal voltage, charge, current, and ground-
effect factors. `summary.json` field `powertrain` reports final charge,
final/minimum voltage, energy used, limited commands, and depletion time.

### Temperature effects

One ambient temperature, `environment.ambient_temperature_c` (°C, default
25, range −40…60), drives two optional models. Both start at ambient. Keys
[ / ] change ambient during keyboard flight ([4 Control](4_control.md#keyboard-effect-toggles)).

**Battery (`battery.thermal`, `simulation/battery.py`).** The pack heats
from its losses I²R and cools towards ambient:
`C dT/dt = I²R(T) − h (T − T_ambient)`. Resistance grows in the cold,
`R(T) = R₂₅ · exp(k_R (25 − T))`, and usable capacity shrinks below 25 °C,
`factor = max(0.3, 1 − k_C · (25 − T))`.

| Key | Units | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | bool | false | Turn the thermal model on (`internal_resistance_ohm` is then the 25 °C value) |
| `heat_capacity_j_per_k` | J/K | 300 | Pack heat capacity C |
| `cooling_w_per_k` | W/K | 1.5 | Heat loss to air h (C/h = 200 s time constant) |
| `resistance_coefficient_per_c` | 1/°C | 0.028 | k_R; resistance doubles at about 0 °C |
| `capacity_coefficient_per_c` | 1/°C | 0.006 | k_C; 15 % less capacity at 0 °C |

**IMU (`sensors.imu.temperature`, `sensors/imu_temperature.py`).** The die
warms after power-on, `T = T_ambient + rise · (1 − exp(−t/τ))` with t from
the first sample, and each axis gains a bias `coefficient · (T − calibration_c)`.
It is applied after the Allan-variance errors and before vibration and
faults.

| Key | Units | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | bool | false | Turn the model on |
| `warmup_rise_c` | °C | 10 | Self-heating above ambient |
| `warmup_time_constant_s` | s | 120 | Warm-up time constant τ |
| `calibration_c` | °C | 25 | Temperature at which the biases are zero |
| `accel_bias_per_c` | m/s²/°C | [0, 0, 0] | Per-axis accelerometer coefficient (typical MEMS 0.003–0.01) |
| `gyro_bias_per_c` | rad/s/°C | [0, 0, 0] | Per-axis gyro coefficient (typical 1e-4–3e-4) |

`configs/environment/cold_weather.json` sets 0 °C with both models on. A
60 s lockstep hover from 15 % charge on `battery_low.json` (1500 mAh 4S):

| Ambient | Depleted at | Energy delivered | Minimum voltage | Pack heating |
| --- | --- | --- | --- | --- |
| 25 °C | 48.6 s | 3.18 Wh | 12.72 V | +0.8 °C |
| 0 °C | 40.5 s | 2.66 Wh | 12.35 V | +1.4 °C |

In the same pair of runs the mean IMU difference (cold − warm, 5–35 s) was
(−0.150, 0.100, −0.200) m/s² and (−3.75, 2.50, −5.00)·10⁻³ rad/s, exactly
coefficient × (−25 °C). Rerun logs `battery/temperature_c` and
`imu/temperature_c`; `summary.json` reports `powertrain.battery.temperature_range_c`
and `imu_temperature.final_temperature_c`. These are plausible teaching
values, not a calibrated pack or IMU. Not modelled: motor and ESC heating,
air temperature effects on air density, barometer, or magnetometer.
Observed with `configs/vehicles/battery_low.json` (15 % of 1500 mAh): steady 1 m hover
until the pack emptied at 48.5 s (model prediction about 49 s), minimum
loaded voltage 12.7 V, then loss of lift.

## Low-battery failsafe (`failsafe.low_battery`)

Mission logic (`experiments/failsafe.py`) for rotor-thrust and keyboard
modes; requires `battery.enabled`. It reads the **measured** battery state
like a real flight controller: the battery monitor's remaining percentage
and voltage when `sensors.battery_monitor` is enabled (so its counting
errors matter), otherwise ideal values from the battery model.

| Key | Default | Meaning |
| --- | --- | --- |
| `action` | `land` | `warn` (log only), `land` (where it is), or `return_and_land` (fly to the spawn x, y, then land) |
| `warn_remaining_pct`, `land_remaining_pct` | 30, 15 | Thresholds on remaining charge |
| `critical_cell_voltage_v` | 3.3 | Below this per-cell loaded voltage, land immediately (no return) |
| `descent_mps`, `return_speed_mps`, `home_radius_m` | 0.5, 3, 1 | Landing descent, return speed, arrival radius |
| `touchdown_height_m`, `touchdown_speed_mps`, `touchdown_hold_s` | 0.05, 0.1, 0.5 | Touchdown: body bottom within this height, vertical speed below this, held this long |

States `normal → warned → (returning →) landing → landed`. From
`returning` on, pilot and scripted commands are overridden. Landing lifts
the controller's minimum-altitude floor (`ManualController.allow_landing`)
and descends at a constant rate; at touchdown the motors are disarmed
(`RotorControlLoop.disarm`). Events go to `summary.json` (`failsafe`), Rerun's
`events/failsafe` log, and `status/failsafe_state` (0–4).

Verified in lockstep with `vehicles/battery_low.json` (15 % charge, which
otherwise runs out after about 48 s): with `land` at 10 % it began landing at
16.6 s, touched down and disarmed at 19.2 s with 9 % left; with
`return_and_land` after flying 6.7 m away it returned to within 0.77 m of
home and landed with 7 % left. Example layer:
`configs/control/failsafe_rth.json`.

## Battery monitor (`sensors.battery_monitor`)

A power-module model (`sensors/battery_monitor.py`) measuring the simulated
pack, off by default; requires `battery.enabled`.

| Key | Units | Default | Meaning |
| --- | --- | --- | --- |
| `enabled`, `rate_hz` | | off, 10 Hz | Fixed simulation-time grid, at most the odometry rate |
| `voltage_noise_std_v` | V | 0 | White noise on measured voltage |
| `current_noise_std_a` | A | 0 | White noise on measured current |
| `current_offset_a` | A, any sign | 0 | Constant current-sensor offset |
| `seed` | integer | 0 | Local RNG |

Like a simple flight controller it integrates measured current into
`consumed_mah` and reports `remaining_pct = 100 · (1 − consumed / capacity)`
assuming the pack was **full at start**: starting a partly charged pack
overestimates remaining charge, and a current offset makes the estimate
drift. Rerun's **Battery** tab shows measured voltage, current, and remaining
percentage next to the true values. In a 19 s racing run the measured
remaining charge (97.3 %) tracked the true state of charge (97.4 %).

## Aerodynamics (`aerodynamics`)

All effects are off by default; `air_density_kg_m3` (default 1.225) is
shared. Models are in `simulation/aerodynamics.py`.

### Quadratic body drag (`aerodynamics.body_drag`)

`F = −½ ρ (Cd·A)_i v_i |v_i|` per body axis, with v the airspeed (velocity
minus the current wind target) and `drag_area_m2` the drag area Cd·A per
body axis (default [0.012, 0.012, 0.03] m²). Python applies it every odometry
sample through Gazebo's ApplyLinkWrench system
(`simulation/gazebo/body_drag.py`). Measured on gz-sim 8: persistent wrenches
are in world axes and **accumulate**, so each update publishes only the
change. Works in every control mode; use lockstep with a high odometry rate
for accurate force updates.

Verified in lockstep with all motors stopped: falling from 60 m with
Cd·A = 0.3 m², the drone fell 21.50 m in 3 s against 21.44 m for the exact
quadratic-drag solution, reaching 9.04 m/s (terminal velocity 9.07 m/s);
without drag it matched free fall within 1.6 cm.

With wind enabled, Gazebo's linear WindEffects drag also acts; reduce
`wind.force_scale_per_s` when using quadratic drag to avoid counting drag
twice.

### Translational lift (`aerodynamics.translational_lift`)

A rotor moving edgewise meets clean air and produces more thrust at the
same speed. Per rotor, `1 + max_gain · (1 − v_i / sqrt(v_i² + v_h²))` with
v_h the edgewise airspeed and v_i the momentum-theory hover induced velocity
`sqrt(T_hover / (2 ρ π R²))` (5.6 m/s for the default vehicle, 7.0 m/s for
the racing preset). `max_gain` default 0.15. Applied like ground effect, by
scaling published motor speed (rotor-thrust and keyboard modes). In a 6 m/s
racing burst the vehicle flew 4 cm higher than without it.

### Vortex ring state (`aerodynamics.vortex_ring`)

Descending along the rotor axis at about the induced velocity makes rotors
re-ingest their own wake and lose thrust. Mean factor
`1 − thrust_loss · exp(−((d / v_i − 1) / 0.4)²)` for descent speed d
(`thrust_loss` default 0.3), multiplied by seeded per-rotor fluctuation of
standard deviation `fluctuation_std` (0.1) scaled by depth in the state.
Commanding a 5.6 m/s descent with the default vehicle, thrust fell to the
full 30 % loss and the vehicle descended at 6.1 m/s on average instead of
5.1 m/s, ending 8.7 m lower.

### Blade flapping

Not a separate model: Gazebo's motor plugin already applies a rotor drag
force and rolling moment proportional to rotor speed and edgewise airspeed
(`motors.rotor_drag_coefficient`, `rolling_moment_coefficient`), which is how
flapping-induced drag is usually approximated.

### Payload (`payload`)

A box hanging below the body, attached at start by Gazebo's DetachableJoint
system (`simulation/gazebo/payload.py`); off by default.

| Key | Default | Meaning |
| --- | --- | --- |
| `mass_kg`, `size_m` | 0.3, [0.12, 0.12, 0.08] | Payload mass and box size |
| `offset_m` | [0, 0, −0.12] | Payload centre from the body origin (the hook point); must hang below the body |
| `drop_s` | null | Scheduled drop time (s) |
| `pickup_radius_m` | 0.3 | Maximum hook-to-payload distance for pickup |

Dropping detaches it and it falls under physics. Gazebo re-attaches a child
wherever it currently is, so pickup is only requested when the payload is
within `pickup_radius_m` of the hook point, and it stays at that relative
pose (it does not snap onto the hook). With ground start the drone stands on
its payload. The hover check includes payload mass. During keyboard flight
key 8 drops it or, within reach, picks it up.

**The controller is not told about the payload, and has no integral term**,
so carrying mass m makes the drone settle about m·g / (M·K_p) below its
setpoint: 1.6 m for 0.3 kg on the default vehicle, more than its 1 m hover
height. Verified in Gazebo: carrying the default payload the drone sank
until the payload rested on the ground; after a scheduled drop at 6 s it
climbed back to 0.99 m. With keyboard keys, a drop, a descent over the
payload, and a pickup at 0.22 m from the hook all worked, but the drone
could not lift it again. Integral action (planned controller work) removes
this sag.

### Tethered payload on a winch (`payload.tether`)

The payload can instead hang from a hook at the end of a winch cable,
modelled as a variable-length spherical pendulum: a ball joint at
`anchor_m` under the body lets the cable swing freely, and a prismatic joint
along it is the winch, driven by Gazebo's JointPositionController. The
swing pulls back on the drone, the main difficulty of real slung loads.
The cable is rigid (it cannot go slack), so a ground start is rejected.

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | false | Use the tether instead of the rigid hook (`offset_m` is then unused) |
| `anchor_m` | [0, 0, −0.07] | Winch point under the body |
| `initial_length_m`, `max_length_m` | 1.0, 3.0 | Cable length at start and maximum (minimum 0.1 m) |
| `winch_speed_mps` | 0.5 | Reel speed |
| `swing_damping` | 0.0005 | Ball-joint damping (N·m·s/rad); swing decay |
| `length_schedule` | `[]` | Timed `{time_s, length_m}` targets for repeatable runs |

During keyboard flight **W** lowers and **S** raises the hook (held keys);
key 8 releases the payload or, within `pickup_radius_m` of its top, hooks
it. The cable is drawn in Rerun and its length is plotted under `payload`.
Example: `configs/scenarios/tethered_delivery.json` (light 0.05 kg payload,
spawn at 2.5 m).

Verified in lockstep and with scripted keys: after a forward burst a 1 m
cable with a 0.05 kg payload swung with a 2.02 s period (fixed-pivot pendulum
2.08 s; the moving drone shortens it), 0.40 m amplitude decaying about 9 %
per period, and shook the drone 0.52 m peak to peak; the winch followed its
schedule at 0.5 m/s; a drop, lowering to 3 m, pickup 0.07 m from the
payload's top, and reeling in lifted the payload from the ground to 0.97 m.
With the default 0.3 kg the controller's sag lowered the drone until the
payload rested on the ground, so use light payloads until integral action
exists.

## Fixed wing and quadplanes (`fixed_wing`)

Rotor-thrust mode only. `simulation/fixed_wing.py` computes wing, tail, and
control-surface forces in Python and applies them, with body drag, as one
persistent ApplyLinkWrench. That wrench acts at the link origin, which was
verified with an offset centre of mass. The wing sees airspeed relative to
DroneLab's wind.

The model is the coefficient model of Beard & McLain, *Small Unmanned
Aircraft* (2012), ch. 4:
- **Lift:** linear lift blended into flat-plate lift past the stall angle
  (`stall_alpha_rad`, sharpness `stall_sharpness`).
- **Drag:** parasitic plus induced drag (`oswald_efficiency`, aspect ratio
  from `wingspan_m`²/`wing_area_m2`).
- **Side force and moments:** side force, and roll, pitch, and yaw moments
  with rate damping and aileron/elevator/rudder terms.

Defaults are the Aerosonde coefficients of the book's companion simulator;
`fixed_wing.coefficients` overrides individual entries (names in
`DEFAULT_COEFFICIENTS`).

Conventions (FLU body axes):
- positive elevator pitches nose down,
- positive aileron rolls the right wing down,
- positive rudder yaws the nose left.

Servos (`fixed_wing.servos`) apply a travel limit (`max_deflection_rad`), a
rate limit (`rate_limit_rad_s`), and first-order lag (`time_constant_s`) to
the controller's surface commands. A controller provides them through an
optional `surface_commands()` ([7 Experiments](7_experiments.md#algorithm-plug-ins-algorithms)).
Rerun logs `aero/fixed_wing` (airspeed, α, β) and `aero/surfaces_deg`. Wing,
tail, and fin are drawn as visuals only; contacts use the fuselage box.

**Pusher power.** Non-vertical rotors see axial inflow, so the powertrain
uses actuator-disk momentum theory for them:
v_i = −V/2 + √((V/2)² + T/(2ρA)) and P = T(V + v_i)/FM, with
`motors.forward_figure_of_merit` (0.8). Lift rotors keep the static
c_m·k_t·ω³ model, so quad results are unchanged.

### Airspeed sensor (`sensors.airspeed`)

A synthetic pitot-static sensor along body +x (`sensors/airspeed.py`); off
by default. It measures Δp = `scale`·½ρu² + `offset_pa` + N(0,
`noise_std_pa`²) and reports indicated airspeed √(2Δp/ρ₀) with ρ₀ =
1.225 kg/m³. `outages_s` intervals model a blocked pitot (Δp reads only
offset and noise).
- **Logging:** Rerun shows `airspeed/measured`.
- **Bus:** the sensor bus carries `airspeed`.
- **Controller state:** with `control_state_source: estimate`,
  `ControlState.airspeed_mps` is the latest delivered reading; with truth
  control it is the true body-x airspeed.

### Kite-like quadplane preset

`configs/vehicles/kite_quadplane.json` is a plausible quadplane in the class
of the IMS Kite, using that aircraft's published figures. It is not a model
of the real Kite. Where the published figures stop, values are derived or
assumed:

| Quantity | Value | Source |
| --- | --- | --- |
| Total mass | 22 kg | 17 kg empty + 5 kg standard payload (published MTOW < 25 kg) |
| Wingspan | 2.99 m | Published |
| Wing area, chord | 0.932 m², 0.312 m (AR 9.6) | From the published 60 km/h stall with CLmax 1.36 |
| Stall angle | 0.27 rad (CLmax at 12.7°) | Chosen so CLmax is typical of small UAV wings (the Aerosonde set reaches 2.4) |
| Pitch trim | Cm0 = 0.25 | Assumed tail incidence: trims near zero elevator at 80 km/h |
| Inertia | [1.8, 2.4, 3.9] kg·m² | Aerosonde values scaled by mass |
| Lift rotors | 4 × 22-inch at (±0.65, ±0.75) m, 110 N max each (T/W 2.04) | Assumed |
| Pusher | Same motor and propeller at x = −0.85 m along +x; the autopilot caps it at 66 N | Assumed |
| Battery | 12S 45 Ah, 85 % powertrain efficiency, 40 W avionics | Sized for the published 180 min at 80 km/h (≈ 0.6 kW cruise) |

The model predicts:
- **Cruise at 80 km/h:** C_L 0.77, α 5.5°, L/D 11.8, drag 18 N, about
  595 W electrical.
- **Hover:** about 3.7 kW electrical. In lockstep the preset hovered stably
  at 3 m, drawing 21.1 Wh in 20 s (3.8 kW).

**Verified in Gazebo** (lockstep, airfield, baseline autopilot), against
the published figures:

| Check | Model | Gazebo | Published |
| --- | --- | --- | --- |
| Cruise at 80 km/h: α, elevator | 5.5°, ~0° | 5.33°, 0.6° | – |
| Cruise drag (= pusher thrust) | 18.2 N | 18.3 N | – |
| Cruise electrical power | ~635 W | 672 W (pusher induced loss included) | – |
| Endurance on 2.05 kWh (12S × 3.8 V × 45 Ah) | – | ≈ 183 min | 180 min |
| Minimum level-flight speed (airspeed ramp 22 → 14 m/s) | 16.7 m/s (60 km/h) at CLmax | ≈ 17.2 m/s (62 km/h); stall at α > 12.7°, 30 m lost in 10 s | 60 km/h |
| Hover power | 3.7 kW | 3.8 kW | – |

The endurance agreement follows from sizing the battery to the published
endurance. What the check verifies is that the simulated drag, propeller,
and power chain deliver the predicted cruise power. Speeds and power are
plausible-model outputs, not measurements of the real aircraft.

`worlds/airfield.json` is a 3 km × 3 km ground (scene key `ground_size_m`,
10–20000 m) with a hangar, a tower, and a tree line. Beyond the ground's
edge nothing collides, and a vehicle there falls forever. Runs print a
warning and record `left_ground_area_s` in `summary.json`.

## Ground effect (`aerodynamics.ground_effect`)

Near the ground a rotor produces more thrust at the same speed.
`simulation/aerodynamics.py` uses the Cheeseman-Bennett model
`T_IGE / T_OGE = 1 / (1 − (R / 4z)²)` for each rotor's height z above flat
ground at z = 0 (from the true pose, including tilt) and rotor radius
R = `vehicle.propeller_diameter_m` / 2 (default 0.125 m), using
`vehicle.rotor_height_m` for the rotor plane. Height is clamped at R/2 (factor 4/3).
The effect is about 2 % at z = 1.8 R and under 1 % above 2.5 R, so it only
matters within roughly 0.3 m of the ground for the default rotors.
Rotor-thrust and keyboard modes only; off by default.

Gazebo's motor model cannot change its thrust coefficient, so the published
motor speed is raised by `sqrt(factor)` to produce the extra thrust. Gazebo's
reaction torque rises by the same factor (a small inaccuracy). Obstacles
under the vehicle are ignored: only the ground plane counts. With
`configs/scenarios/ground_effect_hover.json` the vehicle commanded to 0.12 m settled at
0.22 m (0.13 m without ground effect), because the controller has no
integral term to cancel the extra thrust.

## Vehicle presets

Each preset in `configs/vehicles/` sets the airframe, motors, battery, and
controller gains for one class of quadrotor; combine it with sensor,
environment, and fault layers. Values are plausible for the class, not a
measured vehicle. `hexa_x.json` adds a hexacopter through the general rotor
layout ([Vehicle and motors](#vehicle-and-motors)).

| Preset | Mass | Thrust/weight | Hover speed | Motor lag (up) | Battery | Attitude ωn, ζ |
| --- | --- | --- | --- | --- | --- | --- |
| `micro` | 0.030 kg | 1.95 | 72 % | 72 ms | none (see below) | 6 rad/s, 1.0 |
| `racing_5in` | 0.65 kg | 9.9 | 32 % | 15 ms | 6S 1300 mAh | 20 rad/s, 0.7 |
| `freestyle_5in` | 0.76 kg | 7.6 | 36 % | 18 ms | 6S 1300 mAh | 18 rad/s, 0.7 |
| `10in_4s` | 1.54 kg | 2.3 | 66 % | 40 ms | 4S 5000 mAh | 7.7 rad/s, 0.75 |
| `heavy_lift` | 6.0 kg | 2.1 | 69 % | 80 ms | 6S 16000 mAh | 5 rad/s, 0.7 |

**Gains.** The controller's attitude gains are torques per radian, not
normalized by inertia, so each preset derives them from its inertia I and a
chosen natural frequency ωn and damping ζ: `Kp = I·ωn²`,
`Kd = 2·ζ·ωn·I`, with ωn well below the motor bandwidth `1/τ`. Changing
mass or inertia without retuning gains will make a vehicle sluggish or
unstable.

**Sources and assumptions.**

- `micro`: thrust and yaw-moment coefficients, inertia, arm length, maximum
  speed, and the identified 72 ms motor lag come from RotorPy's Crazyflie
  parameters (Förster 2015; Eschmann et al. 2024). Its Izz (2.89e-5) violates
  the rigid-body triangle inequality (Izz ≤ Ixx + Iyy = 2.86e-5), so the
  preset uses 2.85e-5. With that yaw-moment coefficient the power model
  predicts about 18 W of hover shaft power, roughly 2–3× a real Crazyflie,
  so the battery is left off rather than giving a misleading endurance.
- `racing_5in`: 5.1-inch props, about 1.6 kgf per motor near 32 000 rpm;
  220 mm frame. Includes the 250 Hz attitude / 50 Hz position control timing
  with publish-on-send, still-air drag (0.15/s), and a 10 m/s² acceleration
  limit.
- `freestyle_5in`: the acrobatic sibling of the racer, flown the same way (acro mode on real
  drones) but heavier (sturdier frame and a GoPro-class camera on top, which
  raises roll/pitch inertia), with mid-range motors and props (about 1.45 kgf
  per motor near 28 000 rpm) for smooth throttle control, and slightly softer
  attitude tuning. Includes the 250/50 Hz control timing and still-air drag.
- `10in_4s`: see the motor and propeller notes below.
- `heavy_lift`: 15-inch props, about 3.2 kgf per motor; 700 mm frame; hover
  shaft power about 530 W (momentum theory with figure of merit 0.7 gives
  about 610 W), roughly 28 min on 16 Ah.

**Gazebo verification.** Each preset flew a scripted rotor-thrust manoeuvre
(hover, forward burst at the preset's pilot speed, yaw burst, climb, each
followed by a hold) in an empty scene:

| Preset | Max tilt | Settled error after holds | Notes |
| --- | --- | --- | --- |
| `micro` | 5.4° | 0.03–0.18 m | Same results with a 1 ms physics step, so the default 2 ms step suffices |
| `racing_5in` | 52° | 0.03–0.3 m | Brakes from 6 m/s; brief motor saturation |
| `freestyle_5in` | 49° | 0.01–0.26 m | Brakes from 5 m/s; 3.5 rad/s yaw holds within 3 cm |
| `10in_4s` | 15° | 0.03–0.06 m | |
| `heavy_lift` | 16° | 0.05–0.2 m | Sags about 0.65 m after spawning while slow motors spin up |

All presets spawn airborne with motors stopped, so each sags briefly at
start. **Racing with the FPV camera** (`scenarios/racing_fpv.json`) is the
most timing-sensitive case: processing 640×480 frames at 30 Hz stalls the
single-threaded Python loop for 20–27 ms about 30 times per run. In
real-time mode one of three runs flipped during hard braking at 10 m/s². In
[lockstep](2_simulation.md#lockstep-simulation) both runs were stable (57°
peak tilt) and bit-identical, at 0.83× real time. Use lockstep for
timing-sensitive racing experiments. Presets with a battery work only in
rotor-thrust and keyboard modes.

**Racing behaviour.** Real racing drones fly in acro (rate) mode, where the
pilot commands rotation rates. The built-in controller holds position and
limits acceleration, so `racing_5in` and `freestyle_5in` have the airframes but not the
handling; a rate-mode controller is future work (flips need it). Reversible
"3D" motors for inverted flight would need negative thrust, which the
schema does not allow.

## Motor and propeller preset

`configs/vehicles/10in_4s.json` describes a plausible 1.5 kg quad with
10-inch propellers on a 4S pack. The default thrust coefficient (8.55e-6
N·s², 8.55 N at 1000 rad/s ≈ 9500 rpm) and reaction moment ratio (0.016 m)
are already in the range of published 10-inch propeller data (about 0.9 kgf
per rotor near 10 000 rpm), so the preset keeps them and adds slower motor
response (40/60 ms up/down, typical of larger propellers), a full 5000 mAh
battery, still-air drag, and ground effect. In a 20 s hover it stayed within
7 cm and used 1.3 Wh. These are plausible values, not a measured vehicle;
gains are not retuned.

## Camera realism (`sensors.camera`)

Lens projection is rendered by Gazebo; other effects are applied in Python
to each frame before recording (`sensors/camera_effects.py`).

| Key | Default | Meaning |
| --- | --- | --- |
| `lens.type` | `pinhole` | `pinhole` (Gazebo camera, FOV < π) or a native wide-angle projection: `equidistant`, `stereographic`, `equisolid_angle`, `orthographic` (Gazebo `wideanglecamera`, FOV up to 2π) |
| `lens.k1`, `k2`, `k3`, `p1`, `p2` | 0 | Brown–Conrady distortion, applied in Python: gz-rendering's ogre2 backend reports "ImageBrownDistortionModel is not supported" |
| `exposure_time_s` | 0 | Motion blur over the exposure |
| `rolling_shutter_readout_s` | 0 | Rows read top to bottom over this time; 0 = global shutter |
| `latency_s` | 0 | Frames are recorded at capture time + latency |

Blur and rolling shutter use the image motion caused by the camera's
rotation at capture (yaw and pitch rates from odometry, at the image
centre); translation and roll are ignored, and shifts are whole pixels.
Verified with the racing FPV scenario in lockstep (150° equidistant fisheye,
8 ms exposure, 20 ms readout, 25 ms latency): hover frames were sharp, frames
during a 3 rad/s yaw showed up to 21 px of blur and slanted gate posts, and
all 576 frames were delivered. A fisheye also sees close parts of the
vehicle: the heading marker is therefore drawn as a stripe on top of the
body's front half, scaled to the body, instead of a box in front of it.

Not modelled: auto-exposure, vignetting, chromatic aberration, dynamic range,
equirectangular 360° output (a 2π fisheye gives an all-around view in one
circular image), and multiple cameras (one camera per vehicle).

## Magnetometer (`sensors.magnetometer`)

A Gazebo magnetometer, off by default. Gazebo computes the Earth field from
the world's latitude/longitude with an internal table, ignoring any world
`<magnetic_field>`. Its defaults publish **gauss** in a **north-east-down**
frame despite the `field_tesla` message name; DroneLab sets the plugin's
`use_units_gauss` and `use_earth_frame_ned` to false so messages carry tesla
in the world's ENU frame, rotated into sensor axes. Verified in Gazebo: at
the Madrid reference the field is 25.6 µT north and 37.2 µT down (45.1 µT,
inclination 55°), and with up to 16° of tilt the reading matched the field
rotated into the sensor axes to 0.0005 µT. The table gives **zero
declination** here (the real value is about +1°).

| Key | Units | Default | Meaning |
| --- | --- | --- | --- |
| `enabled`, `rate_hz`, `mount` | | off, 50 Hz, origin | As for other sensors |
| `field_ut.bias`, `field_ut.noise_stddev` | µT, 3 axes | 0 | Gazebo native Gaussian error (not seeded) |
| `error_model.hard_iron_ut` | µT, 3 axes | 0 | Constant offset from magnetized parts |
| `error_model.soft_iron` | 3×3 | identity | Field-dependent distortion/scale, applied as `S @ field` |
| `error_model.motor_interference_ut_per_n` | µT/N, 3 axes | 0 | Offset per newton of total commanded thrust (rotor-thrust mode; zero otherwise) |
| `error_model.current_interference_ut_per_a` | µT/A, 3 axes | 0 | Offset per ampere of battery current (needs the battery model) |
| `error_model.drift_std_ut`, `drift_correlation_s` | µT, s | 0, 300 | Slow Gauss–Markov offset per axis (magnetization, temperature) |
| `dipoles` | list | [] | Point-dipole anomalies `{position_m, moment_am2}` (world); scene obstacles add `magnetic_moment_am2` |
| `error_model.white_std_ut` | µT, 3 axes | 0 | Per-sample white noise |
| `error_model.enabled`, `seed` | | off, 0 | Seeded Python model (`sensors/magnetometer.py`) |

`measured = S @ field + hard_iron + k_T·total_thrust + k_I·battery_current + drift(t) + white`.

**Anomalies near structures** (`sensors/magnetic_anomalies.py`): each
dipole adds `B = μ₀/4π · (3r̂(m·r̂) − m)/|r|³` at the sensor's true position,
rotated into sensor axes with the true attitude, before the sensor errors.
Distances are clamped to 0.5 m. A 1000 A·m² moment, about a parked car,
gives 1.6 µT at 5 m. Scene obstacles with `magnetic_moment_am2` (world
axes) become dipoles at their centres; `summary.json.magnetic_anomalies`
reports the largest anomaly seen. Verified in lockstep: a 2000 A·m²
vertical dipole 2 m beside a hover changed the measured z field by
−24.999 µT, against −25.0 µT predicted (equatorial −μ₀/4π·m/r³).

Rerun's **Magnetometer** tab shows the field and a level-compass yaw
`atan2(m_x, m_y)` (valid only near level, ignoring declination). With
`configs/sensors/magnetometer.json` in a hover the compass read 16.6° for a
true heading of about 0°, typical of an uncalibrated compass (0.02° after calibration, below). Not modelled:
induced (soft-iron) magnetization of structures, which is a fixed moment
here, and anomalies with extent (fields of large bodies are approximated by
one dipole).

### Magnetometer calibration (`sensors.magnetometer.calibration`)

Calibration removes the vehicle's own predictable distortions. It does not
remove noise, drift after calibration, changes to the vehicle (payload,
wiring), or anomalies in the environment, so filters should still treat the
compass as imperfect. Real autopilots split the work three ways, and so does
DroneLab:

**1. Procedure: `dronelab-magcal`** (`experiments/magnetometer_calibration.py`),
the simulated ground-station compass calibration:

```bash
dronelab-magcal --drone configs/sensors/magnetometer.json --output outputs/magcal
dronelab ... --drone configs/sensors/magnetometer.json --drone outputs/magcal/magnetometer-calibration.json
```

- **Orientation dance:** Gazebo runs the configured vehicle with gravity
  off. The drone is turned through `--orientations` (120) seeded random
  orientations with the `set_pose` service, standing in for hand rotation,
  and `--samples` readings are taken at each.
- **Throttle sweep ("compassmot"):** held level, the total thrust command
  steps from zero to maximum.
- **Raw readings:** Gazebo supplies the Earth field; the mission's error,
  battery, and battery-monitor models turn it into raw readings, exactly as
  in flight. The interference regressor is the battery monitor's measured
  current when the vehicle has one, else the commanded thrust.
- **Evaluation:** every fifth orientation is held out and compared with the
  true field.
- **Outputs:** `magnetometer-calibration.json`, a layer to pass with
  `--drone`, and `calibration-report.json` with fit quality, before/after
  errors, and the injected truth.

**2. Fitting** (`estimation/magnetometer_calibration.py`, transport-free):
- **Hard and soft iron:** an ellipsoid fit by total least squares on a
  general quadric. The centre is the hard-iron offset; the symmetric square
  root of the shape, scaled to the local field strength, is the soft-iron
  correction. A pure rotation is indistinguishable from the sensor mount.
- **Interference:** a per-axis linear regression of readings on current or
  thrust.

**3. Use.** Corrected field b̂ = C (m − o − k_I·I − k_T·T):

| Key | Units | Meaning |
| --- | --- | --- |
| `enabled` | | Apply the calibration and expose it to filters |
| `hard_iron_offset_ut` | µT | Offset o |
| `soft_iron_correction` | 3×3 | Correction matrix C |
| `current_compensation_ut_per_a`, `thrust_compensation_ut_per_n` | µT/A, µT/N | Interference coefficients k_I, k_T |
| `field_strength_ut`, `inclination_deg`, `declination_deg` | µT, deg | Local Earth field (defaults: Gazebo's field at the default site, as a geomagnetic model would give) |
| `field_tolerance_ut` | µT | How far \|b̂\| may differ from the field strength before a reading is suspect |
| `heading_std_deg` | deg | Residual heading uncertainty (set from the held-out evaluation) |

When enabled, the autopilot side applies it like a driver:
- **Corrected readings:** it uses the battery monitor's measured current
  and the commanded total thrust, never true values. `MagnetometerReading`
  on the sensor bus then carries `calibrated_field_ut` next to the raw
  `field_ut`.
- **Rerun:** logs `magnetometer/calibrated_ut` and
  `magnetometer/calibrated_level_heading_deg`.
- **Filters:** estimators receive the whole section, including field
  strength, inclination, declination, tolerance, and heading uncertainty,
  as `context.sensors["magnetometer"].calibration`.

Verified with `configs/sensors/magnetometer.json` (default quad, no battery,
so thrust regression):
- **Speed:** the run took 20 s.
- **Recovered values:**
  - hard iron (4.04, −2.54, 1.49) µT against injected (4.0, −2.5, 1.5);
  - soft iron to within 0.2 % (correction × distortion = identity);
  - motor interference (0.145, −0.050, 0.099) µT/N against
    (0.15, −0.05, 0.10).
- **Held-out heading error:** 6.7° RMS (12.2° max) before, 0.84° RMS
  (2.3° max) after. Field-magnitude error dropped from 2.5 to 0.28 µT.
- **In a hover pointed at 0°:** the raw level compass read 16.6°, the
  calibrated one 0.02° ± 0.64°.

## Rangefinder (`sensors.rangefinder`)

A downward Gazebo `gpu_lidar` with one ray along the sensor's +x axis, off by
default. It sees the actual world geometry, including obstacles, and needs
the rendering system (EGL when headless), like the camera and lidar.

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled`, `rate_hz` | off, 20 Hz | |
| `mount` | 0.07 m below the body origin, pitched 90° (pointing down) | Body-relative pose |
| `min_range_m`, `max_range_m` | 0.1, 12 m | Readings outside are dropped (NaN) |
| `range_m.bias`, `range_m.noise_stddev` | 0 m | Gazebo native Gaussian error |

Verified in Gazebo: range matched the geometric distance to the ground along
the ray to within 1 mm during a tilting flight. Rerun: **Rangefinder and
flow** tab.

## Optical flow (`sensors.optical_flow`)

Gazebo has no optical-flow sensor, so `sensors/optical_flow.py` synthesizes
one from the true motion (truth generates the measurement; nothing but the
noisy output would reach an estimator). The optical axis is the sensor's +x
axis; the default mount points it down. For the ground point on that axis at
distance d, with sensor velocity v and angular velocity ω in sensor axes,
the reported raw flow is the image angular rate

`flow_rad_s = (−v_y / d − ω_z,  −v_z / d + ω_y)`

including body rotation and the mount's lever arm, as real sensors report
it. With the default mount, forward motion gives negative `flow_rad_s[1]`.

| Key | Units | Default | Meaning |
| --- | --- | --- | --- |
| `enabled`, `mount` | | off, as rangefinder | |
| `rate_hz` | Hz | 50 | Fixed simulation-time grid, at most the odometry rate |
| `min_distance_m`, `max_distance_m` | m | 0.08, 8 | Valid distance range |
| `max_rate_rad_s` | rad/s | 7 | Larger rates are invalid (sensor saturation) |
| `noise_std_rad_s` | rad/s | 0 | Per-sample white noise per axis |
| `seed` | integer | 0 | Local RNG |

The ground is the plane z = 0 (obstacles are ignored, unlike the
rangefinder). Invalid samples (axis not pointing at the ground, distance out
of range, saturation) report NaN flow. Not modelled: texture and lighting
dependence, a quality metric, rolling shutter, and integration intervals.

## Lidar channels (`sensors.lidar`)

Setting `vertical_samples` above 1 makes the existing lidar a multi-channel
3D lidar: `vertical_min_angle_deg` and `vertical_max_angle_deg` (−90 to 90)
set the elevation range. Default 1 keeps the original horizontal 2D scan.
Rerun shows the 3D point cloud under the drone. Verified in Gazebo with 16
channels at ±15°: decoded elevations spanned −15° to 15° and ground returns
lay exactly at the true height, confirming the row ordering.

### Lidar realism (`sensors.lidar.effects`)

Gazebo's gpu_lidar is close to ideal, so `sensors/lidar_effects.py` adds, in
the lidar's own axes before recording:

| Key | Default | Meaning |
| --- | --- | --- |
| `noise_std_m`, `noise_per_m` | 0 | Range noise along each ray, σ = σ₀ + k·r |
| `dropout_probability` | 0 | Uniform probability p₀ that a return is lost |
| `far_dropout_exponent` | 0 | Extra losses towards maximum range: p = p₀ + (1 − p₀)(r/r_max)ⁿ; 0 disables |
| `scan_time_s` | 0 | Sweep duration for motion distortion (0 = instantaneous, Gazebo's behaviour) |
| `beam_divergence_rad` | 0 | Full-angle beam divergence δ for mixed-pixel edge returns; 0 disables |
| `edge_min_jump_m` | 0.3 | Range jump between neighbouring beams that counts as a depth edge |
| `range_resolution_m` | null | Edge surfaces closer than this merge into one ghost return; farther ones become separate returns. null: always merge |
| `return_mode` | first | Returns reported per beam: `first`, `last`, `strongest`, or `dual` (first plus a distinct strongest) |
| `detection.enabled` | false | Drop returns too weak to detect (reflectivity, range, weather) |
| `detection.reference_range_m`, `detection.reference_reflectivity` | 40, 0.1 | Datasheet detection limit, "40 m at 10 %" |
| `detection.default_reflectivity` | 0.5 | Surfaces without a scene reflectivity |
| `seed` | 0 | Local RNG |

Before noise, dropouts and distortion, `sensors/lidar_returns.py` handles
per-beam return physics on the full scan grid, including beams that hit
nothing:

- **Reflectivity and detection.** Received power ∝ ρ·exp(−2αr)/r²; a return
  is kept when it reaches ρ_ref/R_ref². A target of reflectivity ρ is
  therefore seen to R_ref·√(ρ/ρ_ref) in clear air (50 % grey to 89 m, 2 %
  black to 18 m, before `max_range_m` cuts off). ρ comes from the scene
  (`reflectivity` per obstacle, `ground_reflectivity`), written into the SDF
  as `<laser_retro>` = 100·ρ and reported back by Gazebo's gpu_lidar as the
  beam intensity (checked empirically: unset retro reads 0, which maps to
  `default_reflectivity`). Values above 1 model retroreflectors.
- **Weather** (`environment.weather`, below) attenuates returns and adds
  near-range backscatter clutter.
- **Beam spread (mixed pixels).** Where neighbouring beams (along the sweep
  and across channels) straddle a depth jump, the edge lies uniformly
  between their centres. A beam whose footprint of width δ covers it
  receives a share w of its energy from its own surface and 1 − w from the
  other, and reports the power-weighted mean range: a ghost point between
  foreground and background. Energy falling on open sky is lost, so edge
  returns may drop out, and sky beams grazing an object see it weakly
  (objects look slightly wider). The probability that a beam touches an
  edge is (δ/2)/Δ for angular step Δ. Only one edge per beam is applied,
  and the sweep does not wrap around.

### Weather (`environment.weather`)

| Key | Units | Default | Meaning |
| --- | --- | --- | --- |
| `visibility_m` | m | null (clear) | Meteorological visibility V; lidar extinction α = 3.912/V·(905/550)^(−q) (Kim et al. 2001; q = 0 below 500 m) |
| `lidar_clutter_per_m` | 1/m | 0 | Rate of false returns along each beam (rain drops, dust, fog backscatter) |
| `lidar_clutter_max_range_m` | m | 10 | Clutter only occurs this close to the sensor |
| `lidar_clutter_reflectivity` | – | 0.05 | Return strength of a clutter echo (detection applies to it too) |

Along each beam, the first clutter event is exponentially distributed; it
replaces the real return if it comes first, so 1 − exp(−λ·d_max) of open-sky
beams return clutter. Attenuation acts only through detection, so a
`visibility_m` with an enabled lidar requires `detection.enabled`. Weather
affects the lidar only; camera images are not fogged.

Example layers (illustrative values, not measured weather):

| Layer | Visibility | Clutter rate | Clutter range |
| --- | --- | --- | --- |
| `configs/environment/fog.json` | 80 m | 0.08 /m | 6 m |
| `configs/environment/rain.json` | 1500 m | 0.01 /m | 15 m |
| `configs/environment/dust.json` | 200 m | 0.05 /m | 10 m |

Each also enables lidar detection. Verified in lockstep with the Livox layer
in the circuit for 8 s (about 23k beams per scan, `summary.json`
`lidar_effects`):

| Run | Clutter returns | Undetected returns | Mixed edge returns |
| --- | --- | --- | --- |
| Clear air | 0 | 549 (0.03 % of beams) | 38,735 (2.1 %) |
| Fog | 681,532 (36.5 %; 37 % bound for open sky) | 12,694 | 38,415 |
| Clear, all obstacles at 5 % reflectivity | 0 | 17,689 | 38,490 |

Fog extinction was 0.0489 /m (3.912/80).

**Motion distortion**: a real lidar sweeps over `scan_time_s` while the
vehicle moves, so each point is measured from a slightly different pose.
Point i, at sweep fraction aᵢ, is expressed in the sensor frame at
Δᵢ = (aᵢ − ½)·T from mid-scan: `p' = R(−ω·Δᵢ)(p − v·Δᵢ)`, with the sensor's
velocity (including the mount lever arm) and angular rate from odometry.
Removing it ("deskewing") is left to perception code, as on real hardware.

**Livox Mid-360-like layer** (`configs/sensors/lidar_livox_mid360.json`): 360°
around, −7° to +52° vertically in 32 channels × 720 azimuth steps (about 23k
rays per 10 Hz scan, close to the real sensor's 200k points/s), 0.1–40 m,
2 cm noise plus 0.05 % of range, 2 % dropouts with far-range losses, and a
0.1 s sweep, 4 mrad beam divergence, and detection to 40 m at 10 %
reflectivity. The real sensor's non-repetitive rosette pattern is approximated
by a regular grid. Verified in lockstep in the circuit: about 4,700 returns
per frame (upward rays mostly see sky), 95 % of returns kept, and, isolating
distortion by comparing two otherwise identical deterministic runs, 0.1 cm
mean distortion in hover and 23 cm (74 cm max) during a 0.7 rad/s yaw,
against about 19 cm predicted from rotation alone. 19 s of flight took 9 s
in lockstep.

**Multiple returns and intensity.** Each beam has up to three candidate
returns, each with strength share·ρ·exp(−2αr)/r²:
- the surface it hits,
- the other surface at a depth edge (unless merged by `range_resolution_m`),
- a weather clutter echo, which only occurs in front of the nearest hard
  surface.

Detection drops weak candidates, and `return_mode` picks what the beam
reports. Every point carries a reported reflectivity, the range-normalized
strength P·r² = share·ρ·exp(−2αr), so edges and fog read darker, as in real
intensity images:
- Rerun colours the points by it.
- The sensor bus carries it in `LidarPoints.reflectivity`.
- `summary.json.lidar_effects` counts `second_returns`.

Verified in lockstep with the Livox layer in the circuit (6 s):
- **Rain:** first-return mode reported 181,547 clutter echoes, and
  last-return mode 151,829. Beams with a target behind the rain now report
  the target, while beams aimed at open sky still only see drops.
- **Dual mode with 0.5 m resolution (clear air):** 6,902 resolved second
  returns at depth edges.

Not modelled: angle of incidence (surfaces are Lambertian at any angle),
sunlight noise, and the rosette pattern. Moving obstacles have no
reflectivity setting.

## Barometer (`sensors.baro`)

A Gazebo air-pressure sensor on the body, off by default. It measures static
pressure in Pa; DroneLab converts it with the International Standard
Atmosphere (`sensors/barometer.py`) to **pressure altitude above sea level**,
comparable with GPS altitude (both use the world's 650 m elevation datum).
Near 650 m, pressure falls about 11 Pa per metre.

| Key | Units | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | bool | false | Add the sensor and subscribe to `/dronelab/baro` |
| `rate_hz` | Hz | 50 | Sample rate (≤ physics rate) |
| `mount` | m, deg | origin | Body-relative position/rotation, like other sensors |
| `pressure_pa.bias`, `pressure_pa.noise_stddev` | Pa | 0 | Gazebo native Gaussian error (not seeded) |
| `error_model.enabled` | bool | false | Seeded Python error model below |
| `error_model.offset_m` | m, any sign | 0 | Constant altitude offset (e.g. weather since calibration) |
| `error_model.drift_std_m` | m | 0 | Stationary std of slow Gauss-Markov drift |
| `error_model.drift_correlation_time_s` | s, > 0 | 300 | Drift correlation time |
| `error_model.white_std_m` | m | 0 | Per-sample white noise |
| `error_model.prop_wash_coefficient` | − (any sign) | 0 | Rotor downwash on the static port: error = c · v_i² / 2g, v_i = sqrt(T / 2ρA) from the current commanded thrust (rotor-thrust and keyboard modes) |
| `error_model.airspeed_coefficient` | − (any sign) | 0 | Horizontal airspeed on the static port: error = c · v² / 2g |
| `error_model.seed` | integer | 0 | Local RNG seed |

The error model works in altitude metres (the quantity users reason about)
and converts back to pressure, so the message still carries Pa. Drift is
updated exactly over elapsed simulation time and starts stationary. Real
barometers drift in absolute altitude with weather and temperature while
height *changes* stay accurate; this is the behaviour modelled.
`summary.json` field `baro_errors` reports the offset, final drift, and peak
error, for evaluation only.

Positive coefficients make the barometer read high, as downwash lowering the
static pressure does on most multirotors; real values depend on sensor
placement and foam shielding. Verified in lockstep with the racing preset
and coefficients 0.1 (downwash) and 0.05 (airspeed): hovering it read
0.250 m high (predicted 0.250 m for v_i = 7.0 m/s), and at 6.7 m/s 0.367 m
high (downwash plus 0.116 m from airspeed).

Not modelled: ground effect on the static port, temperature, and altitudes
above 11 km. The EKF
does not use the barometer yet; see [6 Roadmap](6_roadmap.md).

The example `configs/sensors/baro.json` (1.5 m offset, 0.5 m drift,
0.15 m noise) gave, in a 10 s Gazebo hover, a mean error of 1.38 m and a
standard deviation of 0.16 m against 650 m + true height.

## EKF settings

The `ekf` section holds estimator priors and assumptions, deliberately
separate from the actual simulated sensor errors; values are never shared
silently. See [5 Estimation](5_estimation.md#configuration).

## GPS dropouts and wind

Both are off by default. `configs/scenarios/disturbances.json`:

```json
{
  "sensors": {"gps": {"dropout": {"probability": 0.15, "seed": 7, "outages_s": [[10, 15]]}}},
  "wind": {
    "enabled": true, "velocity_enu_mps": [3, 0, 0], "force_scale_per_s": 0.1,
    "rise_time_s": 1, "gust_amplitude_fraction": 0.25, "gust_period_s": 8
  }
}
```

### GPS dropout

Applied by `sensors/dropout.py` to Gazebo fixes at the Python mission
boundary, before Rerun and the EKF; the raw Gazebo topic is unchanged and no
replacement fixes are created.

- `probability`: independent per-fix loss in [0, 1] (default 0).
- `seed`: nonnegative integer for a local RNG (default 0).
- `outages_s`: `[start, end)` intervals in absolute simulation seconds; overlaps
  allowed (default none).
- `zones`: GPS blockout boxes, `[{"min_m": [x, y, z], "max_m": [x, y, z]}]` in
  world metres (default none). Every fix is lost while the vehicle is inside
  one, e.g. under a gate or structure. See [fault zones](#fault-zones).
- Random draws happen for every fix, also during outages and in zones.
  Counter precedence: scheduled, then zone, then random.

`summary.json` field `messages.gps` counts raw arrivals; `gps_dropout` counts
`received`, `delivered`, `random_drops`, `scheduled_drops`, and `zone_drops`. Unexpected
queue losses stay in `dropped`. Dropped-fix timestamps still advance live EKF
ordering, so IMU prediction continues and covariance grows until GPS returns.
Real stalls still trip watchdogs. A completed EKF mission needs at least one
accepted GPS correction.

### Wind

Gazebo's native `WindEffects` system acts on the drone base link (the world
gets the plugin and wind velocity; the link gets `enable_wind`). The force is
`F = body_mass * force_scale_per_s * (wind_velocity - body_velocity)` at the
centre of mass: linear drag, not calibrated aerodynamics
([source](https://github.com/gazebosim/gz-sim/blob/gz-sim8/src/systems/wind_effects/WindEffects.cc)).

| Key | Meaning | Default |
| --- | --- | --- |
| `enabled` | Turn wind on | false |
| `velocity_enu_mps` | Mean world ENU wind, m/s | `[0, 0, 0]` |
| `force_scale_per_s` | Linear force scale, 1/s | 0.1 |
| `rise_time_s` | Smoothing of wind changes, s | 1 |
| `gust_amplitude_fraction` | Horizontal sinusoidal gust, fraction [0, 1] of mean horizontal speed | 0 |
| `gust_period_s` | Gust period, s | 10 |
| `random_seed` | Local RNG seed for random changes | 0 |
| `random_update_period_s` | Hold time of random targets, ≥ physics step | 0.5 |
| `random_speed_std_mps` | Per-update Gaussian horizontal speed variation, clamped at 0 | 0 |
| `random_direction_std_deg` | Per-update Gaussian direction variation | 0 |
| `schedule` | List of `{time_s, velocity_enu_mps}` mean changes, strictly increasing times | `[]` |
| `still_air_drag` | Add WindEffects with zero wind when `enabled` is false, so the body gets linear drag in still air | false |
| `turbulence` | Dryden-style turbulence, see [Turbulence](#turbulence-windturbulence) | off |

```json
{"wind": {"enabled": true, "velocity_enu_mps": [3, 0, 0], "random_seed": 42,
  "random_update_period_s": 0.5, "random_speed_std_mps": 0.5, "random_direction_std_deg": 15,
  "schedule": [{"time_s": 8, "velocity_enu_mps": [0, 3, 0]},
               {"time_s": 18, "velocity_enu_mps": [-2, -2, 0]}]}}
```

- Random speed and direction draws are independent; vertical wind is
  unchanged. With zero mean speed the direction reference is east. The speed
  clamp makes the distribution asymmetric near zero. These are per-update
  standard deviations, not densities.
- Random ticks are fixed in simulation time from zero, so repeatability does
  not depend on odometry polling. Scheduled events replace the mean and do
  not reset the random sequence; periodic gusts act on the smoothed speed.
- `simulation/gazebo/wind.py` publishes dynamic targets to
  `/world/dronelab/wind` at the first odometry sample at or after each
  tick/event; missed ticks consume their RNG draws. Gazebo smooths the change
  (`rise_time_s`) and may take the long way round when crossing ±180°.
- `wind-commands.jsonl` records requested targets and dispatch times, not
  measured wind or forces. Random/scheduled wind needs the Python mission;
  opening the SDF directly keeps only static wind and periodic gusts.
- The EKF receives no wind information. Changing wind requires regenerating
  the world. Generation alone does not verify forces or flight.
- **Body drag.** With wind enabled the force above already includes drag
  relative to the air. Without wind, set `still_air_drag: true` to get the
  same linear drag in still air, using `force_scale_per_s` as the body's drag
  coefficient (0.1/s gives 0.77 N at 5 m/s for the 1.54 kg default, close to
  `½ρC_dAv²` for a small quad at that speed; quadratic drag is not modelled).

### Turbulence (`wind.turbulence`)

`simulation/aerodynamics.py` (`DrydenTurbulence`) adds seeded turbulence to
the published wind targets. Requires `wind.enabled`. Each of the along-wind,
cross-wind, and vertical components is a first-order Gauss-Markov process
with stationary std `intensity_mps[i]` and correlation time
`length_scale_m[i] / U`, where U is the mean horizontal wind speed (at least
`min_advection_speed_mps`): turbulence is carried past the vehicle by the
wind. This matches the Dryden along-wind spectrum; Dryden's lateral and
vertical components are second order and are approximated as first order.

| Key | Units | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | bool | false | Turn turbulence on |
| `intensity_mps` | m/s, ≥ 0 | `[0, 0, 0]` | Along/cross/vertical stationary std |
| `length_scale_m` | m, > 0 | `[20, 20, 5]` | Along/cross/vertical length scales |
| `update_period_s` | s, ≥ physics step | 0.05 | Fixed simulation-time update grid |
| `min_advection_speed_mps` | m/s, > 0 | 1 | Lower bound for U (avoids frozen turbulence in calm air) |
| `seed` | integer | 0 | Local RNG, independent of the random-gust sequence |

Guide values for low altitude (MIL-F-8785C): vertical intensity about 10 %
of the 6 m wind speed, horizontal intensities 1.3–2× vertical, and length
scales of tens of metres near the ground. Gazebo's `rise_time_s` low-pass
filters every target, so keep it short (e.g. 0.1 s) for turbulence. With the
linear drag model, forces are small: in a 20 s hover in 3 m/s wind
(`configs/environment/turbulence.json`) horizontal deviation rose only from 0.58 m to
0.61 m RMS. Updates are published at odometry samples (20 Hz grid by default)
and recorded in `wind-commands.jsonl`.

## Accelerometer vibration

`sensors.imu.vibration` (`sensors/vibration.py`) perturbs Gazebo
accelerometer readings before Rerun and the EKF. Off by default.
`configs/sensors/imu_vibration.json`:

```json
{"sensors": {"imu": {"vibration": {
  "enabled": true, "frequency_hz": 60,
  "accel_amplitude_mps2": [2, 2, 15], "phase_deg": [0, 90, 0],
  "random_std_mps2": [0.3, 0.3, 1], "random_correlation_s": 0.01,
  "clip_limit_mps2": 19.62, "seed": 42}}}}
```

Per sensor axis:
`measured = gazebo_specific_force + amplitude * sin(2π·frequency·t + phase) + random`,
then optionally clamped to `±clip_limit_mps2`.

- Amplitudes are peak m/s², phases degrees, frequency Hz in absolute
  simulation time; applied in IMU sensor axes before the mount rotation.
- `random_std_mps2` are per-axis stationary standard deviations (not
  densities) of a first-order correlated Gaussian process with time constant
  `random_correlation_s`, updated exactly over each sample interval, started
  in its stationary distribution, from a local seeded RNG.
- `clip_limit_mps2` (`null` disables) clips the complete specific force,
  including gravity response and Gazebo noise, so clipping can bias the mean.
  The example ±2 g limit deliberately clips some z peaks.
- Defaults: 60 Hz, zero amplitudes/phases/random stds, 0.01 s correlation,
  no clipping, seed 0.
- Gyro data, timestamps, the raw topic, and physical motion are unchanged.
  The model samples at the IMU rate (frequencies above Nyquist alias) and has
  no oversampling, anti-alias filter, rotor coupling, or gyro vibration.

`summary.json` field `accelerometer_vibration` reports `enabled`, `samples`,
`clipped_samples` (each sample counted once), `clipped_axes_xyz`, and
`peak_before_clip_mps2`. EKF noise assumptions are not retuned, and the
truth-feedback controller does not use these measurements.
