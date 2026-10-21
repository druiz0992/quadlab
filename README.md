# DroneLab

DroneLab is a local drone simulation lab for developing and testing state
estimation and flight-control algorithms. It covers quadrotors, general
multirotors, and a quadplane VTOL. Gazebo Harmonic simulates the vehicle and
sensors, Python adds realistic sensor errors, faults, and timing, Rerun shows
and records the flight, and your estimator and controller plug in through a
typed sensor bus. Runs are scored against truth and can be swept in batches
with regression baselines ([7 Experiments](docs/7_experiments.md)).

Controllers fly on simulator truth by default, or on an estimator's output
when configured. Vehicles are simplified, plausible models, not calibrated
hardware.

*Renamed from QuadLab (2026-10-21): the package, commands (`dronelab`,
`dronelab-batch`), Gazebo world name, topics (`/dronelab/...`), and test
variables (`DRONELAB_TEST_*`) changed accordingly. Existing outputs, saved
worlds, and scripts using the old names need updating.*

Documentation (read in order): [1 Overview](docs/1_overview.md) ·
[2 Simulation](docs/2_simulation.md) · [3 Configuration](docs/3_configuration.md) ·
[4 Control](docs/4_control.md) · [5 Estimation](docs/5_estimation.md) ·
[6 Roadmap](docs/6_roadmap.md) · [7 Experiments](docs/7_experiments.md).
Project rules: [AGENTS.md](AGENTS.md).

## Setup

Prerequisites: Python >= 3.11, and for simulation Gazebo Harmonic with Python
bindings matching your interpreter (here: ROS Jazzy, Python 3.12). pip does
not install Gazebo. Keyboard control needs Tk (`python3-tk`) and a desktop.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,gazebo]'   # use '.[dev]' for tests/world generation only
```

Rerun the install after updating the checkout to refresh console entry points.
Before any Gazebo command, in each new terminal:

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
```

## Running

`dronelab`, `dronelab-gazebo`, and `python -m dronelab` are the same command.
Run from the repository root (scene paths are relative to it) and use a fresh
`--output` directory per run; reusing one overwrites files.

```bash
# Fly with the keyboard in the obstacle circuit (Gazebo headless, Rerun + key window)
dronelab --keyboard --headless --scene worlds/circuit.json --output outputs/keyboard

# Same, with the live EKF estimate shown next to truth
dronelab --keyboard --headless --ekf --scene worlds/circuit.json --output outputs/live-ekf

# Automatic flight to a waypoint with Gazebo GUI, Rerun, camera and lidar
dronelab --render-sensors --duration 20 --output outputs/waypoint

# Window-free recording runs
dronelab --headless --duration 10 --output outputs/headless
dronelab --headless --control-mode rotor-thrust --duration 10 --output outputs/thrust-hover
dronelab --headless --rerun --ekf --control-mode rotor-thrust --duration 15 --output outputs/ekf-hover

# Start on the ground and take off automatically (rotor-thrust/keyboard)
dronelab --keyboard --headless --drone configs/scenarios/ground_start.json --output outputs/ground-start

# Aerodynamics: quadratic body drag, translational lift, vortex ring state
dronelab --headless --rerun --control-mode rotor-thrust --drone configs/vehicles/racing_5in.json \
  --drone configs/environment/aerodynamics.json --drone configs/control/lockstep.json \
  --duration 20 --output outputs/aero

# Tethered payload: W/S lower/raise the hook, 8 releases or hooks the payload
dronelab --keyboard --headless --drone configs/scenarios/tethered_delivery.json --output outputs/delivery

# Terrain and moving (ground and flying) obstacles
dronelab --keyboard --headless --scene worlds/hills.json --drone configs/scenarios/ground_start.json --output outputs/hills
dronelab --headless --rerun --scene worlds/moving.json --control-mode rotor-thrust \
  --drone configs/control/lockstep.json --duration 20 --output outputs/moving

# Vehicle presets: micro, racing 5-inch (with FPV camera scenario), heavy-lift
dronelab --keyboard --headless --drone configs/vehicles/micro.json --output outputs/micro
dronelab --keyboard --headless --scene worlds/circuit.json \
  --drone configs/scenarios/racing_fpv.json --output outputs/racing
dronelab --headless --rerun --control-mode rotor-thrust \
  --drone configs/vehicles/heavy_lift.json --duration 20 --output outputs/heavy
dronelab --keyboard --headless --scene worlds/circuit.json \
  --drone configs/scenarios/freestyle_fpv.json --output outputs/freestyle

# Custom vehicle / sensors / disturbances. Configs are building blocks under
# configs/{vehicles,sensors,faults,environment,control,scenarios}; repeat --drone to layer them.
dronelab --headless --rerun --ekf --control-mode rotor-thrust \
  --drone configs/vehicles/10in_4s.json --drone configs/sensors/sensor_errors.json \
  --drone configs/environment/turbulence.json --duration 30 --output outputs/combined
dronelab --keyboard --headless --drone configs/vehicles/quad_x.json --output outputs/configured
dronelab --headless --rerun --ekf --control-mode rotor-thrust \
  --drone configs/scenarios/disturbances.json --duration 25 --output outputs/disturbances
dronelab --headless --rerun --ekf --control-mode rotor-thrust \
  --drone configs/sensors/imu_vibration.json --duration 20 --output outputs/imu-vibration
dronelab --headless --rerun --ekf --control-mode rotor-thrust \
  --drone configs/sensors/sensor_errors.json --duration 30 --output outputs/sensor-errors

# Vehicle physics: battery running out, turbulence, ground effect, 10-inch/4S preset
dronelab --headless --rerun --control-mode rotor-thrust \
  --drone configs/vehicles/battery_low.json --duration 60 --output outputs/battery
dronelab --headless --rerun --control-mode rotor-thrust \
  --drone configs/environment/turbulence.json --duration 20 --output outputs/turbulence
dronelab --headless --rerun --control-mode rotor-thrust \
  --drone configs/scenarios/ground_effect_hover.json --duration 10 --output outputs/ground-effect
dronelab --keyboard --headless --scene worlds/circuit.json \
  --drone configs/vehicles/10in_4s.json --output outputs/preset

# Livox Mid-360-like 3D lidar with noise, dropouts, and motion distortion (rendering needed)
dronelab --headless --rerun --scene worlds/circuit.json --drone configs/sensors/lidar_livox_mid360.json \
  --drone configs/control/lockstep.json --duration 20 --output outputs/livox

# Lidar in fog: attenuation and near-range clutter (also rain.json, dust.json)
dronelab --headless --rerun --scene worlds/circuit.json --drone configs/sensors/lidar_livox_mid360.json \
  --drone configs/environment/fog.json --drone configs/control/lockstep.json --duration 20 --output outputs/fog

# New sensors: magnetometer, downward rangefinder + optical flow, 3D lidar (rendering needed)
dronelab --headless --rerun --control-mode rotor-thrust \
  --drone configs/sensors/magnetometer.json --drone configs/sensors/downward.json \
  --drone configs/sensors/lidar_3d.json --duration 20 --output outputs/new-sensors

# Low-battery failsafe: return home and land at 15 % (here starting at 15 %)
dronelab --headless --rerun --control-mode rotor-thrust --drone configs/vehicles/battery_low.json \
  --drone configs/control/failsafe_rth.json --drone configs/control/lockstep.json \
  --duration 40 --output outputs/failsafe

# Cold weather (0 °C): cold pack sags more and runs flat sooner; IMU biases drift as it warms
dronelab --headless --rerun --control-mode rotor-thrust --drone configs/vehicles/battery_low.json \
  --drone configs/environment/cold_weather.json --drone configs/control/lockstep.json \
  --duration 60 --output outputs/cold

# Battery monitor on the racing preset; Rerun Battery and Sensor status tabs
dronelab --headless --rerun --control-mode rotor-thrust --drone configs/vehicles/racing_5in.json \
  --drone configs/sensors/battery_monitor.json --duration 20 --output outputs/battery-monitor

# Barometer with offset, drift, and noise (see the Barometer tab in Rerun)
dronelab --headless --rerun --control-mode rotor-thrust \
  --drone configs/sensors/baro.json --duration 20 --output outputs/baro

# Lockstep: Gazebo waits for Python; deterministic, repeatable runs (no --keyboard)
dronelab --headless --control-mode rotor-thrust --drone configs/vehicles/racing_5in.json \
  --drone configs/control/lockstep.json --duration 20 --output outputs/lockstep

# Realistic controller timing: 250 Hz attitude / 50 Hz position loop with latency
dronelab --headless --rerun --control-mode rotor-thrust \
  --drone configs/control/control_timing.json --duration 20 --output outputs/control-timing

# Motor faults (rotor-thrust mode): damaged propeller or a stopped motor, in lockstep
dronelab --headless --rerun --control-mode rotor-thrust --drone configs/faults/motor_degraded.json \
  --drone configs/control/lockstep.json --duration 15 --output outputs/motor-degraded

# Fault injection: fly through circuit gate G1 with GPS blockout/multipath, or IMU faults
dronelab --headless --rerun --ekf --scene worlds/circuit.json --target 11 0 1.8 \
  --drone configs/faults/gps_faults.json --duration 28 --output outputs/gps-faults
dronelab --headless --rerun --ekf --scene worlds/circuit.json --target 11 0 1.8 \
  --drone configs/faults/imu_faults.json --duration 28 --output outputs/imu-faults

# Reference missions scored against truth (metrics.json): step, square laps, circuit lap
dronelab --headless --control-mode rotor-thrust --drone configs/missions/square.json \
  --drone configs/algorithms/ekf_observer.json --drone configs/control/lockstep.json \
  --duration 45 --output outputs/square
dronelab --headless --control-mode rotor-thrust --scene worlds/circuit.json \
  --drone configs/missions/circuit_lap.json --drone configs/control/lockstep.json \
  --duration 120 --output outputs/circuit-lap

# Fly on the EKF estimate instead of truth (estimator in the loop)
dronelab --headless --control-mode rotor-thrust --drone configs/missions/square.json \
  --drone configs/algorithms/ekf_in_the_loop.json --drone configs/control/lockstep.json \
  --duration 45 --output outputs/ekf-in-the-loop

# Batch sweep (seeds × parameters, parallel) with a regression baseline;
# runs whose health verdict failed are excluded from aggregates and fail --check
dronelab-batch configs/experiments/step_wind_seeds.json --output outputs/batch --save-baseline
dronelab-batch configs/experiments/step_wind_seeds.json --output outputs/batch-2 --check

# Kite-like quadplane: hover, transition, fixed-wing square at 40 m, return and land
dronelab --headless --rerun --control-mode rotor-thrust --scene worlds/airfield.json \
  --drone configs/vehicles/kite_quadplane.json --drone configs/algorithms/quadplane_autopilot.json \
  --drone configs/control/lockstep.json --duration 160 --output outputs/quadplane

# Magnetometer calibration (compass dance + throttle sweep in Gazebo), then fly with it
dronelab-magcal --drone configs/sensors/magnetometer.json --output outputs/magcal
dronelab --headless --control-mode rotor-thrust --drone configs/sensors/magnetometer.json \
  --drone outputs/magcal/magnetometer-calibration.json --duration 20 --output outputs/calibrated

# IMU calibration once (static poses at 5/25/45 °C, startup gyro bias), then reuse the layer;
# configs/sensors/no_calibration.json (last) bypasses all calibrations
dronelab-imucal --drone configs/sensors/imu_consumer_mems.json --startup-gyro-s 2 --output outputs/imucal
dronelab --headless --control-mode rotor-thrust --drone configs/sensors/imu_consumer_mems.json \
  --drone configs/scenarios/ground_start.json --drone outputs/imucal/imu-calibration.json \
  --duration 20 --output outputs/imu-calibrated

# Worlds: generate only, edit paused in Gazebo, fly a saved world
dronelab --generate-only --output outputs/world-preview
dronelab --edit --scene worlds/circuit.json --output outputs/circuit-editor
dronelab --world worlds/my-world.sdf --render-sensors --duration 20 --output outputs/saved

# Replay a recording
rerun outputs/headless/flight.rrd
```

### Keyboard controls

Click the **DroneLab — keyboard flight** window and keep it focused.

| Key | Action |
| --- | --- |
| ↑ / ↓ | Forward / backward at 1 m/s along the heading |
| ← / → | Turn counterclockwise / clockwise at 0.7 rad/s |
| Q / A | Climb / descend at 0.6 m/s (descent stops 0.4 m above ground) |
| Release keys or lose focus | Brake and hover |
| Escape or close window | Stop and save |
| 1–8 | Toggle effects: motor stop, damaged propeller, wind, GPS blackout, GPS multipath, gyro bias step, flying object circling the drone, payload drop/pickup ([details](docs/4_control.md#keyboard-effect-toggles)) |
| 9 / 0 | Battery: drain one step / refill to full (needs a battery) |
| [ / ] | Ambient temperature −/+ 10 °C; battery pack and IMU follow at once (needs `battery.thermal` or `sensors.imu.temperature`) |
| W / S | Winch: lower / raise a tethered payload's hook (held) |

The drone starts airborne at (0, 0, 1) m. Closing Gazebo itself ends the run
with an error; use Escape to finish cleanly.

### Command-line options

| Option | Default | Meaning |
| --- | --- | --- |
| `--scene PATH` | `worlds/training.json` | JSON scene used to generate the world SDF |
| `--world PATH` | – | Use a world saved from Gazebo instead of generating one (no `--drone`, `--keyboard`, or `--ekf`) |
| `--drone PATH`, `--config PATH` | built-in | Drone JSON configuration; repeat to layer files, later ones overriding earlier ([combining files](docs/3_configuration.md#combining-files)) |
| `--output DIR` | `outputs/gazebo` | Directory for the world and run artifacts |
| `--duration S` | 20 (keyboard: 3600) | Mission length in simulation seconds |
| `--keyboard` | off | Fly with arrows and Q/A using the per-rotor thrust controller; opens Rerun |
| `--control-mode {velocity,rotor-thrust}` | `velocity` | `velocity`: Gazebo native controller to `--target`. `rotor-thrust`: Python controller holds the initial pose |
| `--target X Y Z` | `3 0 2` | Waypoint in metres (velocity mode only) |
| `--ekf` | off | Run the live GPS/IMU EKF; records `estimates.jsonl` and shows it in Rerun |
| `--headless` | off | Gazebo server only, no Gazebo GUI |
| `--rerun` | off | Open live Rerun even with `--headless` (automatic with `--keyboard`) |
| `--render-sensors` | off | Enable camera and GPU lidar (needs working graphics/EGL) |
| `--generate-only` | off | Write `world.sdf` and `drone-resolved.json`, then exit |
| `--edit` | off | Open the world paused in Gazebo for editing, no mission |

Window behavior: no flags opens Gazebo and Rerun; `--headless` opens nothing;
`--headless --rerun` opens Rerun only; `--keyboard --headless` opens Rerun and
the keyboard window.

### Outputs

Each mission writes `flight.rrd`, `telemetry.jsonl`, `commands.jsonl`,
`command-events.jsonl`, `summary.json`, and `gazebo.log`. Generated worlds add
`world.sdf` and `drone-resolved.json`; every run writes `metrics.json`
(estimation and control scores, [7 Experiments](docs/7_experiments.md#metrics-metricsjson));
`--ekf` or an estimator plug-in adds `estimates.jsonl`; dynamic
wind adds `wind-commands.jsonl`; camera runs save `camera-first.png`. Check
the health verdict (`ok`/`warning`/`failed`, printed after the run),
completion, message counts, and dropped counters in `summary.json` before
trusting a dataset. Details: [2 Simulation](docs/2_simulation.md#run-artifacts).

## Tests

```bash
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q          # unit + transport-free tests
python -m dronelab --generate-only --output /tmp/dronelab-world-check   # world generation check
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids third-party ROS pytest plugins.
Tests needing real Gazebo or a desktop are skipped unless enabled:

```bash
source /opt/ros/jazzy/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DRONELAB_TEST_GAZEBO=1 python -m pytest -q tests/integration/test_gazebo_manual.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DRONELAB_TEST_GAZEBO=1 python -m pytest -q tests/integration/test_live_gazebo_ekf.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DRONELAB_TEST_GUI=1 python -m pytest -q tests/integration/test_keyboard_window.py
# Gazebo regression suite (lockstep, about 2.5 min); add DRONELAB_TEST_GAZEBO_SLOW=1
# for the full quadplane mission (about 17 min)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DRONELAB_TEST_GAZEBO=1 python -m pytest -q tests/integration/test_gazebo_regression.py
```

The regression suite flies short lockstep runs and asserts the numbers
verified when each feature was built: the 1 cm centre-of-mass offset
steady state, a magnetic dipole's −25 µT field change, `dronelab-magcal`
and `dronelab-imucal` recovering injected errors, EKF-in-the-loop tracking
matching truth-fed tracking on the square, ESC telemetry against commanded
speeds, identical repeated lockstep runs, and a battery-depletion crash
reported as `failed` by the health verdict. Run it after changing the
mission loop, physics, sensors, or Gazebo settings.
