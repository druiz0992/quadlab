# 4. Flight control

DroneLab has two control modes, chosen with `--control-mode` (or implied by
`--keyboard`). Both use **ground-truth odometry** explicitly as feedback,
labelled as such in `summary.json`. No estimator feeds control yet.

| Mode | Who closes the inner loop | Python command |
| --- | --- | --- |
| `velocity` (default) | Gazebo native `MulticopterVelocityControl` | Body velocity toward `--target` |
| `rotor-thrust` / `--keyboard` | Python `ManualController` | Four rotor thrusts (N) |

## Velocity mode (waypoint)

`control/waypoint.py` multiplies position error by 0.8, limits it to 1 m/s,
rotates it into the body frame, and the runner publishes it on `/drone/cmd_vel`
at simulation-time intervals of at least 0.04 s. Gazebo's native controller
handles attitude and motors. `--target` (default (3, 0, 2) m) applies only to
this mode. There is no path planning; obstacles cause collisions.

## Rotor-thrust mode

The Gazebo world omits the native velocity controller so it cannot compete
for the motors. Data flow:

```text
truth odometry ─→ StateBuffer ─→ RotorControlLoop (fixed-rate ticks) ─→ DelayLine
                  (state latency)   ManualController:                     (command latency)
KeyboardWindow ─→ FlightCommand ─↗    position loop ─→ attitude loop          │
                                                                              ↓
                       RotorThrustHandle ─→ latest-command slot ─→ publish on send and/or
                                                                   worker (wall clock)
                                          RotorThrustControl adapter → /drone/command/motor_speed
```

- `control/manual.py` (`ManualController`) receives explicit truth state and a
  `FlightCommand`. Its **position loop** (`update_position`) integrates
  position/yaw setpoints and applies position/velocity feedback to produce an
  attitude and thrust setpoint. Its **attitude loop** (`update_attitude`)
  applies geometric attitude feedback to the held setpoint and allocates
  collective thrust and body moments into four bounded thrust requests.
  `update()` runs both, the original single-loop behaviour. The runner
  rotates Gazebo body-frame linear velocity into world coordinates explicitly.
- With a battery or ground effect enabled, `simulation/powertrain.py` limits
  and adjusts the published motor speeds (see
  [3 Configuration](3_configuration.md#battery-and-powertrain-battery)); the
  speeds in `commands.jsonl` are the controller's nominal conversion.
- `experiments/rotor_loop.py` (`RotorControlLoop`) schedules the controller;
  see [Controller timing and latency](#controller-timing-and-latency).
- `visualization/keyboard.py` owns only Tk key, focus, and window state.
  Keys can combine; opposite keys cancel; release or focus loss requests
  braking and hover. Descent setpoints stop 0.4 m above ground; this is not
  a landing controller. The mission loop services the window (keys and
  effect toggles) at most every 20 ms of wall time and holds the last
  pilot command in between ([2 Simulation](2_simulation.md#real-time-loop-timing)).
- With `--control-mode rotor-thrust` and no keyboard, the controller holds
  the initial pose (zero pilot input).
- `core/vehicle.py` holds shared built-in properties; SDF, allocation, and the
  actuator adapter use the same mass, rotor layout, thrust coefficient, and
  limits from the resolved [configuration](3_configuration.md).

### Keyboard effect toggles

During keyboard flight, number keys switch effects on and off
(`experiments/effect_toggles.py`); the keyboard window lists them with their
current state. Each switch drives a manual override in an existing model,
alongside anything scheduled in the configuration.

| Key | Effect | Configured by (`keyboard_effects`) |
| --- | --- | --- |
| 1 | Motor stop | `motor_stop_rotor` (default 0) |
| 2 | Damaged propeller | `damaged_rotor` (1), `damaged_thrust_fraction` (0.7) |
| 3 | Wind on/off (still air when off) | Requires a configured `wind` section; starts on; otherwise shown n/a |
| 4 | GPS blackout (every fix dropped) | — |
| 5 | GPS multipath jump | `gps_multipath_offset_enu_m` (default [4, −3, 2] m) |
| 6 | Gyro bias step | `gyro_bias_step_rad_s` (default [0, 0, 0.02] rad/s) |
| 7 | Flying object: spawn/remove an object circling the drone | `flying_object_size_m` ([0.5, 0.5, 0.3]), `flying_object_radius_m` (3), `flying_object_speed_mps` (2) |
| 8 | Payload: drop, or pick up when within reach | Requires `payload.enabled` ([3 Configuration](3_configuration.md#payload-payload)) |
| W / S (held) | Winch: lower / raise the hook | Requires `payload.tether.enabled` |
| 9 | Battery: drain charge (one press = one step) | `battery_drain_fraction` (0.1); requires `battery.enabled` |
| 0 | Battery: refill to full (recovers an empty pack) | Requires `battery.enabled` |
| [ / ] | Ambient temperature down / up one step (clamped to −40…60 °C); the battery pack and IMU die jump by the same amount, as if swapping to a pack soaked at the new temperature, then self-heating continues | `ambient_step_c` (10); requires `battery.thermal.enabled` or `sensors.imu.temperature.enabled` ([3 Configuration](3_configuration.md#temperature-effects)) |

Key 7 spawns a gravity-free moving obstacle at run time through Gazebo's
`/world/dronelab/create` service (removed with `/world/dronelab/remove`),
circling the drone's position at the key press, at that altitude, along an
8-point loop. Each spawn has a new name (`flying_object_1`, `_2`, …). Service
replies are not trusted (see lockstep notes); the object's pose stream is
the evidence. Verified with scripted key presses: the object appeared 0.1 s
after the press, circled at 2.77–3.03 m (the octagon of a 3 m circle), and
disappeared from Gazebo and Rerun after the second press.

Keys 9 and 0 are one-shot actions rather than toggles; their events record
the new charge (`soc`). Verified with scripted presses on
`vehicles/battery_low.json`: two drains emptied the pack (15 % → 5 % → 0)
and the motors stopped; a refill 0.3 s later restored power, but the drone
had already fallen 0.83 m and touched the ground before climbing back to
1.00 m.

Every change is written to `effect-toggles.jsonl` (simulation time, effect,
on/off), shown in Rerun as `status/effect_<name>` on the **Sensor status** tab
and as lines in the `events/effects` log, and summarized under
`effect_toggles` in `summary.json`. GPS blackout drops are counted as
`gps_dropout.manual_drops`. Toggles come from a person in real time, so these
runs are not repeatable; use scheduled faults with lockstep for repeatable
experiments.

Verified by driving the real keyboard mission path with scripted key
presses: all nine toggles took effect at their times (a 2 s blackout dropped
exactly 20 fixes; wind went to still air and back; rotor 0's available thrust
fell from 1 to 0 at the motor-stop toggle). Tests also caught that removing
the last motor override must restore full health. The interactive Tk window
itself needs a desktop check (`DRONELAB_TEST_GUI=1`).

### Rotors

| Index / ID | Body x, y (m) | Spin (from above) |
| --- | --- | --- |
| 0 / rotor_0 | +0.2, +0.2 | CCW |
| 1 / rotor_1 | −0.2, −0.2 | CCW |
| 2 / rotor_2 | +0.2, −0.2 | CW |
| 3 / rotor_3 | −0.2, +0.2 | CW |

This is the default X quad. Rotor centres are at body z = 0.08 m and thrust
acts along +body z. `vehicle.rotors` defines other layouts (hexacopters,
quadplanes with a pusher), and thrust then acts along each rotor's axis
([3 Configuration](3_configuration.md#vehicle-and-motors)). Directions
are vehicle properties, not command signs. `simulation/gazebo/actuators.py`
converts each thrust to a positive speed,
`speed_rad_s = sqrt(thrust_n / thrust_coefficient_n_s2)`, published in index
order via `Actuators.velocity`. Defaults: 1000 rad/s and 8.55 N per rotor;
hover needs about 3.777 N each. Gazebo simulates motor lag and forces;
requested thrust differs from transient applied thrust.

## Quadplane baseline autopilot

`control/quadplane.py` (`QuadplaneController`) is a minimal
`RotorController` plug-in. It flies a quadplane, with four vertical lift
rotors, a pusher, and `fixed_wing` surfaces, through a whole mission, so
the airframe can be exercised. It is a baseline to replace with your own
controllers, not a tuned autopilot. `configs/algorithms/quadplane_autopilot.json`
selects it with a 300 m square route at 40 m. The mission reference
position (`mission.center_m`) is home.

| Mode | What flies |
| --- | --- |
| `hover` | Multirotor controller climbs to `cruise_altitude_m` over home; waits `settle_s` |
| `transition` | Pusher at `transition_thrust_n`; lift rotors hold altitude and level attitude with no horizontal braking, until airspeed ≥ `transition_airspeed_mps` |
| `cruise` | Lift rotors fade out over `lift_fade_s`. L1 guidance along `route_m` for `laps` (Park, Deyst & How 2004; L1 = `l1_period_s`·V/π; |η| clamped to 90° so a target behind still commands a full turn); pitch from altitude error (elevator loop with a small integral, needed to hold high α); pusher thrust from airspeed error (`cruise_airspeed_mps` or `airspeed_schedule`) with a small integral term; aileron/elevator attitude loops; rudder yaw damper with washout |
| `back_transition` | Pusher off; multirotor brakes to a hover where it is |
| `return`, `land` | Multirotor flies home at cruise altitude, then descends at `land_speed_mps` |

Hover gains come from `controller_options.hover`, because a plug-in does
not see the `controller` section. The example passes the preset's gains.
Its acceleration limit is 5 m/s²: in transition the wing already lifts
about 42 N at α = 0, and with the default 2 m/s² limit the integral-free
hover controller could not hold altitude.
`summary.json.algorithms.controller_summary` records the mode log.

## Rotor-thrust command API

Implemented in `core/commands.py` (types) and `control/drone.py`
(`RotorThrustControl` service and exclusive handle).

```python
@dataclass(frozen=True)
class RotorThrusts:
    thrust_n: tuple[float, ...]

@dataclass(frozen=True)
class CommandReceipt:
    command_id: int
    submitted_sim_time_ns: int
    expires_sim_time_ns: int
    requested: RotorThrusts
    accepted: RotorThrusts
    motor_speed_targets_rad_s: tuple[float, ...]
    limited_rotor_indices: tuple[int, ...]

class RotorThrustHandle:
    rotors: tuple[RotorSpec, ...]          # fixed, ordered by actuator index
    def send(self, command: RotorThrusts, *, lifetime_sim_s: float,
             bounds: Literal["reject", "clip"] = "reject") -> CommandReceipt: ...
    def stop(self) -> None: ...
```

Usage (the runner owns this and simulator cleanup):

```python
control = RotorThrustControl(adapter.send)   # adapter: Gazebo actuator adapter
try:
    with control.acquire() as handle:
        control.update_clock(sim_time_ns)    # integer sim ns on every odometry sample
        receipt = handle.send(RotorThrusts((3.777,) * 4), lifetime_sim_s=0.05)
        control.check_health()
        events = control.drain_events()
finally:
    control.close()
```

One equilibrium command expires; it is not a flight loop. The owner must keep
supplying clock updates and fresh feedback commands, check health, drain
events, and terminate its simulator on faults.

### Contract

- **One atomic vector.** No per-rotor setters, which would allow partial
  updates. Exactly one finite real per rotor; booleans, NaN, infinity, and
  malformed vectors are rejected. Lifetime must be finite, positive, and at
  most the configured maximum.
- **Bounds.** Default `reject` refuses negative or excessive thrust. `clip`
  clamps to `[0, max_thrust_n]` and reports limited indices, keeping both
  requested and accepted vectors.
- **Ownership.** `control.acquire()` gives one exclusive writer; a second
  raises `ControlBusy`, wrong mode raises `ControlModeError`. Never publish
  directly beside the service.
- **Acceptance ≠ publication.** `send` returns after validating and replacing
  the latest-command slot. Invalid sends leave the previous command intact.
  IDs increase per session.
- **Publication.** A worker republishes the latest vector at 50 Hz
  (configurable) of **wall-clock** time; a newer vector supersedes it
  atomically. Republishing never extends lifetime (default 0.1 simulation s
  in the runner). With `publish_on_send=True`, `send` also publishes the
  accepted vector immediately and records a `published` event, so actuation
  follows the caller's simulation-time schedule; the worker still
  republishes and enforces expiry. Without it, each command waits a random
  0 to 1/publish-rate seconds before reaching Gazebo. In lockstep the worker
  is not started: the mission loop calls the same `tick()` (republish and
  expiry check) once per odometry period, in simulation time, so wall time
  never reaches the battery and ESC models through republishing.
- **Expiry and watchdog.** Expiry uses the session clock at submission. Expiry
  or the 1 s wall-clock submission watchdog requests zero motor speeds, which
  removes lift; it does not hover or land.
- **Pause.** `update_clock(t, paused=True)` clears commands; sending while
  paused raises `SimulationPaused`; resuming needs a fresh command. Time
  reversal faults the session. The current runner has no pause subscription:
  its odometry watchdog ends a pause longer than one second.
- **Stop.** `stop()` is idempotent and requests zero speeds under the
  publication lock, so older vectors cannot publish afterwards. Closing a
  handle stops before releasing ownership. Publication failure faults the
  session; later sends raise.
- **Events.** `drain_events()` returns `published`, `superseded`, `expired`,
  `stopped`, and `failed` events with command ID, simulation and monotonic
  wall time, and errors. The buffer holds 4096 events, drops oldest, and
  counts `dropped_events`. "Published" means transport accepted the message,
  not that force was applied.

The controller stays a numerical object: explicit state, setpoint, and time
in, `RotorThrusts` out; no transport handles or hidden truth reads. Any
wrench allocator must be called explicitly and expose saturation.

## Controller timing and latency

Real flight controllers run on fixed timers, typically a 250–1000 Hz
attitude loop and a 50–100 Hz position loop, using the latest available
(slightly old) measurements, and their motor commands take time to act.
`RotorControlLoop` models this in simulation time. Settings live in the
`control` section of the drone JSON and apply to rotor-thrust/keyboard mode
only:

| Key | Default | Meaning |
| --- | --- | --- |
| `odometry_rate_hz` | 50 | Truth state rate; drives the tick clock (≤ physics rate) |
| `control_rate_hz` | 50 | Attitude-loop tick rate, 10 Hz to `odometry_rate_hz` |
| `position_rate_hz` | `null` | Position-loop rate; `null` = every tick; must divide `control_rate_hz` |
| `state_latency_s` | 0 | Controller uses the newest state at least this old (0–0.5 s) |
| `command_latency_s` | 0 | Computed command reaches the thrust service this much later (0–0.5 s) |
| `publish_on_send` | false | Publish each command immediately; required when `control_rate_hz` > `publish_rate_hz` |

**Tick grid.** Ticks lie on a fixed grid anchored at the first odometry
sample (`control/timing.py`, `FixedRateTicker`). A tick runs when an odometry
sample reaches its grid time, within half a physics step. If samples skip
grid times, only the latest runs and the rest count as `missed_ticks`. Each
tick samples the pilot input at the tick time, runs the position loop when
due (dt = time since its previous run) and the attitude loop every tick.

**Latency.** `StateBuffer` supplies the newest sample at least
`state_latency_s` old, standing in for sensing and estimation delay.
`DelayLine` holds each command for `command_latency_s`, standing in for
computation, bus, and ESC delay; only the newest due command is submitted.
Both are quantized to the odometry period: with 250 Hz odometry a 10 ms
state latency selects a 12 ms old sample. Commands are released when an
odometry sample is processed.

**Warm-up.** The vehicle spawns airborne, so before latency has elapsed the
first state is used and the first computed command is repeated (records
marked `"warmup": true`), as if the pipeline were already full.

**Limits.** In the default real-time mode Gazebo runs on its own clock and
Python adds unmodelled processing delay (typically milliseconds; 20–27 ms
stalls with a 30 Hz camera) between a sample's timestamp and publication, on
top of the modelled latency. [Lockstep simulation](2_simulation.md#lockstep-simulation)
removes that delay: Gazebo waits for Python, so only modelled latency
remains and runs are repeatable. Keep `publish_on_send` false only to
reproduce the original behaviour; otherwise the wall-clock worker adds
0–20 ms of random delay.

`configs/control/control_timing.json`: 250 Hz odometry and attitude
loop, 50 Hz position loop, 10 ms state and 8 ms command latency, publish on
send. These are plausible values, not a measured flight stack.

**Observed (20 s hover in gusty wind, one seeded run each):**

| Timing | Horizontal deviation RMS | Tilt RMS | Result |
| --- | --- | --- | --- |
| Default (50 Hz, no latency) | 0.56 m | 2.9° | Stable |
| Example (250/50 Hz, 10 + 8 ms) | 0.56 m | 2.9° | Stable, nearly identical |
| 250/50 Hz, 60 + 60 ms | — | — | Attitude loop unstable: tilt > 10° at 2.0 s, flipped at 3.5 s, crashed |

The current gains have large margin for realistic delays, but about
120 ms of total loop delay destabilizes them. Gains are not retuned
automatically when timing changes.

## Runner safeguards and records

The runner reads truth odometry (default 50 Hz), aborts on one second
without telemetry, stale queued odometry, or odometry intervals outside
(0, 0.1] s, checks worker health, and records events. Exit sends zero motor
targets and terminates the owned simulator. `commands.jsonl` records, per
submitted command, requested thrusts, speed targets, position/yaw setpoints,
saturation flags, command IDs, accepted thrusts, expiry, and timing:
`tick_sim_time_ns`, `state_sim_time_ns` (state used), `release_sim_time_ns`,
`missed_ticks`, and `warmup`. `time` is the tick time in seconds.
`command-events.jsonl` records lifecycle events. The summary records control
mode, ground-truth state source, exit reason, saturation count (ticks),
dropped events, publication settings, and `control_timing`: configured rates
and latencies, ticks, missed ticks, position updates, released, superseded,
warm-up and pending commands, and maximum state age. Rerun's **Rotor thrust
targets (N)** tab shows commanded targets (including zeros on stop/expiry),
not applied forces.

## Verification status

Unit tests cover ownership, rotor order, conversion, bounds, malformed
values, conflicts, replacement, expiry, pause/resume, stop races, failure
cleanup, publish-on-send, tick grids, latency buffers, warm-up, loop-rate
division, and equivalence of default timing with the single-loop controller,
using fake clocks and transport. The opt-in Gazebo test
(`DRONELAB_TEST_GAZEBO=1`, `tests/integration/test_gazebo_manual.py`) exercises
forward/backward, turning, climb/descent, and hover recovery. Per-rotor
perturbation response, lag, and GUI flight still need runtime checks.
