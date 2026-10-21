# DroneLab agent and contributor rules

These rules apply to every session, for both coding agents and human
contributors. Read them before changing anything. How to run the project is
in [README.md](README.md); design, behavior, and reference material are in
[docs/](docs/1_overview.md), which you should read in numbered order.

## 1. Working method

- Read the relevant implementation, tests, and docs before changing behavior.
- Keep each change focused on the request. Preserve unrelated work and never
  delete or overwrite user data, worlds, or run outputs without asking.
- Distinguish implemented features from planned work, in code and in docs.
  Perception, planning, and RL are placeholder packages today.
- Do not describe the simplified vehicle as a calibrated real-world model, or
  backend comparisons as controlled physics comparisons.
- Report results honestly: say which tests ran, which failed, and which
  runtime dependencies (Gazebo, display, GPU) were unavailable.

## 2. Code quality

- **Typed Python.** Annotate every function signature and public attribute
  with type hints (Python >= 3.11 syntax: `list[float]`, `X | None`). Use
  frozen dataclasses for value types and `Protocol`s for pluggable behavior.
  When you touch existing untyped code, add hints to what you change.
- **Clear, well-commented code.** Every module, public class, and public
  function gets a docstring stating purpose, units, frames, and failure
  behavior. Comment the *why* of non-obvious steps, math, and conventions;
  cite equations or references for numerical algorithms. Use descriptive
  names that carry units (`thrust_n`, `rate_hz`, `sim_time_ns`).
- **Short functions.** One responsibility per function; aim for under ~40
  lines. Extract helpers instead of nesting deeply.
- **Easily extensible.** Prefer small composable components with explicit
  inputs over hidden globals. Pass dependencies (clock, transport, RNG,
  state source) in, so they can be replaced by fakes in tests and by new
  implementations later. Introduce shared interfaces only when a second real
  implementation needs them.
- **Explicit failures.** Validate inputs at boundaries and raise clear errors;
  never silently clip, drop, or substitute data without counting/reporting it.
- Format with `black`. Keep imports for optional dependencies (Gazebo
  bindings, Rerun, Tk) deferred so world generation and numerical tests work
  without them.

## 3. Architecture boundaries

- Numerical algorithms live in `estimation`, `control`, `perception`,
  `planning`, or `rl`. They must not depend on simulator transport, viewers,
  CLI parsing, or process management.
- Sensor models and message decoding go in `sensors`; simulator adapters and
  SDF construction in `simulation`; orchestration in `experiments`; plotting,
  recording, and UI in `visualization`. Keep `cli` thin and `core` small.
- Preserve the separate step-based (RotorPy) and asynchronous (Gazebo)
  execution models; do not force them behind one interface.
- Keep ground-truth control and estimator initialization explicit. **Never
  give an estimator hidden access to simulator truth.**

## 4. Numerical and data contracts

- SI units, right-handed z-up world, gravity (0, 0, -9.81) m/s².
- Quaternions are (x, y, z, w) and rotate body vectors into world coordinates.
  Angular rates and accelerometer specific force are body/sensor-frame values.
- Preserve explicit timestamps: integer simulation nanoseconds from the
  source; IMU samples are held from interval starts; state arrays include
  initial and final states; position samples occur on the first tick at or
  after their scheduled time. Never substitute wall-clock time.
- Synthetic noise uses local seeded RNGs. Document units and per-sample versus
  density semantics. Do not silently change gains, vehicle parameters, frame
  conventions, recording schemas, CLI entry points, or artifact formats; such
  changes require a documented migration.

## 5. Worlds and generated files

- Authored scenes and saved worlds live in `worlds/`; generated artifacts go
  in `outputs/` or a temporary directory. Never commit virtual environments,
  caches, recordings, or large flight data.
- When editing reusable worlds keep the world name `dronelab`, vehicle name
  `drone`, sensor topics, and plugins. Collision and visual sizes must match.
- Gazebo is authoritative for collisions and rendered sensors; Rerun world
  sync is partial. Check the health verdict and missing/dropped message
  counts in `summary.json`.
- Use a fresh output directory for every verification run.

## 6. Testing

- Every new feature or changed numerical contract needs tests:
  - **Unit tests** in `tests/unit/<package>/` for numerical behavior,
    validation, and edge cases, using fakes for transport, clocks, and RNGs.
  - **Integration tests** in `tests/integration/` for flows spanning
    components. Tests needing Gazebo or a display are opt-in through
    `DRONELAB_TEST_GAZEBO=1` / `DRONELAB_TEST_GUI=1` and must skip otherwise;
    missions of several minutes also need `DRONELAB_TEST_GAZEBO_SLOW=1`.
  - **Gazebo regression tests** in `tests/integration/test_gazebo_regression.py`
    pin verified simulator numbers in lockstep. Add one when you verify a new
    Gazebo behavior, and run the suite after changing the mission loop,
    physics, sensors, or Gazebo settings.
- Before finishing, from the repository root:

  ```bash
  source .venv/bin/activate
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
  python -m dronelab --generate-only --output /tmp/dronelab-world-check
  ```

- Documentation-only changes need command checks where relevant, not tests.
- Never claim a GUI or Gazebo mission was verified from world generation alone.
- For timing-sensitive Gazebo checks (control tuning, determinism), layer
  `configs/control/lockstep.json` so results are repeatable.

## 7. Documentation

Documentation must match the code at the end of every change.

| File | Contents | Update when |
| --- | --- | --- |
| `AGENTS.md` | These rules | Project rules change |
| `README.md` | Setup, how to run, CLI options, tests | User-facing commands or options change |
| `docs/1_overview.md` | Purpose, status, architecture, conventions | Packages, boundaries, or conventions change |
| `docs/2_simulation.md` | Gazebo missions, worlds, sensors, artifacts | Simulation workflow or outputs change |
| `docs/3_configuration.md` | Drone JSON schema and disturbances | Config keys, defaults, or validation change |
| `docs/4_control.md` | Controllers and the rotor-thrust API | Control behavior changes |
| `docs/5_estimation.md` | EKF math and live estimation | Estimator behavior changes |
| `docs/6_roadmap.md` | Proposed APIs and future work | Plans change or a proposal is implemented |
| `docs/7_experiments.md` | Algorithm plug-ins, sensor bus, missions, metrics, batch runs | The harness or its contracts change |

When a proposal in the roadmap is implemented, move its description into the
relevant numbered doc. Do not create new top-level docs; extend these, or add
a new numbered file if a topic truly needs one. Keep links relative and valid.

## 8. Environment

- Python >= 3.11. Gazebo bindings (`gz.transport13`, `gz.msgs10`) must match
  the system interpreter: currently Python 3.12 from ROS Jazzy
  (`source /opt/ros/jazzy/setup.bash`).
- RotorPy and Rerun are pinned in `pyproject.toml`; change pins deliberately.
- `gazebo_plugins/` holds C++ gz-sim system plugins (currently the lockstep
  barrier), built with `dronelab-build-plugins` into `build/` (never
  committed). Python must keep working without them (fallback and a clear
  error when one is required). Keep them small; numerical algorithms stay
  in Python.
