"""``ground_vehicle`` settings: defaults, per-robot presets, and validation.

``robot.kind`` selects the vehicle family (``ROBOT_KINDS``). For a wheeled
kind, the loader applies that kind's preset (``PRESETS``) on top of the
built-in defaults and below every user config layer, so a layer changes
only what it names. The presets are textbook-plausible vehicles, not
calibrated models:

- ``rover``: a 50 kg four-wheel skid-steer research rover (no steering
  joints; turns by left/right torque difference), ~1 m/s;
- ``car``: a 700 kg rear-drive single-seater race car with slicks, wings
  (drag and downforce), ~210 kW;
- ``tractor``: a 3500 kg electric four-wheel-drive, four-wheel-steer
  tractor modelled on published Voltrac Thor figures (4 × 28 kW wheel
  motors, 4.5 m long, 1.7 m wide, 2.2 m wheelbase, 0.5 m ground
  clearance, 380/85R28 tyres → 0.68 m radius, ~40 km/h). Track width,
  centre of mass, inertia, torque, and tyre/suspension constants are
  assumptions (not published).

Ground vehicles always run in lockstep: their suspension and tyre forces
are computed in Python every odometry period and must never be skipped.
"""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ...core.wheeled_vehicle import STEERING_LAYOUTS, WheeledProperties

ROBOT_KINDS = ("multirotor", "rover", "car", "tractor")
WHEELED_KINDS = ("rover", "car", "tractor")

WHEEL_KEYS = {"name", "position_m", "steered", "driven", "braked"}


def _wheels(
    front_x: float,
    rear_x: float,
    half_track: float,
    z: float,
    *,
    steer_front: bool,
    steer_rear: bool,
    drive_front: bool,
    drive_rear: bool,
) -> list[dict[str, Any]]:
    """Four wheels in the order front-left, front-right, rear-left, rear-right."""
    wheels = []
    for axle, x, steered, driven in (
        ("front", front_x, steer_front, drive_front),
        ("rear", rear_x, steer_rear, drive_rear),
    ):
        for side, y in (("left", half_track), ("right", -half_track)):
            wheels.append(
                dict(
                    name=f"{axle}_{side}",
                    position_m=[x, y, z],
                    steered=steered,
                    driven=driven,
                    braked=True,
                )
            )
    return wheels


def _tyre(
    radius_m: float,
    friction: float,
    b: float,
    c: float,
    e: float,
    relaxation_m: tuple[float, float],
    rolling_resistance: float,
    low_speed_damping_n_s_per_m: float,
) -> dict[str, Any]:
    return dict(
        radius_m=radius_m,
        longitudinal=dict(
            friction=friction, b=b, c=c, e=e, relaxation_length_m=relaxation_m[0]
        ),
        lateral=dict(
            friction=friction, b=b, c=c, e=e, relaxation_length_m=relaxation_m[1]
        ),
        rolling_resistance=rolling_resistance,
        low_speed_mps=2.0,
        low_speed_damping_n_s_per_m=low_speed_damping_n_s_per_m,
    )


# Defaults are the race car (the most demanding numerically); presets
# replace whole sections where a vehicle differs.
GROUND_VEHICLE_DEFAULTS: dict[str, Any] = {
    # Gazebo model name; "" uses the robot kind ("rover", "car", "tractor").
    "model_name": "",
    # Keyboard mapping and validation: ackermann | skid | four_wheel.
    "steering_layout": "ackermann",
    "chassis": {
        "mass_kg": 700.0,
        "inertia_kg_m2": [180.0, 650.0, 750.0],
        # Collision box (Gazebo) relative to the centre of mass.
        "collision_size_m": [3.6, 1.0, 0.45],
        "collision_offset_m": [0.0, 0.0, 0.05],
    },
    # Hub centres at full droop (body axes, m), front-left first.
    "wheels": _wheels(
        1.5,
        -1.3,
        0.8,
        -0.08,
        steer_front=True,
        steer_rear=False,
        drive_front=False,
        drive_rear=True,
    ),
    "wheel": {"width_m": 0.3, "inertia_kg_m2": 1.2},
    # Dry-tarmac Magic Formula shape (Pacejka's common example values:
    # B = 10, C = 1.9, E = 0.97) with racing-slick friction.
    "tyre": _tyre(0.30, 1.6, 10.0, 1.9, 0.97, (0.10, 0.20), 0.012, 6000.0),
    "suspension": {
        "stiffness_n_per_m": 80000.0,
        "damping_n_s_per_m": 3000.0,
        "travel_m": 0.06,
        "bump_stiffness_n_per_m": 800000.0,
    },
    # Per wheel: driven wheels share the engine power equally here.
    "drive": {
        "max_torque_nm": 1500.0,
        "max_power_w": 105000.0,
        "max_wheel_speed_rad_s": 270.0,
        "max_brake_torque_nm": 2500.0,
    },
    "steering": {"max_angle_rad": 0.4, "max_rate_rad_s": 2.0},
    "aero": {
        "drag_area_m2": 1.0,
        "downforce_area_m2": 3.0,
        "downforce_front_fraction": 0.4,
    },
    # Start pose without scene spawn_points (x, y from
    # simulation.spawn_position_m); with them, the ego's grid slot.
    "spawn_yaw_deg": 0.0,
    "spawn_index": 0,
    # Rear three-point-hitch implement (tractor); see
    # simulation/wheeled/implement.py. hitch_position_m: implement box
    # centre when raised (body axes from the centre of mass).
    "implement": {
        "enabled": False,
        "type": "moldboard_plough",
        "texture": "medium",
        "draft_a": 0.0,
        "draft_b": 0.0,
        "draft_c": 0.0,
        "texture_factor": 1.0,
        "width_m": 1.6,
        "working_depth_m": 0.2,
        "mass_kg": 700.0,
        "size_m": [1.0, 1.8, 0.7],
        "hitch_position_m": [-2.4, 0.0, -0.1],
        "lift_travel_m": 0.8,
        "lift_rate_mps": 0.15,
        "start_lowered": False,
    },
    # Towed trailer on a drawbar hitch (tractor); see
    # experiments/trailer.py. hitch_position_m: hitch point on the tractor
    # (body axes from its centre of mass); drawbar_m: hitch to trailer
    # centre of mass along the trailer's −x; cg_below_hitch_m: trailer
    # centre of mass below the hitch; wheel positions in trailer axes (hub
    # at full droop). Tyres and suspension: a load-scaled copy of the
    # tractor's (see trailer_properties).
    "trailer": {
        "enabled": False,
        "mass_kg": 1500.0,
        "inertia_kg_m2": [900.0, 1800.0, 2200.0],
        "size_m": [3.5, 1.9, 1.0],
        "hitch_position_m": [-2.0, 0.0, -0.35],
        "drawbar_m": 3.0,
        "cg_below_hitch_m": -0.1,
        "wheel_radius_m": 0.45,
        # Axle 0.3 m behind the centre of mass (positive drawbar load);
        # hubs at static ride height clear the ground by the tyre radius.
        "wheels": [
            {
                "name": "left",
                "position_m": [-0.3, 0.85, -0.38],
                "steered": False,
                "driven": False,
                "braked": True,
            },
            {
                "name": "right",
                "position_m": [-0.3, -0.85, -0.38],
                "steered": False,
                "driven": False,
                "braked": True,
            },
        ],
        "max_brake_torque_nm": 3000.0,
        "yaw_limit_deg": 80.0,
        "pitch_limit_deg": 25.0,
    },
    # Other cars of the same kind on the scene's next spawn points, each
    # with its own controller (default: the centreline follower); see
    # experiments/opponents.py. per_opponent_options (one dict per
    # opponent, optional) merge over controller_options. mode "kinematic"
    # (moving obstacles: Gazebo velocity commands from the track
    # follower's guidance at update_rate_hz, no wheel model) or "dynamic"
    # (full wheel model every period, any WheeledController).
    "opponents": {
        "count": 0,
        "mode": "kinematic",
        "update_rate_hz": 50.0,
        "controller": "track_follower",
        "controller_options": {},
        "per_opponent_options": [],
    },
    # Tyre/wheel-spin integration substeps per odometry period (1 ms each
    # at the 250 Hz default).
    "substeps": 4,
    # Keyboard driving (see control/wheeled_keyboard.py): fraction of the
    # torque limits and steering lock per second of key hold.
    "keyboard": {
        "throttle_rate_per_s": 2.0,
        "steer_rate_per_s": 1.5,
        "max_speed_mps": 0.0,
    },
}

# Settings every wheeled kind shares (layered under the kind's own preset).
_WHEELED_COMMON: dict[str, Any] = {
    # 250 Hz wheel-force updates with two 2 ms physics steps per lockstep
    # burst (lockstep needs two or more): verified against 500 Hz / 1 ms,
    # a follower lap on the race circuit differed by 0.01 % (56.736 s vs
    # 56.730 s) at about two thirds of the cost. publish_on_send satisfies
    # the shared control-timing checks.
    "control": {
        "odometry_rate_hz": 250.0,
        "control_rate_hz": 125.0,
        "publish_on_send": True,
    },
    "simulation": {
        "lockstep": True,
        "step_s": 0.002,
        "spawn_position_m": [0.0, 0.0, 0.0],
    },
    "sensors": {"wheel_encoders": {"enabled": True}},
}

PRESETS: dict[str, dict[str, Any]] = {
    "car": {"robot": {"kind": "car"}},
    "rover": {
        "robot": {"kind": "rover"},
        "ground_vehicle": {
            "steering_layout": "skid",
            "chassis": {
                "mass_kg": 50.0,
                "inertia_kg_m2": [2.0, 3.5, 4.5],
                "collision_size_m": [0.9, 0.5, 0.25],
                "collision_offset_m": [0.0, 0.0, 0.05],
            },
            "wheels": _wheels(
                0.256,
                -0.256,
                0.28,
                -0.06,
                steer_front=False,
                steer_rear=False,
                drive_front=True,
                drive_rear=True,
            ),
            "wheel": {"width_m": 0.11, "inertia_kg_m2": 0.04},
            "tyre": _tyre(0.165, 0.9, 8.0, 1.5, 0.5, (0.10, 0.15), 0.03, 300.0),
            "suspension": {
                "stiffness_n_per_m": 20000.0,
                "damping_n_s_per_m": 700.0,
                "travel_m": 0.03,
                "bump_stiffness_n_per_m": 200000.0,
            },
            "drive": {
                "max_torque_nm": 30.0,
                "max_power_w": 150.0,
                "max_wheel_speed_rad_s": 6.5,
                "max_brake_torque_nm": 30.0,
            },
            "steering": {"max_angle_rad": 0.0, "max_rate_rad_s": 0.0},
            "aero": {
                "drag_area_m2": 0.0,
                "downforce_area_m2": 0.0,
                "downforce_front_fraction": 0.5,
            },
        },
    },
    "tractor": {
        "robot": {"kind": "tractor"},
        "ground_vehicle": {
            "steering_layout": "four_wheel",
            "chassis": {
                "mass_kg": 3500.0,
                "inertia_kg_m2": [1300.0, 5300.0, 5400.0],
                # Body between the wheels, from 0.5 m clearance to 1.6 m.
                "collision_size_m": [3.4, 0.9, 1.1],
                "collision_offset_m": [0.0, 0.0, 0.25],
            },
            # Wheelbase 2.2 m; track 1.32 m (1.7 m width − 0.38 m tyre).
            "wheels": _wheels(
                1.1,
                -1.1,
                0.66,
                -0.40,
                steer_front=True,
                steer_rear=True,
                drive_front=True,
                drive_rear=True,
            ),
            "wheel": {"width_m": 0.38, "inertia_kg_m2": 40.0},
            "tyre": _tyre(0.68, 0.8, 6.0, 1.6, 0.3, (0.30, 0.60), 0.05, 12000.0),
            "suspension": {
                "stiffness_n_per_m": 150000.0,
                "damping_n_s_per_m": 9000.0,
                "travel_m": 0.12,
                "bump_stiffness_n_per_m": 1500000.0,
            },
            "drive": {
                "max_torque_nm": 5000.0,
                "max_power_w": 28000.0,
                # 40 km/h on 0.68 m tyres.
                "max_wheel_speed_rad_s": 16.3,
                "max_brake_torque_nm": 6000.0,
            },
            "steering": {"max_angle_rad": 0.6, "max_rate_rad_s": 0.6},
            "aero": {
                "drag_area_m2": 2.0,
                "downforce_area_m2": 0.0,
                "downforce_front_fraction": 0.5,
            },
        },
    },
}


def preset_layer(kind: str) -> dict[str, Any]:
    """The built-in settings layer for ``kind`` (empty for multirotor)."""
    if kind not in ROBOT_KINDS:
        raise ValueError(f"robot kind must be one of {', '.join(ROBOT_KINDS)}")
    if kind == "multirotor":
        return {}
    layer = deepcopy(_WHEELED_COMMON)
    for key, value in deepcopy(PRESETS[kind]).items():
        layer[key] = value
    return layer


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value != value
        or value in (float("inf"), float("-inf"))
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        raise ValueError(f"{name} must be {'>= 0' if allow_zero else 'positive'}")


def _finite(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value != value
        or abs(value) == float("inf")
    ):
        raise ValueError(f"{name} must be a finite number")


def _vector(value: Any, name: str, *, positive: bool = False) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} needs three numbers")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} needs three numbers")
        if item != item or abs(item) == float("inf"):
            raise ValueError(f"{name} must be finite")
        if positive and item <= 0:
            raise ValueError(f"{name} must be positive")


def _section(value: Any, template: Mapping[str, Any], name: str) -> None:
    """Check keys and value types of one nested section against defaults."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    if set(value) != set(template):
        missing, extra = set(template) - set(value), set(value) - set(template)
        raise ValueError(f"{name}: missing {sorted(missing)}, unknown {sorted(extra)}")
    for key, default in template.items():
        item, path = value[key], f"{name}.{key}"
        if isinstance(default, dict):
            _section(item, default, path)
        elif isinstance(default, str):
            if not isinstance(item, str):
                raise ValueError(f"{path} must be a string")
        elif isinstance(default, bool):
            if type(item) is not bool:
                raise ValueError(f"{path} must be true or false")
        elif key in FREE_FORM:
            if not isinstance(item, dict):
                raise ValueError(f"{path} must be an object")
        elif isinstance(default, list):
            if key not in LIST_KEYS:
                _vector(item, path, positive=key in POSITIVE_VECTORS)
        elif isinstance(default, int) and not isinstance(default, bool):
            minimum = 0 if key in {"spawn_index", "count"} else 1
            if type(item) is not int or item < minimum:
                raise ValueError(f"{path} must be an integer >= {minimum}")
        elif key in SIGNED_KEYS:
            _finite(item, path)
        else:
            _positive(item, path, allow_zero=key in ZERO_ALLOWED)


POSITIVE_VECTORS = {"inertia_kg_m2", "collision_size_m", "size_m"}
# Validated by their consumers (plug-in options) or by dedicated checks.
FREE_FORM = {"controller_options"}
LIST_KEYS = {"wheels", "per_opponent_options"}


def trailer_properties(
    trailer: Mapping[str, Any], tractor: "WheeledProperties"
) -> "WheeledProperties":
    """Nominal trailer as a wheeled vehicle without drive or steering.

    Tyre: the tractor's curves and friction with the trailer's radius;
    suspension stiffness and damping scaled by the per-wheel load ratio, so
    the static sag matches the tractor's (an assumption, documented).
    """
    from dataclasses import replace

    from ...core.wheeled_vehicle import DriveProperties, WheelProperties

    wheels = tuple(
        WheelProperties(w["name"], tuple(w["position_m"]), False, False, w["braked"])
        for w in trailer["wheels"]
    )
    load_ratio = (trailer["mass_kg"] / len(wheels)) / (
        tractor.mass_kg / tractor.wheel_count
    )
    suspension = replace(
        tractor.suspension,
        stiffness_n_per_m=tractor.suspension.stiffness_n_per_m * load_ratio,
        damping_n_s_per_m=tractor.suspension.damping_n_s_per_m * load_ratio,
        bump_stiffness_n_per_m=tractor.suspension.bump_stiffness_n_per_m * load_ratio,
    )
    return replace(
        tractor,
        model_name=f"{tractor.model_name}::trailer",
        steering_layout="skid",
        mass_kg=trailer["mass_kg"],
        inertia_kg_m2=tuple(trailer["inertia_kg_m2"]),
        collision_size_m=tuple(trailer["size_m"]),
        collision_offset_m=(0.0, 0.0, 0.0),
        wheels=wheels,
        wheel_inertia_kg_m2=tractor.wheel_inertia_kg_m2
        * (trailer["wheel_radius_m"] / tractor.tyre.radius_m) ** 2,
        tyre=replace(tractor.tyre, radius_m=trailer["wheel_radius_m"]),
        suspension=suspension,
        drive=DriveProperties(0.0, 0.0, 1e9, trailer["max_brake_torque_nm"], 0.0, 0.0),
        aero=replace(tractor.aero, drag_area_m2=0.0, downforce_area_m2=0.0),
    )


MAX_OPPONENTS = 7
OPPONENT_MODES = ("kinematic", "dynamic")
SIGNED_KEYS = {"spawn_yaw_deg", "cg_below_hitch_m"}
ZERO_ALLOWED = {
    "rolling_resistance",
    "e",
    "max_angle_rad",
    "max_rate_rad_s",
    "drag_area_m2",
    "downforce_area_m2",
    "downforce_front_fraction",
    "max_speed_mps",
    "max_brake_torque_nm",
    "damping_n_s_per_m",
    "draft_a",
    "draft_b",
    "draft_c",
    "low_speed_damping_n_s_per_m",
}


def validate_wheels(
    wheels: Any, name: str = "ground_vehicle.wheels", minimum: int = 3
) -> None:
    """At least ``minimum`` uniquely named wheels with flags and positions."""
    if not isinstance(wheels, list) or len(wheels) < minimum:
        raise ValueError(f"{name} needs at least {minimum} wheels")
    names = set()
    for index, wheel in enumerate(wheels):
        path = f"{name}[{index}]"
        if not isinstance(wheel, dict) or set(wheel) != WHEEL_KEYS:
            raise ValueError(f"{path} needs exactly {sorted(WHEEL_KEYS)}")
        if not isinstance(wheel["name"], str) or not wheel["name"].isidentifier():
            raise ValueError(f"{path}.name must be an identifier")
        if wheel["name"] in names:
            raise ValueError(f"{path}.name {wheel['name']!r} is duplicated")
        names.add(wheel["name"])
        _vector(wheel["position_m"], f"{path}.position_m")
        for flag in ("steered", "driven", "braked"):
            if type(wheel[flag]) is not bool:
                raise ValueError(f"{path}.{flag} must be true or false")


def validate_ground_vehicle(data: Mapping[str, Any]) -> None:
    """Validate the merged settings of a wheeled robot; raises ValueError."""
    g = data["ground_vehicle"]
    _section(g, GROUND_VEHICLE_DEFAULTS, "ground_vehicle")
    validate_wheels(g["wheels"])
    layout = g["steering_layout"]
    if layout not in STEERING_LAYOUTS:
        raise ValueError(
            f"ground_vehicle.steering_layout must be one of {STEERING_LAYOUTS}"
        )
    if g["model_name"] and (
        not g["model_name"].isidentifier() or g["model_name"] in {"ground", "terrain"}
    ):
        raise ValueError("ground_vehicle.model_name must be an identifier")
    _check_layout(g, layout)
    _check_opponents(g["opponents"], data["control"]["odometry_rate_hz"])
    _check_implement(g["implement"], data["robot"]["kind"])
    _check_trailer(g)
    for key in ("c", "e"):
        for direction in ("longitudinal", "lateral"):
            value = g["tyre"][direction][key]
            if key == "c" and not 0 < value < 2:
                raise ValueError(f"ground_vehicle.tyre.{direction}.c must be in (0, 2)")
            if key == "e" and not value <= 1:
                raise ValueError(f"ground_vehicle.tyre.{direction}.e must be <= 1")
    if g["aero"]["downforce_front_fraction"] > 1:
        raise ValueError("ground_vehicle.aero.downforce_front_fraction must be <= 1")
    _check_lockstep(data)
    _reject_drone_features(data)


def _check_layout(g: Mapping[str, Any], layout: str) -> None:
    wheels = g["wheels"]
    steered = [w for w in wheels if w["steered"]]
    if not any(w["driven"] for w in wheels):
        raise ValueError("ground_vehicle needs at least one driven wheel")
    if layout == "skid" and steered:
        raise ValueError("skid steering_layout: no wheel may be steered")
    if layout != "skid" and not steered:
        raise ValueError(f"{layout} steering_layout needs steered wheels")
    if layout != "skid" and g["steering"]["max_angle_rad"] <= 0:
        raise ValueError("ground_vehicle.steering.max_angle_rad must be positive")
    if layout != "skid" and g["steering"]["max_rate_rad_s"] <= 0:
        raise ValueError("ground_vehicle.steering.max_rate_rad_s must be positive")
    if layout == "four_wheel":
        x = [w["position_m"][0] for w in steered]
        if not (min(x) < 0 < max(x)):
            raise ValueError("four_wheel steering needs steered front and rear wheels")
    if g["steering"]["max_angle_rad"] >= 1.2:
        raise ValueError("ground_vehicle.steering.max_angle_rad must be < 1.2 rad")


def _check_opponents(opponents: Mapping[str, Any], odometry_rate_hz: float) -> None:
    if opponents["mode"] not in OPPONENT_MODES:
        raise ValueError(
            f"ground_vehicle.opponents.mode must be one of {OPPONENT_MODES}"
        )
    if opponents["mode"] == "kinematic":
        # Kinematic opponents follow the track follower's guidance; other
        # controllers produce wheel torques, which need the wheel model.
        if opponents["count"] and opponents["controller"] != "track_follower":
            raise ValueError(
                "ground_vehicle.opponents: kinematic mode drives with "
                "track_follower; use mode 'dynamic' for other controllers"
            )
        ratio = odometry_rate_hz / opponents["update_rate_hz"]
        # Integer ratio: opponent updates coincide with ego periods.
        if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
            raise ValueError(
                "ground_vehicle.opponents.update_rate_hz must divide "
                f"control.odometry_rate_hz ({odometry_rate_hz:g})"
            )
    if opponents["count"] > MAX_OPPONENTS:
        raise ValueError(f"ground_vehicle.opponents.count must be <= {MAX_OPPONENTS}")
    per = opponents["per_opponent_options"]
    if not isinstance(per, list) or not all(isinstance(o, dict) for o in per):
        raise ValueError(
            "ground_vehicle.opponents.per_opponent_options must be a list of objects"
        )
    if len(per) > opponents["count"]:
        raise ValueError(
            "ground_vehicle.opponents: more per_opponent_options than opponents"
        )
    if opponents["count"] and not opponents["controller"]:
        raise ValueError("ground_vehicle.opponents.controller must name a controller")


def _check_implement(implement: Mapping[str, Any], kind: str) -> None:
    from .implement import IMPLEMENT_TYPES, TEXTURES

    if not implement["enabled"]:
        return
    if implement["type"] not in (*IMPLEMENT_TYPES, "custom"):
        raise ValueError(
            "ground_vehicle.implement.type must be one of "
            f"{sorted((*IMPLEMENT_TYPES, 'custom'))}"
        )
    if implement["texture"] not in TEXTURES:
        raise ValueError(f"ground_vehicle.implement.texture must be one of {TEXTURES}")
    if implement["hitch_position_m"][0] >= 0:
        raise ValueError(
            "ground_vehicle.implement.hitch_position_m must be behind the centre of mass (x < 0)"
        )


def _check_trailer(g: Mapping[str, Any]) -> None:
    trailer = g["trailer"]
    if not trailer["enabled"]:
        return
    if g["implement"]["enabled"]:
        raise ValueError(
            "ground_vehicle: choose the implement or the trailer (one rear hitch)"
        )
    validate_wheels(trailer["wheels"], "ground_vehicle.trailer.wheels", minimum=2)
    if any(w["driven"] or w["steered"] for w in trailer["wheels"]):
        raise ValueError("ground_vehicle.trailer.wheels are neither driven nor steered")
    if trailer["hitch_position_m"][0] >= 0:
        raise ValueError(
            "ground_vehicle.trailer.hitch_position_m must be behind the centre of mass (x < 0)"
        )
    if (
        not 0 < trailer["yaw_limit_deg"] < 120
        or not 0 < trailer["pitch_limit_deg"] < 60
    ):
        raise ValueError(
            "ground_vehicle.trailer yaw_limit_deg in (0, 120), pitch_limit_deg in (0, 60)"
        )


def _check_lockstep(data: Mapping[str, Any]) -> None:
    if not data["simulation"]["lockstep"]:
        raise ValueError(
            "wheeled robots need simulation.lockstep (Python computes the "
            "suspension and tyre forces every odometry period)"
        )
    if data["control"]["odometry_rate_hz"] < 250:
        raise ValueError(
            "wheeled robots need control.odometry_rate_hz >= 250 (tyre forces "
            "are held for one odometry period)"
        )


def _reject_drone_features(data: Mapping[str, Any]) -> None:
    """Multirotor-only sections must stay disabled on a wheeled robot."""
    enabled = [
        name
        for name, on in (
            ("payload", data["payload"]["enabled"]),
            ("fixed_wing", data["fixed_wing"]["enabled"]),
            ("battery", data["battery"]["enabled"]),
            ("wind", data["wind"]["enabled"]),
            ("failsafe.low_battery", data["failsafe"]["low_battery"]["enabled"]),
            (
                "aerodynamics.ground_effect",
                data["aerodynamics"]["ground_effect"]["enabled"],
            ),
            ("aerodynamics.body_drag", data["aerodynamics"]["body_drag"]["enabled"]),
            (
                "aerodynamics.translational_lift",
                data["aerodynamics"]["translational_lift"]["enabled"],
            ),
            (
                "aerodynamics.vortex_ring",
                data["aerodynamics"]["vortex_ring"]["enabled"],
            ),
            ("sensors.optical_flow", data["sensors"]["optical_flow"]["enabled"]),
            ("sensors.airspeed", data["sensors"]["airspeed"]["enabled"]),
            ("sensors.battery_monitor", data["sensors"]["battery_monitor"]["enabled"]),
            ("motors.esc.telemetry", data["motors"]["esc"]["telemetry"]["enabled"]),
        )
        if on
    ]
    if data["motors"]["faults"]["events"]:
        enabled.append("motors.faults")
    if enabled:
        raise ValueError(
            f"{data['robot']['kind']}: multirotor-only features enabled: "
            + ", ".join(enabled)
        )
