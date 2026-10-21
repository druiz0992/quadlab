"""Small XML construction helpers for generated SDF models."""

import xml.etree.ElementTree as ET

from ...sensors.lidar_returns import RETRO_PER_REFLECTIVITY


def element(parent, tag, text=None, **attrs):
    child = ET.SubElement(parent, tag, attrs)
    if text is not None:
        child.text = str(text)
    return child


def plugin(parent, library, system):
    return element(
        parent,
        "plugin",
        filename=f"gz-sim-{library}-system",
        name=f"gz::sim::systems::{system}",
    )


def box(link, name, size, color, pose="0 0 0 0 0 0", collision=True, reflectivity=None):
    """Box visual (and collision). ``reflectivity`` sets the lidar laser_retro."""
    for kind in (["visual", "collision"] if collision else ["visual"]):
        shape = element(
            link, kind, name=name if kind == "collision" else name + "_visual"
        )
        element(shape, "pose", pose)
        element(
            element(element(shape, "geometry"), "box"), "size", " ".join(map(str, size))
        )
        if kind == "visual":
            material = element(shape, "material")
            element(material, "ambient", color)
            element(material, "diffuse", color)
            if reflectivity is not None:
                # Gazebo's gpu_lidar reports this as the beam intensity.
                element(shape, "laser_retro", RETRO_PER_REFLECTIVITY * reflectivity)
