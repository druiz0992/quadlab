"""Small XML construction helpers for generated SDF models."""

import xml.etree.ElementTree as ET

from ...sensors.imaging.lidar_returns import RETRO_PER_REFLECTIVITY


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


def textured_box(
    link,
    name,
    size,
    mesh_path,
    albedo_map,
    pose="0 0 0 0 0 0",
    collision=True,
    reflectivity=None,
):
    """Box collision plus a mesh visual (``textures.write_box_obj``) carrying
    the albedo map (white diffuse, so the image colours show unchanged)."""
    if collision:
        shape = element(link, "collision", name=name)
        element(shape, "pose", pose)
        geometry = element(element(shape, "geometry"), "box")
        element(geometry, "size", " ".join(map(str, size)))
    visual = element(link, "visual", name=name + "_visual")
    element(visual, "pose", pose)
    element(element(element(visual, "geometry"), "mesh"), "uri", mesh_path.as_uri())
    material = element(visual, "material")
    element(material, "ambient", "1 1 1 1")
    element(material, "diffuse", "1 1 1 1")
    metal = element(element(material, "pbr"), "metal")
    element(metal, "albedo_map", str(albedo_map))
    element(metal, "roughness", 0.9)
    element(metal, "metalness", 0.0)
    if reflectivity is not None:
        element(visual, "laser_retro", RETRO_PER_REFLECTIVITY * reflectivity)
