"""WGS84 GPS fixes to an explicitly selected local east/north/up frame."""

import numpy as np

from ..core.measurements import GpsReading


def ecef(fix: GpsReading):
    latitude, longitude = np.deg2rad([fix.latitude_deg, fix.longitude_deg])
    eccentricity_squared = 6.6943799901413165e-3
    radius = 6378137.0 / np.sqrt(1 - eccentricity_squared * np.sin(latitude) ** 2)
    return np.array(
        [
            (radius + fix.altitude_m) * np.cos(latitude) * np.cos(longitude),
            (radius + fix.altitude_m) * np.cos(latitude) * np.sin(longitude),
            (radius * (1 - eccentricity_squared) + fix.altitude_m) * np.sin(latitude),
        ]
    )


def gps_to_enu(fix: GpsReading, reference: GpsReading):
    """Both altitudes use the same ellipsoidal datum; output is receiver position."""
    lat, lon = np.deg2rad([reference.latitude_deg, reference.longitude_deg])
    rotation = np.array(
        [
            [-np.sin(lon), np.cos(lon), 0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )
    return rotation @ (ecef(fix) - ecef(reference))
