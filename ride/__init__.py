"""Ride: a first-person street video -> 3D street map, crosswalks, parked vehicles, daylighting / occlusion checks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ELIDE = Path("/Users/sudharshanramesh/Studies/MyProjects/ElideDB")
G = 518                     # LingBot's depth comes back on a G x G grid (the frame squeezed to a square)
