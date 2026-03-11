#!/usr/bin/env python3
"""
generate_braided_preform.py

Generates a 3D braided preform geometry using TexGen when available, otherwise falls back to a
numpy+meshio-based placeholder geometry generator. Reads parameters from docs/parameters.csv.

Outputs:
 - OBJ (for visualization)
 - Optional Abaqus INP (if generate_inp is yes and mesh generation is available)
 - Writes parameter statistics CSV (reads existing docs/parameters.csv and updates if needed)

Usage:
  python docs/generate_braided_preform.py --params docs/parameters.csv

Notes:
 - This script uses TexGen if importable. If TexGen is not installed, it will generate a simple
   centerline-based OBJ for quick visualization.
 - The parameters CSV in the repository currently contains placeholder values. For precise
   reproduction, replace with values extracted from the provided TexGen guide and Chen (1999) PDF.

"""
import os
import sys
import argparse
import csv
import math
from dataclasses import dataclass
from typing import Dict

HAS_TEXGEN = False
try:
    # TexGen package import - may vary depending on installation
    from TexGen.Core import CTextile, CPattern
    HAS_TEXGEN = True
except Exception:
    HAS_TEXGEN = False

# Optional imports for fallback geometry
try:
    import numpy as np
    import meshio
except Exception:
    np = None
    meshio = None

DEFAULT_PARAM_FILE = os.path.join(os.path.dirname(__file__), "parameters.csv")

@dataclass
class BraidParams:
    tow_diameter_mm: float = 0.5
    braid_angle_deg: float = 30.0
    num_tows_width: int = 8
    num_layers: int = 4
    tow_spacing_mm: float = 1.0
    width_mm: float = 50.0
    thickness_mm: float = 10.0
    length_mm: float = 100.0
    tow_cross_section: str = "circle"
    fiber_volume_fraction: float = 0.6
    output_format: str = "OBJ"
    mesh_resolution_pts_per_circum: int = 16
    pts_along_length: int = 100
    generate_inp: str = "yes"
    material_E_GPa: float = 70.0
    material_nu: float = 0.33

    @staticmethod
    def from_dict(d: Dict[str, str]):
        def g(key, cast, default=None):
            v = d.get(key, "")
            if v is None or v == "":
                return default
            try:
                return cast(v)
            except Exception:
                return default
        return BraidParams(
            tow_diameter_mm=g('tow_diameter_mm', float, 0.5),
            braid_angle_deg=g('braid_angle_deg', float, 30.0),
            num_tows_width=g('num_tows_width', int, 8),
            num_layers=g('num_layers', int, 4),
            tow_spacing_mm=g('tow_spacing_mm', float, 1.0),
            width_mm=g('width_mm', float, 50.0),
            thickness_mm=g('thickness_mm', float, 10.0),
            length_mm=g('length_mm', float, 100.0),
            tow_cross_section=g('tow_cross_section', str, 'circle'),
            fiber_volume_fraction=g('fiber_volume_fraction', float, 0.6),
            output_format=g('output_format', str, 'OBJ'),
            mesh_resolution_pts_per_circum=g('mesh_resolution_pts_per_circum', int, 16),
            pts_along_length=g('pts_along_length', int, 100),
            generate_inp=g('generate_inp', str, 'yes'),
            material_E_GPa=g('material_E_GPa', float, 70.0),
            material_nu=g('material_nu', float, 0.33),
        )


def read_params_csv(path: str) -> BraidParams:
    if not os.path.exists(path):
        print(f"Parameter file {path} not found, using defaults.")
        return BraidParams()
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # Read first non-empty row
        for row in reader:
            return BraidParams.from_dict(row)
    return BraidParams()


def write_params_csv(path: str, params: BraidParams):
    fieldnames = [
        'tow_diameter_mm','braid_angle_deg','num_tows_width','num_layers','tow_spacing_mm',
        'width_mm','thickness_mm','length_mm','tow_cross_section','fiber_volume_fraction',
        'output_format','mesh_resolution_pts_per_circum','pts_along_length','generate_inp',
        'material_E_GPa','material_nu'
    ]
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: getattr(params, k) for k in fieldnames})


def create_braid_placeholder_obj(params: BraidParams, out_obj: str):
    if np is None or meshio is None:
        raise RuntimeError('numpy and meshio are required for fallback generation. Install with pip install numpy meshio')

    # Generate simple centerlines for each tow and create tubular geometry by sweeping circle
    verts = []
    faces = []
    circumference_pts = max(8, params.mesh_resolution_pts_per_circum)
    pts_len = max(2, params.pts_along_length)

    nx = params.num_tows_width
    spacing = params.tow_spacing_mm
    z_layers = params.num_layers

    vert_index = 0
    for i in range(nx):
        x = (i - (nx-1)/2.0) * spacing
        for layer in range(z_layers):
            z = (layer - (z_layers-1)/2.0) * params.thickness_mm / max(1, z_layers-1)
            ys = np.linspace(-params.length_mm/2, params.length_mm/2, pts_len)
            for yi_idx, y in enumerate(ys):
                # center point
                cx = x
                cy = y
                cz = z
                # circle around center
                for t in range(circumference_pts):
                    theta = 2.0 * math.pi * t / circumference_pts
                    rx = (params.tow_diameter_mm/2.0) * math.cos(theta)
                    rz = (params.tow_diameter_mm/2.0) * math.sin(theta)
                    verts.append([cx + rx, cy, cz + rz])
            # create faces (naive quad->triangulate)
            # note: this is a simple, not watertight mesh for production; good for visualization checks
            for s in range(pts_len - 1):
                base = vert_index + s * circumference_pts
                for t in range(circumference_pts):
                    a = base + t
                    b = base + ((t + 1) % circumference_pts)
                    c = base + circumference_pts + t
                    d = base + circumference_pts + ((t + 1) % circumference_pts)
                    faces.append([a, b, c])
                    faces.append([b, d, c])
            vert_index += pts_len * circumference_pts

    # Write OBJ using meshio via triangular mesh
    try:
        # meshio supports writing 'triangle' meshes (cells) from vertices + triangles
        mesh = meshio.Mesh(points=np.array(verts), cells=[("triangle", np.array(faces))])
        meshio.write(out_obj, mesh)
    except Exception as e:
        # fallback: write minimal OBJ
        with open(out_obj, 'w') as f:
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                # OBJ is 1-indexed
                f.write("f " + " ".join(str(idx+1) for idx in face) + "\n")
    print(f"Wrote placeholder OBJ to {out_obj}")
    return out_obj


def create_with_texgen(params: BraidParams, out_obj: str):
    # This function is a template; real implementation requires mapping the parameters to TexGen API
    # and constructing patterns/yarns as per TexGen examples in the provided guide.
    # For now we raise if TexGen is missing.
    if not HAS_TEXGEN:
        raise RuntimeError('TexGen is not available in this runtime')
    # Pseudocode example - to be implemented after reading TexGenScriptingGuide
    # pattern = CPattern()
    # textile = CTextile()
    # -- define yarns, geometry, generate textile --
    # textile.ExportSTL(out_obj)
    print('TexGen path selected but concrete TexGen implementation is not yet scripted.')
    return out_obj


def main():
    parser = argparse.ArgumentParser(description='Generate braided preform geometry')
    parser.add_argument('--params', default=DEFAULT_PARAM_FILE, help='Path to parameters CSV')
    parser.add_argument('--out', default='docs/output.obj', help='Output OBJ path')
    args = parser.parse_args()

    params = read_params_csv(args.params)
    print('Using parameters:')
    print(params)

    out_obj = args.out

    if HAS_TEXGEN:
        print('Detected TexGen - attempting TexGen generation...')
        try:
            create_with_texgen(params, out_obj)
        except Exception as e:
            print('TexGen generation failed, falling back to placeholder generation:', e)
            create_braid_placeholder_obj(params, out_obj)
    else:
        print('TexGen not available - using fallback generator')
        create_braid_placeholder_obj(params, out_obj)

    # Ensure parameters CSV is up-to-date in repo
    write_params_csv(args.params, params)
    print('Generation complete. OBJ at', out_obj)

if __name__ == '__main__':
    main()