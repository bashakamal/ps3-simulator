"""
S3Simulator-Physics: Blender Batch Render Script
=================================================
Generates synthetic top-down images of ship models on seabed textures.
Simulates side-scan sonar geometry:
  - Camera is FIXED at origin (nadir centre)
  - Ships are offset to PORT (left) or STARBOARD (right)
  - Shadow falls toward nadir zone — just like real SSS imagery

Run in Blender 4.x: Scripting tab → Open → Run Script

Usage:
    Open Blender → Scripting tab → Open this file → Run Script
    All images render automatically and save to OUTPUT_DIR

Author: S3Simulator-Physics Pipeline
"""

import bpy
import os
import json
import math
import random
from mathutils import Vector, Euler

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — edit these paths only
# ═══════════════════════════════════════════════════════════════

# Ship STL models — using 3 ships as decided
SHIP_MODELS = [
    r"\\wsl.localhost\Ubuntu\home\kamalbasha\blender\b_s_1.stl",
    r"\\wsl.localhost\Ubuntu\home\kamalbasha\blender\b_s_2.stl",
    r"\\wsl.localhost\Ubuntu\home\kamalbasha\blender\b_s_3.stl",
]

# Seabed texture PNG files from PolyHaven
SEABED_TEXTURES = {
    "sand": r"C:\Users\kamal\Downloads\sand_01_4k\textures\sand_01_diff_4k.png",
    # "gravel": r"C:\Users\kamal\Downloads\gravel_4k\textures\gravel_diff_4k.png",
    # "rock":   r"C:\Users\kamal\Downloads\rock_4k\textures\rock_diff_4k.png",
}

# Output directory
OUTPUT_DIR = r"C:\Users\kamal\Downloads\blender_renders"

# ═══════════════════════════════════════════════════════════════
# DATASET PARAMETERS
# ═══════════════════════════════════════════════════════════════

# Camera altitudes in metres (Z height above seabed)
# Camera always stays directly above origin (nadir centre)
ALTITUDES = [50, 70, 100]

# Sun angles: (name, azimuth_deg, elevation_deg)
# Sun comes from the SAME side as the ship so shadow falls toward nadir (centre)
# This is handled dynamically per position in setup_sun()
SUN_CONDITIONS = [
    ("morning_low",   90,  25),
    ("morning_mid",   90,  50),
    ("noon",           0,  85),
    ("afternoon_mid", 270, 50),
    ("afternoon_low", 270, 25),
]

# ── SSS-Realistic Ship Positions ────────────────────────────────
# Camera is FIXED at (0,0). Ships are offset left (port) or right (starboard).
# Y offsets add along-track variation (forward/aft position in swath).
# (x_offset_m, y_offset_m, label)
#
#   Negative X = PORT  side (left  in image)
#   Positive X = STARBOARD side (right in image)
#
# Offsets chosen so ship stays clearly within swath at all altitudes.
# At altitude=50m, ortho_scale=90m → swath half-width ≈ 45m
# At altitude=100m, ortho_scale=180m → swath half-width ≈ 90m
# Ship offsets of 30–70m keep ship visible at all three altitudes.

POSITIONS = [
    # Port side (ship on left of nadir)
    (-30,  20, "port_fwd_near"),
    (-30,   0, "port_mid_near"),
    (-30, -20, "port_aft_near"),
    (-55,  20, "port_fwd_far"),
    (-55,   0, "port_mid_far"),
    (-55, -20, "port_aft_far"),
    # Starboard side (ship on right of nadir)
    ( 30,  20, "stbd_fwd_near"),
    ( 30,   0, "stbd_mid_near"),
    ( 30, -20, "stbd_aft_near"),
    ( 55,  20, "stbd_fwd_far"),
    ( 55,   0, "stbd_mid_far"),
    ( 55, -20, "stbd_aft_far"),
]

# Ship rotation angles (degrees) — varied orientations
SHIP_ROTATIONS = [0, 15, 30, 45, 60, 75, 90, 110, 135, 150,
                  165, 180, 195, 210, 225, 240, 260, 270, 290,
                  315, 330, 350]

# Render settings
RENDER_WIDTH    = 1728
RENDER_HEIGHT   = 1929
RENDER_SAMPLES  = 32   # 32 = good balance speed/quality for dataset

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def clear_scene():
    """Remove all objects and orphan data from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def setup_render_engine():
    """Configure EEVEE Next render engine — fast, realistic shadows."""
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    bpy.context.scene.render.resolution_x = RENDER_WIDTH
    bpy.context.scene.render.resolution_y = RENDER_HEIGHT
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'

    eevee = bpy.context.scene.eevee
    eevee.taa_render_samples = RENDER_SAMPLES
    eevee.use_shadows = True

    # Dark underwater world background
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.02, 0.05, 0.08, 1.0)
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.1


def create_seabed(texture_png_path, seabed_name):
    """
    Create large seabed plane centred at origin with PNG diffuse texture.
    Compatible with any PolyHaven diffuse PNG.
    """
    bpy.ops.mesh.primitive_plane_add(size=2000, location=(0, 0, 0))
    seabed = bpy.context.active_object
    seabed.name = "seabed"

    mat   = bpy.data.materials.new(name=f"seabed_{seabed_name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_output  = nodes.new("ShaderNodeOutputMaterial")
    node_bsdf    = nodes.new("ShaderNodeBsdfPrincipled")
    node_tex     = nodes.new("ShaderNodeTexImage")
    node_mapping = nodes.new("ShaderNodeMapping")
    node_coord   = nodes.new("ShaderNodeTexCoord")

    node_coord.location   = (-800, 0)
    node_mapping.location = (-600, 0)
    node_tex.location     = (-300, 0)
    node_bsdf.location    = (0,    0)
    node_output.location  = (300,  0)

    try:
        img = bpy.data.images.load(texture_png_path)
        node_tex.image = img
        print(f"  Seabed texture loaded: {texture_png_path}")
    except Exception as e:
        print(f"  Texture load failed: {e} — using fallback sand colour")
        node_bsdf.inputs["Base Color"].default_value = (0.76, 0.65, 0.48, 1.0)

    # Tiling — 20x repeat across 2000m plane
    node_mapping.inputs["Scale"].default_value = (20, 20, 1)
    node_bsdf.inputs["Roughness"].default_value = 0.9

    links.new(node_coord.outputs["UV"],       node_mapping.inputs["Vector"])
    links.new(node_mapping.outputs["Vector"], node_tex.inputs["Vector"])
    links.new(node_tex.outputs["Color"],      node_bsdf.inputs["Base Color"])
    links.new(node_bsdf.outputs["BSDF"],      node_output.inputs["Surface"])

    if seabed.data.materials:
        seabed.data.materials[0] = mat
    else:
        seabed.data.materials.append(mat)

    return seabed


def import_ship(stl_path):
    """
    Import STL ship model, scale to target size, place on seabed.
    Ship X/Y position is set later in the render loop.
    """
    bpy.ops.wm.stl_import(filepath=stl_path)
    ship = bpy.context.active_object
    ship.name = "ship"

    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    ship.location = (0, 0, 0)

    # Grey metallic material — mimics SSS acoustic return appearance
    mat  = bpy.data.materials.new(name="ship_material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1.0)
    bsdf.inputs["Metallic"].default_value   = 0.7
    bsdf.inputs["Roughness"].default_value  = 0.4
    if ship.data.materials:
        ship.data.materials[0] = mat
    else:
        ship.data.materials.append(mat)

    # Scale ship so longest dimension = 40m (realistic wreck size)
    max_dim = max(ship.dimensions)
    if max_dim > 0:
        target_size  = 40.0
        scale_factor = target_size / max_dim
        ship.scale   = (scale_factor, scale_factor, scale_factor)
        bpy.ops.object.transform_apply(scale=True)
        # Place ship bottom face exactly on seabed (Z=0)
        ship.location.z = ship.dimensions.z / 2

    ship.cycles.cast_shadow = True
    return ship


def setup_sun(azimuth_deg, elevation_deg, ship_x):
    """
    Add directional sun lamp with sharp shadows.

    SSS shadow logic:
      - Ship on PORT  (x < 0): sun comes from the LEFT  → shadow falls RIGHT toward nadir
      - Ship on STBD  (x > 0): sun comes from the RIGHT → shadow falls LEFT  toward nadir

    The sun azimuth is overridden to always point FROM the ship TOWARD nadir,
    so the acoustic shadow falls realistically toward the nadir zone centre.

    Elevation from the SUN_CONDITIONS tuple is preserved (controls shadow length).
    """
    el_rad = math.radians(90 - elevation_deg)

    if ship_x < 0:
        # Ship is on left (port) — sun from left, shadow points right toward centre
        az_deg = 90     # sun from east/left
    elif ship_x > 0:
        # Ship is on right (starboard) — sun from right, shadow points left toward centre
        az_deg = 270    # sun from west/right
    else:
        # Ship at centre (unlikely in this setup) — use configured azimuth
        az_deg = azimuth_deg

    az_rad = math.radians(az_deg)

    bpy.ops.object.light_add(type='SUN', location=(0, 0, 200))
    sun          = bpy.context.active_object
    sun.name     = "sun"
    sun.rotation_euler   = Euler((el_rad, 0, az_rad), 'XYZ')
    sun.data.energy      = 6.0
    sun.data.angle       = math.radians(0.1)   # sharp shadow edge
    sun.data.use_shadow  = True
    return sun


def setup_camera(altitude_m):
    """
    Fixed nadir-centred orthographic camera.

    Camera is ALWAYS at (0, 0, altitude_m) — the nadir point.
    Ships are offset left/right from this centre, exactly as in real SSS.
    ortho_scale controls how much of the seabed is visible (swath width).
    """
    bpy.ops.object.camera_add(location=(0, 0, altitude_m))
    cam      = bpy.context.active_object
    cam.name = "top_down_camera"

    # Point straight down — no rotation
    cam.rotation_euler = Euler((0, 0, 0), 'XYZ')

    cam.data.type        = 'ORTHO'
    cam.data.ortho_scale = altitude_m * 1.8   # swath width = 1.8 × altitude

    bpy.context.scene.camera = cam
    return cam


def save_metadata(meta_path, image_id, filename, ship_name, seabed_name,
                  altitude_m, position_label, sun_name, sun_azimuth_used,
                  ship_x, ship_y, ship_rotation_deg):
    """Save metadata JSON alongside rendered image."""
    swath_half = (altitude_m * 1.8) / 2.0
    side       = "port" if ship_x < 0 else "starboard"
    range_m    = abs(ship_x)   # cross-track range from nadir

    meta = {
        "image_id"    : image_id,
        "filename"    : filename,
        "ship_model"  : ship_name,
        "position"    : {
            "x_meters"       : ship_x,
            "y_meters"       : ship_y,
            "cross_track_side"  : side,
            "cross_track_range_m": range_m,
        },
        "geometry"    : {
            "altitude_m"            : altitude_m,
            "swath_width_m"         : round(altitude_m * 1.8, 1),
            "swath_half_width_m"    : round(swath_half, 1),
            "nadir_at_image_centre" : True,
            "grazing_angle_deg"     : round(math.degrees(
                math.atan2(altitude_m, max(range_m, 1))), 2),
        },
        "environment" : {
            "sun_condition"     : sun_name,
            "sun_azimuth_used_deg": sun_azimuth_used,
            "seabed_type"       : seabed_name,
        },
        "ship_rotation_deg" : ship_rotation_deg,
        "position_label"    : position_label,
        "nadir_applied"     : False,
        "pipeline"          : "S3Simulator-Physics-Blender-v1",
        "note"              : (
            "Camera fixed at nadir (0,0). Ship offset to port/starboard. "
            "Shadow falls toward nadir as in real SSS imagery. "
            "Nadir void applied separately via ICPR pipeline."
        ),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


# ═══════════════════════════════════════════════════════════════
# MAIN BATCH RENDER LOOP
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("S3Simulator-Physics: Blender Batch Render")
    print("SSS Mode: Camera fixed at nadir, ships offset L/R")
    print("="*60)

    img_dir  = os.path.join(OUTPUT_DIR, "images")
    meta_dir = os.path.join(OUTPUT_DIR, "metadata")
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    setup_render_engine()

    total_jobs = (len(SHIP_MODELS)        *
                  len(SEABED_TEXTURES)    *
                  len(ALTITUDES)          *
                  len(SUN_CONDITIONS)     *
                  len(POSITIONS))

    print(f"Ships          : {len(SHIP_MODELS)}")
    print(f"Seabed types   : {len(SEABED_TEXTURES)}")
    print(f"Altitudes      : {ALTITUDES}")
    print(f"Sun conditions : {len(SUN_CONDITIONS)}")
    print(f"Positions      : {len(POSITIONS)}  ({len(POSITIONS)//2} port, {len(POSITIONS)//2} starboard)")
    print(f"Total images   : {total_jobs}")
    print(f"Output dir     : {OUTPUT_DIR}\n")

    shot_num = 0

    for ship_path in SHIP_MODELS:
        ship_name = os.path.splitext(os.path.basename(ship_path))[0]

        for seabed_name, seabed_path in SEABED_TEXTURES.items():

            for altitude_m in ALTITUDES:

                for sun_name, sun_az, sun_el in SUN_CONDITIONS:

                    for pos_x, pos_y, pos_label in POSITIONS:

                        # ── File naming ───────────────────────────────
                        image_id  = f"sss_{shot_num:04d}"
                        filename  = (f"{image_id}_{ship_name}_{seabed_name}_"
                                     f"z{altitude_m:03d}_{sun_name}_{pos_label}.png")
                        img_path  = os.path.join(img_dir,  filename)
                        meta_path = os.path.join(meta_dir, f"{image_id}_meta.json")

                        print(f"[{shot_num+1}/{total_jobs}] {filename}")

                        if os.path.exists(img_path):
                            print(f"  Already exists — skipping")
                            shot_num += 1
                            continue

                        # ── Build scene ───────────────────────────────
                        clear_scene()

                        # Seabed centred at origin
                        create_seabed(seabed_path, seabed_name)

                        # Ship — scaled and placed at SSS offset position
                        ship = import_ship(ship_path)
                        ship.location.x = pos_x
                        ship.location.y = pos_y
                        # Random rotation for orientation variety
                        ship_rot = random.choice(SHIP_ROTATIONS)
                        ship.rotation_euler.z = math.radians(ship_rot)

                        # Sun — direction always pushes shadow toward nadir
                        # Returns the actual azimuth used (overrides configured value)
                        sun_az_used = 90 if pos_x < 0 else 270
                        setup_sun(sun_az, sun_el, ship_x=pos_x)

                        # Camera FIXED at nadir (0, 0) — NOT above ship
                        setup_camera(altitude_m)

                        # ── Render ────────────────────────────────────
                        bpy.context.scene.render.filepath = img_path
                        bpy.ops.render.render(write_still=True)

                        # ── Metadata ──────────────────────────────────
                        save_metadata(
                            meta_path, image_id, filename,
                            ship_name, seabed_name, altitude_m,
                            pos_label, sun_name, sun_az_used,
                            pos_x, pos_y, ship_rot)

                        shot_num += 1

    print("\n" + "="*60)
    print(f"DONE — {shot_num} images rendered")
    print(f"Images   : {img_dir}")
    print(f"Metadata : {meta_dir}")
    print("="*60)


# Run
main()
