"""
S3Simulator-Physics: Blender Batch Render Script
=================================================
Generates synthetic top-down images of ship models on seabed textures.
Run in Blender 4.x: Scripting tab → Open → Run Script

Usage:
    Open Blender → Scripting tab → Open this file → Run Script
    All 648+ images render automatically and save to OUTPUT_DIR

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

# Ship STL models (Windows paths accessible from Blender)
SHIP_MODELS = [
    r"\\wsl.localhost\Ubuntu\home\kamalbasha\blender\b_s_1.stl",
    r"\\wsl.localhost\Ubuntu\home\kamalbasha\blender\b_s_2.stl",
    r"\\wsl.localhost\Ubuntu\home\kamalbasha\blender\b_s_3.stl",
    r"\\wsl.localhost\Ubuntu\home\kamalbasha\blender\b_s_4.stl",
    r"\\wsl.localhost\Ubuntu\home\kamalbasha\blender\b_s_5.stl",
    r"\\wsl.localhost\Ubuntu\home\kamalbasha\blender\b_s_6.stl",
    r"\\wsl.localhost\Ubuntu\home\kamalbasha\blender\b_s_7.stl",
]

# Seabed texture blend files from PolyHaven
# Add more seabed blend files here as you download them
SEABED_TEXTURES = {
    "sand": r"C:\Users\kamal\Downloads\sand_01_4k\textures\sand_01_diff_4k.png",
    # "gravel": r"C:\Users\kamal\Downloads\gravel_4k\gravel_4k.blend",
    # "rock":   r"C:\Users\kamal\Downloads\rock_4k\rock_4k.blend",
}

# Output directory
OUTPUT_DIR = r"C:\Users\kamal\Downloads\blender_renders"

# ═══════════════════════════════════════════════════════════════
# DATASET PARAMETERS
# ═══════════════════════════════════════════════════════════════

# Camera altitudes in metres (Z height above seabed)
ALTITUDES = [50, 70, 100]   # 20m removed — ship too large at low altitude

# Sun angles: (name, azimuth_deg, elevation_deg)
SUN_CONDITIONS = [
    ("morning_low",   90,  25),   # low morning sun — long shadow extending right
    ("morning_mid",   90,  50),   # mid morning — medium shadow
    ("noon",           0,  85),   # overhead — very short shadow directly below
    ("afternoon_mid", 270, 50),   # mid afternoon — medium shadow
    ("afternoon_low", 270, 25),   # low afternoon — long shadow extending left
]

# Ship positions relative to camera centre
# (x_offset, y_offset, label)
POSITIONS = [
    (-60,  60, "left_top"),
    (-60,   0, "left_mid"),
    (-60, -60, "left_low"),
    (  0,  60, "centre_top"),
    (  0,   0, "centre_mid"),
    (  0, -60, "centre_low"),
    ( 60,  60, "right_top"),
    ( 60,   0, "right_mid"),
    ( 60, -60, "right_low"),
]

# Ship rotation angles (degrees) — mixed orientations
# Ship rotations — varied angles including tilted positions
# Not just cardinal directions — adds natural variety
SHIP_ROTATIONS = [0, 15, 30, 45, 60, 75, 90, 110, 135, 150, 165, 180,
                  195, 210, 225, 240, 260, 270, 290, 315, 330, 350]

# Nadir zone sets
# Nadir zone handled separately by ICPR pipeline — not in Blender render

# Nadir widths assigned by ICPR pipeline post-processing

# Render settings
RENDER_WIDTH  = 1728
RENDER_HEIGHT = 1929
RENDER_SAMPLES = 64    # Cycles GPU — 64 samples + denoising gives clean result fast

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def setup_render_engine():
    """Configure Cycles with GPU — fast renders, proper shadows, stable."""
    scene = bpy.context.scene

    # Use Cycles — proper ray-traced shadows, GPU accelerated
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'

    # GPU settings
    cycles = scene.cycles
    cycles.device = 'GPU'
    cycles.samples = RENDER_SAMPLES          # low samples = fast render
    cycles.use_denoising = True              # clean image even with low samples
    cycles.denoiser = 'OPENIMAGEDENOISE'     # CPU denoiser — works without CUDA

    # Enable GPU in preferences
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'       # NVIDIA GPU
    prefs.get_devices()
    for device in prefs.devices:
        device.use = True                    # enable all available devices
        print(f"  Device: {device.name} — {device.type} — enabled={device.use}")

    # Dark world background
    world = scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.05, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.05


def create_seabed(texture_png_path, seabed_name):
    """
    Create seabed plane with PNG texture loaded directly.
    Works with any PolyHaven diffuse PNG.
    """
    # Large flat plane
    bpy.ops.mesh.primitive_plane_add(size=2000, location=(0, 0, 0))
    seabed = bpy.context.active_object
    seabed.name = "seabed"

    # Create material with image texture node
    mat = bpy.data.materials.new(name=f"seabed_{seabed_name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Add nodes: Texture Coordinate -> Mapping -> Image Texture -> BSDF -> Output
    node_output  = nodes.new("ShaderNodeOutputMaterial")
    node_bsdf    = nodes.new("ShaderNodeBsdfPrincipled")
    node_tex     = nodes.new("ShaderNodeTexImage")
    node_mapping = nodes.new("ShaderNodeMapping")
    node_coord   = nodes.new("ShaderNodeTexCoord")

    # Positions
    node_coord.location   = (-800, 0)
    node_mapping.location = (-600, 0)
    node_tex.location     = (-300, 0)
    node_bsdf.location    = (0, 0)
    node_output.location  = (300, 0)

    # Load texture image
    try:
        img = bpy.data.images.load(texture_png_path)
        node_tex.image = img
        print(f"  Texture loaded: {texture_png_path}")
    except Exception as e:
        print(f"  Texture load failed: {e}")
        print(f"  Using fallback sand colour")
        node_bsdf.inputs["Base Color"].default_value = (0.76, 0.65, 0.48, 1.0)

    # Texture tiling
    node_mapping.inputs["Scale"].default_value = (20, 20, 1)

    # Roughness
    node_bsdf.inputs["Roughness"].default_value = 0.9

    # Link nodes
    links.new(node_coord.outputs["UV"],       node_mapping.inputs["Vector"])
    links.new(node_mapping.outputs["Vector"], node_tex.inputs["Vector"])
    links.new(node_tex.outputs["Color"],      node_bsdf.inputs["Base Color"])
    links.new(node_bsdf.outputs["BSDF"],      node_output.inputs["Surface"])

    # Apply material
    if seabed.data.materials:
        seabed.data.materials[0] = mat
    else:
        seabed.data.materials.append(mat)

    return seabed


def import_ship(stl_path):
    """Import STL ship model, centre it, place on seabed."""
    bpy.ops.wm.stl_import(filepath=stl_path)
    ship = bpy.context.active_object
    ship.name = "ship"

    # Centre origin to geometry
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    # Will be repositioned after scaling — set to 0 for now
    ship.location = (0, 0, 0)

    # Apply grey material — SSS images are grayscale
    mat = bpy.data.materials.new(name="ship_material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.7
    bsdf.inputs["Roughness"].default_value = 0.4
    if ship.data.materials:
        ship.data.materials[0] = mat
    else:
        ship.data.materials.append(mat)

    # Scale ship to reasonable size relative to seabed
    # Target: ship occupies ~10-15% of image width, like real SSS
    max_dim = max(ship.dimensions)
    if max_dim > 0:
        target_size = 40.0   # ship length target — larger for visibility at dataset scale
        scale_factor = target_size / max_dim
        ship.scale = (scale_factor, scale_factor, scale_factor)
        bpy.ops.object.transform_apply(scale=True)
        # Place ship base exactly on seabed Z=0
        # After scaling, move ship so its BOTTOM face is at Z=0
        # dims.z = full height of ship — bottom is at -dims.z/2 from origin
        # So set location.z = dims.z/2 to put bottom at Z=0
        dims = ship.dimensions
        ship.location.z = dims.z / 2   # bottom face on seabed, no floating

    ship.cycles.cast_shadow = True
    return ship


def setup_sun(azimuth_deg, elevation_deg, ship_x=0, ship_y=0):
    """
    Add directional sun lamp with shadow.
    Sun direction is biased opposite to ship position
    so shadow falls away from ship toward image centre.
    If ship is on the left  → sun comes from left  → shadow goes right
    If ship is on the right → sun comes from right → shadow goes left
    """
    # Base azimuth from session sun condition
    az_rad = math.radians(azimuth_deg)
    el_rad = math.radians(90 - elevation_deg)

    # Sun direction from azimuth — shadow falls opposite to sun direction
    # Ship position biases azimuth slightly so shadow falls toward image centre
    # Small bias only — keeps shadow attached to object
    if ship_x > 0:
        az_rad = math.radians(azimuth_deg + 30)   # ship right → shadow left
    elif ship_x < 0:
        az_rad = math.radians(azimuth_deg - 30)   # ship left  → shadow right
    else:
        az_rad = math.radians(azimuth_deg)

    bpy.ops.object.light_add(type='SUN', location=(0, 0, 200))
    sun = bpy.context.active_object
    sun.name = "sun"
    sun.rotation_euler = Euler((el_rad, 0, az_rad), 'XYZ')
    sun.data.energy = 6.0
    sun.data.angle = math.radians(0.1)   # very sharp shadow edge
    sun.data.use_shadow = True
    return sun


def setup_camera(altitude_m, ship_x, ship_y):
    """
    Orthographic top-down camera directly above ship position.
    Camera always centres on the ship — no cropping.
    ortho_scale controls how much ground is visible around the ship.
    """
    bpy.ops.object.camera_add(location=(ship_x, ship_y, altitude_m))
    cam = bpy.context.active_object
    cam.name = "top_down_camera"

    # Point straight down (-Z direction)
    cam.rotation_euler = Euler((0, 0, 0), 'XYZ')

    cam.data.type = 'ORTHO'
    # Scale: metres visible across image width
    # Higher altitude = wider view = ship appears smaller
    cam.data.ortho_scale = altitude_m * 1.8   # balanced — ship visible but seabed context preserved

    bpy.context.scene.camera = cam
    return cam


def add_nadir_overlay(output_path, nadir_width_m, altitude_m, image_w, image_h):
    """
    Post-process: add nadir zone black void to rendered image.
    Uses your ICPR pipeline logic applied via PIL.
    """
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(output_path).convert('L')   # grayscale
        arr = np.array(img).astype(np.float32) / 255.0
        h, w = arr.shape

        # Nadir zone
        swath_m    = altitude_m * 2.0
        nadir_frac = nadir_width_m / swath_m
        sq         = int(nadir_frac * min(h, w))
        si         = (w - sq) // 2
        ei         = si + sq + 2
        mask       = np.ones_like(arr)

        for i in range(h):
            ss = max(0,   si - int(random.uniform(0, 8)))
            se = min(w,   ei + int(random.uniform(0, 8)))
            mask[i, ss:se] = random.uniform(0, 0.2)

        lp = (si + ei) // 2
        mask[:, lp-1:lp+2] = 1
        arr = np.clip(arr * mask, 0, 1)

        # Save back
        result = Image.fromarray((arr * 255).astype(np.uint8), 'L')
        result.save(output_path)
        return True
    except ImportError:
        print("  PIL not available — saving without nadir overlay")
        return False


def save_metadata(meta_path, image_id, filename, ship_name, seabed_name,
                  altitude_m, position_label, sun_name, sun_azimuth,
                  ship_x, ship_y, ship_rotation_deg=0):
    """Save metadata JSON paired with rendered image."""
    meta = {
        "image_id": image_id,
        "filename": filename,
        "ship_model": ship_name,
        "position": {
            "x_meters": ship_x,
            "y_meters": ship_y
        },
        "geometry": {
            "altitude_m": altitude_m,
            "swath_width_m": round(altitude_m * 1.8, 1),
            "grazing_angle_deg": round(math.degrees(
                math.atan2(altitude_m, max(math.sqrt(ship_x**2 + ship_y**2), 1))), 2)
        },
        "environment": {
            "sun_condition": sun_name,
            "sun_azimuth_deg": sun_azimuth,
            "seabed_type": seabed_name
        },
        "ship_rotation_deg": ship_rotation_deg,
        "position_label": position_label,
        "nadir_applied": False,
        "pipeline": "S3Simulator-Physics-Blender-v1",
        "note": "Nadir zone applied separately via ICPR pipeline"
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


# ═══════════════════════════════════════════════════════════════
# MAIN BATCH RENDER LOOP
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("S3Simulator-Physics: Blender Batch Render")
    print("="*60)

    # Create output directories
    img_dir  = os.path.join(OUTPUT_DIR, "images")
    meta_dir = os.path.join(OUTPUT_DIR, "metadata")
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # Setup render engine once
    setup_render_engine()

    shot_num   = 0
    total_jobs = (len(SHIP_MODELS) * len(SEABED_TEXTURES) *
                  len(ALTITUDES) * len(POSITIONS) *
                  len(SUN_CONDITIONS))

    print(f"Total images to render: {total_jobs}")
    print(f"Output: {OUTPUT_DIR}\n")

    for ship_path in SHIP_MODELS:
        ship_name = os.path.splitext(os.path.basename(ship_path))[0]

        for seabed_name, seabed_path in SEABED_TEXTURES.items():

            for altitude_m in ALTITUDES:

                for sun_name, sun_az, sun_el in SUN_CONDITIONS:

                    for pos_x, pos_y, pos_label in POSITIONS:

                        # Build image ID and filename
                        image_id  = f"sss_{shot_num:04d}"
                        filename  = (f"{image_id}_{ship_name}_{seabed_name}_"
                                     f"z{altitude_m:03d}_{sun_name}_{pos_label}.png")
                        img_path  = os.path.join(img_dir, filename)
                        meta_path = os.path.join(meta_dir, f"{image_id}_meta.json")

                        print(f"[{shot_num+1}/{total_jobs}] {filename}")

                        # Skip if already rendered
                        if os.path.exists(img_path):
                            print(f"  Already exists — skipping")
                            shot_num += 1
                            continue

                        # ── Build scene ──────────────────────────────
                        clear_scene()

                        # Seabed
                        create_seabed(seabed_path, seabed_name)

                        # Ship — random rotation for variety
                        ship = import_ship(ship_path)
                        ship.location.x = pos_x
                        ship.location.y = pos_y
                        ship_rot = random.choice(SHIP_ROTATIONS)
                        ship.rotation_euler.z = math.radians(ship_rot)

                        # Sun — position-aware so shadow direction is natural
                        setup_sun(sun_az, sun_el, ship_x=pos_x, ship_y=pos_y)

                        # Camera directly above ship — no cropping
                        setup_camera(altitude_m, pos_x, pos_y)

                        # ── Render ───────────────────────────────────
                        bpy.context.scene.render.filepath = img_path
                        bpy.ops.render.render(write_still=True)

                        # ── Save metadata ────────────────────────────
                        save_metadata(
                            meta_path, image_id, filename,
                            ship_name, seabed_name, altitude_m,
                            pos_label, sun_name, sun_az,
                            pos_x, pos_y, ship_rot)

                        shot_num += 1

    print("\n" + "="*60)
    print(f"DONE — {shot_num} images rendered")
    print(f"Images   : {img_dir}")
    print(f"Metadata : {meta_dir}")
    print("="*60)


# Run
main()
