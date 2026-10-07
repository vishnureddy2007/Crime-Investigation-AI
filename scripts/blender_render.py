import bpy
import sys
import json
import os

def setup_scene(plan):
    """
    Configures the Blender scene based on the StoryboardPlan.
    """
    # 1. Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 2. Create ground plane
    bounds_x, bounds_y, bounds_z = plan["environment_bounds"]
    bpy.ops.mesh.primitive_plane_add(size=max(bounds_x, bounds_y))
    ground = bpy.context.active_object
    ground.name = "GroundPlane"

    # Material for the ground (dark grey)
    mat_ground = bpy.data.materials.new(name="GroundMat")
    mat_ground.use_nodes = True
    nodes = mat_ground.node_tree.nodes
    nodes["Principled BSDF"].inputs[0].default_value = (0.05, 0.05, 0.05, 1.0)
    ground.data.materials.append(mat_ground)

    # 3. Add actors/objects
    actor_map = {}
    for obj_data in plan["objects"]:
        kind = obj_data["kind"]
        pos = obj_data["initial_pos"]
        label = obj_data["label"]

        if kind == "person":
            bpy.ops.mesh.primitive_cylinder_add(radius=0.25, depth=1.7, location=pos)
            color = (0.1, 0.4, 0.8, 1.0) # Blue
        elif kind == "weapon":
            bpy.ops.mesh.primitive_cube_add(size=0.2, location=pos)
            color = (1.0, 0.0, 0.0, 1.0) # Red
        else:
            bpy.ops.mesh.primitive_cube_add(size=0.3, location=pos)
            color = (0.8, 0.8, 0.8, 1.0)

        obj = bpy.context.active_object
        obj.name = obj_data["id"]
        actor_map[obj.name] = obj

        # Assign material
        mat = bpy.data.materials.new(name=f"Mat_{obj.name}")
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = color
        # Default to opaque
        mat.blend_method = 'OPAQUE'
        mat.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 1.0
        obj.data.materials.append(mat)

    # 4. Setup Camera
    if not plan["camera_shots"]:
        bpy.ops.object.camera_add(location=(10, -10, 10))
        cam = bpy.context.active_object
    else:
        cam_data = plan["camera_shots"][0]
        bpy.ops.object.camera_add(location=cam_data["pos"])
        cam = bpy.context.active_object
    
    bpy.context.scene.camera = cam

    # 5. Animation Logic
    scene = bpy.context.scene
    scene.frame_start = 1
    
    # Process Timeline Events
    for event in plan["timeline"]:
        frame = int(event["timestamp"] * scene.render.fps) + 1
        actor_id = event["actor_id"]
        status = event.get("status", "VERIFIED")

        if actor_id in actor_map:
            obj = actor_map[actor_id]
            if status == "PREDICTED":
                mat = obj.data.materials[0]
                mat.blend_method = 'BLEND'
                mat.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.4
            
            if event["action"] == "move_to":
                obj.location = event["target_pos"]
                obj.keyframe_insert(data_path="location", frame=frame)
            elif event["action"] == "HIGHLIGHT":
                obj.scale = (1.2, 1.2, 1.2)
                obj.keyframe_insert(data_path="scale", frame=frame)
                obj.scale = (1.0, 1.0, 1.0)
                obj.keyframe_insert(data_path="scale", frame=frame + 10)

    # Camera Animation
    for i, shot in enumerate(plan["camera_shots"]):
        frame = int((i * 5.0) * scene.render.fps) + 1
        cam.location = shot["pos"]
        cam.keyframe_insert(data_path="location", frame=frame)
        cam.rotation_euler = (1.57, 0, 0) 
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)

    scene.frame_end = int(plan["total_duration"] * scene.render.fps) + 24

def render_animation(output_dir):
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = os.path.join(output_dir, "frame_")
    scene.render.engine = 'BLENDER_EEVEE'
    # Removed the problematic sampling setting
    bpy.ops.render.render(animation=True)

def main():
    try:
        try:
            idx = sys.argv.index("--")
            args = sys.argv[idx + 1:]
        except ValueError:
            print("Error: No '--' separator found in arguments")
            sys.exit(1)

        if len(args) < 2:
            print(f"Error: Expected 2 arguments, got {len(args)}")
            sys.exit(1)

        json_path = args[0]
        output_dir = args[1]

        with open(json_path, 'r') as f:
            plan = json.load(f)

        setup_scene(plan)
        render_animation(output_dir)
        print("Blender render completed successfully")

    except Exception as e:
        import traceback
        print(f"Blender FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
