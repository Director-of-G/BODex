import numpy as np
from visualizer import Visualizer
import torch
import trimesh as tm
import yaml
import os
import json
from utils.utils_calc import posQuat2Isometry3d, quatWXYZ2XYZW

if __name__ == '__main__':
    # ------------- create visualizer -------------
    config_path = 'src/curobo/content/configs/manip/sim_leap_tac3d/fc.yml'
    robot_urdf_path = 'src/curobo/content/assets/robot/leap_description/leap_tac3d_v0.urdf'
    mesh_dir_path = 'src/curobo/content/assets/robot/leap_description/'
    table_yml_path = 'src/curobo/content/configs/world/collision_table.yml'
    grasp_pose_save_dir = 'src/curobo/content/assets/output/sim_leap_tac3d/fc/iros_selected_grasp_poses'
    visualize = Visualizer(robot_urdf_path=robot_urdf_path, mesh_dir_path=mesh_dir_path)

    # ------------- load config -------------
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    opt_initial_configuration = np.concatenate([config['seeder_cfg']['t'], 
                                                config['seeder_cfg']['r'], 
                                                config['seeder_cfg']['q']], axis=0, dtype=np.float32)

    # ------------- load table -------------
    with open(table_yml_path, 'r') as file:
        table_config = yaml.safe_load(file)
    table_size = np.asarray(table_config['cuboid']['table']['dims']) / 2.0
    table_pose = table_config['cuboid']['table']['pose']
    transform = posQuat2Isometry3d(table_pose[:3], quatWXYZ2XYZW(table_pose[3:]))
    table_mesh = tm.creation.box(extents=table_size)
    table_mesh.apply_transform(transform)
    table_mesh.apply_translation([0, 0, -0.76])
    table_mesh.visual.face_colors = [200, 200, 200, 200] 

    # ------------- load grasp results -------------
    grasp_file_dir = "src/curobo/content/assets/output/sim_leap_tac3d/fc/debug/graspdata"
    
    # !!!!
    object_code = 'bottle'
    grasp_file = os.path.join(grasp_file_dir, f"{object_code}/scale012_pose000_grasp.npy")
    grasp_res = np.load(grasp_file, allow_pickle=True).item()

    # command = '0'
    # grasp_pose_file = os.path.join(grasp_pose_save_dir, object_code, f"grasp_pose_{command}.npy")
    # grasp_pose = np.load(grasp_pose_file).reshape(1, -1)

    # object mesh
    obj_mesh_path = str(grasp_res['obj_path'][0])
    obj_mesh = tm.load_mesh(obj_mesh_path, process=False)
    obj_mesh = obj_mesh.copy().apply_scale(grasp_res['obj_scale'][0])
    normalized_info_path = f'/home/mingrui/mingrui/research/project_adaptive_grasping/MeshProcess/assets/object/iros_obj/processed_data/{object_code}/info/normalized.json'
    with open(normalized_info_path, 'r') as f:
        normalized_info = json.load(f)
    center = normalized_info['center']
    obj_mesh = obj_mesh.apply_translation(center)
    
    # grasp_pose_id = 0
    for index in range(0, 10):
        # optimation initial value
        visualize.set_robot_parameters(hand_pose=torch.tensor(opt_initial_configuration.reshape(1, 23)))
        robot_mesh_0 = visualize.get_robot_trimesh_data(i=0, color=[255, 0, 0])
        robot_mesh_0.apply_translation(center)

        robot_pose = grasp_res['robot_pose'][0, [index], 1, :]
        # robot_pose = grasp_pose.reshape(1, -1)
        # robot_pose[:, -4] -= 0.1
        visualize.set_robot_parameters(hand_pose=torch.tensor(robot_pose))
        robot_mesh_1 = visualize.get_robot_trimesh_data(i=0, color=[0, 255, 0])
        robot_mesh_1.apply_translation(center)
        # print("robot_pose: ", robot_pose[0])
        
        # scene = tm.Scene(geometry=[robot_mesh_0, robot_mesh_1, obj_mesh, table_mesh])
        scene = tm.Scene(geometry=[robot_mesh_1, obj_mesh, table_mesh])
        scene.show(smooth=False)

        print("Press 's' if you want to skip this configuration.")
        command = input()
        if command == 's':
            continue
        else:
            selected_robot_pose = robot_pose.copy()
            selected_robot_pose[:, :3] += center
            os.makedirs(os.path.join(grasp_pose_save_dir, object_code), exist_ok=True)
            file_name = os.path.join(grasp_pose_save_dir, object_code, f"grasp_pose_{command}.npy")
            np.save(file_name, robot_pose.reshape(-1, 23))
            # grasp_pose_id += 1
            print("Grasp pose: ", selected_robot_pose)
            print(f"Save the grasp pose to {file_name}.")