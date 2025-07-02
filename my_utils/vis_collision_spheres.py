import os
import numpy as np
from visualizer import Visualizer
import torch
import trimesh as tm
import yaml

if __name__ == '__main__':
    # ------------- create visualizer -------------
    
    # lz_gripper
    # robot_urdf_path = 'src/curobo/content/assets/robot/lz_gripper_description/urdf/right_with_tip_link.urdf'
    # mesh_dir_path = 'src/curobo/content/assets/robot/lz_gripper_description/'
    # robot_config_path = 'src/curobo/content/configs/robot/lz_gripper.yml'
    # hand_pose = torch.zeros((1, 3 + 4 + 11))
    # hand_pose[:, 7:] = torch.tensor([1.0, -0.15, -0.15, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]).reshape(1, -1)  # joint angles

    # shadow
    # robot_urdf_path = 'src/curobo/content/assets/robot/shadow_hand/right_sim.urdf'
    # mesh_dir_path = 'src/curobo/content/assets/robot/shadow_hand/'
    # robot_config_path = 'src/curobo/content/configs/robot/right_shadow_hand_sim.yml'
    # hand_pose = torch.zeros((1, 3 + 4 + 24))
    # hand_pose[:, 7:9] = 0.0 # wrist
    # hand_pose[:, 9:] = torch.tensor([-0.1, 0.3, 0., 0., 0, 0.3, 0., 0., -0.1, 0.3, 0., 0., 0, -0.2, 0.3, 0., 0., 0, 1., 0, -0.2, 0]).reshape(1, -1)  # joint angles

    # leap
    # robot_urdf_path = 'src/curobo/content/assets/robot/leap_description/urdf/leap_hand_simplified.urdf'
    # mesh_dir_path = 'src/curobo/content/assets/robot/leap_description/'
    # robot_config_path = 'src/curobo/content/configs/robot/leap.yml'
    # hand_pose = torch.zeros((1, 3 + 4 + 16))
    # hand_pose[:, 7:] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, -1)  # joint angles

    # allegro
    robot_urdf_path = 'src/curobo/content/assets/robot/allegro_description/allegro_hand_description_right.urdf'
    mesh_dir_path = 'src/curobo/content/assets/robot/allegro_description/'
    robot_config_path = 'src/curobo/content/configs/robot/allegro.yml'
    hand_pose = torch.zeros((1, 3 + 4 + 16))
    hand_pose[:, 7:] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, -1)  # joint angles

    visualize = Visualizer(robot_urdf_path=robot_urdf_path, mesh_dir_path=mesh_dir_path)
    
    hand_pose[:, 3] = 1.0
    # hand_pose[:, 7:] = torch.rand(size=(1, 11)) * 0.6 - 0.3  # random joint angles
    visualize.set_robot_parameters(hand_pose)
    robot_mesh = visualize.get_robot_trimesh_data(i=0, color=[0, 255, 0, 60])

    with open(robot_config_path, 'r') as file:
        robot_config = yaml.safe_load(file)

    sphere_meshes = []

    collision_spheres = robot_config['robot_cfg']['kinematics']['collision_spheres']
    if not isinstance(collision_spheres, dict):
        with open(os.path.join('src/curobo/content/configs/robot', collision_spheres), 'r') as file:
            collision_spheres = yaml.safe_load(file)['collision_spheres']
    for link_name, spheres in collision_spheres.items():
        if spheres is None or len(spheres) == 0:
            continue
        for sphere in spheres:
            center = sphere['center']
            radius = sphere['radius']
            pos = visualize.current_status[link_name].transform_points(torch.tensor(center).reshape(1, 3))

            sphere_mesh = tm.creation.icosphere(subdivisions=4, radius=radius)
            transform = np.eye(4)
            transform[:3, 3] = pos.numpy()
            sphere_mesh.apply_transform(transform)
            sphere_mesh.visual.face_colors = [255, 0, 0, 255] 

            sphere_meshes.append(sphere_mesh)

    # semantic aware frame
    axes_mesh = None
    hand_pose_transfer = robot_config['robot_cfg']['kinematics']['hand_pose_transfer_path']
    if not isinstance(hand_pose_transfer, dict):
        with open(os.path.join('src/curobo/content/configs/robot', hand_pose_transfer), 'r') as file:
            hand_pose_transfer_data = yaml.safe_load(file)
            root_link_name = list(hand_pose_transfer_data.keys())[0]  # e.g., 'rh_palm'
            hand_pose_transfer_data = hand_pose_transfer_data[root_link_name]  # get the first item, e.g., 'rh_palm'
            t, R = np.array(hand_pose_transfer_data['t']), np.array(hand_pose_transfer_data['r'])
            axes_mesh = visualize.get_axes_trimesh_data(t, R, root_link_name)

    scene = tm.Scene(geometry=[robot_mesh] + sphere_meshes + [axes_mesh])
    scene.show()