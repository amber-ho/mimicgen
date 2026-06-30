# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Main data generation script.

Examples:

    # run normal data generation
    python generate_dataset.py --config /path/to/config.json

    # render all data generation attempts on-screen
    python generate_dataset.py --config /path/to/config.json --render

    # render all data generation attempts to a video
    python generate_dataset.py --config /path/to/config.json --video_path /path/to/video.mp4

    # run a quick debug run
    python generate_dataset.py --config /path/to/config.json --debug

    # pause after every subtask to debug data generation
    python generate_dataset.py --config /path/to/config.json --render --pause_subtask
"""

import os
import shutil
import json
import time
import argparse
import traceback
import random
import sys
import imageio
import numpy as np
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

SIM_ROOT = Path(__file__).resolve().parents[3]
for package_root in (SIM_ROOT / "mimicgen", SIM_ROOT / "robosuite"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

import robomimic
from robomimic.utils.file_utils import get_env_metadata_from_dataset
import h5py
from robosuite.utils import RandomizationError
from robosuite.utils import transform_utils as T
from robosuite.utils.placement_samplers import ObjectPositionSampler, UniformRandomSampler

import mimicgen
import mimicgen.utils.file_utils as MG_FileUtils
import mimicgen.utils.robomimic_utils as RobomimicUtils

from mimicgen.configs import config_factory, MG_TaskSpec
from mimicgen.datagen.data_generator import DataGenerator
from mimicgen.env_interfaces.base import make_interface


class AnnulusRandomSampler(ObjectPositionSampler):
    """
    Area-uniform annulus sampler used by the exp0 Lift source / target datasets.
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
        inner_radius=0.0,
        outer_radius=0.10,
        center_xy=(0.0, 0.0),
        rotation=None,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.0,
    ):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.center_xy = np.asarray(center_xy, dtype=float)
        self.rotation = rotation
        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
        )

    def _sample_xy(self):
        lower = self.inner_radius ** 2
        if self.inner_radius > 0.0:
            lower = np.nextafter(lower, np.inf)
        radius = np.sqrt(np.random.uniform(lower, self.outer_radius ** 2))
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        xy = self.center_xy + radius * np.array([np.cos(theta), np.sin(theta)])
        return xy[0], xy[1]

    def _sample_quat(self):
        if self.rotation is None:
            rot_angle = np.random.uniform(0.0, 2.0 * np.pi)
        elif np.iterable(self.rotation):
            rot_angle = np.random.uniform(min(self.rotation), max(self.rotation))
        else:
            rot_angle = self.rotation
        return np.array([np.cos(rot_angle / 2.0), 0.0, 0.0, np.sin(rot_angle / 2.0)])

    def sample(self, fixtures=None, reference=None, on_top=True):
        placed_objects = {} if fixtures is None else dict(fixtures)
        if reference is None:
            base_offset = np.asarray(self.reference_pos, dtype=float)
        elif isinstance(reference, str):
            assert reference in placed_objects
            ref_pos, _, ref_obj = placed_objects[reference]
            base_offset = np.asarray(ref_pos, dtype=float)
            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.asarray(reference, dtype=float)

        for obj in self.mujoco_objects:
            assert obj.name not in placed_objects, "Object '{}' has already been sampled!".format(obj.name)
            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False

            for _ in range(5000):
                dx, dy = self._sample_xy()
                object_x = base_offset[0] + dx
                object_y = base_offset[1] + dy
                object_z = self.z_offset + base_offset[2]
                if on_top:
                    object_z -= bottom_offset[-1]

                location_valid = True
                if self.ensure_valid_placement:
                    for (x, y, z), _, other_obj in placed_objects.values():
                        if (
                            np.linalg.norm((object_x - x, object_y - y))
                            <= other_obj.horizontal_radius + horizontal_radius
                        ) and (object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]):
                            location_valid = False
                            break

                if location_valid:
                    quat = self._sample_quat()
                    if hasattr(obj, "init_quat"):
                        quat = T.quat_multiply(quat, obj.init_quat)
                    placed_objects[obj.name] = ((object_x, object_y, object_z), quat, obj)
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects")

        return placed_objects


def infer_lift_table_height_from_dataset(dataset_path):
    """
    Infer robosuite Lift table-top z from the first demo's saved model XML.
    """
    try:
        with h5py.File(dataset_path, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda key: int(key.rsplit("_", 1)[1]))
            model_xml = f["data/{}".format(demo_keys[0])].attrs.get("model_file", None)
    except Exception:
        return None

    if model_xml is None:
        return None

    try:
        root = ET.fromstring(model_xml)
    except ET.ParseError:
        return None

    for body in root.iter("body"):
        if body.attrib.get("name") != "table":
            continue
        body_pos = [float(v) for v in body.attrib.get("pos", "0 0 0").split()]
        for site in body.iter("site"):
            if site.attrib.get("name") == "table_top":
                site_pos = [float(v) for v in site.attrib.get("pos", "0 0 0").split()]
                return body_pos[2] + site_pos[2]
    return None


def get_exp0_dataset_meta(dataset_path):
    """
    Return exp0-specific metadata stored alongside robomimic env_args.
    """
    try:
        with h5py.File(dataset_path, "r") as f:
            env_args = f["data"].attrs.get("env_args", None)
    except Exception:
        return {}
    if env_args is None:
        return {}
    try:
        return json.loads(env_args)
    except json.JSONDecodeError:
        return {}


def patch_lift_table_height(table_height):
    """
    Match a source robosuite Lift dataset that was collected with a monkey-patched table height.
    """
    if table_height is None:
        return

    import robosuite.environments.manipulation.lift as lift_env

    original_load_model = lift_env.Lift._load_model

    def load_model_with_table_height(self):
        self.table_offset = np.array((0.0, 0.0, table_height))
        return original_load_model(self)

    lift_env.Lift._load_model = load_model_with_table_height


def restore_exp0_lift_generation_behavior(env, dataset_meta):
    """
    Restore non-JSON robosuite behavior that exp0 records as metadata.

    The original source / target demos use a custom cube placement sampler and,
    for Piper, a reset-time base-yaw prealignment toward the sampled cube. Those
    Python objects cannot be serialized in robomimic env_args, so recreate them
    here before MimicGen starts sampling new task instances.
    """
    if not dataset_meta:
        return

    base_env = env.base_env
    placement_meta = dataset_meta.get("cube_placement")
    if placement_meta is not None and hasattr(base_env, "cube"):
        table_height = float(base_env.model.mujoco_arena.table_offset[2])
        placement_type = placement_meta.get("type")
        if placement_type == "area_uniform_annulus":
            base_env.placement_initializer = AnnulusRandomSampler(
                name="ObjectSampler",
                mujoco_objects=base_env.cube,
                inner_radius=float(placement_meta["inner_radius"]),
                outer_radius=float(placement_meta["outer_radius"]),
                center_xy=placement_meta["center_xy"],
                rotation=placement_meta.get("rotation", 0.0),
                ensure_valid_placement=True,
                reference_pos=(0.0, 0.0, table_height),
                z_offset=0.01,
            )
            print("Restored exp0 annulus cube placement: {}".format(placement_meta))
        elif placement_type == "uniform_box":
            base_env.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=base_env.cube,
                x_range=placement_meta["x_range"],
                y_range=placement_meta["y_range"],
                rotation=placement_meta.get("rotation", 0.0),
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=(0.0, 0.0, table_height),
                z_offset=0.01,
            )
            print("Restored exp0 box cube placement: {}".format(placement_meta))

    scripted_reset = dataset_meta.get("scripted_reset", {})
    if not scripted_reset.get("prealign_base_yaw", False):
        return

    original_reset = env.reset
    yaw_limit = float(scripted_reset.get("prealign_yaw_limit", 1.30))

    def reset_with_prealign():
        original_reset()
        robot = base_env.robots[0]
        if not hasattr(robot, "set_robot_joint_positions"):
            return env.get_observation()

        joint_positions = np.asarray(robot._joint_positions).copy()
        if joint_positions.size == 0:
            return env.get_observation()

        cube_pos = np.asarray(base_env.sim.data.body_xpos[base_env.cube_body_id])
        base_pos = np.asarray(robot.base_pos)
        yaw = np.arctan2(cube_pos[1] - base_pos[1], cube_pos[0] - base_pos[0])
        yaw = np.clip(yaw, -yaw_limit, yaw_limit)

        joint_positions[0] = yaw
        robot.set_robot_joint_positions(joint_positions)
        if hasattr(robot, "controller") and hasattr(robot.controller, "update_initial_joints"):
            robot.controller.update_initial_joints(joint_positions)
        base_env.sim.forward()
        return env.get_observation()

    env.reset = reset_with_prealign
    print("Restored exp0 Piper reset prealignment with yaw limit {}".format(yaw_limit))


def get_important_stats(
    new_dataset_folder_path,
    num_success,
    num_failures,
    num_attempts,
    num_problematic,
    start_time=None,
    ep_length_stats=None,
    dataset_path=None,
):
    """
    Return a summary of important stats to write to json.

    Args:
        new_dataset_folder_path (str): path to folder that will contain generated dataset
        num_success (int): number of successful trajectories generated
        num_failures (int): number of failed trajectories
        num_attempts (int): number of total attempts
        num_problematic (int): number of problematic trajectories that failed due
            to a specific exception that was caught
        start_time (float or None): starting time for this run from time.time()
        ep_length_stats (dict or None): if provided, should have entries that summarize
            the episode length statistics over the successfully generated trajectories

    Returns:
        important_stats (dict): dictionary with useful summary of statistics
    """
    important_stats = dict(
        generation_path=new_dataset_folder_path,
        success_rate=((100. * num_success) / num_attempts),
        failure_rate=((100. * num_failures) / num_attempts),
        num_success=num_success,
        num_failures=num_failures,
        num_attempts=num_attempts,
        num_problematic=num_problematic,
    )
    if dataset_path is not None:
        important_stats["dataset_path"] = dataset_path
    if (ep_length_stats is not None):
        important_stats.update(ep_length_stats)
    if start_time is not None:
        # add in time taken
        important_stats["time spent (hrs)"] = "{:.2f}".format((time.time() - start_time) / 3600.)
    return important_stats


def generate_dataset(
    mg_config,
    auto_remove_exp=False,
    render=False,
    video_path=None,
    video_skip=5,
    render_image_names=None,
    pause_subtask=False,
    output_path=None,
):
    """
    Main function to collect a new dataset with MimicGen.

    Args:
        mg_config (MG_Config instance): MimicGen config object

        auto_remove_exp (bool): if True, will remove generation folder if it exists, else
            user will be prompted to decide whether to keep existing folder or not

        render (bool): if True, render each data generation attempt on-screen

        video_path (str or None): if provided, render the data generation attempts to the 
            provided video path

        video_skip (int): skip every nth frame when writing video

        render_image_names (list of str or None): if provided, specify camera names to 
            use during on-screen / off-screen rendering to override defaults

        pause_subtask (bool): if True, pause after every subtask during generation, for
            debugging.
    """

    # time this run
    start_time = time.time()

    # check some args
    write_video = (video_path is not None)
    assert not (render and write_video) # either on-screen or video but not both
    if pause_subtask:
        assert render, "should enable on-screen rendering for pausing to be useful"

    if write_video:
        # debug video - use same cameras as observations
        if len(mg_config.obs.camera_names) > 0:
            assert render_image_names is None
            render_image_names = list(mg_config.obs.camera_names)

    # path to source dataset
    source_dataset_path = os.path.expandvars(os.path.expanduser(mg_config.experiment.source.dataset_path))

    # get environment metadata from dataset
    env_meta = get_env_metadata_from_dataset(dataset_path=source_dataset_path)
    dataset_meta = get_exp0_dataset_meta(source_dataset_path)

    if env_meta.get("env_name") == "Lift":
        table_height = infer_lift_table_height_from_dataset(source_dataset_path)
        if table_height is not None:
            print("Using Lift table height inferred from source dataset: {}".format(table_height))
            patch_lift_table_height(table_height)

    # set seed for generation
    random.seed(mg_config.experiment.seed)
    np.random.seed(mg_config.experiment.seed)

    # create new folder for this data generation run
    base_folder = os.path.expandvars(os.path.expanduser(mg_config.experiment.generation.path))
    new_dataset_folder_name = mg_config.experiment.name
    new_dataset_folder_path = os.path.join(
        base_folder,
        new_dataset_folder_name,
    )
    print("\nData will be generated at: {}".format(new_dataset_folder_path))

    # ensure dataset folder does not exist, and make new folder
    exist_ok = False
    if os.path.exists(new_dataset_folder_path):
        if not auto_remove_exp:
            ans = input("\nWARNING: dataset folder ({}) already exists! \noverwrite? (y/n)\n".format(new_dataset_folder_path))
        else:
            ans = "y"
        if ans == "y":
            print("Removed old results folder at {}".format(new_dataset_folder_path))
            shutil.rmtree(new_dataset_folder_path)
        else:
            print("Keeping old dataset folder. Note that individual files may still be overwritten.")
            exist_ok = True
    os.makedirs(new_dataset_folder_path, exist_ok=exist_ok)

    if output_path is not None:
        output_path = os.path.abspath(os.path.expandvars(os.path.expanduser(output_path)))
        output_parent = os.path.dirname(output_path)
        if len(output_parent) > 0:
            os.makedirs(output_parent, exist_ok=True)
        if os.path.exists(output_path):
            if not auto_remove_exp:
                ans = input("\nWARNING: output dataset ({}) already exists! \noverwrite? (y/n)\n".format(output_path))
            else:
                ans = "y"
            if ans == "y":
                print("Removed old output dataset at {}".format(output_path))
                os.remove(output_path)
            else:
                raise FileExistsError("Refusing to overwrite output dataset {}".format(output_path))

    # log terminal output to text file
    RobomimicUtils.make_print_logger(txt_file=os.path.join(new_dataset_folder_path, 'log.txt'))

    # save config to disk
    MG_FileUtils.write_json(
        json_dic=mg_config,
        json_path=os.path.join(new_dataset_folder_path, "mg_config.json"),
    )

    print("\n============= Config =============")
    print(mg_config)
    print("")

    # some paths that we will create inside our new dataset folder

    # new dataset that will be generated
    new_dataset_path = output_path if output_path is not None else os.path.join(new_dataset_folder_path, "demo.hdf5")
    if output_path is not None:
        print("Successful trajectories will be written to: {}".format(new_dataset_path))

    # tmp folder that will contain per-episode hdf5s that were successful (they will be merged later)
    tmp_dataset_folder_path = os.path.join(new_dataset_folder_path, "tmp")
    os.makedirs(tmp_dataset_folder_path, exist_ok=exist_ok)

    # folder containing logs
    json_log_path = os.path.join(new_dataset_folder_path, "logs")
    os.makedirs(json_log_path, exist_ok=exist_ok)

    if mg_config.experiment.generation.keep_failed:
        # new dataset for failed trajectories, and tmp folder for per-episode hdf5s that failed
        new_failed_dataset_path = os.path.join(new_dataset_folder_path, "demo_failed.hdf5")
        tmp_dataset_failed_folder_path = os.path.join(new_dataset_folder_path, "tmp_failed")
        os.makedirs(tmp_dataset_failed_folder_path, exist_ok=exist_ok)

    # get list of source demonstration keys from source hdf5
    all_demos = MG_FileUtils.get_all_demos_from_dataset(
        dataset_path=source_dataset_path,
        filter_key=mg_config.experiment.source.filter_key,
        start=mg_config.experiment.source.start,
        n=mg_config.experiment.source.n,
    )

    # prepare args for creating simulation environment

    # auto-fill camera rendering info if not specified
    if (write_video or render) and (render_image_names is None):
        render_image_names = RobomimicUtils.get_default_env_cameras(env_meta=env_meta)
    if render:
        # on-screen rendering can only support one camera
        assert len(render_image_names) == 1

    # env args: cameras to use come from debug camera video to write, or from observation collection
    camera_names = (mg_config.obs.camera_names if not write_video else render_image_names)

    # env args: don't use image obs when writing debug video
    use_image_obs = ((mg_config.obs.collect_obs and (len(mg_config.obs.camera_names) > 0)) if not write_video else False)
    use_depth_obs = False
    
    # simulation environment
    env = RobomimicUtils.create_env(
        env_meta=env_meta,
        env_class=None,
        env_name=mg_config.experiment.task.name,
        robot=mg_config.experiment.task.robot,
        gripper=mg_config.experiment.task.gripper,
        env_meta_update_kwargs=mg_config.experiment.task.env_meta_update_kwargs,
        camera_names=camera_names,
        camera_height=mg_config.obs.camera_height,
        camera_width=mg_config.obs.camera_width,
        render=render, 
        render_offscreen=write_video,
        use_image_obs=use_image_obs,
        use_depth_obs=use_depth_obs,
    )
    print("\n==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")
    restore_exp0_lift_generation_behavior(env=env, dataset_meta=dataset_meta)

    # get information necessary to create env interface
    env_interface_name, env_interface_type = MG_FileUtils.get_env_interface_info_from_dataset(
        dataset_path=source_dataset_path,
        demo_keys=all_demos,
    )
    # possibly override from config
    if mg_config.experiment.task.interface is not None:
        env_interface_name = mg_config.experiment.task.interface
    if mg_config.experiment.task.interface_type is not None:
        env_interface_type = mg_config.experiment.task.interface_type

    # create environment interface to use during data generation
    env_interface = make_interface(
        name=env_interface_name,
        interface_type=env_interface_type,
        # NOTE: env_interface takes underlying simulation environment, not robomimic wrapper
        env=env.base_env,
    )
    print("Created environment interface: {}".format(env_interface))

    # make sure we except the same exceptions that we would normally except during policy rollouts
    exceptions_to_except = env.rollout_exceptions

    # get task spec object from config
    task_spec_json_string = mg_config.task.task_spec.dump()
    task_spec = MG_TaskSpec.from_json(json_string=task_spec_json_string)

    # make data generator object
    data_generator = DataGenerator(
        task_spec=task_spec,
        dataset_path=source_dataset_path,
        demo_keys=all_demos,
    )

    print("\n==== Created Data Generator ====")
    print(data_generator)
    print("")

    # we might write a video to show the data generation attempts
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(video_path, fps=20)

    # data generation statistics
    num_success = 0
    num_failures = 0
    num_attempts = 0
    num_problematic = 0
    ep_lengths = [] # episode lengths for successfully generated data
    selected_src_demo_inds_all = [] # selected source demo index in @all_demos for each trial
    selected_src_demo_inds_succ = [] # selected source demo index in @all_demos for each successful trial

    # we will keep generating data until @num_trials successes (if @guarantee_success) else @num_trials attempts
    num_trials = mg_config.experiment.generation.num_trials
    guarantee_success = mg_config.experiment.generation.guarantee

    while True:

        # generate trajectory
        try:
            generated_traj = data_generator.generate(
                env=env,
                env_interface=env_interface,
                select_src_per_subtask=mg_config.experiment.generation.select_src_per_subtask,
                transform_first_robot_pose=mg_config.experiment.generation.transform_first_robot_pose,
                interpolate_from_last_target_pose=mg_config.experiment.generation.interpolate_from_last_target_pose,
                render=render,
                video_writer=video_writer,
                video_skip=video_skip,
                camera_names=render_image_names,
                pause_subtask=pause_subtask,
            )
        except exceptions_to_except as e:
            # problematic trajectory - do not have this count towards our total number of attempts, and re-try
            print("")
            print("*" * 50)
            print("WARNING: got rollout exception {}".format(e))
            print("*" * 50)
            print("")
            num_problematic += 1
            continue

        # remember selection of source demos for each subtask
        selected_src_demo_inds_all.append(generated_traj["src_demo_inds"])

        # check if generated trajectory was successful
        success = bool(generated_traj["success"])

        if success:
            num_success += 1

            # store successful demonstration
            ep_lengths.append(generated_traj["actions"].shape[0])
            MG_FileUtils.write_demo_to_hdf5(
                folder=tmp_dataset_folder_path,
                env=env,
                initial_state=generated_traj["initial_state"],
                states=generated_traj["states"],
                observations=(generated_traj["observations"] if mg_config.obs.collect_obs else None),
                datagen_info=generated_traj["datagen_infos"],
                actions=generated_traj["actions"],
                src_demo_inds=generated_traj["src_demo_inds"],
                src_demo_labels=generated_traj["src_demo_labels"],
            )
            selected_src_demo_inds_succ.append(generated_traj["src_demo_inds"])
        else:
            num_failures += 1

            # check if this failure should be kept
            if mg_config.experiment.generation.keep_failed and \
                ((mg_config.experiment.max_num_failures is None) or (num_failures <= mg_config.experiment.max_num_failures)):
                
                # save failed trajectory in separate folder
                MG_FileUtils.write_demo_to_hdf5(
                    folder=tmp_dataset_failed_folder_path,
                    env=env,
                    initial_state=generated_traj["initial_state"],
                    states=generated_traj["states"],
                    observations=(generated_traj["observations"] if mg_config.obs.collect_obs else None),
                    datagen_info=generated_traj["datagen_infos"],
                    actions=generated_traj["actions"],
                    src_demo_inds=generated_traj["src_demo_inds"],
                    src_demo_labels=generated_traj["src_demo_labels"],
                )

        num_attempts += 1
        print("")
        print("*" * 50)
        print("trial {} success: {}".format(num_attempts, success))
        print("have {} successes out of {} trials so far".format(num_success, num_attempts))
        print("have {} failures out of {} trials so far".format(num_failures, num_attempts))
        print("*" * 50)

        # regularly log progress to disk every so often
        if (num_attempts % mg_config.experiment.log_every_n_attempts) == 0:

            # get summary stats
            summary_stats = get_important_stats(
                new_dataset_folder_path=new_dataset_folder_path,
                num_success=num_success,
                num_failures=num_failures,
                num_attempts=num_attempts,
                num_problematic=num_problematic,
                start_time=start_time,
                ep_length_stats=None,
                dataset_path=new_dataset_path,
            )

            # write stats to disk
            max_digits = len(str(num_trials * 1000)) + 1 # assume we will never have lower than 0.1% data generation SR
            json_file_path = os.path.join(json_log_path, "attempt_{}_succ_{}_rate_{}.json".format(
                str(num_attempts).zfill(max_digits), # pad with leading zeros for ordered list of jsons in directory
                num_success,
                np.round((100. * num_success) / num_attempts, 2),
            ))
            MG_FileUtils.write_json(json_dic=summary_stats, json_path=json_file_path)

        # termination condition is on enough successes if @guarantee_success or enough attempts otherwise
        check_val = num_success if guarantee_success else num_attempts
        if check_val >= num_trials:
            break

    if write_video:
        video_writer.close()

    # merge all new created files
    print("\nFinished data generation. Merging per-episode hdf5s together...\n")
    MG_FileUtils.merge_all_hdf5(
        folder=tmp_dataset_folder_path,
        new_hdf5_path=new_dataset_path,
        delete_folder=True,
    )
    if mg_config.experiment.generation.keep_failed:
        MG_FileUtils.merge_all_hdf5(
            folder=tmp_dataset_failed_folder_path,
            new_hdf5_path=new_failed_dataset_path,
            delete_folder=True,
        )

    # get episode length statistics
    ep_length_stats = None
    if len(ep_lengths) > 0:
        ep_lengths = np.array(ep_lengths)
        ep_length_mean = float(np.mean(ep_lengths))
        ep_length_std = float(np.std(ep_lengths))
        ep_length_max = int(np.max(ep_lengths))
        ep_length_3std = int(np.ceil(ep_length_mean + 3. * ep_length_std))
        ep_length_stats = dict(
            ep_length_mean=ep_length_mean,
            ep_length_std=ep_length_std,
            ep_length_max=ep_length_max,
            ep_length_3std=ep_length_3std,
        )

    stats = get_important_stats(
        new_dataset_folder_path=new_dataset_folder_path,
        num_success=num_success,
        num_failures=num_failures,
        num_attempts=num_attempts,
        num_problematic=num_problematic,
        start_time=start_time,
        ep_length_stats=ep_length_stats,
        dataset_path=new_dataset_path,
    )
    print("\nStats Summary")
    print(json.dumps(stats, indent=4))

    # maybe render videos
    if mg_config.experiment.render_video:
        if (num_success > 0):
            playback_video_path = os.path.join(new_dataset_folder_path, "playback_{}.mp4".format(new_dataset_folder_name))
            num_render = mg_config.experiment.num_demo_to_render
            print("Rendering successful trajectories...")
            RobomimicUtils.make_dataset_video(
                dataset_path=new_dataset_path,
                video_path=playback_video_path,
                num_render=num_render,
            )
        else:
            print("\n" + "*" * 80)
            print("\nWARNING: skipping dataset video creation since no successes")
            print("\n" + "*" * 80 + "\n")
        if mg_config.experiment.generation.keep_failed:
            if (num_failures > 0):
                playback_video_path = os.path.join(new_dataset_folder_path, "playback_{}_failed.mp4".format(new_dataset_folder_name))
                num_render = mg_config.experiment.num_fail_demo_to_render
                print("Rendering failure trajectories...")
                RobomimicUtils.make_dataset_video(
                    dataset_path=new_failed_dataset_path,
                    video_path=playback_video_path,
                    num_render=num_render,
                )
            else:
                print("\n" + "*" * 80)
                print("\nWARNING: skipping dataset video creation since no failures")
                print("\n" + "*" * 80 + "\n")

    # return some summary info
    final_important_stats = get_important_stats(
        new_dataset_folder_path=new_dataset_folder_path,
        num_success=num_success,
        num_failures=num_failures,
        num_attempts=num_attempts,
        num_problematic=num_problematic,
        start_time=start_time,
        ep_length_stats=ep_length_stats,
        dataset_path=new_dataset_path,
    )

    # write stats to disk
    json_file_path = os.path.join(new_dataset_folder_path, "important_stats.json")
    MG_FileUtils.write_json(json_dic=final_important_stats, json_path=json_file_path)

    # NOTE: we are not currently saving the choice of source human demonstrations for each trial,
    #       but you can do that if you wish -- the information is stored in @selected_src_demo_inds_all
    #       and @selected_src_demo_inds_succ

    return final_important_stats


def main(args):

    # load config object
    with open(args.config, "r") as f:
        ext_cfg = json.load(f)
        # config generator from robomimic generates this part of config unused by MimicGen
        if "meta" in ext_cfg:
            del ext_cfg["meta"]
    mg_config = config_factory(ext_cfg["name"], config_type=ext_cfg["type"])

    # update config with external json - this will throw errors if
    # the external config has keys not present in the base config
    with mg_config.values_unlocked():
        mg_config.update(ext_cfg)

        # We assume that the external config specifies all subtasks, so
        # delete any subtasks not in the external config.
        source_subtasks = set(mg_config.task.task_spec.keys())
        new_subtasks = set(ext_cfg["task"]["task_spec"].keys())
        for subtask in (source_subtasks - new_subtasks):
            print("deleting subtask {} in original config".format(subtask))
            del mg_config.task.task_spec[subtask]

        # maybe override some settings
        if args.task_name is not None:
            mg_config.experiment.task.name = args.task_name

        if args.source is not None:
            mg_config.experiment.source.dataset_path = args.source

        if args.source_filter_key is not None:
            mg_config.experiment.source.filter_key = args.source_filter_key

        if args.source_n is not None:
            mg_config.experiment.source.n = args.source_n

        if args.source_start is not None:
            mg_config.experiment.source.start = args.source_start

        if args.folder is not None:
            mg_config.experiment.generation.path = args.folder

        if args.num_demos is not None:
            mg_config.experiment.generation.num_trials = args.num_demos

        if args.seed is not None:
            mg_config.experiment.seed = args.seed

        # maybe modify config for debugging purposes
        if args.debug:
            # shrink length of generation to test whether this run is likely to crash
            mg_config.experiment.source.n = 3
            mg_config.experiment.generation.guarantee = False
            mg_config.experiment.generation.num_trials = 2

            # send output to a temporary directory
            mg_config.experiment.generation.path = "/tmp/tmp_mimicgen"

    # catch error during generation and print it
    res_str = "finished run successfully!"
    important_stats = None
    try:
        important_stats = generate_dataset(
            mg_config=mg_config,
            auto_remove_exp=args.auto_remove_exp,
            render=args.render,
            video_path=args.video_path,
            video_skip=args.video_skip,
            render_image_names=args.render_image_names,
            pause_subtask=args.pause_subtask,
            output_path=args.output,
        )
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)
    if important_stats is not None:
        important_stats = json.dumps(important_stats, indent=4)
        print("\nFinal Data Generation Stats")
        print(important_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to MimicGen config json",
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick generation run for debugging purposes",
    )
    parser.add_argument(
        "--auto-remove-exp",
        action='store_true',
        help="force delete the experiment folder if it exists"
    )
    parser.add_argument(
        "--render",
        action='store_true',
        help="render each data generation attempt on-screen",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="if provided, render the data generation attempts to the provided video path",
    )
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="skip every nth frame when writing video",
    )
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )
    parser.add_argument(
        "--pause_subtask",
        action='store_true',
        help="pause after every subtask during generation for debugging - only useful with render flag",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="path to source dataset, to override the one in the config",
    )
    parser.add_argument(
        "--source_filter_key",
        type=str,
        help="optional source dataset mask key, for example 'successful'",
        default=None,
    )
    parser.add_argument(
        "--source_n",
        type=int,
        help="if provided, use only the first N source trajectories after filtering / start offset",
        default=None,
    )
    parser.add_argument(
        "--source_start",
        type=int,
        help="if provided, skip the first N source trajectories after filtering",
        default=None,
    )
    parser.add_argument(
        "--task_name",
        type=str,
        help="environment name to use for data generation, to override the one in the config",
        default=None,
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="folder that will be created with new data, to override the one in the config",
        default=None,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="optional exact path for the generated successful demo hdf5; logs still go under --folder / experiment name",
        default=None,
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        help="number of demos to generate, or attempt to generate, to override the one in the config",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed, to override the one in the config",
        default=None,
    )

    args = parser.parse_args()
    main(args)
