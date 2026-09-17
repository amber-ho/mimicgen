"""MimicGen variant of the reconstructed Piper peg-insertion task."""

import numpy as np

from robosuite.environments.manipulation.peg_insertion import PegInsertion

from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG


class PegInsertion_D0(PegInsertion, SingleArmEnv_MG):
    """Sample the receptacle over the support measured in the real dataset."""

    def __init__(self, **kwargs):
        kwargs.pop("receptacle_pose", None)
        PegInsertion.__init__(self, receptacle_pose=None, **kwargs)

    def edit_model_xml(self, xml_str):
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def _get_initial_placement_bounds(self):
        return dict(
            receptacle=dict(
                x=self.DEFAULT_RECEPTACLE_X_RANGE,
                y=self.DEFAULT_RECEPTACLE_Y_RANGE,
                z_rot=self.DEFAULT_RECEPTACLE_YAW_RANGE,
                reference=np.array((0.0, 0.0, self.table_offset[2])),
            )
        )
