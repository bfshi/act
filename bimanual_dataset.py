import numpy as np
import torch
from mvp.bimanual_bc.dataset import Bimanual_Dataset

class BimanualDataset:
    def __init__(self):
        self.dataset = Bimanual_Dataset(
            features=False,
            demo_root="/home/bfshi/data/bimanual/",
            demo_dirs=["pick-yellow-right_04-07-2024"],
            inmem=True,
            start_ind=0,
            num_demos=120,
            num_steps=17,
            num_pred=16,
            look_ahead=0,
            im_size=112,
            cams=["left", "head", "right"],
            noisy_skip=0,
            frame_skip=1,
            default_pos_left_arm=[-4.065179173146383, -0.8556114000133057, 1.419995133076803, -3.108495374719137, -1.3419583479510706, 0,],
            default_pos_right_arm=[-2.198981587086813, -2.2018891773619593, -1.534730076789856, -0.1098826688579102, 1.2620022296905518, 0,],
            joint_noise_mean=[0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.],
            joint_noise_std=[0.0213, 0.0246, 0.0211, 0.0378, 0.0382, 0.0255,
                    0.0273, 0.0373, 0.0329, 0.0500, 0.0738, 0.0741,
                    0.0202, 0.0198, 0.0211, 0.0417, 0.0396, 0.0308,
                    0.0795, 0.0652, 0.0638, 0.0567, 0.0532, 0.0575],
            joint_noise_std_scale=1.0,
            feats_noise_std=0.0,
            data_filter={},
            history_repeating=0,
            img_sample_num=-1,
            use_all_features=False,
            action_data_ratio=None,
            use_touch=False,
            skip_failure=True,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ims, pi_obs, _, pi_act = self.dataset[index][:4]

        image_data = torch.stack([x[0] for x in ims], dim=0)
        action_data = pi_act[:16]
        qpos_data = pi_obs[0]
        is_pad = torch.zeros(16).bool()

        return image_data, qpos_data, action_data, is_pad