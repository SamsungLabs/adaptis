import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms


class AdaptISPrediction(object):
    def __init__(self, net, dataset, device,
                 input_transforms=None):

        if input_transforms is None:
            input_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        self.input_transforms = input_transforms

        self.net = net.eval().to(device)
        self.device = device
        self.things_offset = dataset.things_offset
        self.to_dataset_mapping = dataset.to_dataset_mapping
        self.to_dataset_mapping_list = [-1] * (max(self.to_dataset_mapping.keys()) + 1)
        for k, v in self.to_dataset_mapping.items():
            self.to_dataset_mapping_list[k] = v

    def load_parameters(self, weights_path):
        self.net.load_weights(weights_path)

    def get_features(self, image, use_flip=False):
        data = self.input_transforms(image).unsqueeze(0).to(self.device)
        features = self.net.backbone(data)

        if use_flip:
            features_flipped = self.net.backbone(torch.flip(data, dims=[3]))
            return [features, features_flipped]
        else:
            return [features]

    def get_point_invariant_states(self, features):
        states = [self.net.adaptis_head._get_point_invariant_features(f) for f in features]
        return states

    def get_instance_masks(self, features, points, states=None, width=None, height=None):
        points = np.array(points).reshape((1, -1, 2)).astype(np.float32)
        points = torch.from_numpy(points).to(self.device)

        if states is not None:
            out = self.net.adaptis_head._get_instance_maps(points, *states)
        else:
            out = self.net.adaptis_head(features, points)
        out = torch.sigmoid(out)

        if (height != out.shape[2] or width != out.shape[3]) and width is not None:
            out = F.interpolate(out, (height, width), mode='bilinear', align_corners=True)

        return out.squeeze(1).detach().cpu().numpy()

    def get_semantic_segmentation(self, features, use_flip=False, output_height=None, output_width=None):
        smap = self._get_semantic_segmentation(features[0], height=output_height, width=output_width)

        if use_flip:
            smap_flipped = self._get_semantic_segmentation(features[1], height=output_height, width=output_width)
            smap = 0.5 * (smap + smap_flipped[:, :, :, ::-1])

        instances_prob = smap[0, self.things_offset:, :, :]
        stuff_prob = smap[0, :self.things_offset, :, :]

        return instances_prob, stuff_prob

    def _get_semantic_segmentation(self, features, width=None, height=None):
        out = self.net.segmentation_head(features)
        out = torch.nn.functional.softmax(out, dim=1)

        if (height != out.shape[2] or width != out.shape[3]) and width is not None:
            out = F.interpolate(out, (height, width), mode='bilinear', align_corners=True)

        return out.detach().cpu().numpy()

    def get_proposals_map(self, features, width=None, height=None, power=2):
        out = self.net.proposals_head(features)

        if (height != out.shape[2] or width != out.shape[3]) and width is not None:
            out = F.interpolate(out, (height, width), mode='bilinear', align_corners=True)

        result = torch.sigmoid(out).detach().cpu().numpy()
        critic_map = result ** power
        return critic_map
