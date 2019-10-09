import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptIS(nn.Module):
    def __init__(self,
                 backbone,
                 adaptis_head,
                 segmentation_head=None,
                 proposal_head=None,
                 with_proposals=False,
                 spatial_scale=1.0):
        super(AdaptIS, self).__init__()

        self.with_proposals = with_proposals
        self.spatial_scale = spatial_scale

        self.backbone = backbone
        self.adaptis_head = adaptis_head
        self.segmentation_head = segmentation_head
        self.proposals_head = [proposal_head]
        if with_proposals:
            self.add_proposals_head()

    def add_proposals_head(self):
        self.with_proposals = True

        for param in self.parameters():
            param.requires_grad = False
        self.proposals_head = self.proposals_head[0]

    @staticmethod
    def make_named_outputs(outputs):
        keys, values = list(zip(*outputs))
        named_outputs = namedtuple('outputs', keys)(*values)

        return named_outputs

    def forward(self, x, points):
        orig_size = x.size()[2:]
        outputs = []
        backbone_features = self.backbone(x)

        # instances
        instance_out = self.adaptis_head(backbone_features, points)
        if not math.isclose(self.spatial_scale, 1.0):
            instance_out = F.interpolate(instance_out, orig_size, mode='bilinear', align_corners=True)
        outputs.append(('instances', instance_out))

        # semantic
        if self.segmentation_head is not None:
            semantic_out = self.segmentation_head(backbone_features)
            if not math.isclose(self.spatial_scale, 1.0):
                semantic_out = F.interpolate(semantic_out, orig_size, mode='bilinear', align_corners=True)
            outputs.append(('semantic', semantic_out))

        # proposals
        if self.with_proposals:
            backbone_features = backbone_features.detach()
            proposals_out = self.proposals_head(backbone_features)
            proposals_out = self.adaptis_head.EQF(proposals_out, points.detach())
            outputs.append(('proposals', proposals_out))

        return self.make_named_outputs(outputs)

    def load_weights(self, path_to_weights):
        current_state_dict = self.state_dict()
        new_state_dict = torch.load(path_to_weights)
        current_state_dict.update(new_state_dict)
        self.load_state_dict(current_state_dict)
        
    def get_trainable_params(self):
        trainable_params = []

        for param in self.parameters():
            if param.requires_grad:
                trainable_params.append(param)

        return trainable_params
