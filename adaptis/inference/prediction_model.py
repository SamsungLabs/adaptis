import mxnet as mx
import numpy as np
from mxnet.gluon.data.vision import transforms


class AdaptISPrediction(object):
    def __init__(self, net, dataset,
                 input_transforms=None):

        if input_transforms is None:
            input_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        self.input_transforms = input_transforms

        self.net = net
        self.things_offset = dataset.things_offset
        self.to_dataset_mapping = dataset.to_dataset_mapping
        self.to_dataset_mapping_list = [-1] * (max(self.to_dataset_mapping.keys()) + 1)
        for k, v in self.to_dataset_mapping.items():
            self.to_dataset_mapping_list[k] = v

    def load_parameters(self, weights_path, ctx):
        self.ctx = ctx
        self.net.load_parameters(weights_path, ctx=ctx)

    def get_features(self, image, use_flip=False):
        data = mx.nd.reshape(self.input_transforms(mx.nd.array(image)),
                             shape=(-4, 1, -1, -2))
        data = mx.nd.array(data, ctx=self.ctx)
        features = self.net.feature_extractor(data)

        if use_flip:
            features_flipped = self.net.feature_extractor(data[:, :, :, ::-1])
            return [features, features_flipped]
        else:
            return [features]

    def get_point_invariant_states(self, features):
        states = [self.net.adaptis_head.get_point_invariant_states(mx.nd, f[0])
                  for f in features]
        return states

    def get_instance_masks(self, features, points, states=None, width=None, height=None):
        points = np.array(points).reshape((1, -1, 2)).astype(np.float32)
        points = mx.nd.array(points, ctx=self.ctx, dtype=np.float32)

        if states is not None:
            out = self.net.adaptis_head.get_instances_maps(mx.nd, points, *states)
        else:
            out = self.net.adaptis_head(features[0], points)
        out = mx.nd.sigmoid(out)

        if (height != out.shape[2] or width != out.shape[3]) and width is not None:
            out = mx.nd.contrib.BilinearResize2D(out, width=width, height=height)

        return np.squeeze(out.asnumpy(), axis=1)

    def get_semantic_segmentation(self, features, use_flip=False, output_height=None, output_width=None):
        smap = self._get_semantic_segmentation(features[0], height=output_height, width=output_width)

        if use_flip:
            smap_flipped = self._get_semantic_segmentation(features[1], height=output_height, width=output_width)
            smap = 0.5 * (smap + smap_flipped[:, :, :, ::-1])

        instances_prob = smap[0, self.things_offset:, :, :]
        stuff_prob = smap[0, :self.things_offset, :, :]

        return instances_prob, stuff_prob

    def _get_semantic_segmentation(self, features, width=None, height=None):
        out = self.net.segmentation_head(*features)
        out = mx.nd.softmax(out, axis=1)

        if (height != out.shape[2] or width != out.shape[3]) and width is not None:
            out = mx.nd.contrib.BilinearResize2D(out, width=width, height=height)

        return out.asnumpy()

    def get_proposals_map(self, features, width=None, height=None, power=2):
        out = self.net.proposals_head(*features)

        if (height != out.shape[2] or width != out.shape[3]) and width is not None:
            out = mx.nd.contrib.BilinearResize2D(out, width=width, height=height)

        result = mx.nd.sigmoid(out).asnumpy()
        critic_map = result ** power
        return critic_map
