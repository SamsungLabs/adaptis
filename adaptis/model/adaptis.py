import mxnet as mx
from adaptis.utils.block import NamedHybridBlock


class AdaptIS(NamedHybridBlock):
    def __init__(self,
                 feature_extractor,
                 adaptis_head,
                 segmentation_head=None,
                 proposal_head=None,
                 with_proposals=False,
                 spatial_scale=1.0):
        super(AdaptIS, self).__init__()

        self.with_proposals = with_proposals
        self.spatial_scale = spatial_scale
        self._image_shape = None
        object.__setattr__(self, '_proposals_head', proposal_head)

        with self.name_scope():
            self.feature_extractor = feature_extractor
            self.adaptis_head = adaptis_head
            self.segmentation_head = segmentation_head

            if with_proposals:
                self.add_proposals_head()
            else:
                self.proposals_head = None

    def hybrid_forward(self, F, x, points):
        if hasattr(x, 'shape'):
            self._image_shape = x.shape[2:]

        backbone_features = self.feature_extractor(x)
        instance_out = self.adaptis_head(backbone_features[0], points)
        if self.spatial_scale != 1.0:
            instance_out = F.contrib.BilinearResize2D(instance_out,
                                                      height=self._image_shape[0], width=self._image_shape[1])

        outputs = [('instances', instance_out)]
        if self.segmentation_head is not None:
            segmentation_out = self.segmentation_head(*backbone_features)
            if self.spatial_scale != 1.0:
                segmentation_out = F.contrib.BilinearResize2D(segmentation_out,
                                                              height=self._image_shape[0], width=self._image_shape[1])
            outputs.append(('semantic', segmentation_out))

        if self.proposals_head is not None:
            backbone_features = [x.detach() for x in backbone_features]
            proposals_out = self.proposals_head(*backbone_features)
            proposals_out = self.adaptis_head.eqf(proposals_out, points.detach())
            outputs.append(('proposals', proposals_out))
        return self.make_named_outputs(outputs)

    def add_proposals_head(self, ctx=None, initializer=mx.init.Xavier()):
        if getattr(self, 'proposals_head', None) is None:
            self.proposals_head = self._proposals_head
            self.with_proposals = True

            if ctx is not None:
                self.proposals_head.initialize(initializer, ctx=ctx)
