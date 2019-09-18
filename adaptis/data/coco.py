import cv2
import json
import numpy as np
from pathlib import Path
from .base import AdaptISDataset


class CocoDataset(AdaptISDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super(CocoDataset, self).__init__(**kwargs)
        self.split = split
        self.dataset_path = Path(dataset_path)

        self.load_samples()

    def load_samples(self):
        annotation_path = self.dataset_path / 'annotations' / f'panoptic_{self.split}.json'
        self.labels_path = self.dataset_path / 'annotations' / f'panoptic_{self.split}'
        self.images_path = self.dataset_path / self.split

        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        self.dataset_samples = annotation['annotations']

        self._categories = annotation['categories']
        self._stuff_labels = [x['id'] for x in self._categories if x['isthing'] == 0]
        self._things_labels = [x['id'] for x in self._categories if x['isthing'] == 1]
        self._things_labels_set = set(self._things_labels)

    def get_sample(self, index, coco_format=False):
        dataset_sample = self.dataset_samples[index]

        image_path = self.images_path / self.get_image_name(dataset_sample['file_name'])
        label_path = self.labels_path / dataset_sample['file_name']

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED).astype(np.int32)
        label = 256 * 256 * label[:, :, 0] + 256 * label[:, :, 1] + label[:, :, 2]

        if coco_format:
            return {
                'image': image,
                'annotation': label,
                'segments_info': dataset_sample['segments_info']
            }

        label_map = np.full_like(label, -1)
        for segment in dataset_sample['segments_info']:
            label_map[label == segment['id']] = self.from_dataset_mapping[segment['category_id']]

        instance_map = np.full_like(label, 0)
        instances_info = dict()
        for segment in dataset_sample['segments_info']:
            class_id = segment['category_id']
            if class_id not in self._things_labels_set:
                continue

            mapped_class_id = self.from_dataset_mapping[class_id]
            obj_id = segment['id']
            instance_map[label == obj_id] = obj_id

            ignore = segment['iscrowd'] == 1
            instances_info[obj_id] = {
                'class_id': mapped_class_id, 'ignore': ignore
            }

        sample = {
            'image': image,
            'instances_mask': instance_map,
            'instances_info': instances_info,
            'semantic_segmentation': label_map
        }

        return sample

    @classmethod
    def get_image_name(cls, panoptic_name):
        return panoptic_name.replace('.png', '.jpg')

    @property
    def stuff_labels(self):
        return self._stuff_labels

    @property
    def things_labels(self):
        return self._things_labels
