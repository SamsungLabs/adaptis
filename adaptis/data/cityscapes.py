import cv2
import numpy as np
from pathlib import Path
from .base import AdaptISDataset, get_unique_labels


class CityscapesDataset(AdaptISDataset):
    def __init__(self, dataset_path, split='train', use_jpeg=False, **kwargs):
        super(CityscapesDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split

        images_path = self.dataset_path / ('leftImgJPG' if use_jpeg else 'leftImg8bit') / split
        gt_path = self.dataset_path / 'gtFine' / split

        images_mask = '*leftImg8bit.jpg' if use_jpeg else '*leftImg8bit.png'
        images_list = sorted(images_path.rglob(images_mask))

        self.dataset_samples = []
        for image_path in images_list:
            image_name = str(image_path.relative_to(images_path))
            instances_name = image_name.replace(images_mask[1:], 'gtFine_instanceIds.png')
            instances_path = str(gt_path / instances_name)

            semantic_name = image_name.replace(images_mask[1:], 'gtFine_labelIds.png')
            semantic_path = str(gt_path / semantic_name)

            self.dataset_samples.append((str(image_path), instances_path, semantic_path))

        total_classes = 34
        self._semseg_mapping = np.ones(total_classes, dtype=np.int32)
        for i in range(total_classes):
            self._semseg_mapping[i] = self.from_dataset_mapping.get(i, -1)

    def get_sample(self, index):
        image_path, instances_path, semantic_path = self.dataset_samples[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        instance_map = cv2.imread(instances_path, cv2.IMREAD_UNCHANGED).astype(np.int32)
        label_map = cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED)
        label_map = self._semseg_mapping[label_map]

        instances_info = dict()
        instances_ids = get_unique_labels(instance_map)
        for obj_id in instances_ids:
            if obj_id < 1000:
                continue

            class_id = obj_id // 1000
            mapped_class_id = self._semseg_mapping[class_id]
            ignore = mapped_class_id == -1

            instances_info[obj_id] = {
                'class_id': mapped_class_id, 'ignore': ignore
            }

        iscrowd = np.logical_and(instance_map < 1000, label_map >= self.things_offset)
        iscrowd_area = np.sum(iscrowd)
        if iscrowd_area > 0:
            iscrowd_labels = get_unique_labels(label_map[iscrowd])
            iscrowd_labels = [label for label in iscrowd_labels if label >= self.things_offset]

            new_obj_id = max(instances_info.keys(), default=0) + 1
            for class_id in iscrowd_labels:
                iscrowd_mask = np.logical_and(iscrowd, label_map == class_id)
                instance_map[iscrowd_mask] = new_obj_id
                instances_info[new_obj_id] = {'class_id': class_id, 'ignore': True}
                new_obj_id += 1

        instance_map[instance_map < 1000] = 0

        sample = {
            'image': image,
            'instances_mask': instance_map,
            'instances_info': instances_info,
            'semantic_segmentation': label_map
        }

        return sample

    @property
    def stuff_labels(self):
        return [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23]

    @property
    def things_labels(self):
        return [24, 25, 26, 27, 28, 31, 32, 33]
