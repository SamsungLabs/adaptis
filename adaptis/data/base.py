import random

import cv2
import numpy as np
from scipy.ndimage import measurements
import torchvision
from torch.utils.data.dataset import Dataset

from adaptis.coco.utils import IdGenerator, rgb2id
from adaptis.utils.vis import get_palette


class BaseDataset(Dataset):
    def __init__(self,
                 with_segmentation=False,
                 points_from_one_object=False,
                 augmentator=None,
                 num_points=1,
                 input_transform=None,
                 get_image_scale=None,
                 min_object_area=0,
                 min_ignore_object_area=10,
                 keep_background_prob=0.0,
                 sample_ignore_object_prob=0.0,
                 epoch_len=-1):
        super(BaseDataset, self).__init__()
        self.items_per_epoch = epoch_len  # if > 0 samples "items per epoch" random elements every epoch
        self.num_points = num_points  # No of points per each training sample
        self.with_segmentation = with_segmentation  # include semantic segmentation in training or not
        self.points_from_one_object = points_from_one_object  # sample points from one object per image or not
        self.input_transform = input_transform  # transform applied to an image after all augmentations
        self.augmentator = augmentator  # random augmentations function
        self.get_image_scale = get_image_scale  # (float, int) or function f(image_shape) = scale
        self.min_object_area = min_object_area  # minimum area of the object that is not ignored
        self.min_ignore_object_area = min_ignore_object_area  # minimum area of ignored object to be included in
        # ignored_mask that is used to sample points from ignored areas for more robust training
        self.keep_background_prob = keep_background_prob  # prob. of keeping a crop with only background on it
        self.sample_ignore_object_prob = sample_ignore_object_prob  # prob. of sampling a point from ignored_mask

        if isinstance(self.get_image_scale, (float, int)):
            scale = self.get_image_scale
            self.get_image_scale = lambda shape: scale  # default scale is independent from image_shape

        if self.input_transform is None:
            self.input_transform = torchvision.transforms.ToTensor()

        self.dataset_samples = None
        self._precise_masks = ['instances_mask', 'semantic_segmentation']
        self._from_dataset_mapping = None
        self._to_dataset_mapping = None

    def __getitem__(self, index):
        if self.items_per_epoch > 0:
            index = random.randrange(0, len(self.dataset_samples))

        sample = self._get_sample_checked(index)
        sample = self._rescale_sample(sample)
        sample = self._augment_sample(sample)
        sample = self._exclude_small_objects(sample)
        sample = self._determine_ignored_regions(sample)

        points, points_masks = self._get_points_for_object(sample)

        image = self.input_transform(sample['image'])

        output = {
            'images': image,
            'points': points.astype(np.float32),
            'instances': points_masks
        }

        if self.with_segmentation:
            output['semantic'] = sample['semantic_segmentation']

        return output

    def _get_sample_checked(self, index):
        sample = self.get_sample(index)

        assert sample['image'].dtype == 'uint8'
        assert sample['instances_mask'].dtype == 'int32'
        if 'semantic_segmentation' in sample:
            assert sample['semantic_segmentation'].dtype == 'int32'

        return sample

    def _rescale_sample(self, sample):
        if self.get_image_scale is None:
            return sample

        image = sample['image']
        scale = self.get_image_scale(image.shape)
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        new_size = (image.shape[1], image.shape[0])

        sample['image'] = image
        for mask_name in self._precise_masks:
            if mask_name not in sample:
                continue
            sample[mask_name] = cv2.resize(sample[mask_name], new_size, interpolation=cv2.INTER_NEAREST)

        return sample

    def _check_augmented_sample(self, sample, aug_output, masks_names_to_augment):
        if self.keep_background_prob < 0.0 or random.random() < self.keep_background_prob:
            return True

        aug_instances_mask = aug_output['masks'][masks_names_to_augment.index('instances_mask')]
        aug_sample_ids = set(self.get_unique_labels(aug_instances_mask, exclude_zero=True))
        num_objects_after_aug = len([obj_id for obj_id in aug_sample_ids
                                     if not sample['instances_info'][obj_id]['ignore']])

        return num_objects_after_aug > 0

    @staticmethod
    def get_unique_labels(instance_mask, exclude_zero=False):
        class_sizes = np.bincount(instance_mask.flatten())
        unique_labels = np.nonzero(class_sizes)[0]

        if exclude_zero:
            unique_labels = unique_labels[unique_labels > 0]

        return unique_labels.tolist()

    def _augment_sample(self, sample):
        if self.augmentator is None:
            return sample

        # ignore label is -1, this can cause errors after rotation augmentation
        if 'semantic_segmentation' in sample:
            sample['semantic_segmentation'] += 1

        masks_names_to_augment = [mask_name for mask_name in self._precise_masks if mask_name in sample]
        masks = [sample[mask_name] for mask_name in masks_names_to_augment]

        valid_augmentation = False
        while not valid_augmentation:
            aug_output = self.augmentator(image=sample['image'], masks=masks)
            valid_augmentation = self._check_augmented_sample(sample, aug_output, masks_names_to_augment)

        sample['image'] = aug_output['image']
        for mask_name, mask in zip(masks_names_to_augment, aug_output['masks']):
            sample[mask_name] = mask

        if 'semantic_segmentation' in sample:
            sample['semantic_segmentation'] -= 1

        sample_ids = set(self.get_unique_labels(sample['instances_mask'], exclude_zero=True))
        instances_info = sample['instances_info']
        instances_info = {sample_id: sample_info for sample_id, sample_info in instances_info.items()
                          if sample_id in sample_ids}
        sample['instances_info'] = instances_info

        return sample

    def _exclude_small_objects(self, sample):
        if self.min_object_area <= 0:
            return sample

        for obj_id, obj_info in sample['instances_info'].items():
            if not obj_info['ignore']:
                obj_area = (sample['instances_mask'] == obj_id).sum()
                if obj_area < self.min_object_area:
                    obj_info['ignore'] = True

        return sample

    def _determine_ignored_regions(self, sample):
        if self.sample_ignore_object_prob <= 0.0:
            return sample

        ignore_ids = []
        ignore_mask = np.zeros_like(sample['instances_mask'])
        for obj_id, obj_info in sample['instances_info'].items():
            if obj_info['ignore']:
                obj_mask = sample['instances_mask'] == obj_id
                if np.sum(obj_mask) < self.min_ignore_object_area:
                    continue
                ignore_mask[obj_mask] = obj_id
                ignore_ids.append(obj_id)

        if 'semantic_segmentation' in sample:
            current_id = max(ignore_ids, default=0) + 1
            ss_ignore = sample['semantic_segmentation'] == -1
            ss_ignore_labeled, _ = measurements.label(ss_ignore, np.ones((3, 3)))
            ss_ignore_ids = self.get_unique_labels(ss_ignore_labeled, exclude_zero=True)

            for region_id in ss_ignore_ids:
                region_mask = ss_ignore_labeled == region_id

                region_mask_area = region_mask.sum()
                if region_mask_area < self.min_ignore_object_area:
                    continue

                # if ignore_mask[region_mask].sum() > 0:
                #     continue

                if np.any(np.logical_and(ignore_mask, region_mask)):
                    continue

                ignore_mask[region_mask] = current_id
                ignore_ids.append(current_id)
                current_id += 1

        sample['ignore_mask'] = ignore_mask
        sample['ignore_ids'] = ignore_ids

        return sample

    def _get_random_object(self, sample, object_ids):
        if sample.get('ignore_ids', []) and random.random() < self.sample_ignore_object_prob:
            random_id = random.choice(sample['ignore_ids'])
            mask = sample['ignore_mask'] == random_id
            is_ignored = True
        else:
            random_id = random.choice(object_ids)
            mask = sample['instances_mask'] == random_id
            is_ignored = False

        indices = np.argwhere(mask)
        mask = mask.astype(np.float32)[np.newaxis, :]

        return mask, indices, is_ignored

    def _get_points_for_object(self, sample):
        objects_id = [obj_id for obj_id, obj_info in sample['instances_info'].items()
                      if not obj_info['ignore']]

        if len(objects_id) > 0:
            points, points_masks = [], []
            mask, indices, is_ignored = self._get_random_object(sample, objects_id)
            for i in range(self.num_points):
                if is_ignored:
                    mask[mask > 0.5] = -1

                point_coord = indices[np.random.randint(0, indices.shape[0])]
                points_masks.append(mask)
                points.append(point_coord)
                if not self.points_from_one_object:
                    mask, indices, is_ignored = self._get_random_object(sample, objects_id)

            points_masks = np.array(points_masks)
            points = np.array(points)
        else:
            height, width = sample['instances_mask'].shape[:2]
            points_masks = np.full((self.num_points, 1, height, width), -1, dtype=np.float32)
            points = np.zeros((self.num_points, 2), dtype=np.float32)

        return points, points_masks

    def get_sample(self, index):
        # retrieve mask and image
        raise NotImplementedError

    @property
    def stuff_labels(self):
        raise NotImplementedError

    @property
    def things_labels(self):
        raise NotImplementedError

    @property
    def from_dataset_mapping(self):
        if self._from_dataset_mapping is None:
            dataset_labels = self.stuff_labels + self.things_labels
            mapping = {label: indx for indx, label in enumerate(dataset_labels)}
            self._from_dataset_mapping = mapping

        return self._from_dataset_mapping

    @property
    def to_dataset_mapping(self):
        if self._to_dataset_mapping is None:
            mapping = {indx: label for label, indx in self.from_dataset_mapping.items()}
            self._to_dataset_mapping = mapping

        return self._to_dataset_mapping

    @property
    def stuff_offset(self):
        return 0

    @property
    def things_offset(self):
        return len(self.stuff_labels)

    @property
    def num_classes(self):
        return len(self.from_dataset_mapping)

    def __len__(self):
        if self.items_per_epoch > 0:
            return self.items_per_epoch
        else:
            return len(self.dataset_samples)

    def convert_to_coco_format(self, sample, use_id_generator=False, min_segment_area=1):
        instances_mask = sample['instances_mask']
        panoptic_labels = np.zeros(instances_mask.shape[:2] + (3,), dtype=np.uint8)
        segments = []

        semantic_labels = sample['semantic_segmentation'].copy()
        semantic_labels[instances_mask > 0] = -1
        semantic_ids = np.array(self.get_unique_labels(semantic_labels + 1)) - 1

        if use_id_generator:
            if not hasattr(self, '_categories'):
                self._categories = self._generate_coco_categories()
            categories = {el['id']: el for el in self._categories}
            id_generator = IdGenerator(categories)
        else:
            palette = get_palette(len(semantic_labels) + len(sample['instances_info']) + 1)[1:].astype(int)

        def add_segment(label_id, class_id, isthing):
            if class_id == -1 or (not isthing and class_id >= self.things_offset):
                return

            segment_mask = (instances_mask if isthing else semantic_labels) == label_id
            segment_area = segment_mask.sum()
            if segment_area < min_segment_area:
                return

            category_id = self.to_dataset_mapping[class_id]
            if use_id_generator:
                segment_id, color = id_generator.get_id_and_color(category_id)
            else:
                color = palette[len(segments)]
                segment_id = rgb2id(color)

            panoptic_labels[segment_mask] = color
            segments.append({
                'id': segment_id,
                'category_id': category_id,
                'iscrowd': 0,
                'area': int(segment_area)
            })

        for label_id in semantic_ids:
            add_segment(label_id, label_id, False)

        for inst_id, instance in sample['instances_info'].items():
            add_segment(inst_id, instance['class_id'], True)

        coco_sample = {
            'annotation': panoptic_labels,
            'segments_info': segments
        }

        if 'image' in sample:
            coco_sample['image'] = sample['image']
        return coco_sample

    def _generate_coco_categories(self):
        categories = []

        palette = get_palette(self.num_classes + 1)[1:].astype(int)
        for indx, stuff_label in enumerate(self.stuff_labels):
            categories.append({
                'id': stuff_label,
                'isthing': 0,
                'color': tuple(palette[indx])
            })

        for indx, thing_label in enumerate(self.things_labels):
            categories.append({
                'id': thing_label,
                'isthing': 1,
                'color': tuple(palette[self.things_offset + indx])
            })

        return categories
