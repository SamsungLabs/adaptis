import math
import random

import cv2
import numpy as np
from bridson import poisson_disc_samples

from adaptis.inference.cython_utils.utils import find_local_maxima


def get_panoptic_segmentation(pmodel, image, ignore_mask=None, use_flip=True,
                              crop_size=None, min_overlap=0.2,
                              stuff_prob_mult=1.0, min_things_prob=0.0, min_stuff_prob=0.0,
                              sampling_algorithm='proposals', **sampling_params):
    image_height, image_width = image.shape[:2]
    things_prob = np.zeros((len(pmodel.to_dataset_mapping) - pmodel.things_offset, image_height, image_width), dtype=np.float32)
    stuff_prob = np.zeros((pmodel.things_offset, image_height, image_width), dtype=np.float32)
    counts = np.zeros((image_height, image_width), dtype=np.float32)

    if crop_size is None:
        crop_size_h, crop_size_w = image_height, image_width
    elif isinstance(crop_size, tuple):
        crop_size_h, crop_size_w = crop_size
    else:
        crop_size_h, crop_size_w = crop_size, crop_size

    x_offsets, x_overlap = get_offsets(image_width, crop_size_w, min_overlap)
    y_offsets, y_overlap = get_offsets(image_height, crop_size_h, min_overlap)
    x_osize = int(x_overlap * crop_size_w)
    y_osize = int(y_overlap * crop_size_h)

    masks = [None]
    masks_counts = [None]
    proposals_info = None
    occupied = np.full(image.shape[:2], 0, dtype=np.int32)
    for dy in y_offsets:
        for dx in x_offsets:
            crop_image = image[dy:dy + crop_size_h, dx:dx + crop_size_w]

            cfeatures = pmodel.get_features(crop_image, use_flip)

            cthings_prob, cstuff_prob = pmodel.get_semantic_segmentation(cfeatures, use_flip=use_flip,
                                                                         output_height=crop_size_h,
                                                                         output_width=crop_size_w)

            if crop_size is not None:
                things_prob[:, dy:dy + crop_size_h, dx:dx + crop_size_w] += cthings_prob
                stuff_prob[:, dy:dy + crop_size_h, dx:dx + crop_size_w] += cstuff_prob
                counts[dy:dy + crop_size_h, dx:dx + crop_size_w] += 1

            if ignore_mask is not None:
                cignore_mask = ignore_mask[dy:dy + crop_size_h, dx:dx + crop_size_w]
            else:
                cignore_mask = None

            if sampling_algorithm == 'proposals':
                crop_masks, proposals_info = \
                    predict_instances_with_proposals(pmodel, cfeatures, cthings_prob,
                                                     cignore_mask, crop_image.shape,
                                                     use_flip=use_flip, **sampling_params)
            elif sampling_algorithm == 'random':
                crop_masks, proposals_info = \
                    predict_instances_random(pmodel, cfeatures, cthings_prob,
                                             cignore_mask, crop_image.shape,
                                             use_flip=use_flip, **sampling_params)
            else:
                assert False, "Unknown sampling algorithm"

            if crop_masks:
                crop_final_map = np.array(crop_masks).argmax(axis=0)

            if crop_size is None:
                things_prob, stuff_prob = cthings_prob, cstuff_prob
                masks = [None] + crop_masks
                break

            occupied_left = occupied[dy:dy + crop_size_h, dx:dx + x_osize]
            occupied_top = occupied[dy:dy + y_osize, dx:dx + crop_size_w]
            left_labels, left_sizes = np.unique(occupied_left.flatten(), return_counts=True)
            top_labels, top_sizes = np.unique(occupied_top.flatten(), return_counts=True)

            for i in range(len(crop_masks)):
                obj_mask = crop_final_map == i
                matched = False

                if dy > 0:
                    obj_mask_top = obj_mask[:y_osize, :]

                    for ilabel, isize in zip(top_labels, top_sizes):
                        inter_area = np.logical_and(obj_mask_top, occupied_top == ilabel).sum()
                        union_area = np.logical_or(obj_mask_top, occupied_top == ilabel).sum()
                        iou = inter_area / union_area

                        if iou > 0.5:
                            matched = True
                            masks[ilabel][dy:dy + crop_size_h, dx:dx + crop_size_w] += crop_masks[i]
                            masks_counts[ilabel][dy:dy + crop_size_h, dx:dx + crop_size_w] += 1
                            occupied[dy:dy + crop_size_h, dx:dx + crop_size_w][obj_mask] = ilabel
                            break

                if dx > 0 and not matched:
                    obj_mask_left = obj_mask[:, :x_osize]

                    for ilabel, isize in zip(left_labels, left_sizes):
                        inter_area = np.logical_and(obj_mask_left, occupied_left == ilabel).sum()
                        union_area = np.logical_or(obj_mask_left, occupied_left == ilabel).sum()
                        iou = inter_area / union_area

                        if iou > 0.3:
                            matched = True
                            masks[ilabel][dy:dy + crop_size_h, dx:dx + crop_size_w] += crop_masks[i]
                            masks_counts[ilabel][dy:dy + crop_size_h, dx:dx + crop_size_w] += 1
                            occupied[dy:dy + crop_size_h, dx:dx + crop_size_w][obj_mask] = ilabel
                            break

                if not matched:
                    new_mask = np.zeros((image_height, image_width), dtype=np.float32)
                    new_mask[dy:dy + crop_size_h, dx:dx + crop_size_w] = crop_masks[i]
                    new_mask_counts = np.zeros_like(new_mask)
                    new_mask_counts[dy:dy + crop_size_h, dx:dx + crop_size_w] = 1

                    masks.append(new_mask)
                    masks_counts.append(new_mask_counts)
                    occupied[dy:dy + crop_size_h, dx:dx + crop_size_w][obj_mask] = len(masks) - 1

    if crop_size is not None:
        things_prob /= counts
        stuff_prob /= counts

    object_labels = [None]
    masks[0] = stuff_prob_mult * (1 - things_prob.sum(axis=0))
    for i in range(1, len(masks)):
        if crop_size is not None:
            masks[i] /= np.maximum(masks_counts[i], 1)
        obj_mask = masks[i] > sampling_params['thresh1']
        obj_label = things_prob[:, obj_mask].mean(axis=1).argmax()
        object_labels.append(obj_label)

    if len(masks) > 1:
        tmasks = np.array(masks)
        not_found_mask = tmasks[1:, :, :].max(axis=0) < min_things_prob
        final_map = tmasks.argmax(axis=0)
        final_map[not_found_mask] = 0
    else:
        final_map = np.array(masks).argmax(axis=0)

    semantic_segmentation = np.concatenate((stuff_prob, things_prob), axis=0).argmax(axis=0)
    semantic_segmentation[stuff_prob.max(axis=0) < min_stuff_prob] = -1
    if ignore_mask is not None:
        semantic_segmentation[ignore_mask > 0] = -1

    things_offset = stuff_prob.shape[0]
    instances_mask = final_map
    instances_info = {
        indx + 1: {'class_id': things_offset + label, 'ignore': False}
        for indx, label in enumerate(object_labels[1:])
    }

    return {
        'image': None, 'semantic_segmentation': semantic_segmentation,
        'instances_mask': instances_mask, 'instances_info': instances_info,
        'masks': masks, 'proposals_info': proposals_info
    }


def predict_instances_with_proposals(pmodel, features,
                                     instances_prob, ignore_mask,
                                     image_shape,
                                     sampling_mask=None, use_flip=False,
                                     thresh1=0.4, thresh2=0.4, ithresh=0.6,
                                     fl_prob=0.35, fl_eps=0.003, fl_blur=5, fl_step=0.025,
                                     cut_radius=-1, max_iters=500):
    point_invariant_states = pmodel.get_point_invariant_states(features)
    output_height, output_width = image_shape[:2]

    if ignore_mask is not None:
        instances_prob = instances_prob * (1 - ignore_mask)

    proposals_map = pmodel.get_proposals_map(features[0],
                                             width=image_shape[1], height=image_shape[0])[0, 0]
    if use_flip:
        proposals_map_flipped = pmodel.get_proposals_map(features[1],
                                                         width=image_shape[1], height=image_shape[0])[0, 0]
        proposals_map = 0.5 * (proposals_map + proposals_map_flipped[:, ::-1])

    proposals_map[instances_prob.max(axis=0) < ithresh] = 0

    masks = []
    occupied = np.zeros(image_shape[:2], dtype=np.int32)
    if ignore_mask is not None:
        occupied[ignore_mask == 1] = 1

    pmap = proposals_map.copy()
    if sampling_mask is not None:
        pmap *= sampling_mask
        occupied[sampling_mask == 0] = 1

    if fl_blur > 1:
        pmap = cv2.blur(pmap, (fl_blur, fl_blur))
    colors, candidates = find_local_maxima(pmap, prob_thresh=fl_prob, eps=fl_eps, step_size=fl_step)

    for i in range(max_iters):
        if i >= len(candidates):
            break

        best_point = candidates[i]
        if occupied[best_point]:
            continue

        p = pmodel.get_instance_masks(features[0], [best_point], states=point_invariant_states[0],
                                      width=image_shape[1], height=image_shape[0])[0]
        if use_flip:
            flipped_point = (best_point[0], output_width - best_point[1])
            p_flipped = pmodel.get_instance_masks(features[1], [flipped_point], states=point_invariant_states[1],
                                                  width=image_shape[1], height=image_shape[0])[0]
            p = 0.5 * (p + p_flipped[:, ::-1])

        if cut_radius > 0:
            p = cut_prob_with_radius(p, best_point, cut_radius)

        obj_mask = p > thresh1
        obj_area = obj_mask.sum()
        if obj_area < 10:
            continue

        inter_score = occupied[obj_mask].sum() / obj_area
        if inter_score >= thresh2:
            continue

        masks.append(p)
        occupied[obj_mask == 1] = 1
        kernel = np.ones((3, 3), np.uint8)
        obj_mask = cv2.dilate(obj_mask.astype(np.uint8), kernel, iterations=2)
        proposals_map[obj_mask == 1] = 0

    return masks, (pmap, colors, candidates)


def predict_instances_random(pmodel, features, instances_prob, ignore_mask, image_shape,
                             thresh1=0.5, thresh2=0.6,
                             num_candidates=5, num_iters=40,
                             cut_radius=-1, use_flip=False,
                             ithresh=0.5):
    point_invariant_states = pmodel.get_point_invariant_states(features)
    output_height, output_width = image_shape[:2]

    if ignore_mask is not None:
        instances_prob = instances_prob * (1 - ignore_mask)

    result_map = np.full((output_height, output_width), -1, dtype=np.int)
    result_map[instances_prob.max(axis=0) < ithresh] = 0

    last_color = 1
    masks = []
    for i in range(num_iters):
        if last_color == 0:
            points = get_random_points(output_height, output_width, num_candidates)
        else:
            points = sample_points_with_mask(result_map == -1, num_candidates)
            if points is None:
                break

        pmaps = pmodel.get_instance_masks(features[0], points, states=point_invariant_states[0],
                                          width=output_width, height=output_height)
        if use_flip:
            flipped_points = points.copy()
            flipped_points[:, 1] = output_width - flipped_points[:, 1]
            pmaps_flipped = pmodel.get_instance_masks(features[1], flipped_points, states=point_invariant_states[1],
                                                      width=output_width, height=output_height)

            pmaps = 0.5 * (pmaps + pmaps_flipped[:, :, ::-1])

        if last_color == 0:
            best_point_id = pmaps.mean(axis=(1, 2)).argmax()
        else:
            tmp = [get_map_score(x) for x in pmaps]
            best_point_id = np.argmax(tmp)

        pmap = pmaps[best_point_id]
        if cut_radius > 0:
            pmap = cut_prob_with_radius(pmap, points[best_point_id], cut_radius)

        pmap_mask = pmap > thresh1
        pmap_area = pmap_mask.sum()

        uncovered_area = (result_map[pmap_mask] == -1).sum()

        if uncovered_area / pmap_area < thresh2:
            continue

        result_map[pmap_mask] = last_color

        last_color += 1
        masks.append(pmap)

    return masks, None


def get_random_points(width, height, num_points):
    points = poisson_disc_samples(width=width, height=height, r=10)
    random.shuffle(points)
    return np.round(points[:num_points])


def get_map_score(prob_map, thresh=0.2):
    mask = prob_map > thresh

    if mask.sum() > 0:
        return prob_map[mask].mean()
    else:
        return 0


def sample_points_with_mask(mask, num_points):
    possible_points = np.where(mask)
    num_possible_points = possible_points[0].shape[0]

    if num_possible_points == 0:
        return None

    rindx = random.sample(list(range(num_possible_points)),
                          k=min(num_points, num_possible_points))
    points = []
    for j in rindx:
        points.append((possible_points[0][j], possible_points[1][j]))
    points = np.array(points)

    return points


def cut_prob_with_radius(prob, p, radius):
    mask = np.zeros_like(prob, dtype=np.float32)

    mask[max(p[0] - radius, 0):min(p[0] + radius, mask.shape[0]),
    max(p[1] - radius, 0):min(p[1] + radius, mask.shape[1])] = 1

    return prob * mask


def get_offsets(length, crop_size, min_overlap_ratio=0.2):
    if length == crop_size:
        return [0], 1.0

    N = (length / crop_size - min_overlap_ratio) / (1 - min_overlap_ratio)
    N = math.ceil(N)

    overlap_ratio = (N - length / crop_size) / (N - 1)
    overlap_width = int(crop_size * overlap_ratio)

    offsets = [0]
    for i in range(1, N):
        new_offset = offsets[-1] + crop_size - overlap_width
        if new_offset + crop_size > length:
            new_offset = length - crop_size

        offsets.append(new_offset)

    return offsets, overlap_ratio
