# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch.nn.functional as nnF
from util.box_ops import box_xyxy_to_cxcywh, poly_xy_to_cxcyxy
from util.misc import interpolate
import cv2
import numpy as np
import math
import copy


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")
    
    if "polys" in target:
        polys = target["polys"]
        max_size = torch.as_tensor([w,h], dtype=torch.float32)
        cropped_polys = polys - torch.as_tensor([j,i,j,i,j,i,j,i])
        cropped_polys = torch.min(cropped_polys.reshape(-1,4,2), max_size)
        cropped_polys = cropped_polys.clamp(min=0)
        target["polys"] = cropped_polys.reshape(-1,8)
        fields.append("polys")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        # if "polys" in target:
        #     cropped_polys = target["polys"].reshape(-1,4,2)
        #     keep = ~(torch.all(cropped_polys[:,:,0] == cropped_polys[:,0,0],-1) or torch.all(cropped_polys[:,:,1] == cropped_polys[:,0,1],-1))
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "polys" in target:
        polys = target["polys"]
        polys = polys * torch.as_tensor([-1,1,-1,1,-1,1,-1,1]) + torch.as_tensor([w, 0, w, 0, w, 0, w, 0])
        target["polys"] = polys

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "polys" in target:
        polys = target["polys"]
        scaled_polys = polys * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height,ratio_width, ratio_height, ratio_width, ratio_height])
        target["polys"] = scaled_polys

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        
        if "polys" in target:
            polys = target["polys"]
            # polys = poly_xy_to_cxcyxy(polys)
            polys = polys / torch.tensor([ w, h, w, h, w, h, w, h], dtype=torch.float32)
            target["polys"] = polys

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomRotate(object):
    def __init__(self, angle_list, jitter_degree, train=True):
        self.angle_list = angle_list
        self.jitter_degree = jitter_degree
        self.train = train

    def __str__(self):
        s = self.__class__.__name__ + "("
        s += "angle_list={}, ".format(self.angle_list)
        s += "jitter_degree={})".format(self.jitter_degree)
        return s

    def __call__(self, image, target=None):
        prob = random.random()
        if prob < 0.3:
            return image, target
        if isinstance(image, np.ndarray) is False:
            image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        # generate a random rotation angle
        if self.train:
            angle = np.random.choice(np.array(self.angle_list)) + np.random.randint(-self.jitter_degree,self.jitter_degree)
        else:
            angle = np.random.choice(np.array(self.angle_list)) + (np.random.rand() * 2 * self.jitter_degree - self.jitter_degree)
        if angle > 180:
            angle = angle - 360

        # grab the dimensions of the image and then determine the center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        image = cv2.warpAffine(image, M, (nW, nH))

        if target is None:
            return image

        # #process corresponding keys in entry
        target['size'] = torch.tensor([image.shape[0],image.shape[1]])

        # rotate the annotations correspondingly
        # annos = dataset_dict["annotations"]
        # boxes = np.array([obj["bbox"] for obj in annos])
        boxes = target['boxes'] # XYXY
        # assert boxes.shape[1] == 5 # XYWHA_ABS_V2
        # boxes[:, 2] += boxes[:, 0] # x2
        # boxes[:, 3] += boxes[:, 1] # y2
        # assert "quad_coord" not in annos[0], "Need to implement rotatation for quad_coords"
        # if "poly" in annos[0]:
        #     polys = [copy.deepcopy(obj['poly'])[0] for obj in annos]
        #     assert len(boxes) == len(polys)
        # else:
        #     polys = []
        polys = target['polys'] # xy
        if "solid_lines" in target:
            solid_lines = copy.deepcopy(target['solid_lines'])
        else:
            solid_lines = []

        theta = angle * math.pi / 180
        sin = math.sin(theta)
        cos = math.cos(theta)

        # for rotated box
        # for i in range(boxes.shape[0]):
        #     box = boxes[i, :].reshape(-1)
        #     mid_x = (box[0] + box[2]) / 2.0 - cX
        #     mid_y = (box[1] + box[3]) / 2.0 - cY
        #     old_norm_theta = 0 # box[4]
        #     new_norm_theta = old_norm_theta + angle / 45.0
        #     new_theta = new_norm_theta * 45
        #     x1 = mid_x * cos - mid_y * sin + nW / 2.0
        #     y1 = mid_x * sin + mid_y * cos + nH / 2.0
        #     # normalize the rotate angle in -45 to 45
        #     if new_theta > 90:
        #         new_theta = new_theta - 180
        #     if new_theta < -90:
        #         new_theta = new_theta + 180
        #     if new_theta > 45:
        #         new_theta = new_theta - 90
        #         box_half_width = (box[3] - box[1]) / 2.0
        #         box_half_height = (box[2] - box[0]) / 2.0
        #     elif new_theta < -45:
        #         new_theta = new_theta + 90
        #         box_half_width = (box[3] - box[1]) / 2.0
        #         box_half_height = (box[2] - box[0]) / 2.0
        #     else:
        #         box_half_width = (box[2] - box[0]) / 2.0
        #         box_half_height = (box[3] - box[1]) / 2.0
        #     min_x = int(x1 - box_half_width)
        #     max_x = int(x1 + box_half_width)
        #     min_y = int(y1 - box_half_height)
        #     max_y = int(y1 + box_half_height)
        #     assert new_theta >= -45 and new_theta <= 45
        #     new_norm_theta = new_theta / 45.0
        #     # boxes[i, :] = np.array([min_x, min_y, max_x, max_y, new_norm_theta])
        #     target["boxes"][i,:] = torch.tensor(np.array([min_x, min_y, max_x, max_y])) # External Rectangle

        # for poly
        for i, poly in enumerate(polys):
            poly = np.array(poly)
            poly[0::2] = poly[0::2] - cX
            poly[1::2] = poly[1::2] - cY
            for j in range(len(poly)//2):
                j = j * 2
                x = poly[j]
                y = poly[j+1]
                polys[i][j] = int(x * cos - y * sin + nW / 2.0)
                polys[i][j+1] = int(x * sin + y * cos + nH / 2.0)
            target["polys"][i,:] = polys[i].clone().detach()
            tmp_poly =target["polys"][i,:]
            # external horizontal rectangle
            tmp_x = tmp_poly[0::2]
            tmp_y = tmp_poly[1::2]
            min_x = torch.min(tmp_x)
            min_y = torch.min(tmp_y)
            left_top = torch.tensor([min_x,min_y])
            def distance_function(lt_point,point):
                return nnF.pairwise_distance(lt_point, point, p=2)
            distance = []
            for k in range(4):
                distance.append(distance_function(left_top.unsqueeze(0),tmp_poly[k*2:k*2+2].unsqueeze(0))[0])
            distance = torch.tensor(distance)
            sort_distance = distance.sort(-1)
            if sort_distance[0][0] == sort_distance[0][1]:
                if target["polys"][i,((sort_distance[1][0])%4)*2] > target["polys"][i,((sort_distance[1][1])%4)*2]:
                    min_x_index = sort_distance[1][1]
                else:
                    min_x_index = sort_distance[1][0]
            else:
                min_x_index = sort_distance[1][0]

            # tmp_target_x = target["polys"][i,0::2]
            # sort_target_x = tmp_target_x.sort(-1)
            # if sort_target_x[0][0] == sort_target_x[0][1]:
            #     if target["polys"][i,((sort_target_x[1][0])%4)*2+1] > target["polys"][i,((sort_target_x[1][1])%4)*2+1]:
            #         min_x_index = sort_target_x[1][1]
            #     else:
            #         min_x_index = sort_target_x[1][0]
            # else:
            #     min_x_index = sort_target_x[1][0]
            new_poly = []
            for j in range(4):
                new_poly.append(target["polys"][i,((min_x_index+j)%4)*2:((min_x_index+j)%4)*2+2])
            target["polys"][i,:] = torch.cat(new_poly,-1)
        if len(polys) > 0:
            target["boxes"] = torch.cat([torch.min(target["polys"][:,0::2],-1)[0].unsqueeze(-1),torch.min(target["polys"][:,1::2],-1)[0].unsqueeze(-1),torch.max(target["polys"][:,0::2],-1)[0].unsqueeze(-1),torch.max(target["polys"][:,1::2],-1)[0].unsqueeze(-1)],-1)
        else:
            target["boxes"] = boxes

        # for solid line
        for i, solid_line in enumerate(solid_lines):
            line_direction = solid_line[-1]
            solid_line = np.array(solid_line[:-1])
            solid_line[0::2] = solid_line[0::2] - cX
            solid_line[1::2] = solid_line[1::2] - cY
            for j in range(len(solid_line)//2):
                x = solid_line[j * 2]
                y = solid_line[j * 2 + 1]
                solid_lines[i][j * 2] = int(x * cos - y * sin + nW / 2.0)
                solid_lines[i][j * 2 + 1] = int(x * sin + y * cos + nH / 2.0)
            target["solid_lines"][i] = solid_lines[i]
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        return image, target