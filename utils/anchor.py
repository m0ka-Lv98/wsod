import numpy as np
import matplotlib.pyplot as plt
import torch
PYRAMID_LEVEL = [4,5,6,7]

TEST_SCALE = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
TEST_RATIO = np.array([0.5, 1, 2])
strides = [2 ** (level+1) for level in PYRAMID_LEVEL]

boxSizeBaseSizes = [2 ** (level + 2) for level in PYRAMID_LEVEL]
imageShape = (512, 512)

def generate_anchorbox(boxSize, scale, ratios):
    prevBoxsScales = np.tile(scale, (2, len(scale))).T
    prevBoxsScales = prevBoxsScales * boxSize
    
    preBoxAreas = prevBoxsScales[:, 0] * prevBoxsScales[:, 1]

    # w * h = area
    # w * w*ratio = area
    preBoxRatios = np.repeat(ratios, len(scale))
    preBoxW = np.sqrt(preBoxAreas / preBoxRatios)
    preBoxH = preBoxW * preBoxRatios

    anchorBox = np.zeros((len(scale) * len(ratios), 4))

    anchorBox[:, 2] = preBoxW
    anchorBox[:, 3] = preBoxH

    #
    anchorBox[:, 0::2] -= np.tile(anchorBox[:, 2] * 0.5, (2, 1)).T
    anchorBox[:, 1::2] -= np.tile(anchorBox[:, 3] * 0.5, (2, 1)).T
    return anchorBox

def shift_boxes(positionFixedAnchorBoxes, imageShape, stride, boxSizeBaseSize):
    imageWidth = imageShape[1]
    imageHeight = imageShape[0]

    featuresWidth = int((imageWidth + 0.5 * stride) / stride)
    featureHeight = int((imageHeight + 0.5 * stride) / stride)

    featureXCoordinates = np.arange(0, featuresWidth) + 0.5
    featureYCoordinates = np.arange(0, featureHeight) + 0.5

    featureXCoordinates = featureXCoordinates * stride
    featureYCoordinates = featureYCoordinates * stride

    a, b = np.meshgrid(featureXCoordinates, featureYCoordinates)
    m = np.vstack((a.ravel(), b.ravel(), a.ravel(), b.ravel()))
    m = m.transpose()

    positionFixedAnchorBoxes = np.expand_dims(positionFixedAnchorBoxes, 0)
    m = np.expand_dims(m, 1)

    res = m + positionFixedAnchorBoxes

    return m[:, :, :2], res

def run_shift_boxes(idx):
    position_fixed_anchor_boxes = generate_anchorbox(boxSizeBaseSizes[idx], TEST_SCALE, TEST_RATIO)
    centerPositions, transformed_anchor_boxes = shift_boxes(position_fixed_anchor_boxes, imageShape, strides[idx],
                                                            boxSizeBaseSizes[idx])
    transformed_anchor_boxes = np.clip(transformed_anchor_boxes,0,512)
    
    return transformed_anchor_boxes



def make_anchor():
    p_bboxes = []
    for i in range(3):
        boxes = run_shift_boxes(i)
        p_bboxes.append(torch.from_numpy(boxes))

    p = torch.cat(p_bboxes,0)

    
    return p.view(-1,4)
