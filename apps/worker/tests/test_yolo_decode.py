import numpy as np

from terjemahin_worker.detect.onnx_common import letterbox
from terjemahin_worker.detect.yolo import decode_ultralytics


def test_letterbox_shape_and_scale():
    img = np.zeros((300, 600, 3), dtype=np.uint8)
    tensor, scale, pad = letterbox(img, 640)
    assert tensor.shape == (1, 3, 640, 640)
    assert abs(scale - 640 / 600) < 1e-6
    assert pad == (0, (640 - 320) // 2)


def test_decode_maps_back_to_original_coords():
    # one box in letterboxed space, image 600x300 -> scale 640/600, pad_y 160
    scale, pad = 640 / 600, (0, 160)
    cx, cy, w, h = 320, 320, 64, 32  # centre of canvas
    out = np.zeros((1, 5, 3), dtype=np.float32)  # (1, 4+1cls, N)
    out[0, :, 0] = [cx, cy, w, h, 0.9]
    out[0, :, 1] = [cx, cy, w, h, 0.2]  # below threshold
    regions = decode_ultralytics(out, scale, pad, (600, 300), conf=0.4, iou=0.5)
    assert len(regions) == 1
    b = regions[0].bbox
    assert (b.x1, b.y1, b.x2, b.y2) == (270, 135, 330, 165)
