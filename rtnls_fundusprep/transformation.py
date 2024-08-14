import cv2
import numpy as np
from enum import Enum


class Interpolation(Enum):
    NEAREST = cv2.INTER_NEAREST
    BILINEAR = cv2.INTER_LINEAR
    BICUBIC = cv2.INTER_CUBIC


def get_param_xy(param):
    if hasattr(param, '__iter__') and len(param) == 2:
        return param
    else:
        return param, param


class ProjectiveTransform:

    def __init__(self, M, in_size, out_size):
        self.in_size = get_param_xy(in_size)
        self.out_size = get_param_xy(out_size)
        self.M = M
        self.M_inv = np.linalg.inv(M)

    @property
    def scale(self):
        # self.M is a 3x3 matrix
        # return the scaling factor as a single scalar
        return np.sqrt(np.abs(np.linalg.det(self.M[:2, :2])))
    
    def apply(self, points):
        # Add homogeneous coordinate (1) to each point
        points_homogeneous = np.column_stack([points, np.ones(len(points))])
        p = np.dot(points_homogeneous, self.M.T)
        # Normalize by dividing by the last column (homogeneous coordinate)
        return p[:, :2] / p[:, [-1]]

    def apply_inverse(self, points):
        # Add homogeneous coordinate (1) to each point
        points_homogeneous = np.column_stack([points, np.ones(len(points))])
        p = np.dot(points_homogeneous, self.M_inv.T)
        # Normalize by dividing by the last column (homogeneous coordinate)
        return p[:, :2] / p[:, [-1]]

    def get_dsize(self, image, out_size):
        if out_size is None:
            h, w = image.shape[:2]
            corners = np.array([[0, 0], [0, h], [w, h], [w, 0]])
            return np.ceil(self.apply(corners).max(axis=0)).astype(int)
        else:
            h, w = out_size
            return int(np.ceil(w)), int(np.ceil(h))

    def _apply_warp(self, image, out_size, M, mode):
        dsize = self.get_dsize(image, out_size)
        image_in = image.astype(np.uint8) if image.dtype == bool else image
        result = cv2.warpPerspective(
            image_in, M, dsize=dsize, flags=mode.value)
        return result.astype(bool) if image.dtype == bool else result

    def warp(self, image, out_size=None, mode=Interpolation.BILINEAR):
        return self._apply_warp(image, out_size or self.out_size, self.M, mode)

    def warp_inverse(self, image, out_size=None, mode=Interpolation.BILINEAR):
        return self._apply_warp(image, out_size or self.in_size, self.M_inv, mode)

    def _repr_html_(self):
        html = "<h4>Projective Transform:</h4>"
        html += f"<p>Input size: {self.in_size}</p>"
        html += f"<p>Output size: {self.out_size}</p>"
        html += "<p>Matrix:</p>"
        html += "<table>"
        for row in self.M:
            html += "<tr>"
            for val in row:
                html += f"<td>{val:.3f}</td>"
            html += "</tr>"

        html += "</table>"
        return html

    def to_dict(self):
        return {
            "M": self.M.tolist(),
            "in_size": self.in_size,
            "out_size": self.out_size,
        }

    @classmethod
    def from_dict(cls, d):
        return ProjectiveTransform(np.array(d["M"]), d["in_size"], d["out_size"])


def get_affine_transform(in_size, out_size, rotate=0, scale=1, center=None, flip=(False, False)):
    """
    Parameters:
    in_size: size of the input image (h, w)
    out_size: size of the extracted patch (h, w)
    rotate: angle in degrees
    scale: scaling factor s or (sy, sx)
    center: center of the patch (cy, cx)
    flip: apply horizontal/vertical flipping
    """
    # center to top left corner
    if center is None:
        h, w = get_param_xy(in_size)
        cy, cx = h / 2, w / 2
    else:
        cy, cx = center
    C1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=float)

    # rotate
    th = rotate * np.pi / 180
    R = np.array(
        [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]],
        dtype=float,
    )

    # scale
    sy, sx = get_param_xy(scale)
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=float)

    # top left corner to center
    h, w = get_param_xy(out_size)
    ty = h / 2
    tx = w / 2
    C2 = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)

    M = C2 @ S @ R @ C1
    flip_vertical, flip_horizontal = flip

    if flip_horizontal:
        M = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]]) @ M
    if flip_vertical:
        M = np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]]) @ M

    return ProjectiveTransform(M, in_size, out_size)
