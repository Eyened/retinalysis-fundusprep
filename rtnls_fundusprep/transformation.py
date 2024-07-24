import cv2
import numpy as np


class ProjectiveTransform:
    def __init__(self, M):
        self.M = M
        self.M_inv = np.linalg.inv(M)

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
            return w, h

    def warp(self, image, out_size=None):
        dsize = self.get_dsize(image, out_size)

        if image.dtype == bool or image.dtype == np.uint8:
            warped = cv2.warpPerspective(
                image, self.M, dsize=dsize, flags=cv2.INTER_NEAREST
            )
        else:
            warped = cv2.warpPerspective(image, self.M, dsize=dsize)
        return warped

    def warp_inverse(self, image, out_size=None):
        dsize = self.get_dsize(image, out_size)

        if image.dtype == bool or image.dtype == np.uint8:
            warped = cv2.warpPerspective(
                image, self.M_inv, dsize=dsize, flags=cv2.INTER_NEAREST
            )
        else:
            warped = cv2.warpPerspective(image, self.M_inv, dsize=dsize)
        return warped

    def _repr_html_(self):
        html_table = "<h4>Projective Transform:</h4><table>"

        for row in self.M:
            html_table += "<tr>"
            for val in row:
                html_table += f"<td>{val:.3f}</td>"
            html_table += "</tr>"

        html_table += "</table>"
        return html_table


def get_affine_transform(out_size, rotate, scale, center, flip):
    """
    Parameters:
    out_size: size of the extracted patch (h, w)
    rotate: angle in degrees
    scale: scaling factor (sy, sx)
    center: center of the patch (cy, cx)
    flip: apply horizontal/vertical flipping
    """
    # center to top left corner
    cy, cx = center
    C1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=float)

    # rotate
    th = rotate * np.pi / 180
    R = np.array(
        [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]],
        dtype=float,
    )

    # scale
    sy, sx = scale
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=float)

    # top left corner to center
    h, w = out_size
    ty = h / 2
    tx = w / 2
    C2 = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)

    M = C2 @ S @ R @ C1
    flip_vertical, flip_horizontal = flip

    if flip_horizontal:
        M = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]]) @ M
    if flip_vertical:
        M = np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]]) @ M

    return ProjectiveTransform(M)
