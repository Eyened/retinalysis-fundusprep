from functools import cached_property, lru_cache
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from rtnls_fundusprep.transformation import ProjectiveTransform, get_affine_transform
from rtnls_fundusprep.utils import to_uint8


class CFIBounds:
    def __init__(
        self,
        center: Tuple[float],
        radius: float,
        lines={},
        hw: Tuple[float] = None,
        image: np.array = None,
        **kwargs,
    ):
        center = center
        self.cy = center[1]
        self.cx = center[0]
        self.radius = radius
        self.lines = lines

        assert image is not None or hw is not None, (
            "Either hw or image must be provided"
        )

        self.hw = hw
        self.image = image
        if image is not None:
            self.hw = image.shape[:2]

        self.min_y = 0
        self.max_y = self.hw[0]
        self.min_x = 0
        self.max_x = self.hw[1]

        def intersects(location):
            if location not in lines:
                return
            p0, p1 = lines[location]
            intersects = line_circle_intersection(p0, p1, center, radius)
            if len(intersects) == 2:
                return intersects

        line_bottom = intersects("bottom")
        if line_bottom:
            ((_, y0), (_, y1)) = line_bottom
            self.max_y = min(self.max_y, int(np.floor(min(y0, y1))))
        line_top = intersects("top")
        if line_top:
            ((_, y0), (_, y1)) = line_top
            self.min_y = max(self.min_y, int(np.ceil(max(y0, y1))))

        line_left = intersects("left")
        if line_left:
            ((x0, _), (x1, _)) = line_left
            self.min_x = max(self.min_x, int(np.ceil(max(x0, x1))))
        line_right = intersects("right")
        if line_right:
            ((x0, _), (x1, _)) = line_right
            self.max_x = min(self.max_x, int(np.floor(min(x0, x1))))

    @cached_property
    def mask(self):
        return self.make_binary_mask()

    @cached_property
    def contrast_enhanced_2(self):
        return self.make_contrast_enhanced_res256(sigma_fraction=0.02)

    @cached_property
    def contrast_enhanced_5(self):
        return self.make_contrast_enhanced_res256(sigma_fraction=0.05)

    @cached_property
    def sharpened_5(self):
        return self.make_contrast_enhanced_res256(
            sigma_fraction=0.05, contrast_factor=2, sharpen=True
        )

    @cached_property
    def contrast_enhanced_10(self):
        return self.make_contrast_enhanced_res256(sigma_fraction=0.1)

    @cached_property
    def mirrored_image(self):
        return self.make_mirrored_image()

    def make_contrast_enhanced_res256(
        self, image=None, sigma_fraction=0.05, contrast_factor=4, sharpen=False, mirror=True, mask=True
    ):
        image = self.image if image is None else image

        ce_resolution = 256
        T = self.get_cropping_transform(ce_resolution)
        bounds_warped = self.warp(T, image=image)
        image_warped = bounds_warped.mirrored_image / 255 if mirror else bounds_warped.image / 255
        sigma_warped = sigma_fraction * bounds_warped.radius
        blurred_warped = gaussian_filter(image_warped, (sigma_warped, sigma_warped, 0))
        blurred = T.warp_inverse(blurred_warped, self.hw)

        ce = unsharp_masking(image / 255, blurred, contrast_factor, sharpen)

        if mask:
            mask = self.make_binary_mask(0.01)
            ce = to_uint8(ce)
            ce[~mask] = 0

        return ce
    

    def make_binary_mask(self, shrink_ratio=0.01):
        """
        creates a binary image of the bounds (circle and rectangle)
        """
        _, _, r_squared_norm = self.get_coordinates(shrink_ratio)

        d = int(np.round(shrink_ratio * self.radius))

        mask = r_squared_norm < 1
        mask[: self.min_y + d] = False
        mask[self.max_y - d :] = False
        mask[:, : self.min_x + d] = False
        mask[:, self.max_x - d :] = False
        return mask

    @lru_cache(maxsize=1)
    def get_coordinates(self, shrink_ratio=0.01):
        dx = np.arange(self.hw[1])[None, :] - self.cx
        dy = np.arange(self.hw[0])[:, None] - self.cy

        r = (1 - shrink_ratio) * self.radius
        dx_norm = dx / r
        dy_norm = dy / r
        r_squared_norm = dx_norm**2 + dy_norm**2
        return dx, dy, r_squared_norm

    def make_mirrored_image(self, shrink_ratio=0.01):
        """
        mirrors pixels around the box and circle defined by bounds
        Can be used in combination with contrast_enhance to avoid the bright boundary around the rim
        """

        cy, cx = self.cy, self.cx
        h, w = self.hw

        mirrored_image = np.copy(self.image)

        # shrink by d pixels
        d = int(np.round(shrink_ratio * self.radius))
        min_y = self.min_y + d
        max_y = self.max_y - d
        min_x = self.min_x + d
        max_x = self.max_x - d
        # below min_y mirrored to above min_y
        mirrored_image[:min_y] = mirrored_image[2 * min_y - 1 : min_y - 1 : -1]
        # above max_y mirrored to below max_y
        mirrored_image[max_y:] = mirrored_image[max_y : 2 * max_y - h : -1]

        # left of min_x mirrored to right of min_x
        mirrored_image[:, :min_x] = mirrored_image[:, 2 * min_x - 1 : min_x - 1 : -1]
        # right of max_x mirrored to left of max_x
        mirrored_image[:, max_x:] = mirrored_image[:, max_x : 2 * max_x - w : -1]

        dx, dy, r_squared_norm = self.get_coordinates(shrink_ratio)

        # pixels outside the circle
        mask_outside = r_squared_norm > 1
        y0, x0 = np.where(mask_outside)

        # scale factor to be applied to reflect coordinates in circle outline
        scale = 1 / r_squared_norm[mask_outside]

        x1 = np.round(cx + dx[0, x0] * scale).astype(int)
        y1 = np.round(cy + dy[y0, 0] * scale).astype(int)
        x1 = np.clip(x1, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1)

        # assing pixel values outside the circle
        mirrored_image[y0, x0] = mirrored_image[y1, x1]

        return mirrored_image

    def contrast_enhance(self, sigma=None, contrast_factor=4):
        if sigma is None:
            sigma = 0.05 * self.radius
        image = self.mirrored_image / 255
        blurred = gaussian_filter(image, sigma=(sigma, sigma, 0))
        ce = unsharp_masking(image, blurred, contrast_factor)
        return to_uint8(ce)

    def get_cropping_transform(self, target_diameter, patch_size=None):
        if patch_size is None:
            patch_size = target_diameter

        scale = target_diameter / (2 * self.radius)
        in_size = self.hw
        center = self.cy, self.cx
        return get_affine_transform(in_size, patch_size, scale=scale, center=center)

    def warp(self, transform: ProjectiveTransform, image=None):
        if image is None:
            image = self.image
        cx_warped, cy_warped = transform.apply([[self.cx, self.cy]])[0]
        radius_warped = self.radius * transform.scale
        image_warped = transform.warp(image) if image is not None else None
        lines_warped = {k: transform.apply(v) for k, v in self.lines.items()}
        return CFIBounds(
            (cx_warped, cy_warped),
            radius_warped,
            lines_warped,
            hw=transform.out_size,
            image=image_warped,
        )

    def crop(self, target_diameter):
        T = self.get_cropping_transform(target_diameter)
        return T, self.warp(T)

    def _repr_markdown_(self):
        result = f"""
        #### CFIBounds:

        - Center: ({self.cx}, {self.cy})
        - Radius: {self.radius}
        - Top: {self.min_y}
        - Bottom: {self.max_y}
        - Left: {self.min_x}
        - Right: {self.max_x}
        """
        return result.strip()

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        ax.imshow(self.image)
        ax.scatter(self.cx, self.cy, c="w", s=2)
        ax.add_artist(
            plt.Circle((self.cx, self.cy), self.radius, fill=False, color="w")
        )
        for k in ["top", "bottom", "left", "right"]:
            if k in self.lines:
                p0, p1 = self.lines[k]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], c="w")

        ax.set_xlim(0, self.hw[1])
        ax.set_ylim(self.hw[0], 0)
        return fig, ax

    def make_bounds_image(self, ax=None, fig=None):
        import cv2

        im = self.image.copy()

        cv2.circle(
            im,
            (round(self.cx), round(self.cy)),
            radius=0,
            color=(255, 255, 255),
            thickness=-1,
        )
        cv2.circle(
            im,
            (round(self.cx), round(self.cy)),
            radius=round(self.radius),
            color=(255, 255, 255),
            thickness=2,
        )
        for k in ["top", "bottom", "left", "right"]:
            if k in self.lines:
                p0, p1 = self.lines[k]
                cv2.line(
                    im,
                    (round(p0[0]), round(p0[1])),
                    (round(p1[0]), round(p1[1])),
                    (255, 255, 255),
                    thickness=2,
                )

        return im

    def to_dict(self):
        return {
            "hw": self.hw,
            "center": (self.cx, self.cy),
            "radius": self.radius,
            "lines": {
                k: (v.tolist() if isinstance(v, np.ndarray) else list(v))
                for k, v in self.lines.items()
            },
        }

    def to_dict_all(self):
        return {
            "hw": self.hw,
            "center": (self.cx, self.cy),
            "radius": self.radius,
            "lines": {
                k: (v.tolist() if isinstance(v, np.ndarray) else list(v))
                for k, v in self.lines.items()
            },
            "min_y": self.min_y,
            "max_y": self.max_y,
            "min_x": self.min_x,
            "max_x": self.max_x,
        }

    @classmethod
    def from_dict(cls, image, d):
        return CFIBounds(
            (d["center"][0], d["center"][1]), d["radius"], d["lines"], image=image
        )


def line_circle_intersection(P0, P1, C, r):
    # Convert inputs to numpy arrays for vector operations
    P0, P1, C = np.array(P0), np.array(P1), np.array(C)

    # Define the line as a vector equation
    d = P1 - P0

    # Coefficients for the quadratic equation
    a = d.dot(d)
    b = 2 * d.dot(P0 - C)
    c = P0.dot(P0) + C.dot(C) - 2 * P0.dot(C) - r**2

    # Calculate the discriminant
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        # The line and circle do not intersect
        return []
    else:
        # The line and circle intersect at one or two points
        sqrt_discriminant = np.sqrt(discriminant)
        t = [(-b + sqrt_discriminant) / (2 * a), (-b - sqrt_discriminant) / (2 * a)]
        return [P0 + ti * d for ti in t]


def unsharp_masking(image, blurred, contrast_factor=4, sharpen=False):
    if sharpen:
        return np.clip(contrast_factor * (image - blurred) + image, 0, 1)
    else:
        return np.clip(contrast_factor * (image - blurred) + 0.5, 0, 1)
