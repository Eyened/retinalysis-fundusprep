import numpy as np

# Constants
NUM_ITERATIONS = 1000
MAX_ATTEMPTS = 100


def circle_fit(pts):
    """
    Fit a circle to a set of points.

    Parameters:
    pts (np.array): 2D array of points.

    Returns:
    float, np.array: Radius and center of the fitted circle.
    """
    if len(pts) == 0:
        raise ValueError("No points provided")

    B = np.ones((len(pts), 3))
    B[:, :-1] = pts
    d = (pts**2).sum(axis=-1)
    y, *_ = np.linalg.lstsq(B, d, rcond=None)
    center = 0.5 * y[:2]
    radius = np.sqrt(y[-1] + (center**2).sum())
    return radius, center


def abs_dist(pts, center, radius):
    return np.abs(np.sqrt(np.sum((pts - center) ** 2, axis=-1)) - radius)


def find_circle(
    pts_x,
    pts_y,
    min_radius,
    max_radius,
    inlier_dist_threshold,
    min_fraction=0.2,
    random_state=42,
):
    """
    Find a circle in a set of points using RANSAC.

    Parameters:
    pts_x, pts_y (np.array): 1D arrays of x and y coordinates.

    Returns:
    float, np.array: Radius and center of the found circle.
    """

    pts = np.array([pts_x, pts_y]).T
    n = len(pts)
    best_inliers = None
    n_best_inliers = 0
    rng = np.random.default_rng(seed=random_state)
    for _ in range(NUM_ITERATIONS):
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            random_indices = rng.choice(n, 3, replace=False)
            sampled_points = pts[random_indices]

            radius, center = circle_fit(sampled_points)
            if min_radius < radius and radius < max_radius:
                break
            attempts += 1

        distances = abs_dist(pts, center, radius)
        inliers = distances < inlier_dist_threshold

        n_inliers = np.sum(inliers)

        if best_inliers is None or n_inliers > n_best_inliers:
            best_inliers = inliers
            n_best_inliers = n_inliers

        if n_inliers > n / 2:
            break

    if n_best_inliers < min_fraction * n:
        raise ValueError("No circle found")

    radius, center = circle_fit(pts[best_inliers])

    # refine once more to include all inliers
    distances = abs_dist(pts, center, radius)
    inliers = distances < inlier_dist_threshold

    radius, center = circle_fit(pts[best_inliers])
    return radius, center, inliers
