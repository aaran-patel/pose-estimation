"""
floor_plane.py

Floor plane estimation via RANSAC, using only numpy (no open3d dependency).
Run once at node startup (or on-demand) to find the ground plane from the
depth stream, then use it to compute real height-above-floor for any 3D
point -- a much stronger "is this person on the ground" signal than torso
angle alone, since it's directly measuring what we actually care about
instead of inferring it from orientation.

Assumption: the floor is the single largest flat surface visible in the
depth frame. Holds for a mostly-clear greenhouse walkway; would need
re-checking if the camera's view is dominated by a large flat bench/table
instead of floor.
"""

import numpy as np
import pyrealsense2 as rs


def sample_point_cloud(depth_frame, intrinsics, step=10, max_points=3000):
    """
    Deproject a sampled grid of pixels across the whole frame into 3D
    points. step=10 means every 10th pixel in x and y -- coarse enough to
    be fast, dense enough for a reliable plane fit.
    """
    h, w = depth_frame.get_height(), depth_frame.get_width()
    points = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            d = depth_frame.get_distance(x, y)
            if d > 0:
                point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                points.append(point)
            if len(points) >= max_points:
                return np.array(points)
    return np.array(points)


def fit_plane_ransac(points, n_iterations=200, distance_threshold=0.02, min_inlier_ratio=0.3):
    """
    Basic RANSAC plane fit. Returns (normal, d) such that for a point p on
    the plane: dot(normal, p) + d = 0. Returns None if no plane with
    enough inliers was found (e.g. floor not sufficiently visible).

    normal is oriented so that dot(normal, p) + d > 0 means "above the
    floor" for points captured by a camera looking roughly downward/level
    -- verified/flipped using the camera's own Y-axis as a rough prior.
    """
    if len(points) < 50:
        return None

    best_inliers = 0
    best_plane = None

    rng = np.random.default_rng()

    for _ in range(n_iterations):
        sample_idx = rng.choice(len(points), size=3, replace=False)
        p1, p2, p3 = points[sample_idx]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue  # degenerate (collinear) sample
        normal = normal / norm_len
        d = -np.dot(normal, p1)

        distances = np.abs(points @ normal + d)
        inliers = np.sum(distances < distance_threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (normal, d)

    if best_plane is None or best_inliers / len(points) < min_inlier_ratio:
        return None

    normal, d = best_plane

    # Orient normal so it points "up" (away from camera's rough downward
    # view direction) -- camera Y-axis points down in RealSense convention,
    # so floor normal should have a negative Y component pointing back
    # toward the camera/up. Flip if needed so height math is consistent.
    if normal[1] > 0:
        normal = -normal
        d = -d

    return normal, d


def height_above_floor(point_3d, plane):
    """
    Signed distance from a 3D point to the floor plane. Positive = above
    the floor, by however many meters. plane is (normal, d) from
    fit_plane_ransac.
    """
    if point_3d is None or plane is None:
        return None
    normal, d = plane
    return float(np.dot(normal, np.array(point_3d)) + d)
