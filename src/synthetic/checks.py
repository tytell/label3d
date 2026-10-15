"""Sanity checks: triangulation, epipolar geometry, reprojection, aniposelib round-trip."""

from pathlib import Path

import cv2
import numpy as np

from synthetic.projection import project_points


def _undistort_points(pts_2d: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Undistort Nx2 pixel coordinates, returning Nx2 undistorted pixel coordinates."""
    pts = pts_2d.reshape(-1, 1, 2).astype(np.float64)
    # P=K maps back to pixel coordinates after undistortion
    undistorted = cv2.undistortPoints(pts, K, dist, P=K)
    return undistorted.reshape(-1, 2)


def _dlt_triangulate(cameras: dict, pixel_coords: np.ndarray) -> np.ndarray:
    """DLT triangulation from N cameras.

    pixel_coords: (n_cams, 2) — undistorted pixel coordinates.
    """
    A = []
    for ci, cam in enumerate(cameras.values()):
        P = cam["K"] @ np.hstack([cam["R"], cam["t"].reshape(3, 1)])  # 3×4
        u, v = pixel_coords[ci]
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def _epipolar_error(cam_a: dict, cam_b: dict, pt_a: np.ndarray, pt_b: np.ndarray) -> float:
    """Symmetric epipolar distance (pixels) between corresponding points.

    Points should be undistorted before calling this function.
    """
    Pa = cam_a["K"] @ np.hstack([cam_a["R"], cam_a["t"].reshape(3, 1)])
    Pb = cam_b["K"] @ np.hstack([cam_b["R"], cam_b["t"].reshape(3, 1)])

    C_a = -cam_a["R"].T @ cam_a["t"]

    # Epipole e_b = project C_a into cam_b
    e_b = cam_b["K"] @ (cam_b["R"] @ C_a + cam_b["t"])
    e_b /= e_b[2]

    def skew(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    Pa_pinv = np.linalg.pinv(Pa)
    F = skew(e_b) @ Pb @ Pa_pinv

    x_a = np.array([pt_a[0], pt_a[1], 1.0])
    x_b = np.array([pt_b[0], pt_b[1], 1.0])

    l_b = F @ x_a
    dist_b = abs(x_b @ l_b) / np.sqrt(l_b[0] ** 2 + l_b[1] ** 2)

    l_a = F.T @ x_b
    dist_a = abs(x_a @ l_a) / np.sqrt(l_a[0] ** 2 + l_a[1] ** 2)

    return float((dist_a + dist_b) / 2)


def run_sanity_checks(
    cameras: dict,
    all_pts_3d: list[np.ndarray],
    all_projected: dict[str, list[np.ndarray]],
    node_names: list[str],
    calibration_path: Path,
):
    cam_list = list(cameras.values())
    cam_names = list(cameras.keys())
    n_cams = len(cam_list)
    n_frames = len(all_pts_3d)

    # Pre-compute undistorted projections for checks that need the pinhole model
    all_undistorted = {}
    for cn in cam_names:
        cam = cameras[cn]
        undist_frames = []
        for fi in range(n_frames):
            undist = _undistort_points(all_projected[cn][fi], cam["K"], cam["dist"])
            undist_frames.append(undist)
        all_undistorted[cn] = undist_frames

    # 1. Triangulation: recover ground-truth 3D from undistorted projected pixels
    print("\nTriangulation sanity check (all frames)...")
    max_tri_err = 0.0
    for fi in range(n_frames):
        pixel_coords = np.array([all_undistorted[cn][fi] for cn in cam_names])
        for bi, bp in enumerate(node_names):
            X = _dlt_triangulate(cameras, pixel_coords[:, bi, :])
            gt = all_pts_3d[fi][bi]
            err = float(np.linalg.norm(X - gt))
            max_tri_err = max(max_tri_err, err)
            if err > 1e-4:
                print(f"  WARNING frame {fi} {bp}: triangulation error {err:.6f} cm")
    print(f"  Max triangulation error across all frames/bodyparts: {max_tri_err:.2e} cm  {'✓' if max_tri_err < 1e-4 else '✗'}")

    # 2. Epipolar geometry: undistorted point pairs should lie on each other's epipolar lines
    if n_cams >= 2:
        print("\nEpipolar geometry check (frame 0, all camera pairs)...")
        max_epi_err = 0.0
        for i in range(n_cams):
            for j in range(i + 1, n_cams):
                for bi, bp in enumerate(node_names):
                    pt_i = all_undistorted[cam_names[i]][0][bi]
                    pt_j = all_undistorted[cam_names[j]][0][bi]
                    err = _epipolar_error(cam_list[i], cam_list[j], pt_i, pt_j)
                    max_epi_err = max(max_epi_err, err)
                    if err > 1e-4:
                        print(f"  WARNING {cam_names[i]}-{cam_names[j]} {bp}: epipolar dist {err:.4f} px")
        print(f"  Max symmetric epipolar distance: {max_epi_err:.2e} px  {'✓' if max_epi_err < 1e-4 else '✗'}")

    # 3. Reprojection: project GT 3D back and compare to stored 2D (with distortion)
    print("\nReprojection sanity check...")
    max_reproj_err = 0.0
    for cam_name, cam in cameras.items():
        for fi in range(n_frames):
            reproj = project_points(all_pts_3d[fi], cam["K"], cam["R"], cam["t"], cam["dist"])
            stored = all_projected[cam_name][fi]
            err = float(np.max(np.linalg.norm(reproj - stored, axis=1)))
            max_reproj_err = max(max_reproj_err, err)
    print(f"  Max reprojection error: {max_reproj_err:.2e} px  {'✓' if max_reproj_err < 1e-6 else '✗'}")

    # 4. Aniposelib round-trip (if installed)
    try:
        from aniposelib.cameras import CameraGroup  # noqa: PLC0415
        print("\nAniposelib round-trip check...")
        cgroup = CameraGroup.load(str(calibration_path))
        for i, (cam_name, cam) in enumerate(cameras.items()):
            loaded_K = cgroup.cameras[i].get_camera_matrix()
            err_K = float(np.max(np.abs(loaded_K - cam["K"])))
            loaded_rvec = cgroup.cameras[i].get_rotation()
            err_rvec = float(np.max(np.abs(loaded_rvec - cam["rvec"])))
            status = "✓" if err_K < 1e-6 and err_rvec < 1e-6 else "✗"
            print(f"  {cam_name}: K_err={err_K:.2e}, rvec_err={err_rvec:.2e}  {status}")
    except ImportError:
        pass  # aniposelib not installed, skip
