import cv2
import numpy as np


def triangulate(P_XY, P_XZ, P_YZ, point_xy, point_zx, point_zy):
    if point_xy is not None and point_zx is not None and point_zy is not None:
        coord_xy_xz = linear_ls_triangulation(P_XY, P_XZ, point_xy, point_zx)
        coord_xy_yz = linear_ls_triangulation(P_XY, P_YZ, point_xy, point_zy)
        coord_xz_yz = linear_ls_triangulation(P_XZ, P_YZ, point_zx, point_zy)
        return np.mean(np.array([coord_xy_xz[0], coord_xy_yz[0], coord_xz_yz[0]]), axis=0)
    elif point_xy is None and point_zx is not None and point_zy is not None:
        coord_xz_yz = linear_ls_triangulation(P_XZ, P_YZ, point_zx, point_zy)
        return coord_xz_yz[0]
    elif point_xy is not None and point_zx is None and point_zy is not None:
        coord_xy_yz = linear_ls_triangulation(P_XY, P_YZ, point_xy, point_zy)
        return coord_xy_yz[0]
    elif point_xy is not None and point_zx is not None and point_zy is None:
        coord_xy_xz = linear_ls_triangulation(P_XY, P_XZ, point_xy, point_zx)
        return coord_xy_xz[0]
    else:
        return None


def linear_ls_triangulation(P1, P2, u1, u2):
    """Triangulation via Linear-LS method"""
    # build A matrix for homogeneous equation system Ax=0
    # assume X = (x,y,z,1) for Linear-LS method
    # which turns it into AX=B system, where A is 4x3, X is 3x1 & B is 4x1
    A = np.array([u1[0] * P1[2, 0] - P1[0, 0], u1[0] * P1[2, 1] - P1[0, 1],
                  u1[0] * P1[2, 2] - P1[0, 2], u1[1] * P1[2, 0] - P1[1, 0],
                  u1[1] * P1[2, 1] - P1[1, 1], u1[1] * P1[2, 2] - P1[1, 2],
                  u2[0] * P2[2, 0] - P2[0, 0], u2[0] * P2[2, 1] - P2[0, 1],
                  u2[0] * P2[2, 2] - P2[0, 2], u2[1] * P2[2, 0] - P2[1, 0],
                  u2[1] * P2[2, 1] - P2[1, 1],
                  u2[1] * P2[2, 2] - P2[1, 2]]).reshape(4, 3)
    B = np.array([-(u1[0] * P1[2, 3] - P1[0, 3]),
                  -(u1[1] * P1[2, 3] - P1[1, 3]),
                  -(u2[0] * P2[2, 3] - P2[0, 3]),
                  -(u2[1] * P2[2, 3] - P2[1, 3])]).reshape(4, 1)
    ret, X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
    return X.reshape(1, 3)

