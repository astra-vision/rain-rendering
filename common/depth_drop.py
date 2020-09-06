import numpy as np

R_rect, R_rect_4x4, P2, P2_R_rect_inv = None, None, None, None
cam_pos_cr = None
cam_pos_w = None
world_pos_cr = None


def read_calib_data(fpath):
    global P2, R_rect, world_pos_cr, P2_R_rect_inv, cam_pos_cr, cam_pos_w, world_pos_cr, R_rect_4x4
    with open(fpath, "r") as camfile:
        for line in camfile:
            if line[0:3] == "P2:":
                P2_list = line[3:].strip().split(" ")
                P2 = np.array([float(i) for i in P2_list]).reshape((3, 4))
            elif line[0:6] == "R_rect":
                R_rect_list = line[6:].strip().split(" ")
                R_rect = np.array([float(i) for i in R_rect_list]).reshape((3, 3))

    R_rect_4x4 = np.identity(4, R_rect.dtype)
    R_rect_4x4[0:3, 0:3] = R_rect

    # Ground position in camera reference coordinates
    world_pos_cr = np.array([0., 1.65, 0.0]).reshape((3, 1))
    camref_pos_w = -world_pos_cr

    # Compute the camera position to the reference camera (cf. Geiger et al. 2013)
    cam_pos_cr = np.zeros((3, 1))
    cam_pos_cr[0] = P2[0, 3] / (-P2[0, 0])
    cam_pos_w = cam_pos_cr + camref_pos_w

    # [WARNING]
    # NOTE THAT HERE we assume no rotation and same coordinates system orientation in world, cam ref, and cam
    # This is NOT true and will impact slightly the precision

    P2_R_rect = np.dot(P2, R_rect_4x4)
    P2_R_rect_inv = np.linalg.pinv(P2_R_rect)
    if np.allclose(P2_R_rect, np.dot(P2_R_rect, np.dot(P2_R_rect_inv, P2_R_rect))):
        print("Pseudo inverse matrix [OK]")
        print("Distance ", np.sum(np.power(P2_R_rect - np.dot(P2_R_rect, np.dot(P2_R_rect_inv, P2_R_rect)), 2)))
    else:
        Exception("Pseudo inverse matrix [ERROR]")


# Return origin and ray direction in world coordinate from a pixel in image space coordinates
def point_to_cam_ray(u, v):
    pt = np.zeros((3, 1))

    pt[0, 0] = u
    pt[1, 0] = v
    pt[2, 0] = 1.0

    # Do the inverse projection (Result is expressed in camRef coordinates)
    vec = np.dot(P2_R_rect_inv, pt)
    vec /= vec[3]  # Homogenize the coordinate

    # Remove the translation due to cam>CamRef as in the case of a vector this is no more valid
    vec[0:3] -= cam_pos_cr

    # Extract the direction 0:3, and normalize
    d = vec[0:3]
    d /= np.linalg.norm(d)

    return cam_pos_w, d


# Project a 3D point x to an image coordinate
# We use same notation as Geiger et al. 2013
def project_point_from_ref_cam_coords(x_rf):
    y = np.dot(np.dot(P2, R_rect_4x4), x_rf)
    return y


def project_point_from_world_coords(x_w):
    # Convert from World to Ref Cam coords (assume no rotation)
    x_cr = x_w.copy()
    x_cr[0:3, 0] += world_pos_cr[:, 0]

    return project_point_from_ref_cam_coords(x_cr)


def invproject_point_on_ground(u, v):
    O, d = point_to_cam_ray(u, v)

    return ray_ground_intersection(O, d)


# Ray to Ground intersection (O the origin of the ray, d the direction) in world coordinates
def ray_ground_intersection(O, d):
    # Adapted from source:
    # http://math.stackexchange.com/questions/100439/determine-where-a-vector-will-intersect-a-plane
    # Implementation seems correct, has been verified with:
    # http://www.ambrsoft.com/TrigoCalc/Plan3D/PlaneLineIntersection_.htm
    #
    # A, B, C three distinct points on the plane
    A, B, C = np.array([1., 0., 0.]), np.array([0., 0., 0.]), np.array([0., 0., 1.])

    # N the normal of the plane
    N = np.cross(A - B, B - C)

    # Compute the intersection of vector d leaving from point O to the plane formed having normal N
    t = -1 * (N[0] * (O[0] - A[0]) + N[1] * (O[1] - A[1]) + N[2] * (O[2] - A[2])) / (
                N[0] * d[0] + N[1] * d[1] + N[2] * d[2])

    # Intersection point is O + dt
    pt = O + d * t
    return pt
