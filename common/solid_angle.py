import numpy as np
from numpy import tan, arctan, arccos, sqrt


def get_solid_angles(img):
    """Computes the solid angle subtended by each pixel."""

    # Compute coordinates of pixel borders
    cols = np.linspace(0, 1, img.shape[1] + 1)
    rows = np.linspace(0, 1, img.shape[0] + 1)

    u, v = np.meshgrid(cols, rows)
    dx, dy, dz, _ = image2world(u, v)

    # Split each pixel into two triangles and compute the solid angle
    # subtended by the two tetrahedron
    a = np.vstack((dx[:-1, :-1].ravel(), dy[:-1, :-1].ravel(), dz[:-1, :-1].ravel()))
    b = np.vstack((dx[:-1, 1:].ravel(), dy[:-1, 1:].ravel(), dz[:-1, 1:].ravel()))
    c = np.vstack((dx[1:, :-1].ravel(), dy[1:, :-1].ravel(), dz[1:, :-1].ravel()))
    d = np.vstack((dx[1:, 1:].ravel(), dy[1:, 1:].ravel(), dz[1:, 1:].ravel()))
    omega = tetrahedron_solid_angle(a, b, c)
    omega += tetrahedron_solid_angle(b, c, d)

    # Get pixel center coordinates
    _, _, _, valid = world_coordinates(img)
    omega[~valid.ravel()] = np.nan

    angles = omega.reshape(img.shape[0:2])
    return angles


def image2world(u, v):
    u = u * 2

    # lat-long -> world
    thetaLatLong = np.pi * (u - 1)
    phiLatLong = np.pi * v

    x = np.sin(phiLatLong) * np.sin(thetaLatLong)
    y = np.cos(phiLatLong)
    z = -np.sin(phiLatLong) * np.cos(thetaLatLong)

    valid = np.ones(x.shape, dtype='bool')
    return x, y, z, valid


def image_coordinates(img):
    """Returns the (u, v) coordinates for each pixel center."""
    cols = np.linspace(0, 1, img.shape[1] * 2 + 1)
    rows = np.linspace(0, 1, img.shape[0] * 2 + 1)

    cols = cols[1::2]
    rows = rows[1::2]

    return [d.astype('float32') for d in np.meshgrid(cols, rows)]


def world_coordinates(img):
    """Returns the (x, y, z) world coordinates for each pixel center."""
    u, v = image_coordinates(img)
    x, y, z, valid = image2world(u, v)

    return x, y, z, valid


def tetrahedron_solid_angle(a, b, c, lhuillier=True):
    """ Computes the solid angle subtended by a tetrahedron.
       
          omega = tetrahedronSolidAngle(a, b, c)
       
        The tetrahedron is defined by three vectors (a, b, c) which define the
        vertices of the triangle with respect to an origin.
       
        For more details, see:
          http://en.wikipedia.org/wiki/Solid_angle#Tetrahedron
       
        Both methods are implemented, but L'Huillier (default) is easier to
        parallelize and thus much faster. 
       
        ----------
        Jean-Francois Lalonde
    """
    assert a.shape[0] == 3, 'a must be a 3xN matrix'
    assert b.shape[0] == 3, 'b must be a 3xN matrix'
    assert c.shape[0] == 3, 'c must be a 3xN matrix'

    if lhuillier:
        theta_a = arccos(np.sum(b * c, 0))
        theta_b = arccos(np.sum(a * c, 0))
        theta_c = arccos(np.sum(a * b, 0))

        theta_s = (theta_a + theta_b + theta_c) / 2

        product = tan(theta_s / 2) * tan((theta_s - theta_a) / 2) * \
                  tan((theta_s - theta_b) / 2) * tan((theta_s - theta_c) / 2)

        product[product < 0] = 0
        omega = 4 * arctan(sqrt(product))
    else:
        raise NotImplementedError()

    return omega
