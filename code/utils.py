import numpy as np

default_render_params = {
    "width": 640,
    "height": 486,

    "gamma": 0,
    "theta": 0,
    "phi": 0,
    "alpha": 0,
    "t2d": np.array([[0, 0]]).T,
    "camera_pos": np.array([[0, 0, 3400]]).T,
    "scale_scene": 0.,
    "object_size": 0.615 * 512,
    "shift_object": np.array([[0, 0, -46125]]).T,
    "shift_world": np.array([[0, 0, 0]]).T,
    "scale": 0.001,

    "ac_g": np.array([[1, 1, 1]]).T,
    "ac_c": 1,
    "ac_o": np.array([[0, 0, 0]]).T,
    "ambient_col": 0.6 * np.ones((3, 1)),

    "rotm": np.eye(3),
    "use_rotm": False,

    "do_remap": False,
    "dir_light": np.array([]),

    "do_specular": 0.1,
    "phong_exp": 8,
    "specular": 0.1 * 255,

    "do_cast_shadows": True,
    "sbufsize": 200,

    "proj": "perspective",

    "f": 6000,

    "n_chan": 3,
    "backface_culling": 2,

    # Can be 'phong', 'global_illum', 'no_illum'
    "illum_method": 'phong',

    "global_illum": {
        "brdf": 'lambert',
        "envmap": {},
        "light_probe": np.array([])
    },

    "ablend": []
}

class InvalidArgumentError(Exception):
    pass

def create_render_params(**kwargs):

    new_params = dict(default_render_params)

    for k, v in kwargs.items():
        new_params[k] = v

    return new_params


def coef2object(coefs, mu, pc, ev, MM=None, MB=None):
    """
    Based on coef2object.m provided by Sami Romdhani
    in the Univesity of Basel MFM Matlab package.

    Returns a point in the vector space defined by the 
    eigenbasis derived from the principal components and
    their eigenvalues + the mean vector.

    Namely, this function returns (assuming no blending):

    S = mu + \sum_{i = 1}^N coefs_i * ev_i * pc_i

    Parameters:

    coefs - K x 1 float: coefficients of the linear combination
    mu - N x 1 float: mean face vector
    pc - N x K float: orthonormal eigenbasis
    ev - K x 1 float: eigenvalues belonging to the appropriate
                    basis vector

    TODO: implement blending
    """

    n_dim, n_seg = coefs.shape

    seg_ones = np.ones((1, n_seg))

    obj = np.matmul(mu, seg_ones)
    obj = obj + np.matmul(pc[:, :n_dim],
                          coefs * (np.matmul(ev[:n_dim],
                                            seg_ones)))
    return obj
