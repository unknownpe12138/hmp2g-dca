"""
Pure Python fallback for cython_func.pyx
Used when Cython compilation is not available (e.g., on Windows without MSVC)
"""
import numpy as np

PI = np.pi

def reg_rad(rad):
    """Regularize angle to [-PI, PI]"""
    return (rad + PI) % (2 * PI) - PI

def laser_hit_improve3(pos_o, pos_t, fanRadius, fanOpenRad, fanDirRad):
    """
    Check if a target position is within a fan-shaped laser area.

    Args:
        pos_o: Origin position [x, y]
        pos_t: Target position [x, y]
        fanRadius: Radius of the fan
        fanOpenRad: Opening angle of the fan (radians)
        fanDirRad: Direction angle of the fan (radians)

    Returns:
        True if target is within the fan, False otherwise
    """
    delta = pos_t - pos_o
    dis_square = delta[0]*delta[0] + delta[1]*delta[1]

    if dis_square > fanRadius*fanRadius:
        return False

    ori_rad_pos = fanDirRad + fanOpenRad/2
    ori_rad_neg = fanDirRad - fanOpenRad/2

    ori_2tgt = np.arctan2(delta[1], delta[0])

    d1rad = abs(reg_rad(ori_rad_pos - ori_2tgt))
    d2rad = abs(reg_rad(ori_rad_neg - ori_2tgt))

    if d1rad <= fanOpenRad and d2rad <= fanOpenRad:
        return True
    else:
        return False
