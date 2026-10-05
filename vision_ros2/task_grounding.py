"""Conservative RGB-D grounding. Scores from language models are not probabilities."""
import numpy as np


def ground_box(box, shape, k, rotation, translation, depth=None,
               min_depth=.2, max_depth=10., min_samples=20, min_fraction=.6,
               max_relative_spread=.15):
    box = np.asarray(box, dtype=float)
    k = np.asarray(k, dtype=float).reshape(3, 3)
    if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box < 0)
            or np.any(box > 1) or box[2] <= box[0] or box[3] <= box[1]
            or not np.isfinite(k).all() or k[0,0] <= 0 or k[1,1] <= 0):
        raise ValueError('Invalid normalized box or intrinsics')
    h,w = shape[:2]
    u,v = (box[:2]+box[2:])/2 * [w,h]
    ray = np.array([(u-k[0,2])/k[0,0], (v-k[1,2])/k[1,1], 1.])
    direction = rotation @ ray
    direction /= np.linalg.norm(direction)
    result = dict(state='bearing_only', origin=np.asarray(translation).tolist(),
                  direction=direction.tolist(), bbox=box.tolist())
    if depth is None or depth.shape != (h,w):
        result['depth_reason'] = 'no_registered_synchronized_depth'
        return result
    # Interior sampling avoids box borders, but does not prove instance segmentation.
    lo,hi = box[:2]+.2*(box[2:]-box[:2]), box[2:]-.2*(box[2:]-box[:2])
    x0,y0 = np.floor(lo*[w,h]).astype(int)
    x1,y1 = np.ceil(hi*[w,h]).astype(int)
    patch = depth[y0:y1,x0:x1]
    samples = patch[np.isfinite(patch) & (patch > min_depth) & (patch < max_depth)]
    if samples.size < min_samples or samples.size/max(1,patch.size) < min_fraction:
        result['depth_reason'] = 'insufficient_depth_support'
        return result
    z = float(np.median(samples))
    spread = float(np.percentile(samples,90)-np.percentile(samples,10))
    if spread / z > max_relative_spread:
        result['depth_reason'] = 'mixed_depth_surfaces'
        return result
    result.update(state='localized_unverified', position=(rotation @ (ray*z)+translation).tolist(),
                  depth_m=z, depth_spread_m=spread, depth_samples=int(samples.size),
                  depth_reason='coherent_interior_depth')
    return result


def rotation_matrix(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or norm < 1e-6:
        raise ValueError('Invalid camera rotation')
    x,y,z,w = q/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def ground_direction(sector, shape, k, rotation, translation):
    """Project an image third into a bearing cone; never estimate target depth."""
    thirds = {'left': (0., 1/3), 'center': (1/3, 2/3), 'right': (2/3, 1.)}
    if sector not in thirds:
        raise ValueError('No usable image direction')
    h,w = shape[:2]
    k = np.asarray(k, float).reshape(3,3)
    if not np.isfinite(k).all() or k[0,0] <= 0 or k[1,1] <= 0:
        raise ValueError('Invalid camera intrinsics')
    lo,hi = thirds[sector]
    rays = []
    for u in (lo*w, (lo+hi)*w/2, hi*w):
        ray = rotation @ np.array([(u-k[0,2])/k[0,0], (h/2-k[1,2])/k[1,1], 1.])
        if np.linalg.norm(ray[:2]) < 1e-6:
            raise ValueError('View is not horizontally directed')
        rays.append(ray/np.linalg.norm(ray))
    center = np.arctan2(rays[1][1],rays[1][0])
    half = max(abs(np.arctan2(np.sin(np.arctan2(r[1],r[0])-center),
                             np.cos(np.arctan2(r[1],r[0])-center))) for r in (rays[0],rays[2]))
    return dict(state='bearing_only', origin=np.asarray(translation).tolist(),
                direction=rays[1].tolist(), bearing_half_angle_rad=float(half),
                image_direction=sector, depth_reason='coarse_direction_no_target_range')
