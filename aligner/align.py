
import cv2
import numpy as np
from PIL import Image
from .transforms import get_affine_transform_matrix
from .transforms import get_similarity_transform_for_cv2


def get_reference_facial_5pts(output_size=112, default_square=True):

    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if isinstance(output_size, (list, tuple)):
        output_size = np.array(output_size)

    default_facial_5pts = np.array([
        [30.29459953,  51.69630051],
        [65.53179932,  51.50139999],
        [48.02519989,  71.73660278],
        [33.54930115,  92.3655014],
        [62.72990036,  92.20410156]
    ])

    default_size = np.array((96, 112))

    if default_square:
        size_diff = max(default_size) - default_size
        default_facial_5pts += size_diff / 2
        default_size += size_diff

    scale = output_size / default_size
    return scale * default_facial_5pts


def warp_and_crop_face(src_img, src_pts, ref_pts, crop_size=(96, 112), align_type='smilarity'):

    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        raise Exception('ref_pts.shape must be (K,2) or (2,K) and K>2')

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        raise Exception('src_pts.shape must be (K,2) or (2,K) and K>2')

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

    if src_pts.shape != ref_pts.shape:
        raise Exception('src_pts and ref_pts must have the same shape')

    if align_type is 'cv2_affine':
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
    elif align_type is 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
    else:
        tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)

    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    return face_img


def align_face(img, facial_pts, crop_size=112):
    """
    Args:
        img: input PIL image or numpy array
        landmarks: list of tuple (x, y), 5 facial points or numpy array
    """
    if not isinstance(facial_pts, np.ndarray):
        facial_pts = np.array(facial_pts)
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    ref_pts = get_reference_facial_5pts(crop_size)
    warped_face = warp_and_crop_face(np.array(img), facial_pts, ref_pts, crop_size)
    if not isinstance(img, np.ndarray):
        warped_face = Image.fromarray(warped_face)

    return warped_face
