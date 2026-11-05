import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import spatial

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
import tqdm


def get_gaussian_kernel1d(
    kernel_size,
    sigma,
    device,
):
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def gaussian_blur_2d_separable(
    image,
    sigma,
    device,
):
    ky_size = int(2 * math.ceil(4 * sigma[0]) + 1)
    kx_size = int(2 * math.ceil(4 * sigma[1]) + 1)

    pad_y = ky_size // 2
    pad_x = kx_size // 2

    # Create 1D kernels
    ky = get_gaussian_kernel1d(ky_size, sigma[0], device).reshape(1, 1, -1, 1)
    kx = get_gaussian_kernel1d(kx_size, sigma[1], device).reshape(1, 1, 1, -1)

    # Vertical blur
    image = F.conv2d(image, ky, padding=(pad_y, 0))
    # Horizontal blur
    image = F.conv2d(image, kx, padding=(0, pad_x))

    return image


def _prep_sigmas(
    min_sigma,
    max_sigma,
    sigma_ratio,
):
    # if both min and max sigma are scalar, function returns only one sigma
    scalar_max = np.isscalar(max_sigma)
    scalar_min = np.isscalar(min_sigma)
    scalar_sigma = scalar_max and scalar_min

    if scalar_max:
        max_sigma = (max_sigma,) * 2
    if scalar_min:
        min_sigma = (min_sigma,) * 2

    log_ratio = math.log(sigma_ratio)
    k = sum(
        math.log(max_s / min_s) / log_ratio + 1
        for max_s, min_s in zip(max_sigma, min_sigma)
    )
    k /= len(min_sigma)
    k = int(k)

    ratio_powers = tuple(sigma_ratio**i for i in range(k + 1))
    sigma_list = tuple(tuple(s * p for s in min_sigma) for p in ratio_powers)

    return sigma_list, scalar_sigma, k


def _compute_disk_overlap(
    d,
    r1,
    r2,
):
    ratio1 = (d**2 + r1**2 - r2**2) / (2 * d * r1)
    ratio1 = torch.clip(ratio1, -1, 1)
    acos1 = torch.arccos(ratio1)

    ratio2 = (d**2 + r2**2 - r1**2) / (2 * d * r2)
    ratio2 = torch.clip(ratio2, -1, 1)
    acos2 = torch.arccos(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = r1**2 * acos1 + r2**2 * acos2 - 0.5 * torch.sqrt(torch.abs(a * b * c * d))
    return area / (math.pi * (torch.minimum(r1, r2) ** 2))


def blob_overlap(
    blob1,
    blob2,
    *,
    sigma_dim=1,
):
    ndim = blob1.shape[-1] - sigma_dim
    if ndim > 3:
        return np.zeros((blob1.shape[0],))
    root_ndim = math.sqrt(ndim)
    blob1_mask = blob1[:, -1] > blob2[:, -1]
    blob2_mask = blob1[:, -1] <= blob2[:, -1]
    max_sigma = torch.ones((blob1.shape[0], 1)).to(blob1.device)

    max_sigma[blob1_mask] = blob1[:, -sigma_dim:][blob1_mask]
    max_sigma[blob2_mask] = blob2[:, -sigma_dim:][blob2_mask]
    r1 = torch.ones((blob1.shape[0])).to(blob1.device)
    r1[blob2_mask] = (blob1[:, -1] / blob2[:, -1])[blob2_mask]
    r2 = torch.ones((blob1.shape[0])).to(blob1.device)
    r2[blob1_mask] = (blob2[:, -1] / blob1[:, -1])[blob1_mask]

    pos1 = blob1[:, :ndim] / (max_sigma * root_ndim)
    pos2 = blob2[:, :ndim] / (max_sigma * root_ndim)

    d = torch.sqrt(torch.sum((pos2 - pos1) ** 2, dim=-1))

    output = _compute_disk_overlap(d, r1, r2)
    output[d > r1 + r2] = 0
    output[d <= abs(r1 - r2)] = 1
    output[(blob1[:, -1] == 0) * (blob2[:, -1] == 0)] = 0

    return output


def _prune_blobs(
    blobs_array,
    overlap,
    *,
    sigma_dim=1,
):
    sigma = blobs_array[:, -sigma_dim:].max()
    distance = 2 * sigma * math.sqrt(blobs_array.shape[1] - sigma_dim)

    tree = spatial.cKDTree(blobs_array[:, :-sigma_dim].cpu().data.numpy())

    pairs = torch.from_numpy(tree.query_pairs(distance, output_type="ndarray")).to(
        blobs_array.device
    )
    if len(pairs) == 0:
        return blobs_array[:, -1] != 0
    else:
        blob1, blob2 = blobs_array[pairs[:, 0], :], blobs_array[pairs[:, 1], :]
        mask = blob_overlap(blob1, blob2, sigma_dim=sigma_dim) > overlap
        blobs_array[
            torch.unique(pairs[:, 0][mask * (blob2[:, -1] > blob1[:, -1])]), -1
        ] = 0
        blobs_array[
            torch.unique(pairs[:, 1][mask * (blob1[:, -1] > blob2[:, -1])]), -1
        ] = 0

    return blobs_array[:, -1] != 0


def _exclude_border(
    label,
    border_width,
):
    """Set label border values to 0."""
    # zero out label borders
    for i, width in enumerate(border_width):
        if width == 0:
            continue
        label[(slice(None),) * i + (slice(None, width),)] = 0
        label[(slice(None),) * i + (slice(-width, None),)] = 0
    return label


def _get_peak_mask(
    image_1,
    image_2,
    segment,
    footprint,
):
    image_max_1 = F.max_pool2d(
        image_1.max(-1)[0],
        kernel_size=footprint.shape,
        stride=1,
        padding=tuple(np.asarray(footprint.shape) // 2),
    )

    image_max_2 = F.max_pool2d(
        image_2.max(-1)[0],
        kernel_size=footprint.shape,
        stride=1,
        padding=tuple(np.asarray(footprint.shape) // 2),
    )

    out_1 = image_1 == image_max_1[..., None]
    out_2 = image_2 == image_max_2[..., None]

    # no peak for a trivial image
    if torch.all(out_1):  # synchronize
        out_1[:] = False
    if torch.all(out_2):  # synchronize
        out_2[:] = False

    threshold = torch.quantile(
        torch.maximum(image_1, image_2)[
            segment[..., None].expand(-1, -1, -1, -1, 2)
        ].abs()[::25],
        0.5,
    )
    # print(threshold)
    # threshold = max(image.abs().max()*0.5, image.abs().min())
    out_1 &= image_1.abs() > threshold
    out_2 &= image_2.abs() > threshold

    return out_1 * segment[..., None], out_2 * segment[..., None]


def peak_local_max(
    image_1,
    image_2,
    segment=None,
    footprint=None,
):
    border_width = (0, 0, 41, 41, 0)

    mask_1, mask_2 = _get_peak_mask(image_1, image_2, segment, footprint)
    mask_1 = _exclude_border(mask_1, border_width)
    mask_2 = _exclude_border(mask_2, border_width)
    coordinates_1 = torch.nonzero(mask_1[:, 0, :, :])
    coordinates_2 = torch.nonzero(mask_2[:, 0, :, :])

    return coordinates_1, coordinates_2


def blob_dog(
    image_1,
    image_2,
    th,
    min_sigma=1,
    max_sigma=50,
    sigma_ratio=1.6,
    overlap=0.5,
    *,
    device="cuda",
):
    image_1 = image_1.astype(np.float32)
    image_2 = image_2.astype(np.float32)
    image_1 = torch.from_numpy(image_1).to(device)
    image_2 = torch.from_numpy(image_2).to(device)
    sigma_list, scalar_sigma, k = _prep_sigmas(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        sigma_ratio=sigma_ratio,
    )

    dog_image_cube_1 = torch.empty(
        image_1.shape + (k,),
        dtype=torch.float,
        device=device,
    )
    dog_image_cube_2 = torch.empty(
        image_2.shape + (k,),
        dtype=torch.float,
        device=device,
    )

    gaussian_previous_1 = gaussian_blur_2d_separable(
        image_1,
        tuple(sigma_list[0]),
        device=device,
    )

    gaussian_previous_2 = gaussian_blur_2d_separable(
        image_2,
        tuple(sigma_list[0]),
        device=device,
    )

    for i, s in enumerate(sigma_list[1:]):
        gaussian_current_1 = gaussian_blur_2d_separable(
            image_1,
            tuple(s),
            device=device,
        )
        gaussian_current_2 = gaussian_blur_2d_separable(
            image_2,
            tuple(s),
            device=device,
        )

        dog_image_cube_1[..., i] = gaussian_previous_1 - gaussian_current_1
        gaussian_previous_1 = gaussian_current_1

        dog_image_cube_2[..., i] = gaussian_previous_2 - gaussian_current_2
        gaussian_previous_2 = gaussian_current_2

    # normalization factor for consistency in DoG magnitude
    sf = 1 / (sigma_ratio - 1)
    dog_image_cube_1 *= sf
    dog_image_cube_2 *= sf

    local_maxima_1, local_maxima_2 = peak_local_max(
        dog_image_cube_1,
        dog_image_cube_2,
        segment=(image_1 > th) * ((image_2 > th)),
        footprint=np.ones((17,) * (image_1.ndim - 2)),
    )

    # Catch no peaks
    flag_1 = 1
    flag_2 = 1
    if local_maxima_1.numel() == 0:
        flag_1 = 0
    if local_maxima_2.numel() == 0:
        flag_2 = 0

    sigma_list = torch.from_numpy(np.asarray(sigma_list)).to(torch.float).to(device)

    if flag_1:
        lm = local_maxima_1.to(torch.float)
        sigmas_of_peaks = sigma_list[local_maxima_1[:, -1]]

        if scalar_sigma:
            sigmas_of_peaks = sigmas_of_peaks[:, 0:1]

        lm = torch.hstack([lm[:, :-1], sigmas_of_peaks])
        sigma_dim = sigmas_of_peaks.shape[1]
        local_maxima_1 = lm[_prune_blobs(lm[:, 1:], overlap, sigma_dim=sigma_dim), :]

    if flag_2:
        lm = local_maxima_2.to(torch.float)
        sigmas_of_peaks = sigma_list[local_maxima_2[:, -1]]

        if scalar_sigma:
            sigmas_of_peaks = sigmas_of_peaks[:, 0:1]

        lm = torch.hstack([lm[:, :-1], sigmas_of_peaks])
        sigma_dim = sigmas_of_peaks.shape[1]
        local_maxima_2 = lm[_prune_blobs(lm[:, 1:], overlap, sigma_dim=sigma_dim), :]
    return local_maxima_1, local_maxima_2


@torch.no_grad()
def DoG(
    input_1,
    input_2,
    z_spacing,
    xy_spacing,
    device,
    max_p=None,
):
    points_1 = []
    points_2 = []
    b_num = 10
    ss = torch.split(torch.arange(max(input_1.shape[0], input_2.shape[0])), b_num)

    for ind, s in enumerate(tqdm.tqdm(ss, desc="DoG: ")):
        start, end = s[0], s[-1] + 1
        start, end = start.item(), end.item()
        input_batch_1 = np.array(input_1[start:end])
        input_batch_2 = np.array(input_2[start:end])
        if input_batch_1.shape[0] == input_batch_2.shape[0]:

            th = filters.threshold_otsu(
                np.maximum(input_batch_1[:, ::5, ::5], input_batch_2[:, ::5, ::5])
            )

            if ((input_batch_1 > th) * (input_batch_2 > th)).sum() > 0:
                tmp_1, tmp_2 = blob_dog(
                    input_batch_1[:, None, :, :],
                    input_batch_2[:, None, :, :],
                    min_sigma=1.8,
                    max_sigma=1.8 * 1.6 + 1,
                    th=th,
                    device=device,
                )
                tmp_1[:, 0] += ind * b_num
                tmp_2[:, 0] += ind * b_num

                if tmp_1.shape[0] != 0:
                    points_1.append(tmp_1)
                    points_2.append(tmp_2)
            else:
                pass

    points_1 = torch.cat(points_1, 0)
    points_2 = torch.cat(points_2, 0)

    points_scaled_1 = copy.deepcopy(points_1)
    points_scaled_1[:, 0] *= z_spacing / xy_spacing
    points_1 = points_1[_prune_blobs(points_scaled_1, overlap=0.1, sigma_dim=1), :]

    points_scaled_2 = copy.deepcopy(points_2)
    points_scaled_2[:, 0] *= z_spacing / xy_spacing
    points_2 = points_2[_prune_blobs(points_scaled_2, overlap=0.1, sigma_dim=1), :]

    points_1 = points_1.cpu().data.numpy()
    points_2 = points_2.cpu().data.numpy()

    if max_p is not None:
        if points_1.shape[0] > max_p:
            points_1 = points_1[np.argsort(points_1[:, -1])[-max_p:], :]
        if points_2.shape[0] > max_p:
            points_2 = points_2[np.argsort(points_2[:, -1])[-max_p:], :]

    points_1 *= np.array([z_spacing, xy_spacing, xy_spacing, 1])[None]
    points_2 *= np.array([z_spacing, xy_spacing, xy_spacing, 1])[None]

    return points_1[:, :-1], points_2[:, :-1]
