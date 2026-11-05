"""Custom post-processing for the JSON Leonardo runner.

Place this file beside core_json_version.py. After result = model.train(...):
    from custom_post_processing import run_custom_postprocessing
    final_path = run_custom_postprocessing(image, result, cfg)

Uses cfg["destripe"]["angle_offset"] and cfg["destripe"]["is_vertical"].
Writes a separate *_custom.tif beside cfg["output"]["save_path"].
Preserves the accepted v2.1 filtering settings. Supports YX or ZYX arrays,
processing one plane at a time; does not load data from or write to the input Zarr.
Requires numpy, scipy and tifffile (already in the Leonardo environment).
"""
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.fft import fftfreq,rfftfreq,rfft2,irfft2

def _filter_plane(original,corrected,angles,angular_width=3.,period_px=128.,pad=256,threshold=200.,margin=24.):
    x=np.asarray(original,dtype=np.float32).squeeze();y=np.asarray(corrected,dtype=np.float32).squeeze()
    if x.ndim!=2 or x.shape!=y.shape:raise ValueError('Matching 2D arrays required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Non-finite inputs')
    valid=distance_transform_edt(x>threshold)>margin
    if not valid.any():raise ValueError('No valid tissue interior; review threshold and margin')
    c=np.log(np.maximum(y,1))-np.log(np.maximum(x,1))
    indices=distance_transform_edt(~valid,return_distances=False,return_indices=True)
    c=c[tuple(indices)];del indices
    c=np.pad(c,pad,mode='reflect')
    fy=fftfreq(c.shape[0]).astype('float32')[:,None];fx=rfftfreq(c.shape[1]).astype('float32')[None,:]
    r=np.hypot(fx,fy);distance=np.full(r.shape,90,dtype='float32')
    for theta in np.deg2rad(angles):
        d=np.rad2deg(np.arcsin(np.clip(np.abs(fx*np.cos(theta)+fy*np.sin(theta))/np.maximum(r,1e-30),0,1)))
        distance=np.minimum(distance,d)
    transfer=np.exp(-(distance/angular_width)**2)*(1-np.exp(-(r*period_px)**2))
    c=irfft2(rfft2(c)*transfer,s=c.shape)[pad:-pad,pad:-pad]
    c=gaussian_filter(c,0.7,mode='reflect')
    result=np.maximum(x,1)*np.exp(c);result[x==0]=0
    if not np.isfinite(result).all():raise ValueError('Non-finite result')
    return result.astype('float32')


def run_custom_postprocessing(original, corrected, cfg):
    """Use the SAME loaded cfg and input image used for the completed training run.

    Return the new TIFF's Path; refuse to overwrite any existing file.
    Threshold 200 and margin 24 remain specific to the validated WFA data.
    """
    settings = cfg["destripe"]
    angles = np.asarray(settings["angle_offset"], dtype=float)
    if angles.ndim != 1 or angles.size == 0 or not np.isfinite(angles).all():
        raise ValueError('destripe.angle_offset must be a nonempty list of finite angles')
    vertical = settings["is_vertical"]
    if not isinstance(vertical, (bool, np.bool_)):
        raise ValueError('destripe.is_vertical must be true or false')
    if corrected is None:
        raise ValueError('Leonardo returned no corrected image')
    shape = tuple(original.shape)
    if len(shape) not in (2, 3) or any(n == 0 for n in shape):
        raise ValueError('Original must be a nonempty YX plane or ZYX stack')
    if tuple(corrected.shape) != shape:
        raise ValueError(f'Original/output shapes differ: {shape} vs {corrected.shape}')
    base = Path(cfg["output"]["save_path"])
    if base.suffix.lower() not in ('.tif', '.tiff'):
        raise ValueError('output.save_path must end in .tif or .tiff')
    destination = base.with_name(base.stem + '_custom.tif')
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    # Exclusive creation protects both previous results and the original output.
    handle = destination.open('xb')
    try:
        planes = 1 if len(shape) == 2 else shape[0]
        with handle:
            with tifffile.TiffWriter(handle, bigtiff=np.prod(shape) * 4 >= 2**32 - 2**25) as writer:
                for index in range(planes):
                    a = np.asarray(original if len(shape) == 2 else original[index])
                    b = np.asarray(corrected if len(shape) == 2 else corrected[index])
                    if not np.isfinite(a).all() or not np.isfinite(b).all():
                        raise ValueError(f'Plane {index}: non-finite input values')
                    if np.all(a == 0):
                        final = np.zeros(a.shape, dtype=np.float32)
                    else:
                        try:
                            # Leonardo's vertical offsets become horizontal offsets
                            # after transposing into the filter's coordinate system.
                            final = _filter_plane(a.T if vertical else a,
                                                  b.T if vertical else b, angles=angles)
                            if vertical:
                                final = final.T
                        except ValueError as error:
                            raise ValueError(f'Plane {index}: {error}') from error
                    writer.write(np.ascontiguousarray(final), photometric='minisblack',
                                 contiguous=True, metadata=None)
                    print(f'Custom post-processing: {index + 1}/{planes}', flush=True)
    except BaseException:
        # Delete only this newly created, incomplete output; do not touch inputs.
        handle.close()
        destination.unlink(missing_ok=True)
        raise
    print(f'Custom result saved to: {destination}', flush=True)
    return destination
