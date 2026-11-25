"""Experimental, memory-bounded accelerated GuidedUpsample.

Replaces the original repeated full-plane reflect padding + 129 CPU copies
with one reflect padding per angle and a tiled torch median on the selected device.
The offsets, median, order of angles, and output mathematics are unchanged.

Intended for comparison with the original Leonardo guided_filter_upsample.py.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F

try:
    import jax
    import jax.scipy.ndimage
except ImportError:
    jax = None


class GuidedUpsample:
    def __init__(self, rx, device, strip_rows=16, verbose=True):
        self.rx = int(rx)
        self.device = device
        self.strip_rows = int(strip_rows)
        self.verbose = bool(verbose)
        if self.rx < 1 or self.rx % 2 != 1:
            raise ValueError('rx must be a positive odd number')
        if self.strip_rows < 1:
            raise ValueError('strip_rows must be >= 1')

    def _log(self, name, seconds):
        if self.verbose:
            print(f'[GF timing] {name}: {seconds:.2f}s', flush=True)

    def __call__(
        self, yy, hX, targetd, target, coor, fusion_mask,
        angle_offset_individual, backend,
    ):
        t_all = time.perf_counter()
        if backend == 'jax':
            if jax is None:
                raise RuntimeError('JAX backend requested but JAX unavailable')
            t = time.perf_counter()
            recon_jax = (
                jax.scipy.ndimage.map_coordinates(
                    yy - targetd, coor, order=1, mode='reflect'
                )[None, None] + target
            )
            # Device synchronization is implicit in transfer to NumPy.
            recon = torch.from_numpy(np.array(recon_jax, copy=True)).to(self.device)
            hX = torch.from_numpy(np.array(hX)).to(self.device)
            fusion_mask = np.asarray(fusion_mask)
            self._log('JAX coordinate interpolation and transfers', time.perf_counter() - t)
        else:
            recon = (
                F.grid_sample(
                    yy - targetd, coor, mode='bilinear',
                    padding_mode='reflection', align_corners=True,
                ) + target
            )
            fusion_mask = fusion_mask.cpu().data.numpy()

        # Keep original output dtype/values, summation and log transform.
        y = np.ones_like(fusion_mask)
        for i, angle_list in enumerate(angle_offset_individual):
            t = time.perf_counter()
            hX_slice = hX[:, i:i+1, :, :]
            corrected = self.GF(recon, hX_slice, angle_list)
            y[:, i:i+1, :, :] = corrected.detach().cpu().numpy()
            self._log(f'view {i+1} guided median and CPU transfer', time.perf_counter() - t)

        t = time.perf_counter()
        y = (10 ** y) * fusion_mask
        out = np.log10(np.clip(y.sum(1, keepdims=True), 1, None))
        self._log('final fusion / power / log', time.perf_counter() - t)
        self._log('GuidedUpsample total', time.perf_counter() - t_all)
        return out

    @torch.no_grad()
    def GF(self, yy, hX, angle_list):
        _, _, m, n = hX.shape
        rx = self.rx
        for angle_idx, Angle in enumerate((-1 * np.array(angle_list)).tolist()):
            t = time.perf_counter()
            b = yy - hX
            lval = np.arange(rx) - rx // 2
            lval = np.round(lval * np.tan(np.deg2rad(-Angle))).astype(np.int32)
            xpad = int(lval.max())
            # Original Leonardo pads this identical tensor 129 times per angle.
            padded = F.pad(b, (xpad, xpad, rx // 2, rx // 2), mode='reflect')
            offsets = lval - lval.min()
            # Match original implementation's single-view, batch-1 assumptions.
            if padded.shape[0:2] != (1, 1):
                raise ValueError('Optimized GF currently expects shape (1,1,Y,X)')
            result = torch.empty_like(hX)
            for start in range(0, m, self.strip_rows):
                stop = min(start + self.strip_rows, m)
                # A bounded stack: (rx,1,1,strip_rows,n), on GPU, with the
                # same elements/order as the original CPU b_batch.
                samples = torch.stack([
                    padded[:, :, r+start:r+stop, int(shift):int(shift)+n]
                    for r, shift in enumerate(offsets)
                ], dim=0)
                result[:, :, start:stop, :] = torch.median(samples, dim=0).values
                del samples
            hX = hX + result
            self._log(f'angle {angle_idx+1} median (rx={rx}, rows={self.strip_rows})', time.perf_counter()-t)
        return hX
