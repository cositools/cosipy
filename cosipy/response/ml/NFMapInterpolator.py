import torch
import numpy as np
import healpy as hp
from typing import Union, Tuple
from abc import ABC, abstractmethod

class BasePCHIPInterpolator(ABC):
    def __init__(self, maps):
        self._maps = maps

    @abstractmethod
    def _get_y_nodes(self, I_prev: torch.Tensor, I_curr: torch.Tensor, 
                     I_next: torch.Tensor, I_nnext: torch.Tensor, 
                     map_indices_t: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def _pchip_interpolate(self, q_t: torch.Tensor, ref_t: torch.Tensor, 
                           map_indices_t: torch.Tensor, **kwargs) -> torch.Tensor:
        M_len = len(ref_t)
        
        idx = torch.searchsorted(ref_t, q_t, right=True) - 1
        idx = torch.clamp(idx, 0, M_len - 2)
        
        I_prev = torch.clamp(idx - 1, min=0)
        I_curr = idx
        I_next = idx + 1
        I_nnext = torch.clamp(idx + 2, max=M_len - 1)
        
        y_prev, y_curr, y_next, y_nnext = self._get_y_nodes(
            I_prev, I_curr, I_next, I_nnext, map_indices_t, **kwargs
        )
        
        hk = torch.diff(ref_t)
        
        h_prev = hk[I_prev]
        h_curr = hk[I_curr]
        h_next = hk[torch.clamp(idx + 1, max=M_len - 2)]
        
        m_prev = (y_curr - y_prev) / h_prev
        m_curr = (y_next - y_curr) / h_curr
        m_next = (y_nnext - y_next) / h_next
        
        def standard_pchip(h1, h2, m1, m2):
            condition = (torch.sign(m1) != torch.sign(m2)) | (m1 == 0.0) | (m2 == 0.0)
            w1 = 2.0 * h2 + h1
            w2 = h2 + 2.0 * h1
            
            m1_safe = torch.where(m1 == 0.0, torch.tensor(1.0, dtype=torch.float64), m1)
            m2_safe = torch.where(m2 == 0.0, torch.tensor(1.0, dtype=torch.float64), m2)
            
            whmean = (w1 / m1_safe + w2 / m2_safe) / (w1 + w2)
            return torch.where(condition, torch.tensor(0.0, dtype=torch.float64), 1.0 / whmean)
            
        def edge_case(h0, h1, m0, m1):
            d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
            mask_sign = torch.sign(d) != torch.sign(m0)
            mask2 = (torch.sign(m0) != torch.sign(m1)) & (torch.abs(d) > 3.0 * torch.abs(m0))
            mmm = (~mask_sign) & mask2
            d = torch.where(mask_sign, torch.tensor(0.0, dtype=torch.float64), d)
            return torch.where(mmm, 3.0 * m0, d)
            
        d_k = torch.where(
            idx == 0,
            edge_case(h_curr, h_next, m_curr, m_next),
            standard_pchip(h_prev, h_curr, m_prev, m_curr)
        )
        
        d_k1 = torch.where(
            idx == M_len - 2,
            edge_case(h_curr, h_prev, m_curr, m_prev),
            standard_pchip(h_curr, h_next, m_curr, m_next)
        )
        
        t = (q_t - ref_t[idx]) / h_curr
        t2 = t * t
        t3 = t2 * t
        
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        
        return h00 * y_curr + h10 * h_curr * d_k + h01 * y_next + h11 * h_curr * d_k1


class SpatialSpectralInterpolator(BasePCHIPInterpolator):
    def __init__(self, maps, map_nside):
        super().__init__(maps)
        self._map_nside = map_nside

    def _spatial_interpolation(self, q_pol: np.ndarray, q_az: np.ndarray, layer_indices: Union[int, torch.Tensor]) -> torch.Tensor:
        pixels, weights = hp.get_interp_weights(self._map_nside, q_pol, q_az)
        pixels_t = torch.from_numpy(pixels).long()
        weights_t = torch.from_numpy(weights)
        
        y = torch.zeros(len(q_pol), dtype=torch.float64)
        for p in range(4):
            y += self._maps[layer_indices, pixels_t[p]] * weights_t[p]
            
        return y

    def _get_y_nodes(self, I_prev, I_curr, I_next, I_nnext, map_indices_t, **kwargs):
        q_pol = kwargs['q_pol']
        q_az = kwargs['q_az']
        
        y_prev = self._spatial_interpolation(q_pol, q_az, map_indices_t[I_prev])
        y_curr = self._spatial_interpolation(q_pol, q_az, map_indices_t[I_curr])
        y_next = self._spatial_interpolation(q_pol, q_az, map_indices_t[I_next])
        y_nnext = self._spatial_interpolation(q_pol, q_az, map_indices_t[I_nnext])
        
        return y_prev, y_curr, y_next, y_nnext


class TemporalInterpolator(BasePCHIPInterpolator):
    def __init__(self, maps):
        super().__init__(maps)

    def _get_y_nodes(self, I_prev, I_curr, I_next, I_nnext, map_indices_t, **kwargs):
        y_prev = self._maps[map_indices_t[I_prev]]
        y_curr = self._maps[map_indices_t[I_curr]]
        y_next = self._maps[map_indices_t[I_next]]
        y_nnext = self._maps[map_indices_t[I_nnext]]
        
        return y_prev, y_curr, y_next, y_nnext