import numpy as np
import healpy as hp

from typing import Union, Tuple, Dict, Optional, Callable

from .NFBase import CompileMode, build_c_arqs_flow, build_cmlp_diaggaussian_base, NNDensityInferenceWrapper, AreaModel, DensityModel
import sphericart.torch
import normflows as nf
import torch


class UnpolarizedAreaSphericalHarmonicsExpansion(AreaModel):
    def __init__(self, area_input: Dict, worker_device: Union[str, int, torch.device],
                 batch_size: int, compile_mode: CompileMode = "max-autotune-no-cudagraphs"):
        super().__init__(compile_mode, batch_size, worker_device, area_input)
    
    def _init_model(self, input: Dict):
        self._lmax        = input['lmax']
        self._poly_degree = input['poly_degree']
        self._poly_coeffs = input['poly_coeffs']
        
        self._conv_coeffs = self._convert_coefficients().to(self._worker_device)
        self._sh_calculator = sphericart.torch.SphericalHarmonics(self._lmax)
        
        return self._horner_eval
    
    @property
    def context_dim(self) -> int:
        return 3
    
    def _convert_coefficients(self) -> torch.Tensor:
        num_sh = (self._lmax + 1)**2
        conv_coeffs = torch.zeros((num_sh, self._poly_degree + 1), dtype=torch.float64)

        for cnt, (l, m) in enumerate((l, m) for l in range(self._lmax + 1) for m in range(-l, l + 1)):
            idx = hp.Alm.getidx(self._lmax, l, abs(m))
            if m == 0:
                conv_coeffs[cnt] = self._poly_coeffs[0, :, idx]
            else:
                fac = np.sqrt(2) * (-1)**m
                val = self._poly_coeffs[0, :, idx] if m > 0 else -self._poly_coeffs[1, :, idx]
                conv_coeffs[cnt] = fac * val
        return conv_coeffs.T
    
    def _horner_eval(self, x: torch.Tensor) -> torch.Tensor:
        x_64 = x.to(torch.float64).unsqueeze(1)
        result = self._conv_coeffs[0].expand(x.shape[0], -1).clone()
        for i in range(1, self._conv_coeffs.size(0)):
            result.mul_(x_64).add_(self._conv_coeffs[i])
        return result.to(torch.float32)
    
    def _compute_spherical_harmonics(self, dir_az: torch.Tensor, dir_polar: torch.Tensor) -> torch.Tensor:
        sin_p = torch.sin(dir_polar)
        xyz = torch.stack((
            sin_p * torch.cos(dir_az),
            sin_p * torch.sin(dir_az),
            torch.cos(dir_polar)
        ), dim=-1)
        return self._sh_calculator(xyz)
    
    @torch.inference_mode()
    def evaluate_effective_area(self, dir_az: torch.Tensor, dir_polar: torch.Tensor, energy_keV: torch.Tensor,
                                progress_callback: Optional[Callable[[int], None]] = None) -> torch.Tensor:
        N = energy_keV.shape[0]
        
        ei_norm = (torch.log10(energy_keV) / 2 - 1).to(torch.float32)
        result = torch.empty(N, dtype=torch.float32, device="cpu")
        
        def get_batch(start_idx):
            end_idx = min(start_idx + self._batch_size, N)
            return (
                ei_norm[start_idx:end_idx].to(self._worker_device),
                dir_az[start_idx:end_idx].to(self._worker_device),
                dir_polar[start_idx:end_idx].to(self._worker_device)
            )
         
        for start in range(0, N, self._batch_size):
            end = min(start + self._batch_size, N)
            batch_len = end - start
            
            ei_b, az_b, pol_b = get_batch(start)

            poly_b = self._model_op(ei_b)
            ylm_b = self._compute_spherical_harmonics(az_b, pol_b)
            result[start:end] = torch.sum(poly_b * ylm_b, dim=1)
            
            if progress_callback is not None:
                progress_callback(batch_len)
        
        return torch.clamp(result, min=0)

class UnpolarizedDensityCMLPDGaussianCARQSFlow(DensityModel):
    def __init__(self, density_input: Dict, worker_device: Union[str, int, torch.device],
                 batch_size: int, compile_mode: CompileMode = "default"):
        super().__init__(compile_mode, batch_size, worker_device, density_input)
            
    def _init_model(self, input: Dict):
        self._snapshot          = input["model_state_dict"]
        self._bins              = input["bins"]
        self._hidden_units      = input["hidden_units"]
        self._residual_blocks   = input["residual_blocks"]
        self._total_layers      = input["total_layers"]
        self._context_size      = input["context_size"]
        self._mlp_hidden_units  = input["mlp_hidden_units"]
        self._mlp_hidden_layers = input["mlp_hidden_layers"]
        self._menergy_cuts      = input["menergy_cuts"]
        self._phi_cuts          = input["phi_cuts"]
        
        return self._load_model()
    
    @property
    def context_dim(self) -> int: 
        return 3
    
    @property
    def source_dim(self) -> int: 
        return 4

    def _build_model(self) -> nf.ConditionalNormalizingFlow:
        base = build_cmlp_diaggaussian_base(
            self._context_size, 2 * self.source_dim, self._mlp_hidden_units, self._mlp_hidden_layers
            )
        return build_c_arqs_flow(
            base, self._total_layers, self.source_dim, self._context_size, self._bins, self._hidden_units, self._residual_blocks
            )

    def _load_model(self) -> NNDensityInferenceWrapper:
        model = self._build_model()
        
        model.load_state_dict(self._snapshot)
        model = NNDensityInferenceWrapper(model)
        model.eval()
        model.to(self._worker_device)
        
        return model

    @staticmethod
    def _get_vector(phi_sc: torch.Tensor, theta_sc: torch.Tensor) -> torch.Tensor:
        x = theta_sc[:, 0] * phi_sc[:, 1]
        y = theta_sc[:, 0] * phi_sc[:, 0]
        z = theta_sc[:, 1]
        return torch.stack((x, y, z), dim=-1)

    def _convert_conventions(self, dir_az_sc: torch.Tensor, dir_polar_sc: torch.Tensor, 
                             ei: torch.Tensor, em: torch.Tensor, phi: torch.Tensor, 
                             scatt_az_sc: torch.Tensor, scatt_polar_sc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = em / ei - 1

        source = self._get_vector(dir_az_sc, dir_polar_sc)
        scatter = self._get_vector(scatt_az_sc, scatt_polar_sc)

        dot_product = torch.sum(source * scatter, dim=1)
        phi_geo = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
        theta = phi_geo - phi

        xaxis = torch.tensor([1., 0., 0.], device=source.device, dtype=source.dtype)
        pz = -source
        
        px = torch.linalg.cross(pz, xaxis.expand_as(pz))
        px = px / torch.linalg.norm(px, dim=1, keepdim=True)
        
        py = torch.linalg.cross(pz, px)
        py = py / torch.linalg.norm(py, dim=1, keepdim=True)

        proj_x = torch.sum(scatter * px, dim=1)
        proj_y = torch.sum(scatter * py, dim=1)

        zeta = torch.atan2(proj_y, proj_x)
        zeta = torch.where(zeta < 0, zeta + 2 * np.pi, zeta)

        return eps, theta, zeta

    def _inverse_transform_coordinates(self, *args: torch.Tensor) -> torch.Tensor:
        neps, nphi, ntheta, nzeta, dir_az, dir_pol, ei = args
        
        eps   = -neps
        phi   = nphi * np.pi
        theta = (ntheta - 0.5) * (2 * np.pi)
        zeta  = nzeta * (2 * np.pi)

        em = ei * (eps + 1)

        phi_geo = theta + phi
        scatter_phf = self._get_vector(torch.stack((torch.sin(zeta), torch.cos(zeta)), dim=1),
                                       torch.stack((torch.sin(np.pi - phi_geo), torch.cos(np.pi - phi_geo)), dim=1))
        
        dir_az_sc = torch.stack((torch.sin(dir_az), torch.cos(dir_az)), dim=1)
        dir_pol_sc = torch.stack((torch.sin(dir_pol), torch.cos(dir_pol)), dim=1)
        source_vec = self._get_vector(dir_az_sc, dir_pol_sc)
        xaxis = torch.tensor([1., 0., 0.], device=self._worker_device, dtype=source_vec.dtype)

        pz = -source_vec
        px = torch.linalg.cross(pz, xaxis.expand_as(pz))
        px = px / torch.linalg.norm(px, dim=1, keepdim=True)
        py = torch.linalg.cross(pz, px)
        py = py / torch.linalg.norm(py, dim=1, keepdim=True)

        basis = torch.stack((px, py, pz), dim=2)
        scatter_scf = torch.bmm(basis, scatter_phf.unsqueeze(-1)).squeeze(-1)

        psi_cds = torch.atan2(scatter_scf[:, 1], scatter_scf[:, 0])
        psi_cds = torch.where(psi_cds < 0, psi_cds + 2 * np.pi, psi_cds)
        chi_cds = torch.acos(torch.clamp(scatter_scf[:, 2], -1.0, 1.0))

        return torch.stack([em, phi, psi_cds, chi_cds], dim=1)

    def _transform_coordinates(self, *args: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dir_az, dir_pol, ei, em, phi, scatt_az, scatt_pol = args
        
        dir_az_sc = torch.stack((torch.sin(dir_az), torch.cos(dir_az)), dim=1)
        dir_pol_sc = torch.stack((torch.sin(dir_pol), torch.cos(dir_pol)), dim=1)
        scatt_az_sc = torch.stack((torch.sin(scatt_az), torch.cos(scatt_az)), dim=1)
        scatt_pol_sc = torch.stack((torch.sin(scatt_pol), torch.cos(scatt_pol)), dim=1)

        eps_raw, theta_raw, zeta_raw = self._convert_conventions(
            dir_az_sc, dir_pol_sc, ei, em, phi, scatt_az_sc, scatt_pol_sc
        )

        jac = 1.0 / (ei * torch.sin(theta_raw + phi) * 4 * np.pi**3)
        jac[torch.isinf(jac) | (jac < 0)] = 0.0

        ctx = self._transform_context(dir_az, dir_pol, ei)

        src = torch.cat([
            (-eps_raw).unsqueeze(1),
            (phi / np.pi).unsqueeze(1),
            (theta_raw / (2 * np.pi) + 0.5).unsqueeze(1),
            (zeta_raw / (2 * np.pi)).unsqueeze(1)
        ], dim=1)

        return ctx.to(torch.float32), src.to(torch.float32), jac.to(torch.float32)
    
    def _transform_context(self, *args: torch.Tensor) -> torch.Tensor: 
        dir_az, dir_pol, ei = args
        
        dir_az_sc = torch.stack((torch.sin(dir_az), torch.cos(dir_az)), dim=1)
        dir_pol_c = torch.cos(dir_pol).unsqueeze(1)
        
        ctx = torch.cat([
            (dir_az_sc + 1) / 2, 
            (dir_pol_c + 1) / 2, 
            (torch.log10(ei) / 2 - 1).unsqueeze(1)
        ], dim=1)
        
        return ctx.to(torch.float32)
    
    def _valid_samples(self, *args: torch.Tensor) -> torch.Tensor:
        neps, nphi, ntheta, nzeta, _, _, ei = args
        
        phi_geo_norm = nphi + 2 * ntheta - 1.0
        valid_mask = (neps   <  1.0) & \
                     (nphi   >  0.0) & (nphi <= 1.0) & \
                     (ntheta >= 0.0) & (ntheta <= 1.0) & \
                     (nzeta  >= 0.0) & (nzeta <= 1.0) & \
                     (phi_geo_norm  >  0.0) & (phi_geo_norm  <  1.0) & \
                     (neps <= (1 - self._menergy_cuts[0]/ei)) & \
                     (neps >= (1 - self._menergy_cuts[1]/ei)) & \
                     (nphi >= self._phi_cuts[0]/np.pi) & \
                     (nphi <= self._phi_cuts[1]/np.pi)
                     
        return valid_mask
