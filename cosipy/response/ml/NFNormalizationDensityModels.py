import numpy as np

from typing import Tuple, Dict

from .NFResponseModels import UnpolarizedDensityCMLPDGaussianCARQSFlow
import torch


class Unpolarized2DDensityCMLPDGaussianCARQSFlow(UnpolarizedDensityCMLPDGaussianCARQSFlow):
    @property
    def source_dim(self) -> int: 
        return 2

    def _convert_conventions(self, ei: torch.Tensor, em: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = em / ei - 1

        return eps

    def _inverse_transform_coordinates(self, *args: torch.Tensor) -> torch.Tensor:
        neps, nphi, _, _, ei = args
        
        eps = -neps
        phi = nphi * np.pi

        em = ei * (eps + 1)

        return torch.stack([em, phi], dim=1)

    def _transform_coordinates(self, *args: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dir_az, dir_pol, ei, em, phi = args

        eps_raw = self._convert_conventions(ei, em)

        jac = 1.0 / (ei * np.pi)
        jac[torch.isinf(jac) | (jac < 0)] = 0.0

        ctx = self._transform_context(dir_az, dir_pol, ei)

        src = torch.cat([
            (-eps_raw).unsqueeze(1),
            (phi / np.pi).unsqueeze(1),
        ], dim=1)

        return ctx.to(torch.float32), src.to(torch.float32), jac.to(torch.float32)
    
    def _valid_samples(self, *args: torch.Tensor) -> torch.Tensor:
        neps, nphi, _, _, ei = args
        
        valid_mask = (neps   <  1.0) & \
                     (nphi   >  0.0) & (nphi <= 1.0) & \
                     (neps <= (1 - self._menergy_cuts[0]/ei)) & \
                     (neps >= (1 - self._menergy_cuts[1]/ei)) & \
                     (nphi >= self._phi_cuts[0]/np.pi) & \
                     (nphi <= self._phi_cuts[1]/np.pi)
                     
        return valid_mask
    
class Unpolarized1EMDDensityCMLPDGaussianCARQSFlow(UnpolarizedDensityCMLPDGaussianCARQSFlow):
    @property
    def source_dim(self) -> int: 
        return 1

    def _convert_conventions(self, ei: torch.Tensor, em: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = em / ei - 1

        return eps

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
        
        return self._load_model()
    
    def _inverse_transform_coordinates(self, *args: torch.Tensor) -> torch.Tensor:
        neps, _, _, ei = args
        
        eps = -neps

        em = ei * (eps + 1)

        return torch.stack([em,], dim=1)

    def _transform_coordinates(self, *args: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dir_az, dir_pol, ei, em = args

        eps_raw = self._convert_conventions(ei, em)

        jac = 1.0 / (ei)
        jac[torch.isinf(jac) | (jac < 0)] = 0.0

        ctx = self._transform_context(dir_az, dir_pol, ei)

        src = torch.cat([
            (-eps_raw).unsqueeze(1),
        ], dim=1)

        return ctx.to(torch.float32), src.to(torch.float32), jac.to(torch.float32)
    
    def _valid_samples(self, *args: torch.Tensor) -> torch.Tensor:
        neps, _, _, ei = args
        
        valid_mask = (neps   <  1.0) & \
                     (neps <= (1 - self._menergy_cuts[0]/ei)) & \
                     (neps >= (1 - self._menergy_cuts[1]/ei))
                     
        return valid_mask
    