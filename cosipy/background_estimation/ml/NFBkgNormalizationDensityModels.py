import numpy as np
import torch

from typing import Tuple, Dict

from .NFBackgroundModels import TotalBackgroundDensityCMLPDGaussianCARQSFlow

class TotalBackground2DDensityCMLPDGaussianCARQSFlow(TotalBackgroundDensityCMLPDGaussianCARQSFlow):
    @property
    def source_dim(self) -> int: 
        return 2
    
    def _inverse_transform_coordinates(self, *args: torch.Tensor) -> torch.Tensor:
        nem, nphi, _ = args
        
        em  = 10 ** (2 * (nem + 1))
        phi = nphi * np.pi

        return torch.stack([em, phi], dim=1)
    
    def _transform_coordinates(self, *args: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time, em, phi = args

        jac = 1/(np.log(10) * em * 2*np.pi)

        ctx = self._transform_context(time)

        src = torch.cat([
            (torch.log10(em)/2 - 1).unsqueeze(1),
            (phi / np.pi).unsqueeze(1),
        ], dim=1)

        return ctx.to(torch.float32), src.to(torch.float32), jac.to(torch.float32)
    
    def _valid_samples(self, *args: torch.Tensor) -> torch.Tensor:
        nem, nphi, _ = args
        
        valid_mask = (nem  >= 0.0) & \
                     (nphi >  0.0) & (nphi <= 1.0) & \
                     (nem  >= (np.log10(self._menergy_cuts[0])/2 - 1)) & \
                     (nem  <= (np.log10(self._menergy_cuts[1])/2 - 1)) & \
                     (nphi >= self._phi_cuts[0]/np.pi) & \
                     (nphi <= self._phi_cuts[1]/np.pi)
                     
        return valid_mask

class TotalBackground1EMDDensityCMLPDGaussianCARQSFlow(TotalBackgroundDensityCMLPDGaussianCARQSFlow):
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
        
        self._start_time: float     = input["start_time"]
        self._total_time: float     = input["total_time"]
        self._period: float         = input["period"]
        self._slew_duration: float  = input["slew_duration"]
        self._obs_duration: float   = input["obs_duration"]
        self._outlocs: torch.Tensor = input["outlocs"].to(self._worker_device)
        
        return self._load_model()
    
    @property
    def source_dim(self) -> int: 
        return 1
    
    def _inverse_transform_coordinates(self, *args: torch.Tensor) -> torch.Tensor:
        nem, _ = args
        
        em  = 10 ** (2 * (nem + 1))

        return torch.stack([em,], dim=1)
    
    def _transform_coordinates(self, *args: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time, em = args

        jac = 1/(np.log(10) * em * 2)

        ctx = self._transform_context(time)

        src = torch.cat([
            (torch.log10(em)/2 - 1).unsqueeze(1),
        ], dim=1)

        return ctx.to(torch.float32), src.to(torch.float32), jac.to(torch.float32)
    
    def _valid_samples(self, *args: torch.Tensor) -> torch.Tensor:
        nem, _ = args
        
        valid_mask = (nem  >= 0.0) & \
                     (nem  >= (np.log10(self._menergy_cuts[0])/2 - 1)) & \
                     (nem  <= (np.log10(self._menergy_cuts[1])/2 - 1))
                     
        return valid_mask