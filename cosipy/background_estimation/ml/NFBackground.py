from typing import List, Union, Optional, Dict
from pathlib import Path

import torch
import torch.multiprocessing as mp
from cosipy.response.ml.NFBase import NFBase, CompileMode, update_density_worker_settings, init_density_worker, DensityApproximation, DensityModel, RateModel
from .NFBackgroundModels import TotalBackgroundDensityCMLPDGaussianCARQSFlow, TotalDC4BackgroundRate


class BackgroundDensityApproximation(DensityApproximation):

    def _setup_model(self):
        version_map: Dict[int, DensityModel] = {
            1: TotalBackgroundDensityCMLPDGaussianCARQSFlow,
        }
        if self._major_version not in version_map:
            raise ValueError(f"Unsupported major version {self._major_version} for Density Approximation")
        else:    
            model_class = version_map[self._major_version]
            self._model = model_class(self._density_input, self._worker_device, self._batch_size, self._compile_mode)
            self._expected_context_dim = self._model.context_dim
            self._expected_source_dim = self._model.source_dim

class BackgroundRateApproximation:
    def __init__(self, major_version: int, rate_input: Dict):
        self._major_version = major_version
        self._rate_input = rate_input
        
        self._setup_model()

    def _setup_model(self):
        version_map: Dict[int, RateModel] = {
            1: TotalDC4BackgroundRate,
        }
        if self._major_version not in version_map:
            raise ValueError(f"Unsupported major version {self._major_version} for Rate Approximation")
        else:    
            model_class = version_map[self._major_version]
            self._model = model_class(self._rate_input)
            self._expected_context_dim = self._model.context_dim
    
    def evaluate_rate(self, context: torch.Tensor) -> torch.Tensor:
        dim_context = context.shape[1]
        
        if dim_context != self._expected_context_dim:
            raise ValueError(
                f"Feature mismatch: {type(self._model).__name__} expects "
                f"{self._expected_context_dim} features, but context has {dim_context}."
            )
            
        list_context = [context[:, i] for i in range(dim_context)]
        
        return self._model.evaluate_rate(*list_context)

def init_background_worker(device_queue: mp.Queue, progress_queue: mp.Queue, major_version: int,
                           density_input: Dict, density_batch_size: int,
                           density_compile_mode: CompileMode):
    
    init_density_worker(device_queue, progress_queue, major_version,
                        density_input, density_batch_size,
                        density_compile_mode, BackgroundDensityApproximation)

class NFBackground(NFBase):
    def __init__(self, path_to_model: Union[str, Path], density_batch_size: int = 100_000,
                 devices: Optional[List[Union[str, int, torch.device]]] = None,
                 density_compile_mode: CompileMode = "default", show_progress: bool = True):
        
        super().__init__(path_to_model, update_density_worker_settings, init_background_worker, density_batch_size, devices, density_compile_mode, ['rate_input'], show_progress)
        
        self._rate_approximation = BackgroundRateApproximation(self._major_version, self._ckpt['rate_input'])
        
        self._update_pool_arguments()
        
    def _update_pool_arguments(self):
        self._pool_arguments = [
            getattr(self, "_major_version", None),
            getattr(self, "_density_input", None),
            getattr(self, "_density_batch_size", None),
            getattr(self, "_density_compile_mode", None),
            ]
    
    def evaluate_rate(self, context: torch.Tensor) -> torch.Tensor:
        return self._rate_approximation.evaluate_rate(context)
        