from typing import List, Union, Optional, Dict
from pathlib import Path

import torch
import torch.multiprocessing as mp
from cosipy.response.ml.NFBase import DensityApproximation, CompileMode, NFBase, init_density_worker, update_density_worker_settings, DensityModel
from .NFBkgNormalizationDensityModels import TotalBackground1EMDDensityCMLPDGaussianCARQSFlow, TotalBackground2DDensityCMLPDGaussianCARQSFlow


class BkgNormalizationDensityApproximation(DensityApproximation):

    def _setup_model(self):
        version_map: Dict[int, DensityModel] = {
            1: TotalBackground2DDensityCMLPDGaussianCARQSFlow,
            2: TotalBackground1EMDDensityCMLPDGaussianCARQSFlow
        }
        if self._major_version not in version_map:
            raise ValueError(f"Unsupported major version {self._major_version} for Density Approximation")
        else:    
            model_class = version_map[self._major_version]
            self._model = model_class(self._density_input, self._worker_device, self._batch_size, self._compile_mode)
            self._expected_context_dim = self._model.context_dim
            self._expected_source_dim = self._model.source_dim

def init_bkg_normalization_density_worker(device_queue: mp.Queue, progress_queue: mp.Queue, major_version: int,
                                      density_input: Dict, density_batch_size: int, density_compile_mode: CompileMode):
    
    init_density_worker(device_queue, progress_queue, major_version,
                        density_input, density_batch_size,
                        density_compile_mode, BkgNormalizationDensityApproximation)

class NFBkgNormalizationDensity(NFBase):
    def __init__(self, path_to_model: Union[str, Path], density_batch_size: int = 100_000,
                 devices: Optional[List[Union[str, int, torch.device]]] = None,
                 density_compile_mode: CompileMode = "default", show_progress: bool = True):
        
        super().__init__(path_to_model, update_density_worker_settings, init_bkg_normalization_density_worker, density_batch_size, devices, density_compile_mode, ['norm_dimensions'], show_progress)
        
        self._norm_dimensions = self._ckpt['norm_dimensions']
        if self._norm_dimensions not in ["Em", "Phi", "EmPhi"]:
            raise ValueError(f"Unsupported normalization dimensions {self._norm_dimensions}")
        
        self._update_pool_arguments()
    
    @property
    def norm_dimensions(self) -> str:
        return self._norm_dimensions