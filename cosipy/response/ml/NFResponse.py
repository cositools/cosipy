from typing import List, Union, Optional, Dict, Tuple, Callable
from pathlib import Path
from tqdm.auto import tqdm
import queue

import torch
import torch.multiprocessing as mp
from .NFResponseModels import UnpolarizedDensityCMLPDGaussianCARQSFlow, UnpolarizedAreaSphericalHarmonicsExpansion
from .NFBase import DensityApproximation, CompileMode, NFBase, init_density_worker, update_density_worker_settings, AreaModel, DensityModel
import cosipy.response.ml.NFWorkerState as NFWorkerState


class ResponseDensityApproximation(DensityApproximation):

    def _setup_model(self):
        version_map: Dict[int, DensityModel] = {
            1: UnpolarizedDensityCMLPDGaussianCARQSFlow,
        }
        if self._major_version not in version_map:
            raise ValueError(f"Unsupported major version {self._major_version} for Density Approximation")
        else:    
            model_class = version_map[self._major_version]
            self._model = model_class(self._density_input, self._worker_device, self._batch_size, self._compile_mode)
            self._expected_context_dim = self._model.context_dim
            self._expected_source_dim = self._model.source_dim

class AreaApproximation:
    def __init__(self, major_version: int, area_input: Dict, worker_device: Union[str, int, torch.device], batch_size: int, compile_mode: CompileMode):
        self._major_version = major_version
        self._worker_device = worker_device
        self._area_input = area_input
        self._batch_size = batch_size
        self._compile_mode = compile_mode
        
        self._setup_model()

    def _setup_model(self):
        version_map: Dict[int, AreaModel] = {
            1: UnpolarizedAreaSphericalHarmonicsExpansion,
        }
        if self._major_version not in version_map:
            raise ValueError(f"Unsupported major version {self._major_version} for Effective Area Approximation")
        else:    
            model_class = version_map[self._major_version]
            self._model = model_class(self._area_input, self._worker_device, self._batch_size, self._compile_mode)
            self._expected_context_dim = self._model.context_dim
    
    def evaluate_effective_area(self, context: torch.Tensor,
                                progress_callback: Optional[Callable[[int], None]] = None) -> torch.Tensor:
        dim_context = context.shape[1]
        
        if dim_context != self._expected_context_dim:
            raise ValueError(
                f"Feature mismatch: {type(self._model).__name__} expects "
                f"{self._expected_context_dim} features, but context has {dim_context}."
            )
        
        list_context = [context[:, i] for i in range(dim_context)]
        
        return self._model.evaluate_effective_area(*list_context, progress_callback=progress_callback)

def update_response_worker_settings(args: Tuple[str, Union[int, CompileMode]]):
    attr, value = args
    
    if attr == 'area_batch_size':
        NFWorkerState.area_module._model.batch_size = value
    elif attr == 'area_compile_mode':
        NFWorkerState.area_module._model.compile_mode = value
    else:
        update_density_worker_settings(args)

def init_response_worker(device_queue: mp.Queue, progress_queue: mp.Queue, major_version: int, area_input: Dict,
                         density_input: Dict, area_batch_size: int, density_batch_size: int,
                         area_compile_mode: CompileMode, density_compile_mode: CompileMode):
    
    init_density_worker(device_queue, progress_queue, major_version,
                        density_input, density_batch_size,
                        density_compile_mode, ResponseDensityApproximation)
    
    NFWorkerState.area_module = AreaApproximation(major_version, area_input, NFWorkerState.worker_device, area_batch_size, area_compile_mode)

def evaluate_area_task(args: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    context, indices = args
    
    sub_context = context[indices, :]
    
    cb = lambda n: NFWorkerState.progress_queue.put(n) if hasattr(NFWorkerState, 'progress_queue') else None
    return NFWorkerState.area_module.evaluate_effective_area(sub_context, progress_callback=cb)

class NFResponse(NFBase):
    def __init__(self, path_to_model: Union[str, Path], area_batch_size: int = 300_000, density_batch_size: int = 100_000,
                 devices: Optional[List[Union[str, int, torch.device]]] = None, area_compile_mode: CompileMode = "max-autotune-no-cudagraphs",
                 density_compile_mode: CompileMode = "default", show_progress: bool = True):
        
        super().__init__(path_to_model, update_response_worker_settings, init_response_worker, density_batch_size, devices, density_compile_mode, ['is_polarized', 'area_input'], show_progress)
        
        self._is_polarized = self._ckpt['is_polarized']
        self._area_input = self._ckpt['area_input']
        
        self.area_batch_size = area_batch_size
        self.area_compile_mode = area_compile_mode
        
        self._update_pool_arguments()
    
    @property
    def is_polarized(self) -> bool:
        return self._is_polarized
    
    @property
    def area_batch_size(self) -> int:
        return self._area_batch_size
    
    @area_batch_size.setter
    def area_batch_size(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("area_batch_size must be a positive integer")
        self._area_batch_size = value
        self._update_pool_arguments()
        self._update_worker_config('area_batch_size', value)
    
    @property
    def area_compile_mode(self) -> CompileMode: return self._area_compile_mode
    
    @area_compile_mode.setter
    def area_compile_mode(self, value: CompileMode):
        self._area_compile_mode = value
        self._update_pool_arguments()
        self._update_worker_config('area_compile_mode', value)
    
    def _update_pool_arguments(self):
        self._pool_arguments = [
            getattr(self, "_major_version", None),
            getattr(self, "_area_input", None),
            getattr(self, "_density_input", None),
            getattr(self, "_area_batch_size", None),
            getattr(self, "_density_batch_size", None),
            getattr(self, "_area_compile_mode", None),
            getattr(self, "_density_compile_mode", None),
            ]
    
    def evaluate_effective_area(self, context: torch.Tensor, devices: Optional[List[Union[str, int, torch.device]]]=None) -> torch.Tensor:
        temp_pool = False
        if self._pool is None:
            target_devices = devices if devices is not None else self._devices
            if not target_devices:
                raise RuntimeError("No compute pool initialized and no devices provided/set.")
            
            self.init_compute_pool(target_devices)
            temp_pool = True
        
        try:
            if not context.is_shared():
                context.share_memory_()

            n_data = context.shape[0]
            indices = torch.tensor_split(torch.arange(n_data), self._num_workers)
            
            tasks = [(context, idx) for idx in indices]
            
            async_result = self._pool.map_async(evaluate_area_task, tasks)
            with tqdm(total=n_data, disable=(not self.show_progress), desc="Evaluating the effective area", unit="calls", leave=False, smoothing=0.20) as pbar:
                while not async_result.ready():
                    try:
                        while True:
                            completed = self._progress_queue.get_nowait()
                            pbar.update(completed)
                    except queue.Empty:
                        async_result.wait(timeout=0.1)

                while not self._progress_queue.empty():
                    try:
                        pbar.update(self._progress_queue.get_nowait())
                    except queue.Empty:
                        break
            
            results = async_result.get()
            return torch.cat(results, dim=0)

        finally:
            if temp_pool:
                self.shutdown_compute_pool()