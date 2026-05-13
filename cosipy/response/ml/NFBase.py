from typing import List, Union, Optional, Literal, Tuple, Dict, Optional, Callable
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
from tqdm.auto import tqdm
import queue

import torch
from torch import nn
import torch.multiprocessing as mp
import normflows as nf
import cosipy.response.ml.NFWorkerState as NFWorkerState


CompileMode = Optional[Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]]

def build_cmlp_diaggaussian_base(input_dim: int, output_dim: int,
                                 hidden_dim: int, num_hidden_layers: int) -> nf.distributions.BaseDistribution:
    context_encoder = BaseMLP(input_dim, output_dim, hidden_dim, num_hidden_layers)
    return ConditionalDiagGaussian(shape=output_dim//2, context_encoder=context_encoder)

def build_c_arqs_flow(base: nf.distributions.BaseDistribution, num_layers: int,
                      latent_dim: int, context_dim: int, num_bins: int,
                      num_hidden_units: int, num_residual_blocks: int) -> nf.ConditionalNormalizingFlow:
        flows = []
        for _ in range(num_layers):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(num_input_channels = latent_dim,
                                                                     num_blocks = num_residual_blocks,
                                                                     num_hidden_channels = num_hidden_units,
                                                                     num_bins = num_bins,
                                                                     num_context_channels = context_dim)]
            flows += [nf.flows.LULinearPermute(latent_dim)]
        return nf.ConditionalNormalizingFlow(base, flows)

class NNDensityInferenceWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self._model = model

    def forward(self, 
                source: Optional[torch.Tensor] = None, 
                context: Optional[torch.Tensor] = None,
                n_samples: Optional[int] = None, 
                mode: str = "inference") -> torch.Tensor:
        if mode == "inference":
            if context is None:
                return torch.exp(self._model.log_prob(source))
            else:
                return torch.exp(self._model.log_prob(source, context))
        elif mode == "sampling":
            if context is None:
                return self._model.sample(num_samples=n_samples)[0]
            else:
                return self._model.sample(num_samples=n_samples, context=context)[0]
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'inference' or 'sampling'.")

class ConditionalDiagGaussian(nf.distributions.BaseDistribution):
    def __init__(self, shape: Union[int, List[int], Tuple[int, ...]], context_encoder: nn.Module):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        self.context_encoder = context_encoder

    def forward(self, num_samples: int=1, context: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.context_encoder(context)
        split_ind = encoder_output.shape[-1] // 2
        mean = encoder_output[..., :split_ind]
        log_scale = encoder_output[..., split_ind:]
        eps = torch.randn(
            (num_samples,) + self.shape, dtype=mean.dtype, device=mean.device
        )
        z = mean + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1))
        )
        return z, log_p

    def log_prob(self, z: torch.Tensor, context: Optional[torch.Tensor]=None) -> torch.Tensor:
        encoder_output = self.context_encoder(context)
        split_ind = encoder_output.shape[-1] // 2
        mean = encoder_output[..., :split_ind]
        log_scale = encoder_output[..., split_ind:]
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - mean) / torch.exp(log_scale), 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p

class BaseMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layers: int):
        super().__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseModel(ABC):
    
    def __init__(self, compile_mode: CompileMode, batch_size: int, 
                 worker_device: Union[str, int, torch.device], input: Dict):
        self._worker_device = torch.device(worker_device)
        
        self._base_model = self._init_model(input)
        
        self._compile_mode = compile_mode
        self._compiled_cache = {}
        
        self._update_model_op()
        
        self._is_cuda = (self._worker_device.type == 'cuda')
        self.batch_size = batch_size
    
    @abstractmethod
    def _init_model(self, input: Dict) -> Union[nn.Module, Callable]: ...
    
    @property
    @abstractmethod
    def context_dim(self) -> int: ...

    @property
    def compile_mode(self) -> CompileMode:
        return self._compile_mode

    @compile_mode.setter
    def compile_mode(self, value: CompileMode):
        if value != self._compile_mode:
            self._compile_mode = value
            self._update_model_op()
    
    def _update_model_op(self):
        if self._compile_mode is None:
            self._model_op = self._base_model
        else:
            if self._compile_mode not in self._compiled_cache:
                self._compiled_cache[self._compile_mode] = torch.compile(
                    self._base_model, 
                    mode=self._compile_mode
                )
            self._model_op = self._compiled_cache[self._compile_mode]

    @property
    def batch_size(self) -> int: 
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Batch size must be a positive integer, got {value}")
        self._batch_size = value

class AreaModel(BaseModel):
    @abstractmethod
    def evaluate_effective_area(self, *args: torch.Tensor, progress_callback: Optional[Callable[[int], None]] = None) -> torch.Tensor: ...

class DensityModel(BaseModel):
    @property
    @abstractmethod
    def source_dim(self) -> int: ...

    @torch.inference_mode()
    def sample_density(self, *args: torch.Tensor,
                       progress_callback: Optional[Callable[[int], None]] = None) -> torch.Tensor:
        N = args[0].shape[0]
        
        result = torch.empty((N, self.source_dim), dtype=torch.float32, device="cpu")
        failed_mask = torch.zeros(N, dtype=torch.bool, device="cpu")
        
        for start in range(0, N, self._batch_size):
            end = min(start + self._batch_size, N)
            batch_len = end - start
                
            b_ctx = [t[start:end].to(self._worker_device) for t in args]
            n_ctx = self._transform_context(*b_ctx)
            
            n_latent = self._model_op(context=n_ctx, mode="sampling", n_samples=batch_len)
            result[start:end] = self._inverse_transform_coordinates(*(n_latent.T), *b_ctx)
            failed_mask[start:end] = ~self._valid_samples(*(n_latent.T), *b_ctx)
            
            if progress_callback is not None:
                amount = batch_len - torch.sum(failed_mask[start:end]).item()
                progress_callback(amount)

        if torch.any(failed_mask):
            result[failed_mask] = self.sample_density(*[t[failed_mask] for t in args], progress_callback=progress_callback)

        return result

    @abstractmethod
    def _inverse_transform_coordinates(self, *args: torch.Tensor) -> torch.Tensor: ...
    
    @abstractmethod
    def _valid_samples(self, *args: torch.Tensor) -> torch.Tensor: ...
    
    @abstractmethod
    def _transform_context(self, *args: torch.Tensor) -> torch.Tensor: ...
    
    @abstractmethod
    def _transform_coordinates(self, *args: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    
    @torch.inference_mode()
    def evaluate_density(self, *args: torch.Tensor,
                         progress_callback: Optional[Callable[[int], None]] = None) -> torch.Tensor:
        
        N = args[0].shape[0]
        result = torch.empty(N, dtype=torch.float32, device="cpu")
        
        for start in range(0, N, self._batch_size):
            end = min(start + self._batch_size, N)
            batch_len = end - start
            
            ctx, src, jac = self._transform_coordinates(*[t[start:end].to(self._worker_device) for t in args])
            result[start:end] = self._model_op(src, ctx, mode="inference") * jac
            
            if progress_callback is not None:
                progress_callback(batch_len)

        return result
    
class RateModel(ABC):
    def __init__(self, rate_input: Dict):
        self._unpack_rate_input(rate_input)
    
    @property
    @abstractmethod
    def context_dim(self) -> int: ...
    
    @abstractmethod
    def _unpack_rate_input(self, rate_input: Dict): ...
    
    @abstractmethod
    def evaluate_rate(self, *args: torch.Tensor) -> torch.Tensor: ...
    

class DensityApproximation(ABC):
    def __init__(self, major_version: int, density_input: Dict, worker_device: Union[str, int, torch.device], batch_size: int, compile_mode: CompileMode):
        self._major_version = major_version
        self._worker_device = worker_device
        self._density_input = density_input
        self._batch_size = batch_size
        self._compile_mode = compile_mode
        
        self._model: DensityModel
        self._expected_context_dim: int
        self._expected_source_dim: int
        
        self._setup_model()
    
    @abstractmethod
    def _setup_model(self): ...
    
    def evaluate_density(self, context: torch.Tensor, source: torch.Tensor,
                         progress_callback: Optional[Callable[[int], None]] = None) -> torch.Tensor:
        dim_context = context.shape[1]
        dim_source = source.shape[1]
        
        if dim_context != self._expected_context_dim:
            raise ValueError(
                f"Feature mismatch: {type(self._model).__name__} expects "
                f"{self._expected_context_dim} features, but context has {dim_context}."
            )
        elif dim_source != self._expected_source_dim:
            raise ValueError(
                f"Feature mismatch: {type(self._model).__name__} expects "
                f"{self._expected_source_dim} features, but source has {dim_source}."
            )
            
        list_context = [context[:, i] for i in range(dim_context)]
        list_source = [source[:, i] for i in range(dim_source)]
        
        return self._model.evaluate_density(*list_context, *list_source, progress_callback=progress_callback)
    
    def sample_density(self, context: torch.Tensor,
                       progress_callback: Optional[Callable[[int], None]] = None) -> torch.Tensor:
        dim_context = context.shape[1]
        
        if dim_context != self._expected_context_dim:
            raise ValueError(
                f"Feature mismatch: {type(self._model).__name__} expects "
                f"{self._expected_context_dim} features, but context has {dim_context}."
            )
            
        list_context = [context[:, i] for i in range(dim_context)]
        
        return self._model.sample_density(*list_context, progress_callback=progress_callback)

def cuda_cleanup_task(_) -> bool:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True

def update_density_worker_settings(args: Tuple[str, Union[int, CompileMode]]):
    attr, value = args
    
    if attr == 'density_batch_size':
        NFWorkerState.density_module._model.batch_size = value
    elif attr == 'density_compile_mode':
        NFWorkerState.density_module._model.compile_mode = value
    else:
        raise ValueError(f"Unknown attribute: {attr}")

def init_density_worker(device_queue: mp.Queue, progress_queue: mp.Queue, major_version: int,
                        density_input: Dict, density_batch_size: int,
                        density_compile_mode: CompileMode, density_approximation: DensityApproximation):
    
    NFWorkerState.worker_device = torch.device(device_queue.get())
    NFWorkerState.progress_queue = progress_queue
    if NFWorkerState.worker_device.type == 'cuda':
        torch.cuda.set_device(NFWorkerState.worker_device)
    
    NFWorkerState.density_module = density_approximation(major_version, density_input, NFWorkerState.worker_device, density_batch_size, density_compile_mode)

def evaluate_density_task(args: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    context, source, indices = args
    
    sub_context = context[indices, :]
    sub_source = source[indices, :]
    
    cb = lambda n: NFWorkerState.progress_queue.put(n) if hasattr(NFWorkerState, 'progress_queue') else None
    return NFWorkerState.density_module.evaluate_density(sub_context, sub_source, progress_callback=cb)

def sample_density_task(args: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    context, indices = args
    
    sub_context = context[indices, :]
        
    cb = lambda n: NFWorkerState.progress_queue.put(n) if hasattr(NFWorkerState, 'progress_queue') else None
    return NFWorkerState.density_module.sample_density(sub_context, progress_callback=cb)

class NFBase():
    def __init__(self, path_to_model: Union[str, Path], update_worker, pool_worker, density_batch_size: int = 100_000,
                 devices: Optional[List[Union[str, int, torch.device]]] = None, density_compile_mode: CompileMode = "default",
                 additional_required_keys: List[str] = None, show_progress: bool = True):
        self._ckpt = torch.load(str(path_to_model), map_location=torch.device('cpu'), weights_only=False)
        
        required_keys = ['version', 'density_input'] + (additional_required_keys or [])
        
        for key in required_keys:
            if key not in self._ckpt:
                raise KeyError(
                    f"Invalid Checkpoint: Metadata key '{key}' not found in {str(path_to_model)}. "
                    f"Ensure you saved the model as a dictionary, not just the state_dict."
                )
        
        self._version = self._ckpt['version']
        self._major_version = int(self._version.split('.')[0])
        self._density_input = self._ckpt['density_input']
        
        self._pool = None
        self._has_cuda = False
        self._num_workers = 0
        self._ctx = mp.get_context("spawn")
        self._pool_worker = pool_worker
        self._update_worker = update_worker
        self.show_progress = show_progress
        self._progress_queue = None
        
        self.density_batch_size = density_batch_size
        self.density_compile_mode = density_compile_mode
        
        self._update_pool_arguments()
        
        if devices is not None:
            self.devices = devices
        else:
            self._devices = []
        
    @property
    def show_progress(self) -> bool:
        return self._show_progress
    
    @show_progress.setter
    def show_progress(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("show_progress must be a boolean")
        self._show_progress = value
    
    @property
    def devices(self) -> List[Union[str, int, torch.device]]:
        return self._devices
    
    @devices.setter
    def devices(self, value: List[Union[str, int, torch.device]]):
        if not isinstance(value, list):
            raise ValueError("devices must be a list of device identifiers")
        self._devices = value
    
    @property
    def density_batch_size(self) -> int:
        return self._density_batch_size
    
    @density_batch_size.setter
    def density_batch_size(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("density_batch_size must be a positive integer")
        self._density_batch_size = value
        self._update_pool_arguments()
        self._update_worker_config('density_batch_size', value)
    
    @property
    def density_compile_mode(self) -> CompileMode: return self._density_compile_mode
    
    @density_compile_mode.setter
    def density_compile_mode(self, value: CompileMode):
        self._density_compile_mode = value
        self._update_pool_arguments()
        self._update_worker_config('density_compile_mode', value)
    
    @property
    def active_pool(self) -> bool: return self._pool is not None
    
    def _update_worker_config(self, attr: str, value: Union[int, CompileMode]):
        if self._pool is not None:
            self._pool.map(self._update_worker, [(attr, value)] * self._num_workers)
    
    def _update_pool_arguments(self):
        self._pool_arguments = [
            getattr(self, "_major_version", None),
            getattr(self, "_density_input", None),
            getattr(self, "_density_batch_size", None),
            getattr(self, "_density_compile_mode", None),
            ]
    
    def clean_compute_pool(self):
        if self._pool:
            self._pool.map(cuda_cleanup_task, range(self._num_workers))
        
    def shutdown_compute_pool(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
        self._pool = None
        
        self._num_workers = 0
        self._has_cuda = None
        
        if self._progress_queue is not None:
            self._progress_queue.close()
            self._progress_queue.join_thread()
        self._progress_queue = None
    
    def init_compute_pool(self, devices: Optional[List[Union[str, int, torch.device]]]=None):
        active_devices = devices if devices is not None else self._devices
        
        if not active_devices:
            raise RuntimeError("Cannot initialize pool: no devices provided as argument or set as fallback.")
        
        if self._pool:
            print("Warning: Pool already initialized. Shutting down old pool first.")
            self.shutdown_compute_pool()
        
        self._num_workers = len(active_devices)
        self._has_cuda = any(torch.device(d).type == 'cuda' for d in active_devices)
        
        device_queue = self._ctx.Queue()
        for d in active_devices:
            device_queue.put(d)
        self._progress_queue = self._ctx.Queue()
        
        self._pool = self._ctx.Pool(
            processes=self._num_workers,
            initializer=self._pool_worker,
            initargs=(device_queue, self._progress_queue,*self._pool_arguments),
        )
    
    def sample_density(self, context: torch.Tensor, devices: Optional[List[Union[str, int, torch.device]]]=None) -> torch.Tensor:
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
            
            async_result = self._pool.map_async(sample_density_task, tasks)
            with tqdm(total=n_data, disable=(not self.show_progress), desc="Sampling the density", unit="calls", leave=False, smoothing=0.20) as pbar:
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
    
    def evaluate_density(self, context: torch.Tensor, source: torch.Tensor, devices: Optional[List[Union[str, int, torch.device]]]=None) -> torch.Tensor:
        temp_pool = False
        if self._pool is None:
            target_devices = devices if devices is not None else self._devices
            if not target_devices:
                raise RuntimeError("No compute pool initialized and no devices provided/set.")
            
            self.init_compute_pool(target_devices)
            temp_pool = True
        
        try:    
            if not context.is_shared(): context.share_memory_()
            if not source.is_shared(): source.share_memory_()
            
            n_data = context.shape[0]
            indices = torch.tensor_split(torch.arange(n_data), self._num_workers)
            
            tasks = [(context, source, idx) for idx in indices]
            
            async_result = self._pool.map_async(evaluate_density_task, tasks)
            with tqdm(total=n_data, disable=(not self.show_progress), desc="Evaluating the density", unit="calls", leave=False, smoothing=0.20) as pbar:
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
