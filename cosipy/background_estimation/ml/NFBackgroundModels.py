import numpy as np

from typing import Union, Tuple, Dict

from cosipy.response.ml.NFBase import CompileMode, build_c_arqs_flow, build_cmlp_diaggaussian_base, NNDensityInferenceWrapper, DensityModel, RateModel
import normflows as nf
import torch


class TotalDC4BackgroundRate(RateModel):
    @property
    def context_dim(self) -> int: 
        return 1
    
    def _unpack_rate_input(self, rate_input: Dict):
        self._slew_duration = rate_input["slew_duration"]
        self._obs_duration  = rate_input["obs_duration"]
        self._start_time    = rate_input["start_time"]
        
        self._offset: float                            = rate_input["offset"]
        self._slope: float                             = rate_input["slope"]
        self._buildup_A: Tuple[float, float]           = rate_input["buildup"][0]
        self._buildup_T: Tuple[float, float]           = rate_input["buildup"][1]
        self._scale: float                             = rate_input["scale"]
        self._cutoff_T: float                          = rate_input["cutoff"][0]
        self._cutoff_A: Tuple[float, float, float]     = rate_input["cutoff"][1]
        self._cutoff_kappa: Tuple[float, float, float] = rate_input["cutoff"][2]
        self._cutoff_mu: Tuple[float, float, float]    = rate_input["cutoff"][3]
        self._outlocs: torch.Tensor                    = rate_input["outlocs"]
        self._saa_decay_A: Tuple[float, float]         = rate_input["saa_decay"][0]
        self._saa_decay_T: Tuple[float, float]         = rate_input["saa_decay"][1]
    
    @staticmethod
    def _buildup(time: torch.Tensor, A: float, T: float) -> torch.Tensor:
        return A * (1 - torch.exp(-time * np.log(2) / T))
    
    def _pointing_scale(self, time: torch.Tensor, scale: float, k0: float=1.0) -> torch.Tensor:
        half_slew = self._slew_duration / 2.0
        full_cycle = 2 * (self._obs_duration + self._slew_duration)
        rel_t = time % full_cycle
        k = k0 / self._slew_duration
        
        t1 = self._obs_duration + half_slew
        t2 = full_cycle - half_slew
        
        s1 = 1 / (1 + torch.exp(-k * (rel_t - t1)))
        s2 = 1 / (1 + torch.exp(-k * (rel_t - t2)))
        s0 = 1 / (1 + torch.exp(-k * (rel_t - (t2 - full_cycle))))
        
        return scale * (s0 - s1 + s2)
    
    @staticmethod
    def _von_mises(time: torch.Tensor, T: float, A: float, kappa: float, mu: float) -> torch.Tensor:
        return A * torch.exp(kappa * torch.cos(2 * np.pi * (time - mu) / T))
    
    def _base_cutoff(self, time, T: float, A: Tuple[float, float, float], 
                     kappa: Tuple[float, float, float], mu: Tuple[float, float, float]) -> torch.Tensor:
        return self._von_mises(time, T, A[0], kappa[0], mu[0]) + \
               self._von_mises(time, T, A[1], kappa[1], mu[1]) + \
               self._von_mises(time, T, A[2], kappa[2], mu[2])
    
    def _orbital_period(self, time, scale: float, T: float, A: Tuple[float, float, float], 
                        kappa: Tuple[float, float, float], mu: Tuple[float, float, float]) -> torch.Tensor:
        sample_times = torch.linspace(0, T, 1000)
        fmin = torch.min(self._base_cutoff(sample_times, T, A, kappa, mu))
        
        fval = self._base_cutoff(time, T, A, kappa, mu)
        
        return fmin + (fval - fmin) * (1 + scale)
    
    @staticmethod
    def _decay(time: torch.Tensor, A: float, T: float) -> torch.Tensor:
        return A * torch.exp(-time * np.log(2) / T)
    
    def _saa_decay(self, time: torch.Tensor, A: Tuple[float, float], T: Tuple[float, float]) -> torch.Tensor:
        exit_times = (self._outlocs - self._start_time)/60
        last_exit = exit_times[torch.searchsorted(exit_times, time, right=True) - 1]
        
        return self._decay(time - last_exit, A[0], T[0]) + self._decay(time - last_exit, A[1], T[1])
    
    def evaluate_rate(self, *args: torch.Tensor) -> torch.Tensor:
        time  = (args[0] - self._start_time)/60
        rate  = self._offset + self._slope * time
        rate += self._buildup(time, self._buildup_A[0], self._buildup_T[0])
        rate += self._buildup(time, self._buildup_A[1], self._buildup_T[1])
        rate += self._orbital_period(time, self._pointing_scale(time, self._scale),
                                     self._cutoff_T, self._cutoff_A,
                                     self._cutoff_kappa, self._cutoff_mu)
        rate += self._saa_decay(time, self._saa_decay_A, self._saa_decay_T)
        
        return rate
    
class TotalBackgroundDensityCMLPDGaussianCARQSFlow(DensityModel):
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
        
        self._start_time: float     = input["start_time"]
        self._total_time: float     = input["total_time"]
        self._period: float         = input["period"]
        self._slew_duration: float  = input["slew_duration"]
        self._obs_duration: float   = input["obs_duration"]
        self._outlocs: torch.Tensor = input["outlocs"].to(self._worker_device)
        
        return self._load_model()
    
    @property
    def context_dim(self) -> int: 
        return 1
    
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

    def _inverse_transform_coordinates(self, *args: torch.Tensor) -> torch.Tensor:
        nem, nphi, npsi, nchi, _ = args
        
        em  = 10 ** (2 * (nem + 1))
        phi = nphi * np.pi
        az  = npsi * 2 * np.pi
        pol = torch.acos(2 * nchi - 1)

        return torch.stack([em, phi, az, pol], dim=1)

    def _transform_coordinates(self, *args: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time, em, phi, scatt_az, scatt_pol = args

        jac = 1/(np.log(10) * em * 8*np.pi**2)

        ctx = self._transform_context(time)

        src = torch.cat([
            (torch.log10(em)/2 - 1).unsqueeze(1),
            (phi / np.pi).unsqueeze(1),
            (scatt_az / (2 * np.pi)).unsqueeze(1),
            ((torch.cos(scatt_pol) + 1) / 2).unsqueeze(1)
        ], dim=1)

        return ctx.to(torch.float32), src.to(torch.float32), jac.to(torch.float32)

    def _sigmoid_switch(self, t: torch.Tensor, k0: float=1.0) -> torch.Tensor:
        half_slew = self._slew_duration / 2.0
        full_cycle = 2 * (self._obs_duration + self._slew_duration) 
        rel_t = (t - self._start_time) % full_cycle
        k = k0 / self._slew_duration 

        t1 = self._obs_duration + half_slew
        t2 = full_cycle - half_slew

        s1 = 1 / (1 + torch.exp(-k * (rel_t - t1)))
        s2 = 1 / (1 + torch.exp(-k * (rel_t - t2)))
        s0 = 1 / (1 + torch.exp(-k * (rel_t - (t2 - full_cycle))))

        return s0 - s1 + s2

    def _transform_context(self, *args: torch.Tensor) -> torch.Tensor: 
        time = args[0]
        
        last_exits = self._outlocs[torch.searchsorted(self._outlocs, time, right=True) - 1]
        time_since_start = (time - self._start_time)/self._total_time
        pointing_phase = self._sigmoid_switch(time, k0 = 1.0)
        time_since_saa = (time - last_exits)/self._period
        phase_c = (torch.cos((time - self._start_time)/self._period * 2 * np.pi) + 1) / 2 
        phase_s = (torch.sin((time - self._start_time)/self._period * 2 * np.pi) + 1) / 2 
        
        ctx = torch.hstack([
            (time_since_start).unsqueeze(1),
            (pointing_phase).unsqueeze(1),
            (time_since_saa).unsqueeze(1),
            (phase_c).unsqueeze(1),
            (phase_s).unsqueeze(1)
        ])
        
        return ctx.to(torch.float32)
    
    def _valid_samples(self, *args: torch.Tensor) -> torch.Tensor:
        nem, nphi, npsi, nchi, _ = args
        
        valid_mask = (nem  >= 0.0) & \
                     (nphi >  0.0) & (nphi <= 1.0) & \
                     (npsi >= 0.0) & (npsi <= 1.0) & \
                     (nchi >= 0.0) & (nchi <= 1.0) & \
                     (nem  >= (np.log10(self._menergy_cuts[0])/2 - 1)) & \
                     (nem  <= (np.log10(self._menergy_cuts[1])/2 - 1)) & \
                     (nphi >= self._phi_cuts[0]/np.pi) & \
                     (nphi <= self._phi_cuts[1]/np.pi)
                     
        return valid_mask
