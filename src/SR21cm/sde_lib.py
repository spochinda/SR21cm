import torch 
import numpy as np

class VPSDE():
  def __init__(self, beta_min=0.1, beta_max=20, timesteps=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__()
    self.timesteps = timesteps
    self.beta_min = beta_min
    self.beta_max = beta_max
    
    self.discrete_betas = torch.linspace(beta_min / timesteps, beta_max / timesteps, timesteps)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    b,*d  = x.shape
    beta_t = (self.beta_min + t * (self.beta_max - self.beta_min)).view(b,*[1]*len(d)).to(x.device)
    #print("beta shape", beta_t.shape)
    drift = -0.5 * beta_t * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion
  
  def rsde(self, x, t, score, probability_flow=False):
    """Create the drift and diffusion functions for the reverse SDE/ODE."""
    drift, diffusion = self.sde(x, t)
    #score = self.score_fn(t, model_output)
    drift = drift - diffusion ** 2 * score * (0.5 if probability_flow else 1.)
    # Set the diffusion function to zero for ODEs.
    diffusion = torch.zeros_like(diffusion) if probability_flow else diffusion
    return drift, diffusion
  
  def marginal_prob(self, x, t):
    #print devices:
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
    mean = torch.exp(log_mean_coeff) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.timesteps - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None, None, None] * x - x
    G = sqrt_beta
    return f, G