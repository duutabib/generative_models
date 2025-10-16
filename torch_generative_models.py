import torch


# ============================================================================
#  GAUSSIAN MODEL
# ============================================================================

class GaussianGenerator:
    def __init__(self) -> None:
        self.mean = None
        self.std = None
    
    def fit(self, data:torch.Tensor) -> None:
        self.mean  = torch.mean(data)
        self.std = torch.std(data)
        print(f"Learned: mean={self.mean:.4f}, std={self.std:.4f}")

    def sample(self, n_samples:int = 1) -> torch.Tensor:
        return torch.normal(self.mean, self.std, size=(n_samples,))

# ============================================================================
# MIXTURE OF GAUSSIANS (EM ALGORITHM)
# ============================================================================

class MixtureOfGaussian:
    """Fit mixture of Gaussians using EM algorithm."""

    def __init__(self, n_components:int= 2) -> None:
        self.n_components = n_components
        self.means = None
        self.stds = None
        self.weights = None

    def fit(self, data:torch.Tensor, max_iters:int = 50) -> None:
        data = data.ravel()
        n = len(data)

        # Initialize parameters randomly
        self.means = torch.rand(self.n_components) * (torch.max(data) - torch.min(data)) + torch.min(data)
        self.stds = torch.ones(self.n_components) * torch.std(data)
        self.weights = torch.ones(self.n_components) / self.n_components

        for iteration in range(max_iters):
            # E-step: compute responsibilities
            responsibilities = torch.zeros((n, self.n_components))
            for k in range(self.n_components):
                responsibilities[:, k] = self.weights[k] * self._gaussian(
                    data, self.means[k], self.stds[k]
                )
            responsibilities /= responsibilities.sum(dim=1, keepdim=True) + 1e-10 


            # M-step: update parameters 
            NK = responsibilities.sum(dim=0)
            self.weights = NK / n
            self.means = (responsibilities.T @ data)/ NK
            for k in range(self.n_components):
                diff = data - self.means[k]
                self.stds[k] = torch.sqrt((responsibilities[:, k] * diff**2).sum()/ NK[k])


        print(f"Learned {self.n_components} components:")
        for k in range(self.n_components):
            print(f"Components {k+1}: mean={self.means[k]:.4f}, std={self.stds[k]:.4f}, weight={self.weights[k]:.4f}")


    def _gaussian(self, x:torch.Tensor, mean:torch.Tensor, std:torch.Tensor) -> torch.Tensor:
       """Gaussian probability density""" 
       return torch.exp(-0.5 * ((x - mean)/std)**2) / (std * torch.sqrt(2 * torch.pi))
            
                    

    def sample(self, n_samples:int =1) -> torch.Tensor:
        """ Generate n Mixtures samples

        Args:
            n_samples (int, optional): Number of samples to generate. Defaults to 1.

        Returns:
            torch.Tensor: Generated samples

        """
        components = torch.multinomial(self.weights, n_samples, replacement=True)
        

        # Sample from components
        for k in range(self.n_components):
            mask = components == k
            if n_k > 0:
                samples[mask] = torch.normal(mean=self.means[k].item(), std=self.stds[k].item(), size = (n_k, ))
        return samples 


# ============================================================================
# HISTOGRAM BASED (NON-PARAMETRIC)
# ============================================================================

class HistogramGenerator:
    def __init__(self, n_bins:int = 50) -> None:
        """
         Even simpler: build a histogram and sample from it. 
         This is non-parametric  - just empirical distribution.
        """
        self.n_bins = n_bins
        self.bin_edges = None
        self.bin_probs = None


    def fit(self, data:torch.Tensor) -> None:
        """ Build histogram form data """
        counts = torch.histc(data, bins=self.n_bins)
        self.bin_edges = torch.linspace(data.min().item(), data.max().item(), self.n_bins + 1)
        self.bin_probs = counts / counts.sum()
        print(f"Built histogram with {self.n_bins} bins")

    def sample(self, n_samples:int =1) -> torch.Tensor:
        """ Generate n samples from histogram """
        bin_indices = torch.multinomial(self.bin_probs, n_samples, replacement=True)

        # Sample uniformly within each bin
        samples = torch.zeros(n_samples)
        for j, bin_idx in enumerate(bin_indices):
            left = self.bin_edges[bin_idx]
            right = self.bin_edges[bin_idx + 1]
            samples[j] = left  + (right - left) * torch.rand(1) 
        
        return samples
        

# ============================================================================
# BOOTSTRAP SAMPLING (TRULY MINIMAL)
# ============================================================================

class BootstrapGenerator:
    """
    The absolute minimal 'generative' model: just resample from the data.
    This is literally just torch.multinomial!
    """
    def __init__(self) -> None:
        self.data = None
    
    def fit(self, data: torch.Tensor) -> torch.Tensor:
        """Store the data """
        self.data = data.flatten()
        print(f"Stored {len(self.data)} data points")
        
    def sample(self, n_samples:int =1) -> torch.Tensor:
        """ Randomly sample from stored data (with replacement) """
        return torch.multinomial(self.data, n_samples, replacement=True)


   
# ============================================================================
#  VISUALIZATION 
# ============================================================================

def compare_models(real_data:torch.Tensor, n_samples:int=2000) -> None:  
    """ Compare all basic generative models """
    print("="*60)
    print("Comparing Basic Generative Models using PyTorch")
    print("="*60)

    # Fit all models
    models = {
        "Gaussian": GaussianGenerator(),
        "Mixture (2)": MixtureOfGaussian(n_components=2),
        "Histogram": HistogramGenerator(n_bins=30),
        "Bootstrap": BootstrapGenerator(),
    }
    for name, model in models.items():
        print(f"\n{name}")
        model.fit(real_data)
    

    # Generate Samples
    _, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # Plot real data
    axes[0].hist(real_data.flatten(), bins=50, alpha=0.7, density=True, color='blue')
    axes[0].set_title('Real Data')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].grid(True, alpha=0.3)


    # Plot each model's samples
    for idx, (name, model) in enumerate(models.items(), start=1):
        samples = model.sample(n_samples)
        axes[idx].hist(samples, bins=50, alpha=0.7, density=True, color='orange')
        axes[idx].set_title(f'{name} Generated')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Density')
        axes[idx].grid(True, alpha=0.3)
    
    # Hide last subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
#  EXAMPLES 
# ============================================================================


def generate_data() -> torch.Tensor:
    """ Generate some bimodal data """
    return torch.cat([
        torch.normal(mean=-2, std=0.5, size=(500,)),
        torch.normal(mean=2, std=0.5, size=(500,)),
    ])

def example_gaussian():
    """ Simplest possible example """
    print("\n" + "="*60)
    print("Example 1: Gaussian Generator (Simplest!")
    print("="*60)

    # Create some bimodal data 
    data = generate_data()

    # Fit and sample
    model = GaussianGenerator()
    model.fit(data)
    samples = model.sample(1000)
    
    # plot 
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1) 
    plt.hist(data, bins=50, alpha=0.7, density=True, color="blue")
    plt.title("Real Data (Bimodal)")
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    
    # plot samples
    plt.subplot(1, 2, 2)
    plt.hist(samples, bins=50, alpha=0.7, density=True, color="orange")
    plt.title("Generated Data (Unimodal)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    print(f"\nGenerated  5 samples: {samples[:5].tolist()}")


def example_mixture():
    """ Example with Mixture of Gaussians """
    print("\n" + "="*60)
    print("Example 2: Mixture of Gaussians")
    print("="*60)

    # Create some bimodal data 
    data = generate_data()

    # Fit and sample
    model = SimpleMixtureOfGaussians(n_components=2)
    model.fit(data)
    samples = model.sample(1000)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1) 
    plt.hist(data, bins=50, alpha=0.7, density=True, color="blue")
    plt.title("Real Data (Bimodal)")
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    
    # plot samples
    plt.subplot(1, 2, 2)
    plt.hist(samples, bins=50, alpha=0.7, density=True, color="orange")
    plt.title("Generated Data (Unimodal)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    print(f"\nGenerated  5 samples: {samples[:5].tolist()}")


def example_all():
    """ Run all examples """
    
    # Create datya
    d0 = torch.cat([
        torch.normal(mean=-3, std=0.4, size=(400,)),
        torch.normal(mean=0, std=0.6, size=(600,)), 
        torch.normal(mean=3, std=0.4, size=(400,)),
    ])


    compare_models(d0, n_samples=1400)


if __name__ == "__main__" :
    # Run simplest example
    example_gaussian()


    # Run mixture exampke
    example_mixture()

    # Run all examples
    example_all()


    logging.info("\n" + "="*60)
    logging.info("Summary: Most Basic Generative Models")
    logging.info("="*60)
    logging.info("1. Gaussian: Just Mean + std (2 paramaters)")
    logging.info("2. Mixture of Gaussians: Learn multiple means and stds")
    logging.info("3. Histogram: Non-parametric")
    logging.info("4. Bootstrap: Resample from data")
    logging.info("="*60)
    logging.info(" All of these are way simpler than neural networks...Done")