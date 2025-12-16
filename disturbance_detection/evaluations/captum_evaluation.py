"""
Minimal Captum-based attribution evaluation for forest disturbance detection.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple

try:
    from captum.attr import IntegratedGradients, Saliency
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False


class CaptumEvaluator:
    """Minimal Captum evaluator for band and time step attribution."""
    
    def __init__(self, model, device, config):
        if not CAPTUM_AVAILABLE:
            raise ImportError("Install captum: pip install captum")
        
        self.model = model.eval()
        self.device = device
        self.config = config
        self.features = config.get_features()
        
        # Only the two most useful methods
        self.ig = IntegratedGradients(self.model)
        self.saliency = Saliency(self.model)
    
    def get_attribution(self, input_tensor: torch.Tensor, method: str = 'integrated_gradients') -> np.ndarray:
        """Get attribution for input tensor."""
        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        
        if method == 'integrated_gradients':
            baseline = torch.zeros_like(input_tensor)
            attr = self.ig.attribute(input_tensor, baselines=baseline, target=1)
        else:  # saliency
            attr = self.saliency.attribute(input_tensor, target=1)
        
        return attr.detach().cpu().numpy()
    
    def plot_heatmap(self, attributions: np.ndarray, title: str = "Attribution"):
        """Plot attribution heatmap."""
        # Take first sample if batch
        attr = attributions[0] if len(attributions.shape) == 3 else attributions
        
        # Time labels
        time_labels = [f't-{self.config.window_size-1-i}' for i in range(self.config.window_size)]
        time_labels[-1] = 't (target)'
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(attr, 
                   xticklabels=time_labels,
                   yticklabels=self.features,
                   cmap='RdBu_r', center=0, annot=True, fmt='.3f')
        plt.title(f'{title}\nPositive = increases disturbance probability')
        plt.xlabel('Time Steps')
        plt.ylabel('Bands')
        plt.tight_layout()
        plt.show()
    
    def plot_summary(self, attributions: np.ndarray, title: str = "Attribution"):
        """Plot temporal and band summaries."""
        attr = attributions[0] if len(attributions.shape) == 3 else attributions
        
        # Sum across dimensions
        temporal_attr = np.sum(attr, axis=0)  # Sum across bands
        band_attr = np.sum(attr, axis=1)      # Sum across time
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Temporal importance
        time_labels = [f't-{self.config.window_size-1-i}' for i in range(self.config.window_size)]
        time_labels[-1] = 't (target)'
        
        ax1.bar(range(len(temporal_attr)), temporal_attr, 
                color=['red' if x < 0 else 'blue' for x in temporal_attr])
        ax1.axhline(0, color='black', alpha=0.3)
        ax1.set_title('Temporal Importance')
        ax1.set_xticks(range(len(time_labels)))
        ax1.set_xticklabels(time_labels, rotation=45)
        
        # Band importance  
        ax2.bar(range(len(band_attr)), band_attr,
                color=['red' if x < 0 else 'blue' for x in band_attr])
        ax2.axhline(0, color='black', alpha=0.3)
        ax2.set_title('Band Importance')
        ax2.set_xticks(range(len(self.features)))
        ax2.set_xticklabels(self.features, rotation=45)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


def quick_evaluation(model, test_loader, config, device, num_samples: int = 3):
    """Quick attribution evaluation - just run this function!"""
    if not CAPTUM_AVAILABLE:
        print("Install captum: pip install captum")
        return  # Added missing return statement
    
    evaluator = CaptumEvaluator(model, device, config)
    
    # Find positive samples
    positive_samples = []
    for batch_x, batch_y in test_loader:
        for i in range(batch_x.shape[0]):
            if batch_y[i, -1] == 1:  # Disturbance at last timestep
                positive_samples.append(batch_x[i:i+1])
                if len(positive_samples) >= num_samples:
                    break
        if len(positive_samples) >= num_samples:
            break
    
    if not positive_samples:
        print("No positive samples found!")
        return
    
    print(f"Analyzing {len(positive_samples)} disturbance samples...")
    
    # Analyze each sample
    for i, sample in enumerate(positive_samples):
        print(f"\n--- Sample {i+1} ---")
        
        # Get prediction
        with torch.no_grad():
            pred = torch.sigmoid(model(sample.to(device)))
            print(f"Model prediction: {pred.item():.3f}")
        
        # Get attributions
        ig_attr = evaluator.get_attribution(sample, 'integrated_gradients')
        sal_attr = evaluator.get_attribution(sample, 'saliency')
        
        # Plot results
        evaluator.plot_heatmap(ig_attr, f"Sample {i+1} - Integrated Gradients")
        evaluator.plot_summary(ig_attr, f"Sample {i+1} - Integrated Gradients Summary")
        
        evaluator.plot_heatmap(sal_attr, f"Sample {i+1} - Saliency")
        evaluator.plot_summary(sal_attr, f"Sample {i+1} - Saliency Summary")