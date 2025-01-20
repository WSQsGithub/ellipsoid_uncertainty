import numpy as np
import torch
from torch.utils.data import Dataset

class MCDataset(Dataset):
    def __init__(self, num_samples=1000, batch_size=32, input_dim=3, num_points=100):
        """
        Args:
            num_samples: Total number of samples in the dataset
            batch_size: Number of samples per batch
            input_dim: Dimension of input parameters
            num_points: Number of points to generate per sample
        """
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.num_points = num_points

    def __len__(self):
        return self.num_samples

    def generate_points(self, params):
        """
        Generate points using Monte Carlo sampling based on input parameters
        Args:
            params: Input parameters that define the distribution
        Returns:
            points: Generated points
        """
        # Example: Generate points from a multivariate normal distribution
        mean = params[:3]  # First 3 parameters as mean
        std = np.abs(params[3:]) + 0.1  # Remaining parameters as standard deviation
        
        points = np.random.normal(
            loc=mean,
            scale=std,
            size=(self.num_points, 3)
        )
        return points

    def __getitem__(self, idx):
        # Randomly sample input parameters
        params = np.random.uniform(-1, 1, size=self.input_dim)
        
        # Generate points using Monte Carlo sampling
        points = self.generate_points(params)
        
        # Convert to torch tensors
        params = torch.FloatTensor(params)
        points = torch.FloatTensor(points)
        
        return {
            'parameters': params,
            'points': points
        }
        def generate_points(self, params):
            """
            Generate points using Monte Carlo sampling based on input parameters
            Args:
                params: Input parameters [center(3), radii(3), rotation(9), noise(4)]
            Returns:
                points: Generated points
            """
            # Extract parameters
            center = params[:3]
            radii = np.abs(params[3:6]) + 0.1  # Ensure positive radii
            rotation_matrix = params[6:15].reshape(3, 3)
            noise_params = params[15:]  # Noise parameters

            # Generate random points on unit sphere
            theta = np.random.uniform(0, 2*np.pi, self.num_points)
            phi = np.random.uniform(0, np.pi, self.num_points)
            
            # Convert to Cartesian coordinates
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            points = np.stack([x, y, z], axis=1)
            
            # Scale by radii
            points = points * radii.reshape(1, 3)
            
            # Rotate points
            points = points @ rotation_matrix.T
            
            # Translate points
            points = points + center.reshape(1, 3)
            
            # Add noise
            noise = np.random.normal(0, noise_params[0], points.shape)
            points = points + noise
            
            return points
class UnlimitedMCDataset(Dataset):
    def __init__(self, batch_size=32, input_dim=19, num_points=100):
        """
        Args:
            batch_size: Number of samples per batch
            input_dim: Dimension of input parameters
            num_points: Number of points to generate per sample
        """
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.num_points = num_points

    def __len__(self):
        return int(1e9)  # Practically unlimited size

    def __getitem__(self, idx):
        # Randomly sample input parameters
        params = np.random.uniform(-1, 1, size=self.input_dim)
        
        # Generate points using Monte Carlo sampling
        points = self.generate_points(params)
        
        # Convert to torch tensors
        params = torch.FloatTensor(params)
        points = torch.FloatTensor(points)
        
        return {
            'parameters': params,
            'points': points
        }