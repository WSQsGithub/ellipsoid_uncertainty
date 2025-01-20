import torch
import torch.nn.functional as F
from typing import Tuple, Union

class EllipsoidLoss(torch.nn.Module):
    def __init__(self, center_weight: float = 1.0, containment_weight: float = 1.0, volume_weight: float = 1.0):
        """
        Combined loss for ellipsoid fitting.
        
        Args:
            center_weight: Weight for center loss
            containment_weight: Weight for containment loss
            volume_weight: Weight for volume loss
        """
        super().__init__()
        self.center_weight = center_weight
        self.containment_weight = containment_weight
        self.volume_weight = volume_weight

    def forward(self, pred_params: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass for computing all losses.
        
        Args:
            pred_params: Tensor of shape (batch_size, 9) containing ellipsoid parameters
            points: Tensor of shape (batch_size, num_points, 3) containing point cloud
            
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary containing individual losses
        """
        # Recover center and matrix Q
        cx, cy, cz, Q = self._recover_Q(pred_params)
        pred_center = torch.stack([cx, cy, cz], dim=1)

        # Compute individual losses
        center_loss = self._compute_center_loss(pred_center, points)
        contain_loss = self._compute_containment_loss(points, cx, cy, cz, Q)
        vol_loss = self._compute_volume_loss(Q)

        # Combine losses
        total_loss = (self.center_weight * center_loss + 
                     self.containment_weight * contain_loss + 
                     self.volume_weight * vol_loss)

        # Create loss dictionary for monitoring
        loss_dict = {
            'center_loss': center_loss.item(),
            'containment_loss': contain_loss.item(),
            'volume_loss': vol_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict

    @staticmethod
    def _compute_center_loss(pred_center: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Compute loss between predicted and actual centroid"""
        true_center = torch.mean(points, dim=1)
        return torch.mean(torch.norm(pred_center - true_center, dim=1))

    @staticmethod
    def _compute_containment_loss(points: torch.Tensor, cx: torch.Tensor, 
                                cy: torch.Tensor, cz: torch.Tensor, 
                                Q: torch.Tensor) -> torch.Tensor:
        """Compute containment loss for points within ellipsoid"""
        dx = points[:, :, 0] - cx.unsqueeze(1)
        dy = points[:, :, 1] - cy.unsqueeze(1)
        dz = points[:, :, 2] - cz.unsqueeze(1)
        
        points_centered = torch.stack([dx, dy, dz], dim=2)
        Q_inv = torch.inverse(Q)
        ellipsoid_eq = torch.sum(torch.matmul(points_centered, Q_inv) * points_centered, dim=2)
        
        return torch.mean(F.relu(ellipsoid_eq - 1))

    @staticmethod
    def _compute_volume_loss(Q: torch.Tensor) -> torch.Tensor:
        """Compute volume of ellipsoid"""
        return 4/3 * torch.pi * torch.det(Q)

    @staticmethod
    def _recover_Q(output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recover ellipsoid parameters from network output"""
        cx, cy, cz = output[:, 0], output[:, 1], output[:, 2]
        Q11, Q12, Q13, Q22, Q23, Q33 = output[:, 3:9].chunk(6, dim=1) 
        
        Q = torch.zeros((output.size(0), 3, 3), device=output.device)
        Q[:, 0, 0] = Q11.squeeze()
        Q[:, 0, 1] = Q[:, 1, 0] = Q12.squeeze()
        Q[:, 0, 2] = Q[:, 2, 0] = Q13.squeeze()
        Q[:, 1, 1] = Q22.squeeze()
        Q[:, 1, 2] = Q[:, 2, 1] = Q23.squeeze()
        Q[:, 2, 2] = Q33.squeeze()

        return cx, cy, cz, Q
