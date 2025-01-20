import torch
import torch.nn as nn
import torch.nn.functional as F

class EllipsoidNet(nn.Module):
    def __init__(self):
        super(EllipsoidNet, self).__init__()

        # 定义网络的层
        # 假设我们使用一个简单的全连接网络来处理输入数据
        self.fc1 = nn.Linear(19, 128)  # 输入19个参数（u, v, d, 内参8个, 外参8个）
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 9)    # 输出9个参数：中心坐标和变换矩阵的6个参数

    def forward(self, u, v, d, intrinsics, extrinsics):
        # 拼接输入，形状是 (batch_size, 11)，即(u, v, d, 8内参, 8外参)
        x = torch.cat([u, v, d, intrinsics, extrinsics], dim=1)

        # 前向传播
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # 输出：9个参数（中心坐标 cx, cy, cz 和变换矩阵的6个参数）
        output = self.fc4(x)
        
        return output

if __name__ == '__main__':
    # 示例数据
    batch_size = 32
    u = torch.randn(batch_size, 1)  # u
    v = torch.randn(batch_size, 1)  # v
    d = torch.randn(batch_size, 1)  # d
    intrinsics = torch.randn(batch_size, 8)  # 8个内参
    extrinsics = torch.randn(batch_size, 8)  # 8个外参

    # 创建模型实例
    model = EllipsoidNet()

    # 获取输出
    output = model(u, v, d, intrinsics, extrinsics)
    print(output.shape)  # 应该是 (batch_size, 9)