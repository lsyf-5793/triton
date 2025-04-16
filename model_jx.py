import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import triton
import triton.language as tl

# 定义数据集类，用于加载和处理数据
class JXdataset(Dataset):
    def __init__(self, path):
        # 读取CSV文件并转换为张量
        JX = pd.read_csv(path)
        JX_data = torch.tensor(JX.values, dtype=torch.float32)
        # 分离特征和目标值
        self.value, self.target = JX_data[:, :-1], JX_data[:, -1]
    
    def __getitem__(self, index):
        # 返回指定索引的特征和目标值
        return self.value[index], self.target[index]
    
    def __len__(self):
        # 返回数据集大小
        return len(self.value)

# 设置批处理大小和数据加载器
batch_size = 64
train_dataset = JXdataset("./jixie_train_data.csv")
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = JXdataset("./jixie_test_data.csv")
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# 设置设备和损失函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_func = nn.MSELoss()

# Flash Attention实现
class FlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # 设置模型参数
        self.embed_dim = embed_dim  # 嵌入维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        
        # 确保参数满足要求
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        assert self.head_dim in {16, 32, 64, 128}, "head_dim必须是16、32、64或128之一"
        
        # 设置缩放因子
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, q, k, v):
        # 获取输入维度
        batch_size, seq_len, _ = q.shape
        
        # 重塑为多头格式 [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算缩放点积注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 计算注意力分数
        attn = F.softmax(attn, dim=-1)  # 应用softmax
        output = attn @ v  # 计算加权和
        
        # 重塑回原始形状 [batch_size, seq_len, embed_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return output

# 创建带有Flash Attention的回归模型
class RegressionModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, n_head=4):
        super().__init__()
        # 确保参数满足要求
        assert hidden_dim % n_head == 0, "hidden_dim必须能被n_head整除"
        head_dim = hidden_dim // n_head
        assert head_dim in {16, 32, 64, 128}, "head_dim必须是16、32、64或128之一"
        
        # 定义模型层
        self.input_embedding = nn.Linear(input_dim, hidden_dim)  # 输入嵌入层
        self.flash_attention = FlashAttention(hidden_dim, n_head)  # Flash Attention层
        self.output_layer = nn.Linear(hidden_dim, 1)  # 输出层
        
    def forward(self, x):
        # x: [batch_size, input_features]
        x = x.unsqueeze(1)  # [batch_size, 1, input_features]
        x = self.input_embedding(x)  # [batch_size, 1, hidden_dim]
        
        # 使用相同的张量进行自注意力计算
        x = self.flash_attention(x, x, x)  # [batch_size, 1, hidden_dim]
        
        x = self.output_layer(x)  # [batch_size, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [batch_size]
        return x

# 训练循环函数
def train_loop(model, train_loader, optimizer, loss_fn, device, epoch):
    model.train()  # 设置为训练模式
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
    
    # 遍历训练数据
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 清除梯度
        output = model(data)  # 前向传播
        loss = loss_fn(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()
        pbar.set_postfix({"loss": total_loss / (batch_idx + 1)})
    
    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    print(f"训练损失: {avg_loss:.6f}")
    return avg_loss

# 测试循环函数
def test_loop(model, test_loader, loss_fn, device):
    model.eval()  # 设置为评估模式
    test_loss = 0.0
    
    # 在测试集上进行评估
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            test_loss += loss.item()
    
    # 计算平均测试损失
    avg_loss = test_loss / len(test_loader)
    print(f"测试损失: {avg_loss:.6f}")
    return avg_loss

# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置超参数
    input_dim = 4  # 输入特征维度
    hidden_dim = 64  # 隐藏层维度
    n_head = 4  # 注意力头数
    learning_rate = 1e-3  # 学习率
    weight_decay = 1e-4  # 权重衰减
    num_epochs = 100  # 训练轮数
    
    # 创建模型
    model = RegressionModel(input_dim, hidden_dim, n_head).to(device)
    
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 记录训练历史
    train_losses = []
    test_losses = []
    
    # 训练循环
    for epoch in range(num_epochs):
        train_loss = train_loop(model, train_loader, optimizer, loss_func, device, epoch)
        test_loss = test_loop(model, test_loader, loss_func, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    # 绘制训练和测试损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='training loss')
    plt.plot(test_losses, label='test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (MSE)')
    plt.title('test vs train loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot_test.png')
    
    # 保存模型
    torch.save(model.state_dict(), 'regression_model.pth')
    
    print("训练完成!")

if __name__ == "__main__":
    main()