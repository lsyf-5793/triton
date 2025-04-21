import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os
from sklearn.preprocessing import StandardScaler

# 添加triton的安装路径到Python路径
triton_path = "/root/miniconda3/envs/py12/lib/python3.12/site-packages"
if triton_path not in sys.path:
    sys.path.append(triton_path)

# 导入triton
try:
    import triton_attention
    import triton.language as tl
    from triton_attention import attention  # 导入Triton的attention函数
except ImportError as e:
    print(f"Error importing triton: {e}")
    print("Current Python path:")
    print('\n'.join(sys.path))
    raise

# 定义数据集类
class JXdataset(Dataset):
    def __init__(self, path, scaler=None, is_training=True):
        # 读取CSV文件
        JX = pd.read_csv(path)
        JX_data = JX.values
        
        # 分离特征和目标值
        features = JX_data[:, :-1]
        targets = JX_data[:, -1]
        
        # 标准化特征
        if is_training:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(features)
        else:
            self.features = scaler.transform(features)
        
        # 标准化目标值
        self.target_mean = np.mean(targets)
        self.target_std = np.std(targets)
        self.targets = (targets - self.target_mean) / self.target_std
        
        # 转换为张量
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)
    
    def __getitem__(self, index):
        return self.features[index], self.targets[index]

    def __len__(self):
        return len(self.features)

# 创建回归模型
class RegressionModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, n_head=8):
        super().__init__()
        # 确保隐藏层维度能被注意力头数整除
        assert hidden_dim % n_head == 0, "hidden_dim必须能被n_head整除"
        head_dim = hidden_dim // n_head
        # 确保每个头的维度是16的倍数且大于等于32
        assert head_dim % 16 == 0 and head_dim >= 32, "head_dim必须是16的倍数且大于等于32"
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # 保存配置
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.head_dim = head_dim
        
        # 设置序列长度为256（128的倍数，满足Triton的要求）
        self.seq_len = 256
        
    def forward(self, x):
        # 确保输入是连续的
        x = x.contiguous()
        batch_size = x.shape[0]
        
        # 将输入扩展为固定长度的序列（256，满足Triton要求）
        x = x.unsqueeze(1).expand(-1, self.seq_len, -1)  # [batch_size, seq_len, input_features]
        x = x.contiguous()  # 确保输入张量在内存中是连续的
        
        # 通过嵌入层
        x = self.input_embedding(x)  # [batch_size, seq_len, hidden_dim]
        x = self.norm1(x)
        x = x.contiguous()  # 确保嵌入输出是连续的
        
        # 转换为float16以提高计算效率
        x = x.to(torch.float16)
        
        # 重塑为多头格式 [batch_size, n_head, seq_len, head_dim]
        x = x.view(batch_size, self.seq_len, self.n_head, self.head_dim)
        x = x.transpose(1, 2)  # [batch_size, n_head, seq_len, head_dim]
        x = x.contiguous()  # 确保多头格式是连续的
        
        # 使用Triton Flash Attention进行自注意力计算
        # 对于自注意力，q=k=v=x
        sm_scale = 1.0 / math.sqrt(self.head_dim)  # 缩放因子
        
        # 确保所有输入张量都是连续的
        q = x.contiguous()
        k = x.contiguous()
        v = x.contiguous()
        
        # 调用Triton的attention实现
        x = attention(q, k, v, True, sm_scale)
        x = self.dropout(x)
        
        # 重塑回原始形状
        x = x.transpose(1, 2)  # [batch_size, seq_len, n_head, head_dim]
        x = x.contiguous()
        x = x.view(batch_size, self.seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        
        # 转换回float32以进行后续计算
        x = x.to(torch.float32)
        
        # 残差连接和层归一化
        x = x + x  # 残差连接
        x = self.norm2(x)
        
        # 只取第一个时间步的输出
        x = x[:, 0, :]
        x = x.contiguous()  # 确保最终输出是连续的
        
        # 输出层
        x = self.output_layer(x)
        return x.squeeze(-1)

# 训练模型函数
def train_model(model, train_loader, test_loader, num_epochs=100, learning_rate=0.001, patience=10):
    # 设置设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 定义优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()
    
    # 用于早停的变量
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # 计算平均验证损失
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    
    return model

def evaluate_model(model, test_loader, target_mean, target_std):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    predictions = []
    actuals = []
    
    # 在测试集上进行评估
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # 反标准化预测值和实际值
            outputs = outputs.cpu().numpy() * target_std + target_mean
            targets = targets.cpu().numpy() * target_std + target_mean
            
            loss = criterion(torch.tensor(outputs), torch.tensor(targets))
            total_loss += loss.item()
            
            predictions.extend(outputs)
            actuals.extend(targets)
    
    # 计算平均损失
    avg_loss = total_loss / len(test_loader)
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 计算R2分数
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f'Test Loss: {avg_loss:.4f}')
    print(f'R2 Score: {r2:.4f}')
    
    # 绘制预测值vs实际值散点图
    plt.figure(figsize=(10, 5))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.savefig('predictions.png')
    plt.close()
    
    return avg_loss, r2

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载数据
    batch_size = 64
    train_dataset = JXdataset("./jixie_train_data.csv", is_training=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataset = JXdataset("./jixie_test_data.csv", scaler=train_dataset.scaler, is_training=False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    
    # 创建模型
    model = RegressionModel(input_dim=4, hidden_dim=256, n_head=8)
    
    # 训练模型
    trained_model = train_model(model, train_loader, test_loader)
    
    # 评估模型
    test_loss, r2_score = evaluate_model(trained_model, test_loader, 
                                       train_dataset.target_mean, 
                                       train_dataset.target_std) 