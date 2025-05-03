import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.cuda.amp import autocast, GradScaler
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
scaler = torch.cuda.amp.GradScaler()

class HawkesProcess(nn.Module):
    def __init__(self, n_types):
        super(HawkesProcess, self).__init__()
        self.n_types = n_types

        # 参数定义（直接使用 mu, alpha, beta）
        self.mu = nn.Parameter(torch.rand(n_types, dtype=torch.float64))          # 基础强度 μ ∈ (0, +∞)
        self.alpha = nn.Parameter(torch.rand((n_types, n_types), dtype=torch.float64))  # 影响矩阵 α ∈ (0, +∞)
        self.beta = nn.Parameter(torch.rand((n_types, n_types), dtype=torch.float64))   # 衰减率 β ∈ (0, +∞)

        # 初始化参数（确保物理意义合理）
        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.uniform_(self.mu, 0.01, 0.1)      # 基础强度较小
        nn.init.uniform_(self.alpha, 0.01, 0.1)   # 影响权重较小
        nn.init.uniform_(self.beta, 0.5, 1.5)     # 衰减率适中

    def compute_intensity(self, event_times, event_types, current_time):
        """向量化计算强度函数 λ(t)"""
        if len(event_times) == 0:
            return self.mu  # 无历史事件时仅返回基础强度

        # 计算时间差 [seq_len]
        time_diff = current_time - event_times  # [seq_len]

        # 选择对应事件类型的 alpha 和 beta [seq_len, n_types]
        alpha_selected = self.alpha[event_types]  # [seq_len, n_types]
        beta_selected = self.beta[event_types]    # [seq_len, n_types]

        # 计算衰减项：exp(-β * Δt) [seq_len, n_types]
        decay = torch.exp(-beta_selected * time_diff.unsqueeze(-1))

        # 总影响 = sum(α * exp(-β * Δt)) [n_types]
        total_influence = torch.sum(alpha_selected * decay, dim=0)

        # 最终强度 = μ + 总影响
        return self.mu + total_influence  # [n_types]

    def hawkes_loss(self, event_times, event_types, T):
        """计算负对数似然损失（向量化+数值积分）"""
        batch_size, seq_len = event_times.shape
        log_likelihood = 0.0
        integral = 0.0

        for b in range(batch_size):
            times = event_times[b][event_times[b] >= 0]  # 过滤填充事件
            types = event_types[b][event_times[b] >= 0]

            # --- 计算对数似然 ---
            if len(times) > 0:
                # 预计算所有事件点的强度 [seq_len, n_types]
                intensities = torch.stack([
                    self.compute_intensity(times[:i], types[:i], times[i])
                    for i in range(len(times))
                ])
                log_likelihood += intensities[torch.arange(len(types)), types].sum()

            # --- 计算积分 ---
            # 定义积分区间：[0, t1], [t1, t2], ..., [tN, T]
            intervals = torch.cat([
                torch.tensor([0.0], device=device), 
                times, 
                torch.tensor([T], device=device)
            ])
            for i in range(len(intervals) - 1):
                start, end = intervals[i], intervals[i+1]
                hist_times = times[times <= start]
                hist_types = types[times <= start]

                # 数值积分（梯形法）
                steps = 10
                t_points = torch.linspace(start, end, steps+1, device=device)
                values = torch.stack([self.compute_intensity(hist_times, hist_types, t) for t in t_points])
                integral += values.sum() * (end - start) / steps

        return -(log_likelihood - integral)
class HawkesProcessModel(nn.Module):
    def __init__(self, n_types):
        super(HawkesProcessModel, self).__init__()
        # self.hawkes_process = HawkesProcess(n_types, mu_init, alpha_init, beta_init)
        self.hawkes_process = HawkesProcess(n_types)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def train_batch(self, batch, T):
        event_types, event_times = batch
        event_types = event_types.long()
        loss = self.hawkes_process.hawkes_loss(event_times, event_types, T)
    
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            return torch.inf
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.hawkes_process.mu.data = torch.sigmoid(self.hawkes_process.mu.data)  # 强制[0,1]
            self.hawkes_process.alpha.data = torch.tanh(self.hawkes_process.alpha.data)  
            self.hawkes_process.beta.data = torch.tanh(self.hawkes_process.beta.data) 
        return loss.item()

class EHawkesProcess(nn.Module):
    def __init__(self, n_types, embedding_size):
        super(EHawkesProcess, self).__init__()
        self.n_types = n_types
        self.embedding_size = embedding_size

        self.type_embeddings = nn.Parameter(torch.randn((n_types, embedding_size), dtype=torch.float32))
        self.a = nn.Parameter(torch.randn((n_types, embedding_size), dtype=torch.float32))
        self.A = nn.Parameter(torch.randn((n_types, n_types, embedding_size), dtype=torch.float32))
        self.P = nn.Parameter(torch.randn((n_types, n_types, embedding_size), dtype=torch.float32))


        self._initialize_parameters()


    def _initialize_parameters(self):
        nn.init.normal_(self.type_embeddings, mean=0.0, std=0.1)
        nn.init.normal_(self.a, mean=0.0, std=0.1)
        nn.init.normal_(self.A, mean=0.0, std=0.1)
        nn.init.normal_(self.P, mean=0.0, std=0.1)

    def compute_intensity(self, event_times, event_types, current_time):
        # a = torch.sigmoid(self.a)  # 映射到(0,1)
        # A = torch.tanh(self.A)     # 映射到(-1,1)
        # P = torch.tanh(self.P)     # 映射到(-1,1)
        """计算当前时间点的强度函数（适配序列类别输入）"""
        n_types = self.n_types
        emb_dim = self.embedding_size

        # 处理空历史情况
        if len(event_times) == 0:
            # 基础强度 + 类别基础强度（使用默认或零值）
            intensity = self.type_embeddings * self.a  # [n_types, emb_dim]
            return torch.sum(F.softplus(intensity), dim=1)  # [n_types]

        seq_len = len(event_times)
        time_diff = current_time - event_times  # [seq_len]

        # 获取历史事件的嵌入向量
        f_eth = self.type_embeddings[event_types]  # [seq_len, emb_dim]

        # --- 事件到事件影响 (m_{u,e}) ---
        # f_e = self.type_embeddings.unsqueeze(1)  # [n_types, 1, emb_dim]
        A_selected = self.A[event_types]  # [seq_len, n_types, emb_dim]
        f_e = self.type_embeddings
        # 向量化计算
        # A_terms = (f_e.unsqueeze(1) * A_selected.unsqueeze(0)).sum(dim=2)  # [n_types, seq_len, emb_dim]
        # P_terms = (f_e.unsqueeze(1) * self.P[event_types].unsqueeze(0)).sum(dim=2)
        A_terms = torch.einsum('te,bte->tbe', f_e, A_selected)  # [n_types, seq_len, emb_dim]
        P_terms = torch.einsum('te,bte->tbe', f_e, self.P[event_types])
        # 指数衰减项
        # decay = torch.exp(
        #     -(P_terms * f_eth.unsqueeze(0)) * time_diff.unsqueeze(0).unsqueeze(-1))  # [n_types, seq_len, emb_dim]

        decay = torch.exp(
            -torch.einsum('tse,se,s->tse', P_terms, f_eth, time_diff)
        )

        # 计算m项
        # m = (A_terms * f_eth.unsqueeze(0) * decay).sum(dim=1)  # [n_types, emb_dim]
        m = torch.einsum('tse,se->te', A_terms * decay, f_eth)
        # --- 基础强度项 ---
        base_term = self.type_embeddings * self.a  # [n_types, emb_dim]

        # 合并所有项
        intensity = base_term  + m  # [n_types, emb_dim]
        return torch.sum(F.softplus(intensity), dim=1)  # [n_types]
        # return torch.sum(F.relu(intensity), dim=1)  # [n_types]
        # return torch.sum(F.softsign(intensity)+1, dim=1)  # [n_types]
        # return torch.sum(F.sigmoid(intensity), dim=1)  # [n_types]
        # return torch.sum(intensity, dim=1)
    def compute_loss(self, event_times, event_types, T):
        """计算负对数似然损失（适配序列类别输入）"""
        batch_size, seq_len = event_times.shape
        log_likelihood = torch.tensor(0.0, dtype=torch.float32, device=device)
        integral = torch.tensor(0.0, dtype=torch.float32, device=device) 

        for b in range(batch_size):
            times = event_times[b]  # [seq_len]
            types = event_types[b]  # [seq_len]
            

            # 计算对数似然
            for i in range(seq_len):
                if times[i] < 0:  # 跳过填充事件
                    continue

                # 使用历史事件计算强度
                intensity = self.compute_intensity(
                    event_times=times[:i],
                    event_types=types[:i],
                    current_time=times[i]
                )

                # 当前事件的类型
                current_type = types[i]

                # 获取当前类型的强度值
                log_likelihood += torch.log(intensity[current_type] + 1e-16)
                # log_likelihood += intensity[current_type]
            # 计算积分项（使用完整序列）
            valid_mask = times >= 0
            if valid_mask.any():
                valid_times = times[valid_mask]
                valid_types = types[valid_mask]

                # 最后时间点到T的积分
                last_time = valid_times[-1]
                total_intensity = self.compute_intensity(
                    event_times=valid_times,
                    event_types=valid_types,
                    current_time=T
                )
                integral += torch.sum(total_intensity) * (T - last_time)

                # 0到第一个事件的积分
                first_time = valid_times[0]
                initial_intensity = self.compute_intensity(
                    event_times=times[0:0],  # 空历史
                    event_types=types[0:0],
                    current_time=first_time
                )
                integral += torch.sum(initial_intensity) * first_time
            else:
                # 无有效事件的积分
                initial_intensity = self.compute_intensity(
                    event_times=times[0:0],
                    event_types=types[0:0],
                    current_time=T
                )
                integral += torch.sum(initial_intensity) * T

        return -(log_likelihood - integral)


class EHawkesProcessModel(nn.Module):
    def __init__(self, n_types,  embedding_size, learning_rate=0.01):
        super(EHawkesProcessModel, self).__init__()
        self.hawkes_process = EHawkesProcess(n_types, embedding_size).to(device)
        self.optimizer = optim.Adam(self.hawkes_process.parameters(), lr=learning_rate) 
    def train_batch(self, batch, T):
        event_types, event_times = batch

        event_types = event_types.long().to(device)
        event_times = event_times.to(torch.float32).to(device)
        T = torch.tensor(T, dtype=torch.float32, device=device)

        self.optimizer.zero_grad()

        loss = self.hawkes_process.compute_loss(event_times, event_types, T)
        # if torch.isnan(loss).any() or torch.isinf(loss).any():
        #     return torch.inf
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.hawkes_process.a.data = torch.sigmoid(self.hawkes_process.a.data)  # 强制[0,1]
            self.hawkes_process.A.data = torch.tanh(self.hawkes_process.A.data)  
            self.hawkes_process.P.data = torch.tanh(self.hawkes_process.P.data)     # 映射到(-1,1)
        return loss

class CEHawkesProcess(nn.Module):
    def __init__(self, n_types, n_categories, embedding_size):
        super(CEHawkesProcess, self).__init__()
        self.n_types = n_types
        self.n_categories = n_categories
        self.embedding_size = embedding_size

        self.type_embeddings = nn.Parameter(torch.randn((n_types, embedding_size), dtype=torch.float32))
        self.category_embeddings = nn.Parameter(torch.randn((n_categories, embedding_size), dtype=torch.float32))
        self.a = nn.Parameter(torch.randn((n_types, embedding_size), dtype=torch.float32))
        self.b = nn.Parameter(torch.randn((n_categories, embedding_size), dtype=torch.float32))
        # self.b = nn.Parameter(torch.randn((n_categories, 1), dtype=torch.float64))
        self.A = nn.Parameter(torch.randn((n_types, n_types, embedding_size), dtype=torch.float32))
        self.P = nn.Parameter(torch.randn((n_types, n_types, embedding_size), dtype=torch.float32))
        self.B = nn.Parameter(torch.randn((n_categories, n_categories, embedding_size), dtype=torch.float32))
        self.Q = nn.Parameter(torch.randn((n_categories, n_categories, embedding_size), dtype=torch.float32))
        # self.B = nn.Parameter(torch.randn((n_categories, n_categories, 1), dtype=torch.float64))
        # self.Q = nn.Parameter(torch.randn((n_categories, n_categories, 1), dtype=torch.float64))

        self._initialize_parameters()


    def _initialize_parameters(self):
        nn.init.normal_(self.type_embeddings, mean=0.0, std=0.1)
        nn.init.normal_(self.category_embeddings, mean=0.0, std=0.1)
        nn.init.normal_(self.a, mean=0.0, std=0.1)
        nn.init.normal_(self.b, mean=0.0, std=0.1)
        nn.init.normal_(self.A, mean=0.0, std=0.1)
        nn.init.normal_(self.P, mean=0.0, std=0.1)
        nn.init.normal_(self.B, mean=0.0, std=0.1)
        nn.init.normal_(self.Q, mean=0.0, std=0.1)

    def compute_intensity(self, event_times, event_types, event_categories, current_time):
        """计算当前时间点的强度函数（适配序列类别输入）"""

        n_types = self.n_types
        emb_dim = self.embedding_size

        # 处理空历史情况
        if len(event_times) == 0:
            # 基础强度 + 类别基础强度（使用默认或零值）
            # intensity = self.type_embeddings * self.a  # [n_types, emb_dim]
            intensity = self.type_embeddings * self.a
            return torch.sum(F.softplus(intensity), dim=1)  # [n_types]

        seq_len = len(event_times)
        time_diff = current_time - event_times  # [seq_len]

        # 获取历史事件的嵌入向量
        f_eth = self.type_embeddings[event_types]  # [seq_len, emb_dim]
        g_eth = self.category_embeddings[event_categories]  # [seq_len, emb_dim]

        # --- 事件到事件影响 (m_{u,e}) ---
        f_e = self.type_embeddings  # [n_types, emb_dim]
        A_selected = self.A[event_types]  # [seq_len, n_types, emb_dim]

        # 向量化计算
        A_terms = torch.einsum('te,bte->tbe', f_e, A_selected)  # [n_types, seq_len, emb_dim]
        P_terms = torch.einsum('te,bte->tbe', f_e, self.P[event_types])
        # 指数衰减项
        # decay = torch.exp(
        #     -(P_terms * f_eth.unsqueeze(0)) * time_diff.unsqueeze(0).unsqueeze(-1))  # [n_types, seq_len, emb_dim]

        decay = torch.exp(
            -torch.einsum('tse,se,s->tse', P_terms, f_eth, time_diff)
        )

        # 计算m项
        # m = (A_terms * f_eth.unsqueeze(0) * decay).sum(dim=1)  # [n_types, emb_dim]
        m = torch.einsum('tse,se->te', A_terms * decay, f_eth)

        # --- 类别到类别影响 (n_{u,e}) ---
        g_e = self.category_embeddings  # [n_categories, 1, emb_dim]
        B_selected = self.B[event_categories]  # [seq_len, n_categories, emb_dim]

        # 向量化计算
        # B_terms = (g_e.unsqueeze(1) * B_selected.unsqueeze(0)).sum(dim=2)  
        B_terms = torch.einsum('te,bte->tbe', g_e, B_selected)  # [n_categories, seq_len, emb_dim]
        # Q_terms = (g_e.unsqueeze(1) * self.Q[event_categories].unsqueeze(0)).sum(dim=2)
        Q_terms = torch.einsum('te,bte->tbe', g_e, self.Q[event_categories])

        # 类别衰减项
        cat_decay = torch.exp(
            -torch.einsum('tse,se,s->tse', Q_terms, g_eth, time_diff)
            # -(Q_terms * g_eth.unsqueeze(0)) * time_diff.unsqueeze(0).unsqueeze(-1)
            )  # [n_categories, seq_len, emb_dim]
            
        # 计算n项
        # n = (B_terms * g_eth.unsqueeze(0) * cat_decay).sum(dim=1)  # [n_categories, emb_dim]
        n = torch.einsum('tse,se->te', B_terms * cat_decay, g_eth)

        # 将n映射到事件类型 [n_types, emb_dim]
        if len(event_categories) > 0:
            # 使用当前事件的类别（最后一个历史事件）
            current_cat = event_categories[-1]
            n_per_type = self.type_embeddings * n[current_cat].unsqueeze(0)  # 广播相乘
        else:
            n_per_type = torch.zeros_like(self.type_embeddings)

        # --- 基础强度项 ---
        base_term = self.type_embeddings * self.a  # [n_types, emb_dim]

        # --- 类别项 ---
        if len(event_categories) > 0:
            # 使用当前事件的类别
            current_cat = event_categories[-1]
            category_term = self.type_embeddings * self.b[current_cat].unsqueeze(0)  # [n_types, emb_dim]
        else:
            category_term = torch.zeros_like(self.type_embeddings)

        # 合并所有项
        intensity = base_term + category_term + m + n_per_type  # [n_types, emb_dim]
        # return torch.sum(F.softplus(intensity), dim=1)  # [n_types]
        # return torch.sum(F.relu(intensity), dim=1)  # [n_types]
        # return torch.sum(F.softsign(intensity)+1, dim=1)  # [n_types]
        # return torch.sum(F.sigmoid(intensity), dim=1)  # [n_types]
        return torch.sum(intensity, dim=1)
    
    def compute_loss(self, event_times, event_types, event_categories, T):
        """计算负对数似然损失（适配序列类别输入）"""
        batch_size, seq_len = event_times.shape
        log_likelihood = torch.tensor(0.0, dtype=torch.float32, device=device)
        integral = torch.tensor(0.0, dtype=torch.float32, device=device)

        for bb in range(batch_size):
            times = event_times[bb]  # [seq_len]
            types = event_types[bb]  # [seq_len]
            cats = event_categories[bb]  # [seq_len]

            # 计算对数似然
            for i in range(seq_len):
                if times[i] < 0:  # 跳过填充事件
                    continue

                # 使用历史事件计算强度
                intensity = self.compute_intensity(
                    event_times=times[:i],
                    event_types=types[:i],
                    event_categories=cats[:i],  # 历史事件的类别
                    current_time=times[i]
                )

                # 当前事件的类型和类别
                current_type = types[i]
                current_cat = cats[i]

                # 获取当前类型的强度值
                log_likelihood += torch.log(intensity[current_type] + 1e-16)
                log_likelihood += intensity[current_type]
            # 计算积分项（使用完整序列）
            valid_mask = times >= 0
            if valid_mask.any():
                valid_times = times[valid_mask]
                valid_types = types[valid_mask]
                valid_cats = cats[valid_mask]

                # 最后时间点到T的积分
                last_time = valid_times[-1]
                total_intensity = self.compute_intensity(
                    event_times=valid_times,
                    event_types=valid_types,
                    event_categories=valid_cats,
                    current_time=T
                )
                integral += torch.sum(total_intensity) * (T - last_time)

                # 0到第一个事件的积分
                first_time = valid_times[0]
                initial_intensity = self.compute_intensity(
                    event_times=times[0:0],  # 空历史
                    event_types=types[0:0],
                    event_categories=cats[0:0],
                    current_time=first_time
                )
                integral += torch.sum(initial_intensity) * first_time
            else:
                # 无有效事件的积分
                initial_intensity = self.compute_intensity(
                    event_times=times[0:0],
                    event_types=types[0:0],
                    event_categories=cats[0:0],
                    current_time=T
                )
                integral += torch.sum(initial_intensity) * T

        return -(log_likelihood - integral)


class CEHawkesProcessModel(nn.Module):
    def __init__(self, n_types, n_categories,  embedding_size, learning_rate=0.01):
        super(CEHawkesProcessModel, self).__init__()
        self.hawkes_process = CEHawkesProcess(n_types, n_categories, embedding_size).to(device)
        self.optimizer = optim.Adam(self.hawkes_process.parameters(), lr=learning_rate)

    def train_batch(self, batch, T):
        event_types, event_categories, event_times = batch

        event_types = event_types.long().to(device)
        event_categories = event_categories.long().to(device)
        event_times = event_times.to(torch.float32).to(device)
        T = torch.tensor(T, dtype=torch.float32, device=device)

        self.optimizer.zero_grad()

        loss = self.hawkes_process.compute_loss(event_times, event_types, event_categories, T)
        # if torch.isnan(loss).any() or torch.isinf(loss).any():
        #     return torch.inf

        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.hawkes_process.a.data = torch.sigmoid(self.hawkes_process.a.data)  # 强制[0,1]
            self.hawkes_process.A.data = torch.tanh(self.hawkes_process.A.data)  
            self.hawkes_process.P.data = torch.tanh(self.hawkes_process.P.data)     # 映射到(-1,1)
            self.hawkes_process.b.data = torch.sigmoid(self.hawkes_process.b.data)  # 映射到(0,1)
            self.hawkes_process.B.data = torch.tanh(self.hawkes_process.B.data)     # 映射到(-1,1)
            self.hawkes_process.Q.data = torch.tanh(self.hawkes_process.Q.data)     # 映射到(-1,1)
        return loss


if __name__ == "__main__":
    torch.manual_seed(42)
    train_event = torch.tensor([[4, 0, 3, 2, 1, 0, 3, 2, 1], [4, 0, 3, 2, 1, 0, 3, 0, 0]]).to(device)
    train_dtime = torch.tensor([[0, 0.2, 0.2, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5], [0, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5]]).to(device)
    train_type  = torch.tensor([[0, 1, 3, 1 ,2, 1, 3, 1, 2], [0, 1, 3, 1, 2, 1, 3, 1, 1]]).to(device)
    # event_to_category = torch.tensor([1, 2, 1, 3, 0], device=device)
    model = HawkesProcessModel(5)
    model.to(device)
    # model = HawkesProcessModel(5)
    a_batch = (train_event, train_dtime)
    loss = model.train_batch(a_batch, 1.0)
    print(f"Loss: {loss}")