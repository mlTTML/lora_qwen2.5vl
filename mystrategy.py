"""我的训练策略实现

基于SlidingWindowStrategy的简化版本：
- 风险分数 >= 2.0：标记为异常样本，跳过反向传播
- 风险分数 < 2.0：正常训练一次，不重复训练

适用于大模型训练，认为高loss样本更可能是数据质量问题而非学习机会。
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
import time
import json
import os


class BaseTrainingStrategy:
    """训练策略基类"""
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics_history = defaultdict(list)
        self.train_time = 0
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self.current_epoch = int(epoch)

    def train_batch(self, inputs, targets):
        """训练一个批次的数据"""
        raise NotImplementedError("子类必须实现train_batch方法")

    def evaluate(self, data_loader):
        """在测试集上评估模型"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in data_loader:
                # 处理数据格式
                if isinstance(batch, dict):
                    inputs = {
                        'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                        'labels': batch['labels'],
                        'pixel_values': batch['pixel_values']
                    }
                    # 兼容可选图像辅助键
                    for opt_key in ('image_grid_thw','image_sizes','image_input_size'):
                        if opt_key in batch and batch[opt_key] is not None:
                            inputs[opt_key] = batch[opt_key]
                    targets = batch['labels']
                else:
                    # 兼容旧的元组格式
                    inputs, targets = batch
                
                outputs = self.model(**inputs)
                
                # 处理输出格式
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # 计算损失
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss_per_token = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
                total_loss += loss_per_token.sum().item()
                
                # 计算准确率
                predictions = logits.argmax(dim=-1)
                labels = targets
                
                # 只计算非忽略的token
                non_ignore_mask = labels != -100
                correct_predictions = (predictions == labels) & non_ignore_mask
                total_correct += correct_predictions.sum().item()
                total_tokens += non_ignore_mask.sum().item()

        avg_loss = total_loss / max(1, total_tokens)
        accuracy = total_correct / max(1, total_tokens)

        return avg_loss, accuracy

    def save_metrics(self, epoch, train_loss, train_rmse, train_mae, test_loss, test_rmse, test_mae):
        """保存训练和测试指标"""
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['train_rmse'].append(train_rmse)
        self.metrics_history['train_mae'].append(train_mae)
        self.metrics_history['test_loss'].append(test_loss)
        self.metrics_history['test_rmse'].append(test_rmse)
        self.metrics_history['test_mae'].append(test_mae)

    def save_metrics_with_accuracy(self, epoch, train_loss, train_rmse, train_mae, train_accuracy, test_loss, test_rmse, test_mae, test_accuracy):
        """保存训练和测试指标（包含accuracy/R2）"""
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['train_rmse'].append(train_rmse)
        self.metrics_history['train_mae'].append(train_mae)
        self.metrics_history['train_accuracy'].append(train_accuracy)
        self.metrics_history['test_loss'].append(test_loss)
        self.metrics_history['test_rmse'].append(test_rmse)
        self.metrics_history['test_mae'].append(test_mae)
        self.metrics_history['test_accuracy'].append(test_accuracy)


class MyStrategy(BaseTrainingStrategy):
    """我的训练策略：风险分数>=2丢弃，<2正常训练
    
    核心思想：
    - 对于大模型，高loss样本更可能是数据质量问题
    - 风险分数>=2时标记为异常并跳过
    - 风险分数<2时正常训练一次
    - 不进行重复训练，保持训练效率
    """
    
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        window_size=5,
        loss_threshold=0.3,
        trend_threshold=0.01,
        vol_threshold=0.1,
        window_min_size=3,
        volatility_mode='suppress',
        weight_trend=1.0,
        weight_zloss=0.5,
        weight_vol=0.5,
        outlier_threshold=2.0,  # 异常检测阈值
        save_outlier_log=True,   # 是否保存异常样本日志
        outlier_log_path='outlier_log.json'  # 异常样本日志路径
    ):
        """初始化我的训练策略
        
        Args:
            model: 要训练的模型
            criterion: 损失函数
            optimizer: 优化器
            window_size: 滑动窗口长度
            loss_threshold: 损失阈值
            trend_threshold: 趋势阈值
            vol_threshold: 波动阈值
            window_min_size: 最小窗口大小
            volatility_mode: 波动处理模式 ('suppress' 或 'encourage')
            weight_trend: 趋势权重
            weight_zloss: Z分数权重
            weight_vol: 波动权重
            outlier_threshold: 异常检测阈值（风险分数>=此值将被丢弃）
            save_outlier_log: 是否保存异常样本日志
            outlier_log_path: 异常样本日志文件路径
        """
        super(MyStrategy, self).__init__(model, criterion, optimizer)
        
        # 使用适合语言模型的损失函数
        self.language_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 滑窗参数
        self.window_size = window_size
        self.loss_threshold = loss_threshold
        self.trend_threshold = trend_threshold
        self.vol_threshold = vol_threshold
        self.window_min_size = window_min_size
        self.volatility_mode = volatility_mode
        
        # 风险权重
        self.weight_trend = float(weight_trend)
        self.weight_zloss = float(weight_zloss)
        self.weight_vol = float(weight_vol)
        
        # 异常检测参数
        self.outlier_threshold = float(outlier_threshold)
        self.save_outlier_log = bool(save_outlier_log)
        self.outlier_log_path = outlier_log_path
        
        # 内部状态
        self.loss_window = deque(maxlen=window_size)
        self.batch_history = []
        self.outlier_log = []
        self.batch_count = 0
        self.discarded_count = 0
        self.normal_count = 0
        
        # 添加数组存储用于对比分析
        self.loss_array = []
        self.risk_scores_array = []
        self.discard_flags_array = []
        self.batch_timestamps = []
        
        # 创建异常日志目录
        if self.save_outlier_log:
            os.makedirs(os.path.dirname(self.outlier_log_path) if os.path.dirname(self.outlier_log_path) else '.', exist_ok=True)

    def _compute_trend_std(self, series):
        """计算趋势斜率和标准差"""
        n = len(series)
        if n < 2:
            return 0.0, 0.0
        trend = (series[-1] - series[0]) / max(1, (n - 1))
        std = float(np.std(series))
        return float(trend), std

    def _calculate_risk_score(self, inputs, targets):
        """计算当前批次的风险分数"""
        # 初始前向传播，获取当前批次初始损失（用于决策），不参与反传
        with torch.no_grad():
            initial_outputs = self.model(**inputs)
            # 提取logits
            if hasattr(initial_outputs, 'logits'):
                initial_logits = initial_outputs.logits
            else:
                initial_logits = initial_outputs
            
            # 重塑logits和targets以匹配CrossEntropyLoss的期望
            # logits: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
            # targets: [batch_size, seq_len] -> [batch_size * seq_len]
            batch_size, seq_len, vocab_size = initial_logits.size()
            initial_logits = initial_logits.view(-1, vocab_size)
            targets = targets.view(-1)
            
            initial_loss = float(self.language_criterion(initial_logits, targets).item())

        # 如果窗口数据不足，返回基础风险分数
        if len(self.loss_window) < self.window_min_size:
            # 基于当前损失与阈值的比较
            if initial_loss > self.loss_threshold:
                return 0.5  # 中等风险
            else:
                return 0.0  # 低风险

        # 基于窗口计算趋势/波动
        full_losses = list(self.loss_window)
        recent_losses = full_losses[-self.window_size:]
        
        trend, std_dev = self._compute_trend_std(recent_losses)
        mean_loss = float(np.mean(recent_losses))
        safe_std = max(std_dev, 1e-8)
        
        # 标准化当前损失与趋势
        z_loss = (initial_loss - mean_loss) / safe_std
        norm_trend = trend / safe_std

        # 风险评分
        risk = 0.0
        
        # 1. 趋势风险
        if norm_trend > self.trend_threshold:
            risk += self.weight_trend * (norm_trend - self.trend_threshold)
        
        # 2. 损失风险
        if initial_loss > self.loss_threshold:
            risk += max(0.0, self.weight_zloss * z_loss)
        
        # 3. 波动风险
        if std_dev > self.vol_threshold:
            vol_term = self.weight_vol * (std_dev - self.vol_threshold) / max(self.vol_threshold, 1e-8)
            if self.volatility_mode == 'suppress':
                risk -= vol_term  # 抑制模式：波动大时减少风险
            else:
                risk += vol_term  # 鼓励模式：波动大时增加风险

        return max(0.0, risk)

    def _mark_as_outlier(self, inputs, targets, risk_score, image_path: str = None):
        """标记异常样本，记录更详细的信息"""
        # 安全的统计信息计算
        def safe_mean(tensor):
            if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
                return float(torch.mean(tensor).item())
            else:
                return float(torch.mean(tensor.float()).item())
        
        def safe_std(tensor):
            if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
                return float(torch.std(tensor).item())
            else:
                return float(torch.std(tensor.float()).item())
        
        # 计算窗口统计信息
        window_mean = float(np.mean(list(self.loss_window)) if self.loss_window else 0)
        window_std = float(np.std(list(self.loss_window)) if self.loss_window else 0)
        
        inp_tensor = inputs['input_ids'] if isinstance(inputs, dict) else inputs
        outlier_info = {
            'timestamp': time.time(),
            'epoch': self.current_epoch,
            'batch_idx': self.batch_count,
            'risk_score': float(risk_score),
            'outlier_threshold': float(self.outlier_threshold),
            'image_path': image_path,
            'input_shape': list(inp_tensor.shape),
            'target_shape': list(targets.shape),
            # 输入数据统计
            'input_stats': {
                'mean': safe_mean(inp_tensor),
                'std': safe_std(inp_tensor),
                'min': float(torch.min(inp_tensor.float()).item()) if inp_tensor.dtype != torch.float else float(torch.min(inp_tensor).item()),
                'max': float(torch.max(inp_tensor.float()).item()) if inp_tensor.dtype != torch.float else float(torch.max(inp_tensor).item())
            },
            # 目标数据统计
            'target_stats': {
                'mean': safe_mean(targets),
                'std': safe_std(targets),
                'min': float(torch.min(targets.float()).item()) if targets.dtype != torch.float else float(torch.min(targets).item()),
                'max': float(torch.max(targets.float()).item()) if targets.dtype != torch.float else float(torch.max(targets).item())
            },
            # 窗口统计信息
            'window_stats': {
                'size': len(self.loss_window),
                'mean': window_mean,
                'std': window_std,
                'min': float(np.min(list(self.loss_window)) if self.loss_window else 0),
                'max': float(np.max(list(self.loss_window)) if self.loss_window else 0)
            },
            # 异常检测分析
            'outlier_analysis': {
                'threshold_exceeded_by': risk_score - self.outlier_threshold,
                'relative_risk': risk_score / max(self.outlier_threshold, 1e-8),
                'is_severe_outlier': risk_score > self.outlier_threshold * 2
            }
        }
        
        self.outlier_log.append(outlier_info)
        self.discarded_count += 1
        
        # 定期保存异常日志
        if len(self.outlier_log) % 100 == 0 and self.save_outlier_log:
            self._save_outlier_log()

    def _save_outlier_log(self):
        """保存异常样本日志到文件"""
        try:
            with open(self.outlier_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.outlier_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存异常日志失败: {e}")

    def _normal_training(self, inputs, targets):
        """正常训练一次"""
        self.optimizer.zero_grad()
        outputs = self.model(**inputs)
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # 重塑logits和targets
        batch_size, seq_len, vocab_size = logits.size()
        logits = logits.view(-1, vocab_size)
        targets = targets.view(-1)
        
        loss = self.language_criterion(logits, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_batch(self, inputs, targets, image_path=None):
        """训练一个批次的数据
        
        策略：
        - 风险分数 >= outlier_threshold：标记为异常，跳过训练
        - 风险分数 < outlier_threshold：正常训练一次
        
        Returns:
            tuple: (损失值, 训练次数, 是否被丢弃)
        """
        self.model.train()
        self.batch_count += 1
        
        # 计算风险分数
        risk_score = self._calculate_risk_score(inputs, targets)
        
        # 获取当前批次初始损失
        with torch.no_grad():
            initial_outputs = self.model(**inputs)
            if hasattr(initial_outputs, 'logits'):
                initial_logits = initial_outputs.logits
            else:
                initial_logits = initial_outputs
            
            # 重塑logits和targets
            batch_size, seq_len, vocab_size = initial_logits.size()
            initial_logits = initial_logits.view(-1, vocab_size)
            targets_reshaped = targets.view(-1)
            
            initial_loss = float(self.language_criterion(initial_logits, targets_reshaped).item())
        
        # 根据风险分数决定处理策略
        is_discarded = risk_score >= self.outlier_threshold
        
        if is_discarded:
            # 异常样本：标记丢弃，不进行反向传播
            self._mark_as_outlier(inputs, targets, risk_score, image_path=image_path)
            
            # 不更新损失窗口，保持窗口的"纯净性"
            # self.loss_window.append(initial_loss)  # 删除这行
            
            # 记录历史
            record = {
                'batch_idx': self.batch_count,
                'initial_loss': initial_loss,
                'risk_score': risk_score,
                'action': 'discard',
                'repeat_count': 0,
                'final_loss': None,
                'is_outlier': True,
                'image_path': image_path
            }
            self.batch_history.append(record)
            
            # 记录数组数据用于对比分析
            self.loss_array.append(initial_loss)
            self.risk_scores_array.append(risk_score)
            self.discard_flags_array.append(1)
            self.batch_timestamps.append(time.time())
            
            return None, 0, True  # 返回None表示无损失，0表示训练次数，True表示被丢弃
            
        else:
            # 正常样本：正常训练一次
            final_loss = self._normal_training(inputs, targets)
            self.normal_count += 1
            
            # 只有正常训练的样本才更新损失窗口
            self.loss_window.append(initial_loss)
            
            # 记录历史
            record = {
                'batch_idx': self.batch_count,
                'initial_loss': initial_loss,
                'risk_score': risk_score,
                'action': 'normal',
                'repeat_count': 1,
                'final_loss': final_loss,
                'is_outlier': False,
                'image_path': image_path
            }
            self.batch_history.append(record)
            
            # 记录数组数据用于对比分析
            self.loss_array.append(initial_loss)
            self.risk_scores_array.append(risk_score)
            self.discard_flags_array.append(0)
            self.batch_timestamps.append(time.time())
            
            return final_loss, 1, False  # 返回损失值，1表示训练次数，False表示未被丢弃

    def get_statistics(self):
        """获取训练统计信息"""
        total_batches = self.batch_count
        discarded_rate = self.discarded_count / max(1, total_batches) * 100 if total_batches > 0 else 0
        
        stats = {
            'total_batches': total_batches,
            'normal_batches': self.normal_count,
            'discarded_batches': self.discarded_count,
            'discarded_rate_percent': discarded_rate,
            'current_epoch': self.current_epoch,
            'window_size': len(self.loss_window),
            'outlier_threshold': self.outlier_threshold
        }
        
        return stats

    def print_statistics(self):
        """打印训练统计信息"""
        stats = self.get_statistics()
        print(f"\n=== 训练统计信息 (Epoch {stats['current_epoch']}) ===")
        print(f"总批次数: {stats['total_batches']}")
        print(f"正常训练: {stats['normal_batches']}")
        print(f"丢弃样本: {stats['discarded_batches']}")
        print(f"丢弃率: {stats['discarded_rate_percent']:.2f}%")
        print(f"当前窗口大小: {stats['window_size']}")
        print(f"异常阈值: {stats['outlier_threshold']}")
        print("=" * 50)

    def save_final_logs(self):
        """保存最终的训练日志"""
        if self.save_outlier_log and self.outlier_log:
            self._save_outlier_log()
            print(f"异常样本日志已保存到: {self.outlier_log_path}")
        
        # 保存训练历史
        history_path = 'training_history.json'
        try:
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.batch_history, f, indent=2, ensure_ascii=False)
            print(f"训练历史已保存到: {history_path}")
        except Exception as e:
            print(f"保存训练历史失败: {e}")

    def save_numpy_data(self, output_dir):
        """保存NumPy数组数据，包含丢失数据的详细信息"""
        results_dir = os.path.join(output_dir, 'mystrategy_numpy')
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存各种数组
        np.save(os.path.join(results_dir, 'loss_array.npy'), np.array(self.loss_array))
        np.save(os.path.join(results_dir, 'risk_scores.npy'), np.array(self.risk_scores_array))
        np.save(os.path.join(results_dir, 'discard_flags.npy'), np.array(self.discard_flags_array))
        np.save(os.path.join(results_dir, 'batch_timestamps.npy'), np.array(self.batch_timestamps))
        
        # 保存损失窗口
        if self.loss_window:
            np.save(os.path.join(results_dir, 'loss_window.npy'), 
                   np.array(list(self.loss_window)))
        
        # 新增：保存丢失数据的详细信息
        if self.outlier_log:
            outlier_risk_scores = [item['risk_score'] for item in self.outlier_log]
            outlier_epochs = [item['epoch'] for item in self.outlier_log]
            outlier_batch_indices = [item['batch_idx'] for item in self.outlier_log]
            outlier_threshold_exceeded = [item['outlier_analysis']['threshold_exceeded_by'] for item in self.outlier_log]
            
            np.save(os.path.join(results_dir, 'outlier_risk_scores.npy'), np.array(outlier_risk_scores))
            np.save(os.path.join(results_dir, 'outlier_epochs.npy'), np.array(outlier_epochs))
            np.save(os.path.join(results_dir, 'outlier_batch_indices.npy'), np.array(outlier_batch_indices))
            np.save(os.path.join(results_dir, 'outlier_threshold_exceeded.npy'), np.array(outlier_threshold_exceeded))
        
        # 保存统计信息
        stats = self.get_statistics()
        np.save(os.path.join(results_dir, 'final_stats.npy'), 
               np.array([stats['total_batches'], stats['normal_batches'], 
                        stats['discarded_batches'], stats['discarded_rate_percent']]))
        
        print(f"MyStrategy NumPy数据已保存到: {results_dir}")
        if self.outlier_log:
            print(f"包含 {len(self.outlier_log)} 个丢失样本的详细信息")


class BaselineStrategy(BaseTrainingStrategy):
    """Baseline策略：每批次训练一次（对照策略）"""
    
    def __init__(self, model, criterion, optimizer):
        super(BaselineStrategy, self).__init__(model, criterion, optimizer)
        
        # 使用适合语言模型的损失函数
        self.language_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 统计信息
        self.batch_count = 0
        self.normal_count = 0
        
        # 添加缺失的属性
        self.batch_history = []  # 添加这个属性
        
        # 添加数组存储用于对比分析
        self.loss_array = []
        self.batch_timestamps = []
    
    def train_batch(self, inputs, targets, image_path=None):
        """训练一个批次的数据"""
        self.model.train()
        self.batch_count += 1
        
        # 直接进行正常训练，不计算风险分数
        self.optimizer.zero_grad()
        outputs = self.model(**inputs)
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # 重塑logits和targets
        batch_size, seq_len, vocab_size = logits.size()
        logits = logits.view(-1, vocab_size)
        targets = targets.view(-1)
        
        loss = self.language_criterion(logits, targets)
        loss.backward()
        self.optimizer.step()
        
        self.normal_count += 1
        
        # 记录训练历史
        record = {
            'batch_idx': self.batch_count,
            'loss': loss.item(),
            'action': 'normal',
            'repeat_count': 1,
            'is_outlier': False,
            'image_path': image_path
        }
        self.batch_history.append(record)
        
        # 记录数组数据用于对比分析
        self.loss_array.append(loss.item())
        self.batch_timestamps.append(time.time())
        
        return loss.item(), 1, False
    
    def get_statistics(self):
        """获取训练统计信息"""
        stats = {
            'total_batches': self.batch_count,
            'normal_batches': self.normal_count,
            'discarded_batches': 0,
            'discarded_rate_percent': 0.0,
            'current_epoch': self.current_epoch,
            'strategy_name': 'Baseline'
        }
        return stats
    
    def print_statistics(self):
        """打印训练统计信息"""
        stats = self.get_statistics()
        print(f"\n=== Baseline策略统计信息 (Epoch {stats['current_epoch']}) ===")
        print(f"总批次数: {stats['total_batches']}")
        print(f"正常训练: {stats['normal_batches']}")
        print(f"丢弃样本: {stats['discarded_batches']}")
        print(f"丢弃率: {stats['discarded_rate_percent']:.2f}%")
        print(f"策略名称: {stats['strategy_name']}")
        print("=" * 50)

    def save_final_logs(self):
        """保存最终的训练日志"""
        # 保存训练历史
        history_path = 'baseline_training_history.json'
        try:
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.batch_history, f, indent=2, ensure_ascii=False)
            print(f"Baseline训练历史已保存到: {history_path}")
        except Exception as e:
            print(f"保存Baseline训练历史失败: {e}")

    def save_numpy_data(self, output_dir):
        """保存NumPy数组数据，包含训练数据的详细信息"""
        results_dir = os.path.join(output_dir, 'baseline_numpy')
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存各种数组
        np.save(os.path.join(results_dir, 'loss_array.npy'), np.array(self.loss_array))
        np.save(os.path.join(results_dir, 'batch_timestamps.npy'), np.array(self.batch_timestamps))
        
        # 新增：保存训练数据的统计信息
        if self.batch_history:
            training_losses = [item['loss'] for item in self.batch_history]
            training_epochs = [item.get('epoch', 0) for item in self.batch_history]
            
            np.save(os.path.join(results_dir, 'training_losses.npy'), np.array(training_losses))
            np.save(os.path.join(results_dir, 'training_epochs.npy'), np.array(training_epochs))
        
        # 保存统计信息
        stats = self.get_statistics()
        np.save(os.path.join(results_dir, 'final_stats.npy'), 
               np.array([stats['total_batches'], stats['normal_batches'], 
                        stats['discarded_batches'], stats['discarded_rate_percent']]))
        
        print(f"Baseline NumPy数据已保存到: {results_dir}")
        print(f"包含 {len(self.batch_history)} 个训练batch的详细信息")

class SlidingWindowMeanStrategy(BaseTrainingStrategy):
    """基于滑动窗口平均值的异常检测策略
    
    核心思想：
    - 维护滑动窗口存储历史loss值
    - 计算当前loss与窗口平均值的差值
    - 如果差值 > 固定阈值，则丢弃该batch
    - 阈值是固定值，不随训练动态调整
    """
    
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        window_size=10,
        threshold=1.0,
        save_outlier_log=True,
        outlier_log_path='sliding_window_outlier_log.json'
    ):
        super().__init__(model, criterion, optimizer)
        
        # 策略参数
        self.window_size = window_size
        self.threshold = threshold  # 固定阈值
        
        # 滑动窗口
        self.loss_window = deque(maxlen=window_size)
        
        # 统计信息
        self.batch_count = 0
        self.normal_count = 0
        self.discarded_count = 0
        self.batch_history = []
        
        # 异常样本日志
        self.save_outlier_log = save_outlier_log
        self.outlier_log_path = outlier_log_path
        self.outlier_log = []
        
        # 语言模型损失函数
        self.language_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        print(f"SlidingWindowMeanStrategy初始化完成:")
        print(f"  窗口大小: {window_size}")
        print(f"  固定阈值: {threshold}")
    
    def _is_outlier(self, current_loss):
        """判断是否为异常样本"""
        if len(self.loss_window) < self.window_size:
            # 窗口未满，无法判断，正常训练
            return False
        
        # 计算窗口平均值
        window_mean = np.mean(list(self.loss_window))
        
        # 计算差值
        loss_diff = abs(current_loss - window_mean)
        
        # 判断是否超过阈值
        is_outlier = loss_diff > self.threshold
        
        return is_outlier
    
    def _mark_as_outlier(self, inputs, targets, loss_value, batch_info):
        """标记异常样本，记录更详细的信息"""
        self.discarded_count += 1
        
        # 计算更详细的统计信息
        window_mean = float(np.mean(list(self.loss_window)) if self.loss_window else 0)
        window_std = float(np.std(list(self.loss_window)) if self.loss_window else 0)
        loss_diff = float(abs(loss_value - window_mean))
        
        # 记录异常信息
        inp_tensor = inputs['input_ids'] if isinstance(inputs, dict) else inputs
        outlier_info = {
            'epoch': self.current_epoch,
            'batch': self.batch_count,
            'timestamp': time.time(),
            'loss_value': float(loss_value),
            'window_mean': window_mean,
            'window_std': window_std,
            'loss_diff': loss_diff,
            'threshold': float(self.threshold),
            'window_size': len(self.loss_window),
            'image_path': batch_info.get('image_path'),
            'batch_info': batch_info,
            # 新增：输入数据的统计信息
            'input_stats': {
                'input_shape': list(inp_tensor.shape),
                'input_mean': float(torch.mean(inp_tensor.float()).item()) if inp_tensor.dtype != torch.float else float(torch.mean(inp_tensor).item()),
                'input_std': float(torch.std(inp_tensor.float()).item()) if inp_tensor.dtype != torch.float else float(torch.std(inp_tensor).item()),
                'input_min': float(torch.min(inp_tensor.float()).item()) if inp_tensor.dtype != torch.float else float(torch.min(inp_tensor).item()),
                'input_max': float(torch.max(inp_tensor.float()).item()) if inp_tensor.dtype != torch.float else float(torch.max(inp_tensor).item())
            },
            # 新增：目标数据的统计信息
            'target_stats': {
                'target_shape': list(targets.shape),
                'target_mean': float(torch.mean(targets.float()).item()) if targets.dtype != torch.float else float(torch.mean(targets).item()),
                'target_std': float(torch.std(targets.float()).item()) if targets.dtype != torch.float else float(torch.std(targets).item()),
                'target_min': float(torch.min(targets.float()).item()) if targets.dtype != torch.float else float(torch.min(targets).item()),
                'target_max': float(torch.max(targets.float()).item()) if targets.dtype != torch.float else float(torch.max(targets).item())
            },
            # 新增：异常检测的详细分析 - 修复布尔值序列化问题
            'outlier_analysis': {
                'is_above_threshold': int(loss_diff > self.threshold),  # 转换为整数
                'threshold_exceeded_by': loss_diff - self.threshold,
                'relative_deviation': loss_diff / max(window_std, 1e-8),
                'z_score': (loss_value - window_mean) / max(window_std, 1e-8) if window_std > 0 else 0
            }
        }
        
        self.outlier_log.append(outlier_info)
        
        # 保存异常日志
        if self.save_outlier_log:
            self._save_outlier_log()
    
    def _save_outlier_log(self):
        """保存异常样本日志"""
        try:
            with open(self.outlier_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.outlier_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存异常日志失败: {e}")
    
    def _normal_training(self, inputs, targets):
        """正常训练流程"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        outputs = self.model(**inputs)
        
        # 处理输出格式
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # 重塑logits和targets
        batch_size, seq_len, vocab_size = logits.size()
        logits = logits.view(-1, vocab_size)
        targets = targets.view(-1)
        
        # 计算损失
        loss = self.language_criterion(logits, targets)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        self.normal_count += 1
        return loss.item()
    
    def train_batch(self, inputs, targets, image_path=None):
        """训练一个batch"""
        self.batch_count += 1
        
        # 计算当前batch的loss（用于异常检测）
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            batch_size, seq_len, vocab_size = logits.size()
            logits = logits.view(-1, vocab_size)
            targets_eval = targets.view(-1)
            
            current_loss = self.language_criterion(logits, targets_eval).item()
        
        # 记录batch信息
        inp_shape = list(inputs['input_ids'].shape) if isinstance(inputs, dict) else list(inputs.shape)
        batch_info = {
            'input_shape': inp_shape,
            'target_shape': list(targets.shape),
            'current_loss': current_loss,
            'image_path': image_path
        }
        
        # 异常检测
        if self._is_outlier(current_loss):
            # 标记为异常，跳过训练
            self._mark_as_outlier(inputs, targets, current_loss, batch_info)
            
            # 记录到历史
            self.batch_history.append({
                'epoch': self.current_epoch,
                'batch': self.batch_count,
                'loss': current_loss,
                'status': 'discarded',
                'reason': 'outlier_detected',
                'image_path': image_path,
                'timestamp': time.time()
            })
            
            return current_loss, 1, True  # (loss, repeat_count, is_discarded)
        else:
            # 正常训练
            loss = self._normal_training(inputs, targets)
            
            # 更新滑动窗口（只记录正常训练的loss）
            self.loss_window.append(current_loss)
            
            # 记录到历史
            self.batch_history.append({
                'epoch': self.current_epoch,
                'batch': self.batch_count,
                'loss': loss,
                'status': 'trained',
                'reason': 'normal_sample',
                'image_path': image_path,
                'timestamp': time.time()
            })
            
            return loss, 1, False  # (loss, repeat_count, is_discarded)
    
    def get_statistics(self):
        """获取统计信息"""
        stats = {
            'total_batches': self.batch_count,
            'normal_training': self.normal_count,
            'discarded_samples': self.discarded_count,
            'discard_rate': self.discarded_count / max(1, self.batch_count),
            'window_size': self.window_size,
            'threshold': self.threshold,
            'current_window_size': len(self.loss_window),
            'window_mean': np.mean(list(self.loss_window)) if self.loss_window else 0,
            'window_std': np.std(list(self.loss_window)) if self.loss_window else 0
        }
        return stats
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        print(f"\n=== 滑动窗口均值策略统计信息 (Epoch {self.current_epoch}) ===")
        print(f"总批次数: {stats['total_batches']}")
        print(f"正常训练: {stats['normal_training']}")
        print(f"丢弃样本: {stats['discarded_samples']}")
        print(f"丢弃率: {stats['discard_rate']:.2%}")
        print(f"当前窗口大小: {stats['current_window_size']}")
        print(f"固定阈值: {stats['threshold']}")
        print(f"窗口均值: {stats['window_mean']:.4f}")
        print(f"窗口标准差: {stats['window_std']:.4f}")
        print("=" * 50)
    
    def save_final_logs(self):
        """保存最终日志"""
        # 保存训练历史
        history_path = 'sliding_window_training_history.json'
        try:
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.batch_history, f, indent=2, ensure_ascii=False)
            print(f"滑动窗口策略训练历史已保存到: {history_path}")
        except Exception as e:
            print(f"保存训练历史失败: {e}")
        
        # 保存异常日志
        if self.save_outlier_log:
            self._save_outlier_log()
            print(f"异常样本日志已保存到: {self.outlier_log_path}")
    
    def save_numpy_data(self, output_dir):
        """保存NumPy格式数据，包含丢失数据的详细信息"""
        results_dir = os.path.join(output_dir, 'sliding_window_numpy')
        os.makedirs(results_dir, exist_ok=True)
        
        # 提取数据
        epochs = [item['epoch'] for item in self.batch_history]
        losses = [item['loss'] for item in self.batch_history]
        statuses = [item['status'] for item in self.batch_history]
        
        # 新增：丢失数据的详细信息
        discarded_batches = [item for item in self.batch_history if item['status'] == 'discarded']
        if discarded_batches:
            discarded_losses = [item['loss'] for item in discarded_batches]
            discarded_epochs = [item['epoch'] for item in discarded_batches]
            discarded_reasons = [item.get('reason', 'unknown') for item in discarded_batches]
        else:
            discarded_losses = []
            discarded_epochs = []
            discarded_reasons = []
        
        # 保存数据
        np.save(os.path.join(results_dir, 'epochs.npy'), np.array(epochs))
        np.save(os.path.join(results_dir, 'losses.npy'), np.array(losses))
        np.save(os.path.join(results_dir, 'statuses.npy'), np.array(statuses))
        
        # 新增：保存丢失数据信息
        np.save(os.path.join(results_dir, 'discarded_losses.npy'), np.array(discarded_losses))
        np.save(os.path.join(results_dir, 'discarded_epochs.npy'), np.array(discarded_epochs))
        np.save(os.path.join(results_dir, 'discarded_reasons.npy'), np.array(discarded_reasons))
        
        # 保存异常日志的统计信息
        if self.outlier_log:
            outlier_losses = [item['loss_value'] for item in self.outlier_log]
            outlier_diffs = [item['loss_diff'] for item in self.outlier_log]
            outlier_z_scores = [item['outlier_analysis']['z_score'] for item in self.outlier_log]
            
            np.save(os.path.join(results_dir, 'outlier_losses.npy'), np.array(outlier_losses))
            np.save(os.path.join(results_dir, 'outlier_diffs.npy'), np.array(outlier_diffs))
            np.save(os.path.join(results_dir, 'outlier_z_scores.npy'), np.array(outlier_z_scores))
        
        print(f"滑动窗口策略NumPy数据已保存到: {results_dir}")
        print(f"包含 {len(discarded_batches)} 个丢失batch的详细信息")


class SlidingWindowProportionalStrategy(BaseTrainingStrategy):
    """基于滑动窗口比例阈值的异常检测策略
    
    核心思想：
    - 维护滑动窗口存储历史loss值
    - 计算窗口平均值
    - 将平均值乘以比例系数（如1.2）
    - 如果当前loss > 平均值 * 比例系数，则丢弃该batch
    """
    
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        window_size=10,
        proportion_factor=1.2,
        save_outlier_log=True,
        outlier_log_path='sliding_window_proportional_outlier_log.json'
    ):
        super().__init__(model, criterion, optimizer)
        
        # 策略参数
        self.window_size = window_size
        self.proportion_factor = proportion_factor  # 比例系数
        
        # 滑动窗口
        self.loss_window = deque(maxlen=window_size)
        
        # 统计信息
        self.batch_count = 0
        self.normal_count = 0
        self.discarded_count = 0
        self.batch_history = []
        
        # 异常样本日志
        self.save_outlier_log = save_outlier_log
        self.outlier_log_path = outlier_log_path
        self.outlier_log = []
        
        # 语言模型损失函数
        self.language_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        print(f"SlidingWindowProportionalStrategy初始化完成:")
        print(f"  窗口大小: {window_size}")
        print(f"  比例系数: {proportion_factor}")
    
    def _is_outlier(self, current_loss):
        """判断是否为异常样本"""
        if len(self.loss_window) < self.window_size:
            # 窗口未满，无法判断，正常训练
            # 返回三个值：False表示不是异常，阈值为0，窗口均值为0
            return False, 0.0, 0.0
        
        # 计算窗口平均值
        window_mean = np.mean(list(self.loss_window))
        
        # 计算动态阈值：平均值 * 比例系数
        dynamic_threshold = window_mean * self.proportion_factor
        
        # 判断是否超过阈值
        is_outlier = current_loss > dynamic_threshold
        
        return is_outlier, dynamic_threshold, window_mean
    
    def _mark_as_outlier(self, inputs, targets, loss_value, batch_info, dynamic_threshold, window_mean):
        """标记异常样本，记录更详细的信息"""
        self.discarded_count += 1
        
        # 计算更详细的统计信息
        window_std = float(np.std(list(self.loss_window)) if self.loss_window else 0)
        threshold_exceeded = loss_value - dynamic_threshold
        
        # 记录异常信息
        inp_tensor = inputs['input_ids'] if isinstance(inputs, dict) else inputs
        outlier_info = {
            'epoch': self.current_epoch,
            'batch': self.batch_count,
            'timestamp': time.time(),
            'loss_value': float(loss_value),
            'window_mean': window_mean,
            'window_std': window_std,
            'proportion_factor': float(self.proportion_factor),
            'dynamic_threshold': float(dynamic_threshold),
            'threshold_exceeded': float(threshold_exceeded),
            'window_size': len(self.loss_window),
            'image_path': batch_info.get('image_path'),
            'batch_info': batch_info,
            # 输入数据的统计信息
            'input_stats': {
                'input_shape': list(inp_tensor.shape),
                'input_mean': float(torch.mean(inp_tensor.float()).item()) if inp_tensor.dtype != torch.float else float(torch.mean(inp_tensor).item()),
                'input_std': float(torch.std(inp_tensor.float()).item()) if inp_tensor.dtype != torch.float else float(torch.std(inp_tensor).item()),
                'input_min': float(torch.min(inp_tensor.float()).item()) if inp_tensor.dtype != torch.float else float(torch.min(inp_tensor).item()),
                'input_max': float(torch.max(inp_tensor.float()).item()) if inp_tensor.dtype != torch.float else float(torch.max(inp_tensor).item())
            },
            # 目标数据的统计信息
            'target_stats': {
                'target_shape': list(targets.shape),
                'target_mean': float(torch.mean(targets.float()).item()) if targets.dtype != torch.float else float(torch.mean(targets).item()),
                'target_std': float(torch.std(targets.float()).item()) if targets.dtype != torch.float else float(torch.std(targets).item()),
                'target_min': float(torch.min(targets.float()).item()) if targets.dtype != torch.float else float(torch.min(targets).item()),
                'target_max': float(torch.max(targets.float()).item()) if targets.dtype != torch.float else float(torch.max(targets).item())
            },
            # 异常检测的详细分析 - 修复布尔值序列化问题
            'outlier_analysis': {
                'is_above_threshold': int(loss_value > dynamic_threshold),  # 转换为整数
                'threshold_exceeded_by': threshold_exceeded,
                'relative_to_mean': loss_value / max(window_mean, 1e-8),
                'relative_to_threshold': loss_value / max(dynamic_threshold, 1e-8),
                'z_score': (loss_value - window_mean) / max(window_std, 1e-8) if window_std > 0 else 0
            }
        }
        
        self.outlier_log.append(outlier_info)
        
        # 保存异常日志
        if self.save_outlier_log:
            self._save_outlier_log()
    
    def _save_outlier_log(self):
        """保存异常样本日志"""
        try:
            with open(self.outlier_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.outlier_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存异常日志失败: {e}")
    
    def _normal_training(self, inputs, targets):
        """正常训练流程"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        outputs = self.model(**inputs)
        
        # 处理输出格式
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # 重塑logits和targets
        batch_size, seq_len, vocab_size = logits.size()
        logits = logits.view(-1, vocab_size)
        targets = targets.view(-1)
        
        # 计算损失
        loss = self.language_criterion(logits, targets)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        self.normal_count += 1
        return loss.item()
    
    def train_batch(self, inputs, targets, image_path=None):
        """训练一个batch"""
        self.batch_count += 1
        
        # 计算当前batch的loss（用于异常检测）
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            batch_size, seq_len, vocab_size = logits.size()
            logits = logits.view(-1, vocab_size)
            targets_eval = targets.view(-1)
            
            current_loss = self.language_criterion(logits, targets_eval).item()
        
        # 记录batch信息
        inp_shape = list(inputs['input_ids'].shape) if isinstance(inputs, dict) else list(inputs.shape)
        batch_info = {
            'input_shape': inp_shape,
            'target_shape': list(targets.shape),
            'current_loss': current_loss,
            'image_path': image_path
        }
        
        # 异常检测
        is_outlier, dynamic_threshold, window_mean = self._is_outlier(current_loss)
        
        if is_outlier:
            # 标记为异常，跳过训练
            self._mark_as_outlier(inputs, targets, current_loss, batch_info, dynamic_threshold, window_mean)
            
            # 记录到历史
            self.batch_history.append({
                'epoch': self.current_epoch,
                'batch': self.batch_count,
                'loss': current_loss,
                'status': 'discarded',
                'reason': 'proportional_threshold_exceeded',
                'dynamic_threshold': dynamic_threshold,
                'window_mean': window_mean,
                'image_path': image_path,
                'timestamp': time.time()
            })
            
            return current_loss, 1, True  # (loss, repeat_count, is_discarded)
        else:
            # 正常训练
            loss = self._normal_training(inputs, targets)
            
            # 更新滑动窗口（只记录正常训练的loss）
            self.loss_window.append(current_loss)
            
            # 记录到历史
            self.batch_history.append({
                'epoch': self.current_epoch,
                'batch': self.batch_count,
                'loss': loss,
                'status': 'trained',
                'reason': 'normal_sample',
                'dynamic_threshold': dynamic_threshold if 'dynamic_threshold' in locals() else None,
                'window_mean': window_mean if 'window_mean' in locals() else None,
                'image_path': image_path,
                'timestamp': time.time()
            })
            
            return loss, 1, False  # (loss, repeat_count, is_discarded)
    
    def get_statistics(self):
        """获取统计信息"""
        stats = {
            'total_batches': self.batch_count,
            'normal_training': self.normal_count,
            'discarded_samples': self.discarded_count,
            'discard_rate': self.discarded_count / max(1, self.batch_count),
            'window_size': self.window_size,
            'proportion_factor': self.proportion_factor,
            'current_window_size': len(self.loss_window),
            'window_mean': np.mean(list(self.loss_window)) if self.loss_window else 0,
            'window_std': np.std(list(self.loss_window)) if self.loss_window else 0
        }
        return stats
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        print(f"\n=== 滑动窗口比例策略统计信息 (Epoch {self.current_epoch}) ===")
        print(f"总批次数: {stats['total_batches']}")
        print(f"正常训练: {stats['normal_training']}")
        print(f"丢弃样本: {stats['discarded_samples']}")
        print(f"丢弃率: {stats['discard_rate']:.2%}")
        print(f"当前窗口大小: {stats['current_window_size']}")
        print(f"比例系数: {stats['proportion_factor']}")
        print(f"窗口均值: {stats['window_mean']:.4f}")
        print(f"窗口标准差: {stats['window_std']:.4f}")
        print("=" * 50)
    
    def save_final_logs(self):
        """保存最终日志"""
        # 保存训练历史
        history_path = 'sliding_window_proportional_training_history.json'
        try:
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.batch_history, f, indent=2, ensure_ascii=False)
            print(f"滑动窗口比例策略训练历史已保存到: {history_path}")
        except Exception as e:
            print(f"保存训练历史失败: {e}")
        
        # 保存异常日志
        if self.save_outlier_log:
            self._save_outlier_log()
            print(f"异常样本日志已保存到: {self.outlier_log_path}")
    
    def save_numpy_data(self, output_dir):
        """保存NumPy格式数据，包含丢失数据的详细信息"""
        results_dir = os.path.join(output_dir, 'sliding_window_proportional_numpy')
        os.makedirs(results_dir, exist_ok=True)
        
        # 提取数据
        epochs = [item['epoch'] for item in self.batch_history]
        losses = [item['loss'] for item in self.batch_history]
        statuses = [item['status'] for item in self.batch_history]
        
        # 丢失数据的详细信息
        discarded_batches = [item for item in self.batch_history if item['status'] == 'discarded']
        if discarded_batches:
            discarded_losses = [item['loss'] for item in discarded_batches]
            discarded_epochs = [item['epoch'] for item in discarded_batches]
            discarded_reasons = [item.get('reason', 'unknown') for item in discarded_batches]
            discarded_thresholds = [item.get('dynamic_threshold', 0) for item in discarded_batches]
            discarded_means = [item.get('window_mean', 0) for item in discarded_batches]
        else:
            discarded_losses = []
            discarded_epochs = []
            discarded_reasons = []
            discarded_thresholds = []
            discarded_means = []
        
        # 保存数据
        np.save(os.path.join(results_dir, 'epochs.npy'), np.array(epochs))
        np.save(os.path.join(results_dir, 'losses.npy'), np.array(losses))
        np.save(os.path.join(results_dir, 'statuses.npy'), np.array(statuses))
        
        # 保存丢失数据信息
        np.save(os.path.join(results_dir, 'discarded_losses.npy'), np.array(discarded_losses))
        np.save(os.path.join(results_dir, 'discarded_epochs.npy'), np.array(discarded_epochs))
        np.save(os.path.join(results_dir, 'discarded_reasons.npy'), np.array(discarded_reasons))
        np.save(os.path.join(results_dir, 'discarded_thresholds.npy'), np.array(discarded_thresholds))
        np.save(os.path.join(results_dir, 'discarded_means.npy'), np.array(discarded_means))
        
        # 保存异常日志的统计信息
        if self.outlier_log:
            outlier_losses = [item['loss_value'] for item in self.outlier_log]
            outlier_thresholds = [item['dynamic_threshold'] for item in self.outlier_log]
            outlier_means = [item['window_mean'] for item in self.outlier_log]
            outlier_exceeded = [item['threshold_exceeded'] for item in self.outlier_log]
            
            np.save(os.path.join(results_dir, 'outlier_losses.npy'), np.array(outlier_losses))
            np.save(os.path.join(results_dir, 'outlier_thresholds.npy'), np.array(outlier_thresholds))
            np.save(os.path.join(results_dir, 'outlier_means.npy'), np.array(outlier_means))
            np.save(os.path.join(results_dir, 'outlier_exceeded.npy'), np.array(outlier_exceeded))
        
        print(f"滑动窗口比例策略NumPy数据已保存到: {results_dir}")
        print(f"包含 {len(discarded_batches)} 个丢失batch的详细信息")


# 使用示例
if __name__ == "__main__":
    # 示例：如何创建和使用MyStrategy
    print("MyStrategy 使用示例:")
    print("1. 创建策略实例")
    print("2. 在训练循环中调用 train_batch()")
    print("3. 根据返回值判断是否被丢弃")
    print("4. 定期调用 print_statistics() 查看统计信息")
    print("5. 训练结束后调用 save_final_logs() 保存日志")
