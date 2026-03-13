import os
import json
import datetime
import pandas as pd
from libcity.utils import ensure_dir
from libcity.model import loss
from libcity.evaluator.traffic_state_evaluator import TrafficStateEvaluator


class TrafficStateGridEvaluator(TrafficStateEvaluator):

    def __init__(self, config):
        super().__init__(config)
        self.output_dim = self.config.get('output_dim', 1)
        self.mask_val = self.config.get('mask_val', 10)

    def collect(self, batch):
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        y_true = batch['y_true']
        y_pred = batch['y_pred']
        if y_true.shape != y_pred.shape:
            raise ValueError("batch['y_true'].shape is not equal to batch['y_pred'].shape")
        self.len_timeslots = y_true.shape[1]
        for j in range(self.output_dim):
            for i in range(1, self.len_timeslots+1):
                for metric in self.metrics:
                    if str(j)+'-'+metric+'@'+str(i) not in self.intermediate_result:
                        self.intermediate_result[str(j)+'-'+metric+'@'+str(i)] = []
        if self.mode.lower() == 'average':
            for j in range(self.output_dim):
                for i in range(1, self.len_timeslots+1):
                    for metric in self.metrics:
                        if metric == 'masked_MAE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mae_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j], 0, self.mask_val).item())
                        elif metric == 'masked_MSE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mse_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j], 0, self.mask_val).item())
                        elif metric == 'masked_RMSE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_rmse_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j], 0, self.mask_val).item())
                        elif metric == 'masked_MAPE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mape_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j], 0).item())
                        elif metric == 'MAE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mae_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j]).item())
                        elif metric == 'MSE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mse_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j]).item())
                        elif metric == 'RMSE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_rmse_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j]).item())
                        elif metric == 'MAPE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mape_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j]).item())
                        elif metric == 'R2':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.r2_score_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j]).item())
                        elif metric == 'EVAR':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.explained_variance_score_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j]).item())
        elif self.mode.lower() == 'single':
            for j in range(self.output_dim):
                for i in range(1, self.len_timeslots+1):
                    for metric in self.metrics:
                        if metric == 'masked_MAE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mae_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j], 0, self.mask_val).item())
                        elif metric == 'masked_MSE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mse_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j], 0, self.mask_val).item())
                        elif metric == 'masked_RMSE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_rmse_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j], 0, self.mask_val).item())
                        elif metric == 'masked_MAPE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mape_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j], 0).item())
                        elif metric == 'MAE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mae_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j]).item())
                        elif metric == 'MSE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mse_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j]).item())
                        elif metric == 'RMSE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_rmse_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j]).item())
                        elif metric == 'MAPE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mape_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j]).item())
                        elif metric == 'R2':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.r2_score_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j]).item())
                        elif metric == 'EVAR':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.explained_variance_score_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j]).item())
        else:
            raise ValueError('Error parameter evaluator_mode={}, please set `single` or `average`.'.format(self.mode))

    def evaluate(self):
        for j in range(self.output_dim):
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    self.result[str(j)+'-'+metric+'@'+str(i)] = sum(self.intermediate_result[str(j)+'-'+metric+'@'+str(i)]) / \
                                                                len(self.intermediate_result[str(j)+'-'+metric+'@'+str(i)])
        return self.result

    def save_result(self, save_path, filename=None):
        self._logger.info('Note that you select the {} mode to evaluate!'.format(self.mode))
        self.evaluate()
        ensure_dir(save_path)
        if filename is None:
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                       self.config['model'] + '_' + self.config['dataset']

        if 'json' in self.save_modes:
            self._logger.info('Evaluate result is ' + json.dumps(self.result))
            with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
                json.dump(self.result, f)
            self._logger.info('Evaluate result is saved at ' +
                              os.path.join(save_path, '{}.json'.format(filename)))

        dataframe = {}
        if 'csv' in self.save_modes:
            for j in range(self.output_dim):
                for metric in self.metrics:
                    dataframe[str(j)+"-"+metric] = []
                for i in range(1, self.len_timeslots + 1):
                    for metric in self.metrics:
                        dataframe[str(j)+"-"+metric].append(self.result[str(j)+'-'+metric+'@'+str(i)])
            dataframe = pd.DataFrame(dataframe, index=range(1, self.len_timeslots + 1))
            dataframe.to_csv(os.path.join(save_path, '{}.csv'.format(filename)), index=False)
            self._logger.info('Evaluate result is saved at ' +
                              os.path.join(save_path, '{}.csv'.format(filename)))
            self._logger.info("\n" + str(dataframe))
            
            # 额外输出 inflow/outflow 格式的表格
            if self.output_dim == 2:
                feature_names = ['inflow', 'outflow']
                
                # 输出平均值汇总表
                summary_table = self._create_summary_table(feature_names)
                self._logger.info("\n========== Summary Table (Average across all timesteps) ==========")
                self._logger.info("\n" + summary_table)
                
                # 输出每个时间步长的详细表格
                timestep_table = self._create_timestep_table(feature_names)
                self._logger.info(f"\n========== Per-Timestep Metrics ({self.len_timeslots} timestep{'s' if self.len_timeslots > 1 else ''}) ==========")
                self._logger.info("\n" + timestep_table)
                
                with open(os.path.join(save_path, '{}_summary.txt'.format(filename)), 'w') as f:
                    f.write("========== Summary Table (Average across all timesteps) ==========\n")
                    f.write(summary_table)
                    f.write(f"\n\n========== Per-Timestep Metrics ({self.len_timeslots} timestep{'s' if self.len_timeslots > 1 else ''}) ==========\n")
                    f.write(timestep_table)
        return dataframe
    
    def _find_metric_key(self, base_name):
        """在 self.metrics 中找到匹配的 metric key（支持 masked_ 前缀）"""
        for m in self.metrics:
            if m == base_name or m == 'masked_' + base_name:
                return m
        return None

    def _create_summary_table(self, feature_names):
        """创建类似论文表格格式的汇总表"""
        mae_key = self._find_metric_key('MAE')
        mape_key = self._find_metric_key('MAPE')
        rmse_key = self._find_metric_key('RMSE')

        # 计算所有时间步的平均值
        summary_data = {}
        for j in range(self.output_dim):
            summary_data[feature_names[j]] = {}
            for base_name, metric_key in [('MAE', mae_key), ('MAPE', mape_key), ('RMSE', rmse_key)]:
                if metric_key is not None:
                    values = []
                    for i in range(1, self.len_timeslots + 1):
                        key = str(j) + '-' + metric_key + '@' + str(i)
                        if key in self.result:
                            values.append(self.result[key])
                    if values:
                        summary_data[feature_names[j]][base_name] = sum(values) / len(values)
        
        # 格式化输出
        lines = []
        lines.append("=" * 75)
        lines.append(f"{'Dataset':<15} | {'Feature':<10} | {'MAE':<10} | {'MAPE(%)':<12} | {'RMSE':<10}")
        lines.append("-" * 75)
        
        dataset_name = self.config.get('dataset', 'Unknown')
        
        for j, feature in enumerate(feature_names):
            mae_val = summary_data[feature].get('MAE', 0.0)
            mape_val = summary_data[feature].get('MAPE', 0.0) * 100
            rmse_val = summary_data[feature].get('RMSE', 0.0)
            
            if j == 0:
                lines.append(f"{dataset_name:<15} | {feature:<10} | {mae_val:<10.3f} | {mape_val:<12.3f} | {rmse_val:<10.3f}")
            else:
                lines.append(f"{'':15} | {feature:<10} | {mae_val:<10.3f} | {mape_val:<12.3f} | {rmse_val:<10.3f}")
        
        lines.append("=" * 75)
        return '\n'.join(lines)
    
    def _create_timestep_table(self, feature_names):
        """创建每个时间步长的详细指标表格"""
        mae_key = self._find_metric_key('MAE')
        mape_key = self._find_metric_key('MAPE')
        rmse_key = self._find_metric_key('RMSE')

        lines = []
        
        # 为每个feature（inflow/outflow）创建单独的表格
        for j, feature in enumerate(feature_names):
            lines.append("=" * 60)
            lines.append(f"{feature.upper()} - Metrics per Timestep")
            lines.append("=" * 60)
            lines.append(f"{'Timestep':<10} | {'MAE':<12} | {'MAPE(%)':<12} | {'RMSE':<12}")
            lines.append("-" * 60)
            
            for i in range(1, self.len_timeslots + 1):
                mae = self.result.get(f"{j}-{mae_key}@{i}", 0.0) if mae_key else 0.0
                mape = (self.result.get(f"{j}-{mape_key}@{i}", 0.0) if mape_key else 0.0) * 100
                rmse = self.result.get(f"{j}-{rmse_key}@{i}", 0.0) if rmse_key else 0.0
                
                lines.append(f"T={i:<7} | {mae:<12.3f} | {mape:<12.3f} | {rmse:<12.3f}")
            
            lines.append("=" * 60)
            if j < len(feature_names) - 1:
                lines.append("")  # 空行分隔
        
        return '\n'.join(lines)

