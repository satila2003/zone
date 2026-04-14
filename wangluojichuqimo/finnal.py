import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple

# 设置随机种子保证可复现性
np.random.seed(42)
class SignalDataGenerator:
    """信号数据生成类（逻辑不变）"""
    def __init__(self, config: Dict):
        self.time_steps = config["time_steps"]  # 总时间步数
        self.position_noise_amplitude = config["position_noise_amplitude"]  # 位置噪声幅度
        self.signal_noise_amplitude = config["signal_noise_amplitude"]  # 信号参数噪声幅度
        self.position_sequence = None  # 移动台位置序列（0~1）
        self.signal_strength_cell_a = None  # 小区A信号强度(dBm)
        self.signal_strength_cell_b = None  # 小区B信号强度(dBm)
        self.signal_noise_ratio_cell_a = None  # 小区A信噪比(dB)
        self.signal_noise_ratio_cell_b = None  # 小区B信噪比(dB)
        self.transmission_delay_cell_a = None  # 小区A传输延迟(ms)
        self.transmission_delay_cell_b = None  # 小区B传输延迟(ms)

    def generate_position_sequence(self) -> np.ndarray:
        base_position = 0.5 + 0.45 * np.sin(np.linspace(0, 20 * np.pi, self.time_steps))
        position_noise = np.random.normal(0, self.position_noise_amplitude, self.time_steps)
        self.position_sequence = np.clip(base_position + position_noise, 0, 1)
        return self.position_sequence

    def generate_signal_parameters(self) -> Dict[str, np.ndarray]:
        if self.position_sequence is None:
            self.generate_position_sequence()

        # 1. 信号强度（RSRP）计算
        base_ss_a = -140 + 80 * (1 - self.position_sequence)
        self.signal_strength_cell_a = base_ss_a + np.random.normal(0, self.signal_noise_amplitude, self.time_steps)
        self.signal_strength_cell_a = np.clip(self.signal_strength_cell_a, -140, -60)

        base_ss_b = -140 + 80 * self.position_sequence
        self.signal_strength_cell_b = base_ss_b + np.random.normal(0, self.signal_noise_amplitude, self.time_steps)
        self.signal_strength_cell_b = np.clip(self.signal_strength_cell_b, -140, -60)

        # 2. 信噪比（SINR）计算
        base_sinr_a = 30 * (1 - self.position_sequence)
        self.signal_noise_ratio_cell_a = base_sinr_a + np.random.normal(0, self.signal_noise_amplitude / 3,
                                                                        self.time_steps)
        self.signal_noise_ratio_cell_a = np.clip(self.signal_noise_ratio_cell_a, 0, 30)

        base_sinr_b = 30 * self.position_sequence
        self.signal_noise_ratio_cell_b = base_sinr_b + np.random.normal(0, self.signal_noise_amplitude / 3,
                                                                        self.time_steps)
        self.signal_noise_ratio_cell_b = np.clip(self.signal_noise_ratio_cell_b, 0, 30)

        # 3. 传输延迟计算
        base_delay_a = 1 + 19 * self.position_sequence
        self.transmission_delay_cell_a = base_delay_a + np.random.normal(0, self.signal_noise_amplitude / 5,
                                                                         self.time_steps)
        self.transmission_delay_cell_a = np.clip(self.transmission_delay_cell_a, 1, 20)

        base_delay_b = 1 + 19 * (1 - self.position_sequence)
        self.transmission_delay_cell_b = base_delay_b + np.random.normal(0, self.signal_noise_amplitude / 5,
                                                                         self.time_steps)
        self.transmission_delay_cell_b = np.clip(self.transmission_delay_cell_b, 1, 20)

        signal_data = {
            "position": self.position_sequence,
            "signal_strength_cell_a": self.signal_strength_cell_a,
            "signal_strength_cell_b": self.signal_strength_cell_b,
            "signal_noise_ratio_cell_a": self.signal_noise_ratio_cell_a,
            "signal_noise_ratio_cell_b": self.signal_noise_ratio_cell_b,
            "transmission_delay_cell_a": self.transmission_delay_cell_a,
            "transmission_delay_cell_b": self.transmission_delay_cell_b
        }
        return signal_data


class HandoverRule:
    """切换规则类（综合得分的实例属性）"""
    def __init__(self, config: Dict):
        self.traditional_ss_threshold = config["traditional_ss_threshold"]
        self.hysteresis_threshold_for_anti_pingpong = config["hysteresis_threshold"]
        self.continuous_time_steps = config["continuous_time_steps"]
        self.weight_signal_strength = config["weight_ss"]
        self.weight_signal_noise_ratio = config["weight_sinr"]
        self.weight_transmission_delay = config["weight_delay"]

        total_weight = (self.weight_signal_strength + self.weight_signal_noise_ratio +
                        self.weight_transmission_delay)
        assert abs(total_weight - 1) < 1e-6, "参数权重和必须为1（当前和：{:.4f}）".format(total_weight)

        self.traditional_handover_records = None
        self.improved_handover_records = None
        self.score_a = None  # 小区A综合质量得分
        self.score_b = None  # 小区B综合质量得分

    def min_max_normalize(self, data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        if max_val == min_val:
            return np.zeros_like(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        return np.clip(normalized_data, 0, 1)

    def calculate_composite_quality_score(self, signal_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        ss_a, ss_b = signal_data["signal_strength_cell_a"], signal_data["signal_strength_cell_b"]
        sinr_a, sinr_b = signal_data["signal_noise_ratio_cell_a"], signal_data["signal_noise_ratio_cell_b"]
        delay_a, delay_b = signal_data["transmission_delay_cell_a"], signal_data["transmission_delay_cell_b"]

        ss_a_norm = self.min_max_normalize(ss_a, -140, -60)
        ss_b_norm = self.min_max_normalize(ss_b, -140, -60)

        sinr_a_norm = self.min_max_normalize(sinr_a, 0, 30)
        sinr_b_norm = self.min_max_normalize(sinr_b, 0, 30)

        delay_a_norm = 1 - self.min_max_normalize(delay_a, 1, 20)
        delay_b_norm = 1 - self.min_max_normalize(delay_b, 1, 20)

        self.score_a = (self.weight_signal_strength * ss_a_norm +
                   self.weight_signal_noise_ratio * sinr_a_norm +
                   self.weight_transmission_delay * delay_a_norm)
        self.score_b = (self.weight_signal_strength * ss_b_norm +
                   self.weight_signal_noise_ratio * sinr_b_norm +
                   self.weight_transmission_delay * delay_b_norm)
        return self.score_a, self.score_b

    def traditional_handover(self, signal_data: Dict[str, np.ndarray]) -> np.ndarray:
        ss_a = signal_data["signal_strength_cell_a"]
        ss_b = signal_data["signal_strength_cell_b"]
        time_steps = len(ss_a)

        handover_records = [0]
        current_cell = 0
        for i in range(1, time_steps):
            if ss_b[i] > ss_a[i]:
                current_cell = 1
            else:
                current_cell = 0
            handover_records.append(current_cell)

        self.traditional_handover_records = np.array(handover_records)
        return self.traditional_handover_records

    def improved_handover(self, signal_data: Dict[str, np.ndarray]) -> np.ndarray:
        self.calculate_composite_quality_score(signal_data)  # 先计算综合得分
        time_steps = len(self.score_a)
        handover_records = []
        current_cell = 0
        continuous_count = 0

        def score_generator():
            for sa, sb in zip(self.score_a, self.score_b):
                yield sa, sb

        score_gen = score_generator()
        for _ in range(time_steps):
            sa, sb = next(score_gen)
            score_diff = sb - sa if current_cell == 0 else sa - sb

            if score_diff > self.hysteresis_threshold_for_anti_pingpong:
                continuous_count += 1
                if continuous_count >= self.continuous_time_steps:
                    current_cell = 1 if current_cell == 0 else 0
                    continuous_count = 0
            else:
                continuous_count = 0

            handover_records.append(current_cell)

        self.improved_handover_records = np.array(handover_records)
        return self.improved_handover_records

    def detect_pingpong(self, handover_records: np.ndarray, window_size: int = 3) -> int:
        pingpong_count = 0
        self.pingpong_windows = []  # 乒乓切换的窗口位置
        for i in range(len(handover_records) - window_size + 1):
            window = handover_records[i:i + window_size]
            if np.array_equal(window, [0, 1, 0]) or np.array_equal(window, [1, 0, 1]):
                pingpong_count += 1
                self.pingpong_windows.append((i, i+window_size))  # 记录乒乓窗口的起始/结束索引
        return pingpong_count

    def get_handover_metrics(self, signal_data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        traditional_records = self.traditional_handover(signal_data)
        traditional_total = np.sum(np.diff(traditional_records) != 0)
        traditional_pingpong = self.detect_pingpong(traditional_records)  # 会生成pingpong_windows
        traditional_pingpong_rate = traditional_pingpong / traditional_total if traditional_total > 0 else 0

        improved_records = self.improved_handover(signal_data)
        improved_total = np.sum(np.diff(improved_records) != 0)
        improved_pingpong = self.detect_pingpong(improved_records)
        improved_pingpong_rate = improved_pingpong / improved_total if improved_total > 0 else 0

        pingpong_reduction = (
                                         traditional_pingpong - improved_pingpong) / traditional_pingpong if traditional_pingpong > 0 else 1.0
        total_reduction = (traditional_total - improved_total) / traditional_total if traditional_total > 0 else 1.0

        metrics = {
            "traditional": {
                "total_handovers": traditional_total,
                "pingpong_count": traditional_pingpong,
                "pingpong_rate": traditional_pingpong_rate,
                "success_rate": 1.0
            },
            "improved": {
                "total_handovers": improved_total,
                "pingpong_count": improved_pingpong,
                "pingpong_rate": improved_pingpong_rate,
                "success_rate": 1.0
            },
            "reduction_ratios": {
                "pingpong_reduction_ratio": pingpong_reduction,
                "total_handovers_reduction_ratio": total_reduction
            }
        }
        return metrics


class ResultAnalyzer:

    def __init__(self, signal_data: Dict[str, np.ndarray], handover_rule: HandoverRule,
                 metrics: Dict[str, Dict[str, float]]):
        self.signal_data = signal_data
        self.handover_rule = handover_rule
        self.metrics = metrics
        self.time_steps = len(signal_data["position"])
        self.time_axis = np.arange(self.time_steps)
        self.color_a = "#003366"
        self.color_b = "#990000"
        self.color_trad = "orange"
        self.color_impr = "green"

    def print_metrics(self):
        print("  传统切换规则（单一RSRP阈值，无滞回）")
        print(f"   总切换次数：{self.metrics['traditional']['total_handovers']} 次")
        print(f"   乒乓切换次数：{self.metrics['traditional']['pingpong_count']} 次")
        print(f"   乒乓切换率：{self.metrics['traditional']['pingpong_rate']:.2%}（乒乓次数/总切换次数）")
        print(f"   切换成功率：{self.metrics['traditional']['success_rate']:.2%}")

        print("\n  改进切换规则（多参数融合+滞回+持续时间）")
        print(f"   总切换次数：{self.metrics['improved']['total_handovers']} 次")
        print(f"   乒乓切换次数：{self.metrics['improved']['pingpong_count']} 次")
        print(f"   乒乓切换率：{self.metrics['improved']['pingpong_rate']:.2%}（乒乓次数/总切换次数）")
        print(f"   切换成功率：{self.metrics['improved']['success_rate']:.2%}")

        print("\n  优化效果")
        print(f"   乒乓切换减少比例：{self.metrics['reduction_ratios']['pingpong_reduction_ratio']:.2%}")
        print(f"   总切换次数减少比例：{self.metrics['reduction_ratios']['total_handovers_reduction_ratio']:.2%}")

    def visualize_results(self):
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        # ---------------------- 图1：小区A/B信号强度时序图 ----------------------
        fig1 = plt.figure(figsize=(12, 6))
        ax1 = fig1.add_subplot(111)
        ax1.plot(self.time_axis, self.signal_data["signal_strength_cell_a"],
                 label="小区A信号强度(dBm)", color=self.color_a, linewidth=1.5)
        ax1.plot(self.time_axis, self.signal_data["signal_strength_cell_b"],
                 label="小区B信号强度(dBm)", color=self.color_b, linewidth=1.5)
        ax1.set_title("小区A/B信号强度（RSRP）时序变化", fontsize=14, fontweight="bold")
        ax1.set_xlabel("时间步")
        ax1.set_ylabel("信号强度(dBm)")
        ax1.legend(loc="upper right", fontsize=10)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("图1-小区AB信号强度.svg", format="svg", bbox_inches="tight")
        plt.close(fig1)

        # ---------------------- 图2：小区A/B传输延迟时序图 ----------------------
        fig2 = plt.figure(figsize=(12, 6))
        ax2 = fig2.add_subplot(111)
        ax2.plot(self.time_axis, self.signal_data["transmission_delay_cell_a"],
                 label="小区A传输延迟(ms)", color=self.color_a, linewidth=1.5)
        ax2.plot(self.time_axis, self.signal_data["transmission_delay_cell_b"],
                 label="小区B传输延迟(ms)", color=self.color_b, linewidth=1.5)
        ax2.set_title("小区A/B传输延迟时序变化", fontsize=14, fontweight="bold")
        ax2.set_xlabel("时间步")
        ax2.set_ylabel("传输延迟(ms)")
        ax2.legend(loc="upper right", fontsize=10)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("图2-小区AB传输延迟.svg", format="svg", bbox_inches="tight")
        plt.close(fig2)

        # ---------------------- 图3：小区A/B信噪比时序图 ----------------------
        fig3 = plt.figure(figsize=(12, 6))
        ax3 = fig3.add_subplot(111)
        ax3.plot(self.time_axis, self.signal_data["signal_noise_ratio_cell_a"],
                 label="小区A信噪比(dB)", color=self.color_a, linewidth=1.5)
        ax3.plot(self.time_axis, self.signal_data["signal_noise_ratio_cell_b"],
                 label="小区B信噪比(dB)", color=self.color_b, linewidth=1.5)
        ax3.set_title("小区A/B信噪比（SINR）时序变化", fontsize=14, fontweight="bold")
        ax3.set_xlabel("时间步")
        ax3.set_ylabel("信噪比(dB)")
        ax3.legend(loc="upper right", fontsize=10)
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("图3-小区AB信噪比.svg", format="svg", bbox_inches="tight")
        plt.close(fig3)

        # ---------------------- 图4：小区A/B信号质量差异对比 ----------------------
        fig4 = plt.figure(figsize=(12, 8))
        # 子图1：信号强度差（B-A）
        ax4_1 = fig4.add_subplot(311)
        ss_diff = self.signal_data["signal_strength_cell_b"] - self.signal_data["signal_strength_cell_a"]
        ax4_1.plot(self.time_axis, ss_diff, color="#666666", linewidth=1.2)
        ax4_1.axhline(y=0, color="red", linestyle="--", linewidth=1)  # 阈值线（传统规则切换条件）
        ax4_1.set_title("小区B与A的信号强度差（B-A）", fontsize=12, fontweight="bold")
        ax4_1.set_ylabel("信号强度差(dBm)")
        ax4_1.grid(True, alpha=0.3)
        # 子图2：信噪比差（B-A）
        ax4_2 = fig4.add_subplot(312)
        sinr_diff = self.signal_data["signal_noise_ratio_cell_b"] - self.signal_data["signal_noise_ratio_cell_a"]
        ax4_2.plot(self.time_axis, sinr_diff, color="#666666", linewidth=1.2)
        ax4_2.axhline(y=0, color="red", linestyle="--", linewidth=1)
        ax4_2.set_title("小区B与A的信噪比差（B-A）", fontsize=12, fontweight="bold")
        ax4_2.set_ylabel("信噪比差(dB)")
        ax4_2.grid(True, alpha=0.3)
        # 子图3：延迟差（A-B）（延迟越低质量越高）
        ax4_3 = fig4.add_subplot(313)
        delay_diff = self.signal_data["transmission_delay_cell_a"] - self.signal_data["transmission_delay_cell_b"]
        ax4_3.plot(self.time_axis, delay_diff, color="#666666", linewidth=1.2)
        ax4_3.axhline(y=0, color="red", linestyle="--", linewidth=1)
        ax4_3.set_title("小区A与B的传输延迟差（A-B）", fontsize=12, fontweight="bold")
        ax4_3.set_xlabel("时间步")
        ax4_3.set_ylabel("延迟差(ms)")
        ax4_3.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("图4-小区AB信号质量差异.svg", format="svg", bbox_inches="tight")
        plt.close(fig4)

        # ---------------------- 图5：改进规则-小区A/B综合质量得分对比 ----------------------
        fig5 = plt.figure(figsize=(12, 6))
        ax5 = fig5.add_subplot(111)
        ax5.plot(self.time_axis, self.handover_rule.score_a,
                 label="小区A综合质量得分", color=self.color_a, linewidth=1.5)
        ax5.plot(self.time_axis, self.handover_rule.score_b,
                 label="小区B综合质量得分", color=self.color_b, linewidth=1.5)
        # 绘制滞回阈值线（改进规则切换条件）
        ax5.axhline(y=self.handover_rule.hysteresis_threshold_for_anti_pingpong,
                    color="purple", linestyle="--", linewidth=1.2, label="滞回阈值")
        ax5.set_title("改进规则-小区A/B综合质量得分对比", fontsize=14, fontweight="bold")
        ax5.set_xlabel("时间步")
        ax5.set_ylabel("综合质量得分（0~1）")
        ax5.legend(loc="upper right", fontsize=10)
        ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("图5-改进规则综合得分对比.svg", format="svg", bbox_inches="tight")
        plt.close(fig5)

        # ---------------------- 图6：传统规则-乒乓切换事件定位图 ----------------------
        fig6 = plt.figure(figsize=(12, 6))
        ax6 = fig6.add_subplot(111)
        ax6.step(self.time_axis, self.handover_rule.traditional_handover_records,
                 where="mid", color=self.color_trad, linewidth=2, label="当前服务小区")
        # 标记乒乓切换窗口
        for (start, end) in self.handover_rule.pingpong_windows:
            ax6.axvspan(start, end, color="red", alpha=0.2, label="乒乓切换窗口" if start == self.handover_rule.pingpong_windows[0][0] else "")
        ax6.set_title("传统规则-乒乓切换事件定位", fontsize=14, fontweight="bold")
        ax6.set_xlabel("时间步")
        ax6.set_ylabel("服务小区（0=A/1=B）")
        ax6.set_yticks([0, 1])
        ax6.set_yticklabels(["小区A", "小区B"])
        ax6.legend(loc="upper right", fontsize=10)
        ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("图6-传统规则乒乓事件定位.svg", format="svg", bbox_inches="tight")
        plt.close(fig6)

        # ---------------------- 图7：切换触发条件满足情况（传统vs改进）----------------------
        fig7 = plt.figure(figsize=(12, 6))
        ax7 = fig7.add_subplot(111)
        # 传统规则触发条件（B信号强度>A）
        trad_trigger = (self.signal_data["signal_strength_cell_b"] > self.signal_data["signal_strength_cell_a"]).astype(int)
        # 改进规则触发条件（B综合得分 - A综合得分 > 滞回阈值）
        impr_trigger = (self.handover_rule.score_b - self.handover_rule.score_a > self.handover_rule.hysteresis_threshold_for_anti_pingpong).astype(int)
        ax7.plot(self.time_axis, trad_trigger, color=self.color_trad, linewidth=1.2, label="传统规则触发条件（1=满足）")
        ax7.plot(self.time_axis, impr_trigger, color=self.color_impr, linewidth=1.2, linestyle="--", label="改进规则触发条件（1=满足）")
        ax7.set_title("切换触发条件满足情况", fontsize=14, fontweight="bold")
        ax7.set_xlabel("时间步")
        ax7.set_ylabel("触发条件状态（1=满足/0=不满足）")
        ax7.legend(loc="upper right", fontsize=10)
        ax7.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("图7-切换触发条件对比.svg", format="svg", bbox_inches="tight")
        plt.close(fig7)

        # ---------------------- 图8：传统规则切换记录图 ----------------------
        fig8 = plt.figure(figsize=(12, 4))
        ax8 = fig8.add_subplot(111)
        ax8.step(self.time_axis, self.handover_rule.traditional_handover_records,
                 where="mid", color=self.color_trad, linewidth=2, label="当前服务小区")
        ax8.set_title("传统切换规则切换记录", fontsize=14, fontweight="bold")
        ax8.set_xlabel("时间步")
        ax8.set_ylabel("服务小区")
        ax8.set_yticks([0, 1])
        ax8.set_yticklabels(["小区A", "小区B"])
        ax8.legend(loc="upper right", fontsize=10)
        ax8.grid(True, alpha=0.3)
        ax8.text(0.02, 0.95, f"乒乓次数：{self.metrics['traditional']['pingpong_count']}",
                 transform=ax8.transAxes, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5), fontsize=10)
        plt.tight_layout()
        plt.savefig("图8-传统规则切换记录.svg", format="svg", bbox_inches="tight")
        plt.close(fig8)

        # ---------------------- 图9：改进规则切换记录图 ----------------------
        fig9 = plt.figure(figsize=(12, 4))
        ax9 = fig9.add_subplot(111)
        ax9.step(self.time_axis, self.handover_rule.improved_handover_records,
                 where="mid", color=self.color_impr, linewidth=2, label="当前服务小区")
        ax9.set_title("改进切换规则切换记录", fontsize=14, fontweight="bold")
        ax9.set_xlabel("时间步")
        ax9.set_ylabel("服务小区")
        ax9.set_yticks([0, 1])
        ax9.set_yticklabels(["小区A", "小区B"])
        ax9.legend(loc="upper right", fontsize=10)
        ax9.grid(True, alpha=0.3)
        ax9.text(0.02, 0.95, f"乒乓次数：{self.metrics['improved']['pingpong_count']}",
                 transform=ax9.transAxes, bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5), fontsize=10)
        plt.tight_layout()
        plt.savefig("图9-改进规则切换记录.svg", format="svg", bbox_inches="tight")
        plt.close(fig9)

        # ---------------------- 图10：切换次数对比图 ----------------------
        fig10 = plt.figure(figsize=(10, 6))
        ax10 = fig10.add_subplot(111)
        categories = ["总切换次数", "乒乓切换次数"]
        traditional_vals = [self.metrics['traditional']['total_handovers'],
                            self.metrics['traditional']['pingpong_count']]
        improved_vals = [self.metrics['improved']['total_handovers'], self.metrics['improved']['pingpong_count']]
        x = np.arange(len(categories))
        width = 0.35
        bars1 = ax10.bar(x - width / 2, traditional_vals, width, label="传统规则", color=self.color_trad, alpha=0.8)
        bars2 = ax10.bar(x + width / 2, improved_vals, width, label="改进规则", color=self.color_impr, alpha=0.8)
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax10.text(bar.get_x() + bar.get_width() / 2., height + 0.2, f"{int(height)}",
                     ha="center", va="bottom", fontsize=10)
        ax10.set_title("切换次数对比", fontsize=14, fontweight="bold")
        ax10.set_xlabel("指标类型")
        ax10.set_ylabel("次数")
        ax10.set_xticks(x)
        ax10.set_xticklabels(categories)
        ax10.legend(loc="upper right", fontsize=10)
        ax10.grid(True, alpha=0.3, axis="y")
        ax10.text(0.7, 0.9,
                  f"乒乓减少比例：{self.metrics['reduction_ratios']['pingpong_reduction_ratio']:.2%}\n总切换减少比例：{self.metrics['reduction_ratios']['total_handovers_reduction_ratio']:.2%}",
                  transform=ax10.transAxes, bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5), fontsize=10)
        plt.tight_layout()
        plt.savefig("图10-切换次数对比.svg", format="svg", bbox_inches="tight")
        plt.close(fig10)



    def export_to_excel(self, filename: str = "handover_analysis_data.xlsx"):
        export_data = {
            "时间步": self.time_axis,
            "移动台位置(0=A,1=B)": self.signal_data["position"],
            "小区A信号强度(dBm)": self.signal_data["signal_strength_cell_a"],
            "小区B信号强度(dBm)": self.signal_data["signal_strength_cell_b"],
            "小区A信噪比(dB)": self.signal_data["signal_noise_ratio_cell_a"],
            "小区B信噪比(dB)": self.signal_data["signal_noise_ratio_cell_b"],
            "小区A传输延迟(ms)": self.signal_data["transmission_delay_cell_a"],
            "小区B传输延迟(ms)": self.signal_data["transmission_delay_cell_b"],
            "小区A综合质量得分": self.handover_rule.score_a,
            "小区B综合质量得分": self.handover_rule.score_b,
            "传统规则服务小区(0=A,1=B)": self.handover_rule.traditional_handover_records,
            "改进规则服务小区(0=A,1=B)": self.handover_rule.improved_handover_records
        }
        df = pd.DataFrame(export_data)
        df.to_excel(filename, index=False, engine="openpyxl")


if __name__ == "__main__":
    config = {
        "time_steps": 500,  # 总时间步数
        "position_noise_amplitude": 0.1,  # 位置噪声幅度
        "signal_noise_amplitude": 3.0,  # 信号参数噪声幅度
        "traditional_ss_threshold": -100,  # RSRP切换阈值
        "hysteresis_threshold": 0.15,  # 滞回阈值
        "continuous_time_steps": 5,  # 持续时间步数
        "weight_ss": 0.5,  # 信号强度权重
        "weight_sinr": 0.3,  # 信噪比权重
        "weight_delay": 0.2,  # 延迟权重
    }

    # 生成信号数据
    signal_gen = SignalDataGenerator(config)
    signal_data = signal_gen.generate_signal_parameters()

    # 执行切换规则并计算指标
    handover_rule = HandoverRule(config)
    metrics = handover_rule.get_handover_metrics(signal_data)

    # 结果分析
    result_analyzer = ResultAnalyzer(signal_data, handover_rule, metrics)
    result_analyzer.print_metrics()
    result_analyzer.visualize_results()
    result_analyzer.export_to_excel()