# 同宏观量、不同局部感知分析

> 说明：本报告对应早期 outputs/uniform_speed_integrated/ 数据；本次 rough 20 回合散点图及最新统计见 THREE_CLAIMS_REPORT.md 和 outputs/three_claims_rough20/same_macro_different_local.json。两批数据的回合数、配对规则参数和统计数字不同，不应混用。

输入数据为 `outputs/uniform_speed_integrated/` 下 rough、slope2、stairs 三种地形的带时间戳 CSV。分析在同一地形内配对不同回合的样本：对每个样本寻找踝部六维接触力、双脚 CoP 相近的最近样本，再比较足底 8x4 三轴单元向量和双脚踝部 IMU。由于配对来自不同回合，因此不是简单地比较相邻时间点。

完整数值见 [outputs/uniform_speed_integrated/same_macro_different_local.json](outputs/uniform_speed_integrated/same_macro_different_local.json)。本次仅核对保存报告与位置，没有重新计算配对。

## 结果

- 配对数：1944。
- 宏观状态标准化距离中位数：0.0506。
- 足底分布标准化距离中位数：0.0260；rough 地形为 0.2380。
- IMU 标准化距离中位数：0.0691。
- rough 地形中，22.6% 的配对足底分布距离大于 1，说明相近宏观扳手/CoP 下仍存在明显不同的受力空间分布。
- 最大示例来自 rough：宏观距离 0.691，而足底分布距离 5.963；两时刻分别为 episode 0 的 4.06 s 和 episode 1 的 1.18 s。

## 如何解释

宏观踝部扳手和 CoP 是足底力场的低维汇总。它们的相近只说明总力和一阶力矩相近，并不保证压力/切向力在足底上的位置、接触区域数量和左右脚局部状态相同。连续足底积分阵列保留了这些空间分布信息；踝部 IMU 则补充了相同宏观接触下的局部角速度与比力信息。

当前数据支持“足底分布 + IMU 含有宏观量没有显式保留的新信息”。rough 地形的证据最强；slope2 和 stairs 的跨回合运动较重复，配对差异较小。因此这不是已经完成的因果证明或泛化性能结论，后续还应增加随机地形、外力扰动和更多回合，并直接比较由传感器推断出的接触扳手边界。
