# 阶段 B：不完整连续支撑下的准静态单足试验

## 试验目的

本试验验证第一阶段的局部接触扳手模型，而不是模拟落地碰撞。刚性足以零速度贴合支撑面开始，竖直外载先缓慢增加，再沿足底方向移动作用点，直至发生持续倾转。MuJoCo 中的支撑区域是连续共面几何；足底的 `10x5=50` 个小接触几何只用于离散记录法向载荷，二者不一一对应。

配置中的完整、后部、外侧和缺角场景只是验证条件，不是算法预定义的接触模式。环境知道真实支撑几何，但局部集合求解器不能读取它。

## 信息流与集合

算法只使用参考保持阶段的 50 点法向载荷：

1. 以 `pressure_active_threshold_n` 去除接触求解残量，得到当前承载点；
2. 对同一批阈值化压力求总法向载荷和当前扳手；
3. 使用单点上限 `98.1 N` 构造固定载荷的法向局部集合；
4. 冻结该集合，沿实际加载路径计算理论径向边界；
5. 将理论边界与最后稳定、持续超阈值区间的起点和确认点比较。

本阶段正式验证的是压力可观测投影 `[Mx, My, Fz]`。完整六维 MuJoCo 真值、IMU 动力学估计和融合估计仍被保存，但没有把未辨识摩擦系数用于扩张边界，因此这里不宣称完成了六维摩擦能力验证。

## 运行与结果

```powershell
$env:PYTHONPATH='src'
conda run --no-capture-output -n foot-wrench python experiments/stage_b_partial_contact/run_partial_contact.py
conda run --no-capture-output -n foot-wrench python experiments/stage_b_partial_contact/plot_partial_contact.py
conda run --no-capture-output -n foot-wrench python experiments/stage_b_partial_contact/run_partial_viewer.py
```

结果保存在 `outputs/stage_b_partial_contact`：逐时刻记录包含压力、IMU、完整扳手真值、倾角和加载进度；摘要给出理论边界、最后稳定与首次失效；三张图分别显示时序、参考压力分布和边界对比。

查看器中按 `1`--`5` 切换实际接触扳手、IMU/压力模型扳手、压力扳手、参考局部集合和外部加载；`Space` 暂停，`R` 重置，`N` 切换工况。所有扳手箭头均以足部质心为参考点。
