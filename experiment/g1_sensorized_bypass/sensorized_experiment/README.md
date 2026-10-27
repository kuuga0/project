# G1虚拟足底旁路：入口与数据批次

更新日期：2026-10-25。本单元是**前期G1信息结构实验**，不是当前H2 Plus主平台。它复用 `../external/candidates/humanoid-sim-eval-learning/` 的12自由度G1预训练策略与MuJoCo场景，被动记录新增观测，不将这些观测反馈给控制器。第三方身份见 [来源与记录边界](../来源与记录边界.md)。本次只核对保存内容，没有重新采集或复现分析。

## 两条路线先分开

| 路线 | 模型／观测 | 入口与结果 |
|---|---|---|
| 原足接触分桶 | 不改变候选模型的碰撞几何；按接触点位置把求解器三维力分到虚拟4×2或8×4网格。 | `run_sensorized_walk.py`、`sensor_suite.py`；本目录 `outputs/`。当前代码使用8×4，4×2结果来自早期版本。 |
| 连续足板积分 | 关闭原足碰撞，在每个踝部添加一块连续薄足板；8×4只是后处理分区，不是32个独立碰撞单元。 | [continuous_sole/README.md](continuous_sole/README.md)；其模型、场景与 `outputs/` 独立保存。 |

两条路线的物理模型及列结构有差别，不把它们当作同一轨迹的纯分辨率消融。网格更密也不代表MuJoCo提供了更多独立接触样本。

## 观测定义与分量顺序

每个控制周期记录双脚，代码目标记录率50 Hz：

1. **虚拟阵列**：以 `left_ankle_roll_link`／`right_ankle_roll_link` 为足部坐标系，每单元记录三维合力（N）。超出范围的接触另存 `outside_*`，不静默丢弃。它不是实体压力阵列或FOOTRIX标定读数。
2. **踝部扳手与CoP**：同一批接触的合力、力臂矩及接触力矩运输到踝部原点。日志顺序是 **力在前、力矩在后**，字段为 `ankle_fx/fy/fz/tx/ty/tz`；研究正文和单足工程默认力矩在前、力在后，跨单元使用须显式重排。CoP按局部法向载荷加权，须结合 `cop_valid` 阅读。
3. **理想IMU通道**：代码意图记录足部局部角速度与加速度计比力，而不是静止为零的线加速度。代码使用 $\mathbf f_B=\mathbf R_{WB}^{T}(\mathbf a_W-\mathbf g_W)$ 的约定，静止时向上轴约 $+9.81\,\mathrm{m/s^2}$。后端加速度语义和实际静止值仍需运行验证，不能由公式说明认定真实IMU标定完成。

阵列和宏观扳手来自同一接触求解结果。合力闭合只说明该积分／汇总链的一致性，不能充当独立传感器真值。

## 保存的数据与报告

| 本目录下的位置 | 批次／用途 |
|---|---|
| `outputs/uniform_speed_4x2/` | 早期原足4×2基线；rough、slope2、stairs，目标0.4 m/s，各2回合、最长8 s。 |
| `outputs/uniform_speed_8x4/` | 原足8×4对应批次；参数同上。与4×2 CSV列数不同，不混在一次分析中。 |
| `outputs/speed_examples_4x2/` | 4×2补充：rough 0.8 m/s、slope2 1.0 m/s、stairs 0.4 m/s。 |
| `continuous_sole/outputs/uniform_speed_integrated/` | 连续足板多地形批次；与原足模型分开，具体行数和分析口径见连续足板说明。 |
| `continuous_sole/outputs/warning_validation/` | rough 0.4 m/s、20回合，4332行；4回合跑满、16回合按原策略阈值判fall。 |

原足两组报告中，有接触足帧的平均非零单元数约2.66（4×2）和2.69（8×4），占用率约33.2%和8.4%。这是对保存报告的引用，不是新增采集。细分主要增加了接口维数，不能把空单元解释为新增受力信息。

## 运行与分析

以下命令从 **project根目录** 执行，使用已准备好的MuJoCo、PyTorch、NumPy环境；绘图脚本另需Matplotlib。原依赖声明位于外部候选的 `pyproject.toml`／`uv.lock`，不意味着本次验证了当前环境。原文中的 `biped_walking/` 是迁移前位置，现行命令不用该前缀。

```bash
python experiment/g1_sensorized_bypass/sensorized_experiment/run_sensorized_walk.py --scene rough --speed 0.8 --duration 8 --episodes 2 --output experiment/g1_sensorized_bypass/sensorized_experiment/outputs/rough_forward_sensorized.csv
```

场景可选 `flat`、`slope1`、`slope2`、`slope3`、`stairs`、`perlin`、`rough`。运行生成CSV及同名JSON摘要，记录判fall／跑满时长和行数；`timeout`只表示到达时间上限，不自动代表任务成功。选择输出位置时保留已有证据，避免覆盖同名结果。

分析8×4保存批次的调用形式：

```bash
python experiment/g1_sensorized_bypass/sensorized_experiment/analyze_sensor_information.py experiment/g1_sensorized_bypass/sensorized_experiment/outputs/uniform_speed_8x4/rough_04_sensorized.csv experiment/g1_sensorized_bypass/sensorized_experiment/outputs/uniform_speed_8x4/slope2_04_sensorized.csv experiment/g1_sensorized_bypass/sensorized_experiment/outputs/uniform_speed_8x4/stairs_04_sensorized.csv --stride 5 --json experiment/g1_sensorized_bypass/sensorized_experiment/outputs/uniform_speed_8x4/sensor_information_report.json --markdown experiment/g1_sensorized_bypass/sensorized_experiment/outputs/uniform_speed_8x4/sensor_information_report.md
```

该命令会重写指定报告，本次未执行。脚本中的旧示例路径和历史JSON中的旧绝对路径仍保留，本页给出现行调用位置。重新生成Markdown时，须继续保留以下证据限制。

## 分析能支持到哪里

`analyze_sensor_information.py`提供跨地形宏观近邻、宏观量到阵列的线性重建和按“地形＋回合”分组的最近质心分类。线性重建残差不能证明所有非线性方法都无法恢复，分类准确率也不证明足底反馈能改善控制。报告中的“未解释比例”是脚本对归一化RMSE平方截断到1后的派生值，不是互信息或严格方差解释率。

[THREE_CLAIMS_REPORT.md](continuous_sole/THREE_CLAIMS_REPORT.md) 保存同地形跨回合差异、离线预警及假设模型下的方向性加载余量；四个正常回合用于阈值和误报率估计，不能当独立测试。姿态示例冻结当前载荷和宏观扳手，只计算几何反事实；“达标”不是已执行、站稳或降低跌倒率。

文件职责：`sensor_suite.py`提取观测，`run_sensorized_walk.py`按原策略时序采集，`analyze_sensor_information.py`离线比较；`continuous_sole/`保存连续足板生成、采集和后处理；所有 `outputs/` 按模型与批次区分，不能与第三方候选的上游 `results/` 混用。
