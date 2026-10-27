# G1连续足板积分：模型、数据与离线分析

更新日期：2026-10-25。本页承接 [G1总入口](../README.md)，记录连续足板这条前期路线。本次核对已有代码和结果，没有重跑仿真。旧引用的 `../../../docs/foot_sensing_project_overview_cn.md` 不在当前project中；阅读本页及下列现存报告即可，不把缺失的旧总览当作前置材料。

## 模型与观测

`build_continuous_model.py`从本单元外部候选派生 `model/g1_12dof_continuous.xml` 和 `scenes/`。原足碰撞被禁用，每个踝部增加一块连续薄矩形足板。第三方原件不改；派生模型改变了足部碰撞几何，因此不能和原足分桶路线混称为同一物理模型。

8×4网格只用于按接触位置积分：

$$
\mathbf F_{ij}=\sum_{k\in A_{ij}}\mathbf f_k\approx\int_{A_{ij}}\mathbf t(\mathbf r)\,dA.
$$

`fx/fy/fz`是单元合力；`pressure_avg_pa`是正法向合力除以完整名义单元面积，即 `max(Fz, 0) / A_cell`，不是实际接触斑内的压力密度；`contact_count`为参与该单元的求解器接触样本数。碰撞面连续不代表求解器直接给出了连续牵引力场。

同时保存踝部扳手／CoP、理想IMU、足外合力和 `force_closure_error_n`。G1扳手字段顺序是力在前、力矩在后；IMU加速度字段意图表示比力，见总入口。力闭合误差比较“网格内力＋足外力”与同源宏观合力，约机器精度的误差不是独立实测证据。

## 结果目录按批次阅读

| 位置（相对本页） | 内容／口径 |
|---|---|
| `outputs/uniform_speed_integrated/` | rough、slope2、stairs的0.4 m/s、各2回合批次，原始行数分别662／800／800；信息报告按stride降采样后共453行。`integration_report.json`、`sensor_information_report.*`和早期边界／预警JSON属于这一批。 |
| [same_macro_different_local_report.md](same_macro_different_local_report.md) | 上述三地形数据的同地形跨回合配对，共1944对；对应JSON在 `outputs/uniform_speed_integrated/`。 |
| `outputs/three_claims/` | 早期离线分析输出，不能替代rough 20回合批次。 |
| `outputs/warning_validation/` | rough 20回合原始CSV和运行摘要；4332行，4回合跑满8 s、16回合fall。 |
| `outputs/three_claims_rough20/` | 20回合的统计、方向余量CSV与预警／配对图；入口为 [THREE_CLAIMS_REPORT.md](THREE_CLAIMS_REPORT.md)。 |
| [outputs/posture_adjustment_demo/POSITION_ADJUSTMENT_REPORT.md](outputs/posture_adjustment_demo/POSITION_ADJUSTMENT_REPORT.md) | 从20回合数据选两帧，比较冻结载荷下的姿态几何反事实；JSON和两张图同目录。`posture_adjustment_demo_run.txt`为历史执行日志。 |
| `outputs/project_overview_figures/` | G1场景截图和 `results_summary_cn.png`。汇总图拼合不同统计口径：多地形单元占用、20回合预警、单帧反事实；不是一场新的统一实验，也不是H2 Plus图。 |

**非零单元统计的分母不同：**多地形 `integration_report.json` 对每只脚的全部记录帧取平均（包括无接触帧），结果约1.47—2.18；`sensor_information_report.md` 对降采样后有接触足帧取平均，约2.98。两者不能直接作提升幅度比较。

## 脚本职责与调用

| 文件 | 职责 |
|---|---|
| `build_continuous_model.py` | 从外部候选生成连续足板模型和场景副本。 |
| `continuous_sensor_suite.py`、`run_continuous_sole_walk.py` | 提取上述观测，按原控制时序采集；不把新增输入反馈给策略。 |
| `analyze_integrated_sole.py` | 汇总合力闭合与占用。 |
| `../analyze_sensor_information.py` | 宏观量、阵列、IMU的信息比较。 |
| `analyze_same_macro_different_local.py` | 同地形跨回合宏观近邻配对。 |
| `analyze_boundary_and_warning.py` | 离线预警、方向性加载余量及配对差异。 |
| `analyze_posture_adjustment_examples.py` | 冻结快照下的足姿反事实。 |
| `generate_project_overview_figures.py` | 从既有模型／摘要生成G1示意图和汇总图，不产生新闭环结果。 |

以下在已具备MuJoCo、PyTorch、NumPy及绘图所需Matplotlib的环境中，从 **project根目录先进入本目录**；本次未运行，输出可能覆盖同名旧文件：

```bash
cd experiment/g1_sensorized_bypass/sensorized_experiment/continuous_sole
python run_continuous_sole_walk.py --scene rough --speed 0.4 --duration 8 --episodes 2 --output outputs/uniform_speed_integrated/rough_04_integrated.csv
python analyze_posture_adjustment_examples.py outputs/warning_validation/rough_04_20episodes.csv --output-dir outputs/posture_adjustment_demo --examples 2
```

## 反事实与实际动作的边界

姿态示例保持法向载荷和踝部扳手不变，比较 `hold`、俯仰／横滚正负3°候选，按活动单元支撑范围计算四向余量；IMU标记动态紧迫性。第一帧是宏观模型建议保持、活动区域模型低于阈值的例子，第二帧是宏观余量本已不足的例子。

这是单步几何计算，没有从同一MuJoCo状态分叉重放，没有执行足姿修正，也未求解调整后的实际接触分布或到达过程。未加载区域不必然不存在潜在支撑，因此活动单元范围不自动是真实物理能力边界。后续若验证动作收益，需要保存足够的完整状态、实际干预并独立评价；当前文件只支持相应模型假设下的线索。
