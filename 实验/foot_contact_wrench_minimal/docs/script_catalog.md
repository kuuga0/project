# 脚本与数据通道目录

本文档是工程的上手索引。脚本按实验目标归类，不按文件的创建时间归类。现有脚本不需要互相调用；通常先运行 `run_*.py` 生成记录，再运行对应的 `plot_*.py` 生成图表，`*_viewer.py` 只负责交互式观察。

## 先分清三类量

工程中有三种容易混淆的量：

1. **理论接触模型**：由 50 个离散候选单元、法向力上限和摩擦约束组成的优化模型。它产生理论上的 `W_loc`，不是传感器读数。
2. **MuJoCo 接触约束真值**：MuJoCo 接触求解器在每个仿真接触约束上得到的接触力/接触力矩。`mj_contactForce` 给出的是约束反力，表达在 MuJoCo 接触坐标系，再被转换并汇总到足部质心。它是数字仿真的接触真值，不是物理压力传感器的输出。
3. **压力阵列理想化代理**：代码从上述接触约束力中取支撑法向分量，按 `contact_00` 到 `contact_49` 的 50 个预定义位置累加，得到 `normal_forces_n`，再计算 `[Mx, My, Fz]`。这等价于“理想、无噪声、已知位置的法向压力单元”，不是 MuJoCo 原生提供的真实压力阵列。

因此，基础台架中的 `contact_wrench_*` 是接触约束真值，`pressure_model_wrench_*` 是由法向分量重建的压力代理；二者必须分别报告。

## 推荐运行顺序

```text
Stage A 数学集合
  -> Stage B 接触约束台架
  -> Stage B 边界扫描
  -> Stage B 部分连续支撑
  -> Stage B 理想 IMU/压力融合
```

所有命令从仓库根目录执行，并先设置 `PYTHONPATH=src`、激活 `foot-wrench` 环境。

## 脚本索引

### Stage A：数学局部能力集

| 文件 | 用途 | 主要输入 | 主要输出 |
|---|---|---|---|
| `experiments/stage_a_local_set/run_local_set.py` | 读取基准 YAML，建立 `DiscreteContactModel` 和 `LocalWrenchSet`，执行支撑/径向查询 | `configs/local_set/stage_a_baseline.yml` | `working_point.json`、`support_queries.csv`、`radial_queries.csv` |
| `experiments/stage_a_local_set/plot_local_set.py` | 读取 Stage A 查询结果并绘制集合投影、网格图 | `outputs/stage_a_baseline/*.csv/json` | `baseline_projections.png`、`contact_grid_layout.png` |

`stage_a_four_point_validation.yml` 只用于小规模回归校验；`stage_a_parameter_sweep.yml` 用于扫描摩擦参数。它们不是另外两套算法。

### Stage B：MuJoCo 接触约束台架

| 文件 | 用途 | 主要输入 | 主要输出 |
|---|---|---|---|
| `experiments/stage_b_contact_bench/run_contact_bench.py` | 让自由刚性足在重力下落到地面，记录 MuJoCo 接触约束和法向分量 | `models/bench/single_foot_bench.xml`、`stage_b_bench.yml` | `contact_truth.jsonl`、`summary.csv` |
| `experiments/stage_b_contact_bench/plot_contact_bench.py` | 绘制接触约束合力、合力矩、法向单元分布 | Stage B JSONL/CSV | `bench_contact_wrench.png` |
| `experiments/stage_b_contact_bench/plot_contact_comparison.py` | 比较完整 MuJoCo 接触约束扳手与法向压力代理扳手 | Stage B 记录 | `wrench_comparison.csv`、误差 JSON、对比图 |
| `experiments/stage_b_contact_bench/run_contact_viewer.py` | 交互式显示接触约束真值、法向代理和局部集合见证 | MuJoCo 模型、配置 | 不覆盖离线输出 |

`src/foot_contact_wrench/mujoco_io/bench.py` 是这一组脚本的底层适配器。它通过 `mj_contactForce` 读取约束反力，并按接触位置计算关于足部质心的合力矩；这一步不是压力阵列仿真。

### Stage B：理论边界扫描

| 文件 | 用途 | 主要输入 | 主要输出 |
|---|---|---|---|
| `experiments/stage_b_boundary_validation/run_boundary_sweep.py` | 在 `+/-Fx,+/-Fy,+/-Mx,+/-My,+/-Mz` 十个方向逐渐施加质心外部扳手，比较理论边界与仿真失效 | `stage_b_wrench_sweep.yml` | 十个 JSONL、`validation_summary.csv/json` |
| `experiments/stage_b_boundary_validation/plot_boundary_sweep.py` | 绘制十个方向的边界和失效轨迹 | validation JSONL/summary | `boundary_validation.png`、`sweep_traces.png` |
| `experiments/stage_b_boundary_validation/run_boundary_viewer.py` | 动态观察沉降、加载、越过边界和失效 | 同上 | 仅交互显示 |

这组实验验证的是“当前离散接触模型与 MuJoCo 约束定律的一致性”，不是硬件摩擦辨识。

### Stage B：部分连续支撑

| 文件 | 用途 | 主要输入 | 主要输出 |
|---|---|---|---|
| `experiments/stage_b_partial_contact/run_partial_contact.py` | 生成完整、后部、外侧、缺角四种连续支撑面，做准静态加载和倾转检测 | `stage_b_partial_contact.yml` | 逐时刻记录、四场景摘要、生成 XML |
| `experiments/stage_b_partial_contact/plot_partial_contact.py` | 绘制压力图、时序和理论边界对比 | partial-contact JSONL/summary | 三类分析图 |
| `experiments/stage_b_partial_contact/plot_display_summary.py` | 生成适合汇报的上下布局总结图 | partial-contact 输出 | `display_partial_contact_summary.png` |
| `experiments/stage_b_partial_contact/render_contact_scenes.py` | 渲染四个 MuJoCo 场景总览 | 生成的 XML | `display_mujoco_contact_scenes.png` |
| `experiments/stage_b_partial_contact/run_partial_viewer.py` | 交互式切换四种支撑条件和扳手显示 | 生成的 XML、记录 | 仅交互显示 |

底层 `src/foot_contact_wrench/mujoco_io/partial_contact.py` 刻意不把环境支撑几何传给集合求解器；求解器只使用阈值化后的法向压力代理。因此“缺角”来自仿真环境，而不是算法预先知道的模式标签。

### Stage B：理想传感器融合

| 文件 | 用途 | 主要输入 | 主要输出 |
|---|---|---|---|
| `experiments/stage_b_sensor_fusion/run_sensor_fusion.py` | 施加已知正弦外部激励，比较理想 IMU、法向压力代理和融合扳手 | `stage_b_sensor_bench.yml` | 记录、RMSE CSV/JSON |
| `experiments/stage_b_sensor_fusion/plot_sensor_fusion.py` | 绘制三种估计值与 MuJoCo 接触约束真值 | sensor JSONL/metrics | 对比图、误差图 |
| `experiments/stage_b_sensor_fusion/run_sensor_viewer.py` | 实时显示真值、IMU、压力代理、融合和外部激励 | 同上 | 仅交互显示 |

底层 `src/foot_contact_wrench/estimation/sensor_wrench.py` 实现比力、角速度差分、压力法向力矩和融合公式。这里的 IMU 和压力都是理想仿真输入，不能直接外推到真实传感器精度。

## 底层模块与脚本的对应关系

```text
geometry/grid.py + contact/discrete.py
        -> optimization/local_set.py
        -> Stage A / 边界查询

mujoco_io/bench.py
        -> 基础台架 / 边界扫描 / 传感器实验

mujoco_io/partial_contact.py
        -> 部分支撑实验

estimation/sensor_wrench.py + mujoco_io/sensor_bench.py
        -> 理想传感器融合实验

analysis/comparison.py + visualization/wrench_scene.py
        -> 比较指标和所有查看器
```

## 哪些文件不应当当作实验入口

- `src/**` 是被实验脚本调用的库代码，不应直接运行。
- `configs/**` 是参数，不是结果；改参数后应重新运行对应的 `run_*.py`。
- `outputs/**` 是生成结果，不是输入真值数据库；旧输出可能对应旧代码或旧配置。
- `models/**` 是 MuJoCo XML 模型。`models/x1_modified` 当前为空，X1 仍未接入。
- `external/agibot_x1_infer` 是只读外部 Git 子模块，不属于本课题的接触算法。
- `__pycache__`、`.pytest_cache`、`MUJOCO_LOG.TXT` 是缓存或日志。

## 当前真实性等级

| 等级 | 对应内容 | 能说明什么 |
|---|---|---|
| L1 理论 | Stage A、`LocalWrenchSet` | 数学约束下哪些扳手可行 |
| L2 数字接触 | MuJoCo `contact_wrench_*` | 当前 MuJoCo 接触约束产生的合力和合力矩 |
| L3 理想观测 | `normal_forces_n`、`pressure_model_wrench_*`、理想 IMU | 在无噪声和已知几何条件下，观测方程是否自洽 |
| L4 真实实验 | 当前不存在 | 需要真实压力/触觉阵列、标定和独立力/力矩真值 |

项目目前完成到 L1-L3，没有 L4。
