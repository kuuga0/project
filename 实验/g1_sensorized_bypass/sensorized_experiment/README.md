# 旁路传感采集实验

这里是当前双足项目的独立传感实验层。它调用
`../external/candidates/humanoid-sim-eval-learning/` 的 G1 预训练策略和
MuJoCo 场景，但不修改外部仓库的 XML、策略或结果文件，也不把传感器
输入反馈给控制器。控制器仍然只是原项目的速度命令行走策略；新增内容
全部是被动采集。

## 传感器定义

每个控制周期（50 Hz）记录双脚：

1. **足底分布阵列**：以 `left_ankle_roll_link` /
   `right_ankle_roll_link` 为足部坐标系，将足底局部区域划成当前实验的
   8×4 个单元（每只脚 32 个单元）。
   每个单元记录 (f_x,f_y,f_z) 三维合力（N）。接触点按局部位置落入
   单元；超出阵列范围的接触另记在 `outside_*`，不静默丢弃。
2. **踝部宏观量**：把同一批足-地接触力和接触点力矩运输到踝部坐标系
   原点，记录六维扳手 (F_x,F_y,F_z,M_x,M_y,M_z)，并按竖直支撑力
   加权计算足部 CoP。
3. **踝部 IMU**：记录踝部坐标系的角速度和去重力线加速度。它是理想
   MuJoCo IMU 通道，静止时向上轴约为 (+9.81\,m/s^2)。

阵列是**虚拟旁路传感器**：它用 MuJoCo 解算的接触点三维力来模拟有限
感知区域的单元输出，不把几何单元直接当作真实压力传感器，也不改变动力学。
这是为了先验证信息结构；后续做硬件等价研究时仍需替换成有足底实体和
有限感知区域的模型。

## 运行采集

在仓库根目录执行：

```text
uv run --no-project --with mujoco --with torch --with numpy python ^
  biped_walking\sensorized_experiment\run_sensorized_walk.py ^
  --scene rough --speed 0.8 --duration 8 --episodes 2 ^
  --output biped_walking\sensorized_experiment\outputs\rough_forward_sensorized.csv
```

可选场景：`flat`、`slope1`、`slope2`、`slope3`、`stairs`、`perlin`、`rough`。
每次运行同时生成同名 JSON 摘要，包含摔倒/跑满时间和实际记录行数。

## 当前已采集数据

`outputs/uniform_speed_4x2/` 是此前 4×2 分辨率的主证据组，保留作为低分辨率
基线。新的 `outputs/uniform_speed_8x4/` 是当前 8×4 分辨率主证据组：粗糙
地形、中坡、楼梯，均以 0.4 m/s 前进、每个场景 2 条 8 s 轨迹。两组 CSV
不能放在同一次分析命令中，因为列结构不同。

`outputs/speed_examples_4x2/` 是此前 4×2 分辨率的补充组：粗糙 0.8 m/s、中坡 1.0 m/s、楼梯
0.4 m/s，用来检查较高速度下的传感通道和摔倒前变化。

## 信息对比

```text
uv run --no-project --with numpy python ^
  biped_walking\sensorized_experiment\analyze_sensor_information.py ^
  biped_walking\sensorized_experiment\outputs\uniform_speed_8x4\rough_04_sensorized.csv ^
  biped_walking\sensorized_experiment\outputs\uniform_speed_8x4\slope2_04_sensorized.csv ^
  biped_walking\sensorized_experiment\outputs\uniform_speed_8x4\stairs_04_sensorized.csv ^
  --stride 5 ^
  --json biped_walking\sensorized_experiment\outputs\uniform_speed_8x4\sensor_information_report.json ^
  --markdown biped_walking\sensorized_experiment\outputs\uniform_speed_8x4\sensor_information_report.md
```

8×4 数据重新生成报告后，会同时给出单元数量、有效单元占用率以及与此前
4×2 结果的分辨率对比。实际结果显示：4×2 时每只脚平均约 2.66 个单元非零，
8×4 时约 2.69 个非零；也就是说，细分增加了输出维数，但没有增加 MuJoCo
求解器实际提供的独立接触点数量。8×4 的每只脚平均占用率只有约 8.4%，
4×2 约 33.2%。因此 8×4 适合作为更高分辨率的记录接口，但不能把空单元
误解为新增真实受力信息。报告中的分类器只是“存在新信息”的可复现实验证据，
不是最终地形识别器。

## 文件用途

- `sensor_suite.py`：接触筛选、阵列分桶、踝部扳手/CoP、IMU 计算。
- `run_sensorized_walk.py`：加载原策略，按原控制时序行走并逐周期写 CSV。
- `analyze_sensor_information.py`：宏观近邻、宏观到阵列重建误差、简单
  地形分类三种对比指标。
- `outputs/`：本实验生成的 CSV、JSON 和 Markdown 报告；不应与外部候选
  项目的 `results/` 混用。

## 连续足底与三项证据

`continuous_sole/` 是当前推荐的后处理层：一块连续碰撞足底板对应一只脚，
再把接触样本积分到 8×4 虚拟单元，而不是把每个单元建成独立碰撞几何。该层
包含单元三维合力、名义单元面积平均压力、接触样本数、踝部扳手/CoP 和 IMU。

`continuous_sole/THREE_CLAIMS_REPORT.md` 汇总了不改控制器情况下的三项结果：
宏观量无法唯一确定局部/动态状态、足底+IMU 的跌倒前离线预警、以及由活动
受力区域更新的单足方向性加载余量。`continuous_sole/outputs/` 与旧的
`outputs/` 分开保存，不能混用不同列结构的数据。
# Continuous sole integration (current layer)

`continuous_sole/` is the physically cleaner follow-up to the earlier virtual
4x2/8x4 contact-bin experiments. It adds one continuous collision plate per
foot and integrates all solver contact samples into virtual grid cells. Each
cell stores integrated `fx/fy/fz`, nominal-area average normal pressure, and
contact-sample count; the grid cells are not separate collision geoms. See
`continuous_sole/README.md` for the exact definition and caveats.

The completed comparison data are in
`continuous_sole/outputs/uniform_speed_integrated/` for rough, slope2, and
stairs terrain. The same sensor-information analyzer can compare macro ankle
wrench/CoP against the integrated plantar array and ankle IMU.
