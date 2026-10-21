# Stage A/B 仿真说明

## Stage A：纯数学局部集合

`configs/local_set/stage_a_baseline.yml` 给出矩形足底离散网格、足底法向、切向基、总法向载荷、单元法向载荷上限和候选摩擦参数。正式基准为 `10×5=50` 个、边长 `20 mm` 的有限面积单元；四点配置 `stage_a_four_point_validation.yml` 仅用于回归校验。`DiscreteContactModel` 按

\[
\mathbf f_j=\lambda_j\mathbf n+\mathbf T\mathbf u_j,
\qquad
\|\mathbf u_j\|_2\leq\mu_j\lambda_j
\]

构造离散接触力，并用

\[
\mathbf w=\sum_j
\begin{bmatrix}[\mathbf r_j]_{\times}\\\mathbf I_3\end{bmatrix}\mathbf f_j
\]

映射到六维扳手 `[tau_x, tau_y, tau_z, f_x, f_y, f_z]`。`LocalWrenchSet` 提供：

- `contains(wrench)`：是否存在满足约束的逐点接触力见证；
- `support(direction)`：给定方向上的最大投影；
- `radial_capacity(working_wrench, direction)`：从工作点沿加载方向到边界的余量；
- `support_reduced(direction)`：解析消去切向力后的线性规划交叉验证。

固定 `normal_load_n` 只是一个条件截面。需要研究法向载荷变化时，应扫描多个配置得到集合族，不能把某一截面当成整机可实现集合。

### Stage A 输出如何阅读

运行 `experiments/stage_a_local_set/run_local_set.py` 和 `plot_local_set.py` 后，结果写入 `outputs/stage_a_baseline`：

- `baseline_projections.png` 的蓝色区域是局部接触扳手集合的二维投影，橙色点是均匀分配法向载荷得到的当前工作扳手；
- `contact_grid_layout.png` 给出 50 个有限面积单元的空间位置和固定编号；
- `support_queries.csv` 给出沿六个扳手坐标正、负方向的集合边界；
- `radial_queries.csv` 给出从当前工作扳手沿指定方向到边界的余量。

基准算例取总法向载荷 `400 N`，50 个单元均分为 `8 N/单元`。每单元上限为 `10 kgf = 98.1 N`，对应平均压力上限 `245.25 kPa`。固定 `400 N` 只是本次载荷截面，不表示法向力不能变化。

## Stage B：简化 MuJoCo 数字实验台

这里必须区分两种“真值”。MuJoCo 的 `mj_contactForce` 返回的是接触求解器对每个碰撞约束计算的约束反力/反力矩，表达在接触坐标系，随后被转换并汇总到足部质心；它不是物理压力阵列在某个真实传感器单元上测到的三维力。工程中的 `normal_forces` 只是从这些约束反力中取支撑法向分量，并按 50 个预定义 `contact_00`--`contact_49` 位置汇总形成的理想压力阵列代理。`contact_wrench_*` 和 `pressure_model_wrench_*` 应分别理解和报告。

`models/bench/single_foot_bench.xml` 是一个带自由基座的刚性足体。理论上将 `200 mm×100 mm` 足底划分为 `10×5=50` 个 `20 mm×20 mm` 有限面积单元；MuJoCo 验证模型在每个单元中心放置一个半径 `1 mm` 的点接触近似，使仿真作用位置与理论等效合力点一致。地面只用于生成仿真接触。`mujoco_io.bench` 读取 MuJoCo 的逐接触力，转换到世界坐标并汇总到足部质心；这是真值通道，不是压力阵列算法输入。

Stage B 先验证接触力提取、法向合力和接触扳手。下一步可在相同配置下加入受控外力/外力矩加载，并将每个时刻的法向点载荷送入 Stage A 成员检验。完整 X1、整机动力学和瞬时可行扳手集合放在后续阶段。

### Stage B 输出如何阅读

运行 `experiments/stage_b_contact_bench/run_contact_bench.py` 和 `plot_contact_bench.py` 后，结果写入 `outputs/stage_b_bench`。`bench_contact_wrench.png` 上图为合力，中图为合力矩，下图为末时刻 `10×5` 阵列的法向载荷热图。当前自由落足工况最终进入对称静止状态，总法向力约为 `39.24 N`，对应 `4 kg` 足体的重力。

当前基准从水平、全足接触状态初始化，因此主要用于检查稳态载荷分布和接触真值提取链路；尚未验证部分落足、压力阵列重建、整机动力学或动作可行扳手集合。

### Stage B 交互式查看器和对比图

`experiments/stage_b_contact_bench/run_contact_viewer.py` 打开 MuJoCo 被动查看器并实时绘制三组量：

- 红/橙：MuJoCo 逐接触力汇总得到的完整六维真值扳手；
- 蓝/青：只使用 50 个法向单元载荷、按理论力臂映射重建的法向扳手；
- 绿/紫：同一个局部扳手包络中的若干完整可行扳手见证，每个见证同时绘制一支合力箭头和一支合力矩箭头；绿、紫仅用于区分两类分量。

三组量统一使用足部刚体质心作为扳手参考点。接触力仍由足底接触平面上的 50 个单元提供，因此每个接触力都用“接触点到质心”的力臂计算力矩。一个实际六维扳手只显示一支合力箭头和一支合力矩箭头，不绘制六个标量分量。局部集合分别沿三维合力方向和三维合力矩方向进行支持优化，但每次都保留求解器返回的完整六维扳手；每个见证同时绘制其合力和合力矩箭头，从而形成一个从质心发散的锥状包络。这样避免把不同优化结果中的力和力矩任意拼接，也避免在同一六维欧氏范数中直接混合 N 与 N·m。由于固定了总法向载荷，该集合严格说是 CWC 的有界截面而不是无界数学锥；锥状箭头只是六维集合的三维可视化。同一编号的合力和合力矩必须作为一对理解，不能跨编号任意组合。按 `1`、`2`、`3` 分别切换三组显示，空格暂停，`R` 重置。

`plot_contact_comparison.py` 生成 `theory_simulation_comparison.png`、逐时刻 `wrench_comparison.csv` 和误差指标 JSON。图中比较 MuJoCo 真值与法向阵列重建值的合力大小、合力矩大小和末时刻三维方向，不再把六个坐标分量当作六支独立物理箭头。法向阵列重建不包含未知切向力，因此不得把蓝色结果解释为一般工况下的完整六维当前扳手。

### IMU 与足底阵列融合扳手估计

`stage_b_sensor_bench.yml`、`experiments/stage_b_sensor_fusion/run_sensor_fusion.py` 和 `plot_sensor_fusion.py` 实现了理想传感器基准。足部质心处施加小幅、已知的正弦外部扳手；其作用是让接触扳手随时间变化，而不是模拟踝关节。估计阶段明确扣除该已知激励，因此比较对象始终是地面对足的接触扳手。

MuJoCo 的足端陀螺仪读数为 \(\boldsymbol\omega_B\in\mathbb R^3\)，加速度计读数为比力 \(\mathbf f_B=\mathbf R_{WB}^T(\mathbf a_W-\mathbf g_W)\in\mathbb R^3\)。用足体质量 \(m\)、足体系主惯量 \(\mathbf I_B\in\mathbb R^{3\times3}\)、采样周期 \(\Delta t\) 和已知非接触外部扳手 \(\mathbf w_{\mathrm{ext}}\in\mathbb R^6\)，IMU 动力学估计为

\[
\hat{\mathbf F}_{\mathrm{imu},W}
=m\mathbf R_{WB}\mathbf f_B-\mathbf F_{\mathrm{ext},W},
\]

\[
\hat{\mathbf M}_{\mathrm{imu},W}
=\mathbf R_{WB}\left[
\mathbf I_B\frac{\boldsymbol\omega_B-\boldsymbol\omega^-_B}{\Delta t}
+\boldsymbol\omega_B\times(\mathbf I_B\boldsymbol\omega_B)
\right]-\mathbf M_{\mathrm{ext},W}.
\]

阵列第 \(j\) 点给出法向载荷 \(\lambda_j\ge0\)，以质心为原点、接触平面 \(z=z_c\) 表示时，阵列直接给出

\[
\hat F_z=\sum_j\lambda_j,\qquad
\hat M_x^{(n)}=\sum_j y_j\lambda_j,\qquad
\hat M_y^{(n)}=-\sum_j x_j\lambda_j.
\]

融合在足坐标系中进行。IMU 提供 \(F_x,F_y,M_z\)，阵列提供 \(F_z\) 和法向压力力矩；切向合力对质心力矩的公共力臂补偿为

\[
\hat M_x=\hat M_x^{(n)}-z_c\hat F_y^{\mathrm{imu}},
\qquad
\hat M_y=\hat M_y^{(n)}+z_c\hat F_x^{\mathrm{imu}}.
\]

这不是假设压力阵列可测切向力，而是利用 IMU 估计的切向合力补偿其已知接触面力臂。若接触点高度并不相同、足变形明显或接触面有未建模的局部力矩，上式不再严格成立，需要升级为逐点接触几何模型。

运行后在 `outputs/stage_b_sensor_bench` 中保存原始记录、分量 RMSE 与两张中文对比图。当前基准为理想无噪声传感器，用于验证观测方程和参考点一致性；不代表真实传感器精度，也不解决未知踝部扳手的可观测性问题。

`experiments/stage_b_sensor_fusion/run_sensor_viewer.py` 提供上述工况的实时 MuJoCo 观察。红/橙表示 MuJoCo 接触真值，橙紫表示 IMU 动力学估计，蓝/青表示仅法向阵列估计，紫色表示融合估计，黄/紫表示已知测试激励。所有扳手均关于足部质心且从质心出发；可以用 `1`--`5` 分别切换五类箭头。融合扳手在当前理想模型中与真值几乎重合，关闭真值或 IMU 显示后更容易观察其他结果。

## Stage B 验证：受控扳手扫描

`configs/bench/stage_b_wrench_sweep.yml` 和 `experiments/stage_b_boundary_validation/run_boundary_sweep.py` 用于验证局部接触扳手集合的边界。每个方向单独运行一次足体：先在重力下沉降到稳定状态，再把一个外部合力或合力矩从零缓慢增加到理论边界的 `1.30` 倍。外部加载施加在足部质心，接触扳手方向取相反方向，因此理论查询与仿真加载使用同一个扳手参考点。

理论边界由

\[
\alpha_{\mathrm{th}}
=
\max\{\alpha\ge0\mid
\mathbf w_0+\alpha\,\Delta\mathbf w
\in\mathscr W_{\mathrm{loc}}\}
\]

得到。仿真失效由持续超过阈值的事件判定：切向速度对应滑移，足体倾角对应滚转/俯仰失稳，绕竖直轴角速度对应偏航滑移。输出包括理论边界、仿真失效载荷、逐工况轨迹和相对误差图。

运行命令：

```powershell
conda run --no-capture-output -n foot-wrench python experiments/stage_b_boundary_validation/run_boundary_sweep.py
conda run --no-capture-output -n foot-wrench python experiments/stage_b_boundary_validation/plot_boundary_sweep.py
```

当前理想点接触基准的结果为：切向合力边界比值约 `1.036--1.046`，滚转/俯仰合力矩约 `1.042--1.048`，偏航合力矩约 `0.989`。这验证的是当前固定载荷、固定 50 个候选单元和 MuJoCo 接触定律下的理论模型，不等同于真实机器人或未知地形的最终验证。

### 动态观察边界逼近过程

`experiments/stage_b_boundary_validation/run_boundary_viewer.py` 将上述批量扫描改成可重复观察的连续过程。每个工况依次经历沉降、线性加载、越过理论边界和最大载荷保持，并自动切换到下一方向。程序不中断失效后的仿真，因此可以直接看到滑移、滚转、俯仰或偏航是怎样发生的。

```powershell
conda run --no-capture-output -n foot-wrench python experiments/stage_b_boundary_validation/run_boundary_viewer.py
```

查看器中红/橙为 MuJoCo 实际接触扳手，蓝/青为法向离散载荷重建扳手，绿色箭头束为固定法向载荷下的局部集合包络，黄/紫为当前施加在质心的外部扳手；当外部加载超过理论边界时变为品红色。左上角的 `Utilization` 是当前外部载荷与该方向理论余量之比，并不是一般六维扳手的统一范数。`Failure` 只有在对应运动指标持续超过阈值后才改变，避免把单步数值波动误判为失稳。

按 `1`--`4` 切换四组箭头，空格暂停/继续，`S` 单步，`R` 重置当前工况，`N/P` 切换下一个/上一个工况，`A` 开关自动轮播。也可以从指定工况开始，例如：

```powershell
conda run --no-capture-output -n foot-wrench python experiments/stage_b_boundary_validation/run_boundary_viewer.py --case moment_y_positive --realtime-factor 0.5
```
