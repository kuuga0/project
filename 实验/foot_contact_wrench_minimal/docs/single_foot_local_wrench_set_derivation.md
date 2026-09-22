# 单足局部接触扳手集合：理论、仿真与实验台模型

## 1. 研究对象与第一阶段边界

第一阶段研究当前单足接触条件下的三个不同对象：

1. 当前动力学扳手集合 \(\mathscr W_{\mathrm{dyn},t}\)：由机器人已经发生的运动和整体动力学得到，回答“当前地面正在对机器人施加什么合扳手”；
2. 当前观测相容集合 \(\mathscr W_{\mathrm{cur},t}\)：同时满足整体动力学、足底法向压力和接触定律，回答“当前扳手及其局部分配有哪些可能”；
3. 局部接触能力集合 \(\mathscr W_{\mathrm{loc},t}\)：允许接触力在当前承载点之间重新分配，回答“当前足地界面还能稳定传递哪些扳手”。

三者不能混用。第一阶段的主线是

\[
\boxed{
\text{整体单足支撑动力学}
\longrightarrow
\mathscr W_{\mathrm{dyn},t}
\longrightarrow
\mathscr W_{\mathrm{cur},t}
\longrightarrow
\Theta_t
\longrightarrow
\{\mathscr W_{\mathrm{loc},t}(\boldsymbol\theta;N)\}_{\boldsymbol\theta\in\Theta_t}.
}
\]

其中 \(\Theta_t\) 是与当前数据相容的未知接触参数集合，\(N\) 是总法向载荷条件。第一阶段不遍历机器人尚未执行的关节力矩和加速度，因此不等同于后续的瞬时可行扳手对集合 \(\mathcal Q_{\mathrm{inst}}\) 或其六维合扳手映射 \(\mathcal W_{\Sigma,\mathrm{inst}}\)。

### 1.1 用一个时刻直观理解整条链路

设机器人处于左脚单足支撑。此时各部分分别回答：

1. 完整机器人动力学根据已经测得的运动，算出此刻地面合扳手，例如 \(\bar{\mathbf w}_{c,t}\)；
2. 压力阵列说明哪些足底位置承受法向力，以及各处法向力是多少；
3. 合扳手与法向压力结合后，可以检查某组候选接触参数是否可能解释当前状态；
4. 不能解释当前状态的参数被排除，剩余参数组成 \(\Theta_t\)；
5. 对每组剩余参数，不再固定当前切向力和法向力分配，而是在物理约束内重新分配，计算足地界面能力 \(\mathscr W_{\mathrm{loc},t}\)；
6. 沿给定加载方向查询距离滑移、翻转或分离边界还有多远。

这里有两个容易混淆的“变化”：

- 重建当前扳手时，使用的是已经发生的运动，不能任意改变加速度；
- 计算局部能力时，为了寻找接触边界，可以在接触约束内改变各点候选力，但不能改变接触点位置、公共法向或机器人动作。

### 1.2 阅读顺序

只掌握基础动力学时，建议先阅读第 1、3、5、6、7、8、11 和 12 节。第 9 节的支持函数与优化方法用于实现算法，第 10 节用于说明本模型与 CWC 的关系，可在理解物理主线后再读。

### 1.3 常用集合与优化符号

| 符号 | 读法 | 本文中的含义 |
| --- | --- | --- |
| \(\mathbf w\in\mathscr W\) | \(\mathbf w\) 属于 \(\mathscr W\) | 一个具体扳手满足集合的全部约束 |
| \(\mathscr A\subseteq\mathscr B\) | \(\mathscr A\) 包含于 \(\mathscr B\) | \(\mathscr A\) 中每个元素也在 \(\mathscr B\) 中 |
| \(\mathscr A\cap\mathscr B\) | 交集 | 同时满足两组条件的元素 |
| \(\mathscr A\cup\mathscr B\) | 并集 | 满足至少一组条件的元素 |
| \(\{x\mid\text{条件}\}\) | 满足条件的 \(x\) 的集合 | 用约束定义集合 |
| \(\exists x\) | 存在某个 \(x\) | 只需找到至少一个可行解 |
| \(\max_x f(x)\) | 对 \(x\) 求最大值 | 在全部可行 \(x\) 中寻找最大能力 |
| \(\|\mathbf x\|_2\) | 二范数 | 向量长度，如二维切向力大小 |

粗体小写量如 \(\mathbf w\) 表示一个向量；花体大写量如 \(\mathscr W\) 表示许多可能向量组成的集合；\(\boldsymbol\theta\) 表示一组具体参数，\(\Theta\) 表示许多候选参数组成的集合。

## 2. 基础模型与明确限制

1. 足部为刚性平板，全部候选受力点位于同一足底平面；
2. 当前承载点共享同一单位法向，不预先判定地形类别；
3. 足地作用由 \(M\) 个有限面积单元的等效受力点表示，点位可以不均匀；
4. 当前承载点由正法向载荷确定，不限定为有限种接触方式；
5. 接触力满足单边接触和选定的接触定律，但其参数不默认已知；
6. 当前未承载点不进入本时刻局部集合；建立新接触属于后续接触调整动作；
7. 理论基线不考虑传感器误差、保护层传力和连续压力重建；
8. 历史和学习不是集合定义的必要条件，只用于参数候选估计、求解加速或生成待验证候选；未经物理约束复核的学习输出不能扩大可行集合。

统一法向是当前刚性平板基础模型的适用条件，不代表识别了具体地形。当前支撑点可以不规则、间断或集中在边缘；一旦足开始绕边转动，应将其视为当前稳定集合的失效边界或新的接触状态。足底顺应性、多法向和非共面接触留作扩展。

## 3. 坐标系、扳手与离散点

在足部刚体上建立右手坐标系 \(\{F\}\)。本阶段将原点固定在足部质心 \(G\)，\(z_F\) 轴沿地面对足的统一压缩方向：

\[
\mathbf n^F=\mathbf e_z=
\begin{bmatrix}0&0&1\end{bmatrix}^T
\in\mathbb R^3.
\]

足姿态 \(\mathbf R_{WF,t}\in SO(3)\) 给出世界系法向

\[
\mathbf n_t^W=\mathbf R_{WF,t}\mathbf n^F.
\]

扳手采用“力矩在前、力在后”的顺序：

\[
\mathbf w=
\begin{bmatrix}
\boldsymbol\tau\\
\mathbf f
\end{bmatrix}
\in\mathbb R^6.
\]

全文的接触扳手均指“地面对足或机器人施加的扳手”，法向 \(\mathbf n\) 指向地面外侧并朝向足部，因此承压时 \(\lambda_j\ge0\)。所有扳手在求交、比较或代入优化问题前必须转换到同一参考点、同一坐标系和同一排列顺序。除特别说明外，本文的足部局部扳手均相对于质心 \(G\) 表达。

第 \(j\) 个离散点代表有限小区域 \(\Omega_j\)，其面积、位置和平均法向压力分别为

\[
A_j>0,
\qquad
\mathbf r_j^G\in\mathbb R^3,
\qquad
p_j\in\mathbb R_{\ge0},
\]

其中 \(\mathbf r_j^G\) 从质心 \(G\) 指向该点，并满足刚性平板共面条件

\[
\mathbf n^T\mathbf r_j^G=-h,
\qquad h>0.
\]

其中 \(h\) 是质心到接触作用平面的距离。当前仿真足体的质心在足体坐标原点，接触平面位于 \(z_F=-0.026\,\mathrm m\)，所以离散单元的三维位置必须包含这一高度；不能把所有点错误地放在质心平面上。

该区域的法向合力为

\[
\lambda_j=A_jp_j
\in\mathbb R_{\ge0}.
\]

直观上，\(p_j\) 描述单位面积上的压力，\(A_jp_j\) 才是这个小区域实际承担的力。例如面积为 \(2\times10^{-4}\,\mathrm{m^2}\)、平均压力为 \(10^5\,\mathrm{Pa}\) 时，该单元法向力为 \(20\,\mathrm N\)。把足底离散化不是认为真实接触只发生在数学点上，而是用每个小区域的等效合力近似连续压力分布。

离散模型的优化变量是单位为 \(\mathrm N\) 的 \(\lambda_j\)，不是零面积点上的压力。若以后引入局部承压上限 \(p_j^{\max}\)，必须令

\[
\lambda_j^{\max}=A_jp_j^{\max},
\]

并在离散加密时同步缩放面积和单元载荷上限。

定义公共切平面正交基

\[
\mathbf T\in\mathbb R^{3\times2},
\qquad
\mathbf T^T\mathbf T=\mathbf I_2,
\qquad
\mathbf T^T\mathbf n=\mathbf 0.
\]

第 \(j\) 点接触力与其关于质心 \(G\) 的扳手为

\[
\mathbf f_j
=\lambda_j\mathbf n+\mathbf T\mathbf u_j
\in\mathbb R^3,
\qquad
\mathbf u_j\in\mathbb R^2,
\]

\[
\mathbf w_j^G=\mathbf G_j^G\mathbf f_j,
\qquad
\mathbf G_j^G=
\begin{bmatrix}
[\mathbf r_j^G]_\times\\
\mathbf I_3
\end{bmatrix}
\in\mathbb R^{6\times3}.
\]

其中 \(\lambda_j\mathbf n\) 是垂直足底的法向力，\(\mathbf T\mathbf u_j\) 是平行足底的二维切向力。矩阵 \([\mathbf r_j^G]_\times\) 满足

\[
[\mathbf r_j^G]_\times\mathbf f_j
=\mathbf r_j^G\times\mathbf f_j,
\]

所以 \(\mathbf G_j^G\) 只是把基础力学关系

\[
\text{力矩}=\text{力臂}\times\text{力}
\]

与接触力合写成一个六维向量。所有离散点的 \(\mathbf w_j^G\) 相加，就是地面对整只脚、且相对于质心的合扳手。

离散点只施加合力，不额外施加独立点扭矩；分布式切向力通过力臂自然形成总偏航力矩。

### 3.1 符号角色速查

| 类型 | 典型符号 | 含义 |
| --- | --- | --- |
| 当前观测量 | \(\bar\lambda_{j,t},\bar N_t\) | 横线表示当前实际测得或重建的工作点 |
| 优化变量 | \(\lambda_j,\mathbf u_j,\alpha\) | 为寻找可行分配或能力边界而变化的候选量 |
| 具体参数 | \(\boldsymbol\theta\) | 一组候选摩擦、承载上限等参数 |
| 参数集合 | \(\Theta_0,\Theta_t\) | 尚未被数据排除的多组候选参数 |
| 扳手集合 | \(\mathscr W\) | 满足指定物理和观测约束的多个六维扳手 |
| 集合族 | \(\mathfrak W\) | 参数不同而得到的多个扳手集合 |

下标 \(t\) 表示当前时刻，下标 \(j\) 表示第 \(j\) 个离散接触单元，上标 \(F,W,G,O\) 分别用于说明坐标系或参考点。公式中省略上标时，默认所有量已经统一到足坐标系 \(\{F\}\) 和足部质心 \(G\)。

## 4. 已知量、未知量和验证真值

### 4.1 在线算法可使用的量

- 机器人模型参数及当前 \(q_t,\mathbf v_t,\dot{\mathbf v}_t\) 或其可接受范围；
- 除目标足地接触外的其他已知外部作用；
- 足端姿态和 IMU 运动量；
- 当前离散法向载荷 \(\bar{\boldsymbol\lambda}_t\in\mathbb R_{\ge0}^M\)；
- 离散点面积、位置和公共法向。

### 4.2 不默认已知的量

- 当前完整六维接触扳手；
- 各点切向力分配；
- 摩擦参数、局部承载上限及其他接触参数；
- 动作改变接触后的未来能力。

将未知接触参数统一记为

\[
\boldsymbol\theta
\in\Theta_0.
\]

例如 \(\boldsymbol\theta\) 可以包含 \(\mu_j\)、\(p_j^{\max}\) 或其他接触定律参数。\(\Theta_0\) 在理论中可保留为符号域；若没有数值范围或边界数据，最终只能得到参数化集合族，不能凭空得到唯一数值能力集合。

这里“参数集合”不是说接触参数会在同一时刻任意变化，而是表示研究者目前还不知道哪组参数是真实的。例如只知道公共摩擦系数可能位于 \([0.4,0.8]\)，则

\[
\Theta_0=\{\mu\mid0.4\le\mu\le0.8\}.
\]

真实系统在一次试验中只有一个实际 \(\mu\)，只是算法尚未确定它。

### 4.3 仅用于仿真或实验验证的特权真值

- MuJoCo 的逐点三维接触力和合扳手；
- 仿真环境内部设置的真实接触参数；
- 单足实验台底部六维力传感器输出；
- 相机或仿真接触状态给出的滑移、边缘转动和分离标签。

除离线参数辨识外，这些真值不作为拟部署算法的在线输入。

“特权真值”是指仿真器或实验测量系统能够提供、但未来机器人在线运行时未必直接拥有的信息。若在开发算法时直接读取 MuJoCo 的真实摩擦系数或逐点三维力，算法就绕过了原本需要解决的估计问题。因此这些量只用于回答“算法算得是否正确”。

## 5. 由整体动力学得到当前接触扳手

### 5.1 为什么扩大系统边界

若只截取足部刚体，动力学包含未知踝部扳手和地面接触扳手，两者无法仅由足端 IMU 分离。若将系统边界扩大到整个单足机器人，或完整机器人处于单足支撑且没有其他未知外部接触，踝部和所有关节作用都成为内力并从整体动量方程中消失。

这与把多个刚体的牛顿--欧拉方程相加完全相同。相邻两个刚体通过关节相互作用：小腿对足的力与足对小腿的力大小相等、方向相反；把两者都包含在系统边界内后，这对内力相消。所有关节作用都如此相消，方程最终只剩重力、足地接触力以及其他真正穿过系统边界的外力。

因此最终机器人仿真优先采用完整 X1 的单足支撑状态。浮动基单腿机器人可以作为简化调试模型，但必须包含完整浮动刚体和腿部系统，且不能通过外部固定装置引入未建模的支撑力。

### 5.2 质心动量平衡

令机器人质心为 \(G\)，质心动量为

\[
\mathbf h_G=
\begin{bmatrix}
\mathbf k_G\\
\mathbf l_G
\end{bmatrix}
\in\mathbb R^6,
\]

其中 \(\mathbf k_G,\mathbf l_G\in\mathbb R^3\) 分别是质心角动量和线动量。关于 \(G\) 的重力扳手为

\[
\mathbf w_g^G=
\begin{bmatrix}
\mathbf 0_3\\
m\mathbf g
\end{bmatrix}
\in\mathbb R^6.
\]

若完整浮动基模型的广义速度为 \(\mathbf v\in\mathbb R^{n_v}\)，质心动量矩阵为 \(\mathbf A_G(q)\in\mathbb R^{6\times n_v}\)，则

\[
\mathbf h_G=\mathbf A_G(q)\mathbf v,
\qquad
\dot{\mathbf h}_G
=\mathbf A_G(q)\dot{\mathbf v}
+\dot{\mathbf A}_G(q,\mathbf v)\mathbf v.
\]

该式说明当前扳手重建需要完整机器人的当前运动和惯性模型，而不是只需要足端 IMU。

整体动量平衡为

\[
\dot{\mathbf h}_{G,t}
=\mathbf w_{c,t}^G
+\mathbf w_g^G
+\mathbf w_{\mathrm{other},t}^G,
\]

故单足接触合扳手为

\[
\boxed{
\mathbf w_{c,t}^G
=\dot{\mathbf h}_{G,t}
-\mathbf w_g^G
-\mathbf w_{\mathrm{other},t}^G.
}
\]

这就是全机器人版本的牛顿--欧拉方程：

- 下三维 \(\dot{\mathbf l}_G=m\mathbf a_G\) 对应合外力等于线动量变化率；
- 上三维 \(\dot{\mathbf k}_G\) 对应关于质心的合外力矩等于角动量变化率。

在只有一只脚接触地面时，地面合扳手是唯一未知外部扳手，因此可以直接移项求出。若两只脚同时接触，整体动力学只能求出左右脚扳手之和，不能仅凭该方程唯一分开左右脚分配。

本阶段不把接触平面中心当作扳手参考点。动力学方程直接给出质心处的世界系扳手
\(\mathbf w_{c,t}^{G,W}\)，再由
\(\mathbf R_{FW,t}=\mathbf R_{WF,t}^T\) 旋转到足坐标系：

\[
\mathbf w_{c,t}^{G,F}
=
\begin{bmatrix}\mathbf R_{FW,t}&0\\0&\mathbf R_{FW,t}\end{bmatrix}
\mathbf w_{c,t}^{G,W}.
\]

若其他模块给出的是接触平面参考点 \(O\) 的扳手，必须先变换到质心：

\[
\boldsymbol\tau^{G}
=\boldsymbol\tau^{O}
+(\mathbf p_O-\mathbf p_G)\times\mathbf f,
\qquad
\mathbf f^{G}=\mathbf f^{O}.
\]

### 5.3 当前动力学扳手集合

将当前状态、加速度、惯性参数和其他外力的可接受组合记为 \(\mathcal Z_t\)。定义

\[
\boxed{
\mathscr W_{\mathrm{dyn},t}^{G}
=\left\{
\mathcal T_{F\leftarrow W}
\left(
\dot{\mathbf h}_G(\mathbf z)
-\mathbf w_g^G(\mathbf z)
-\mathbf w_{\mathrm{other}}^G(\mathbf z)
\right)
\ \middle|\ 
\mathbf z\in\mathcal Z_t
\right\},
}
\]

其中 \(\mathcal T_{F\leftarrow W}\) 只表示从世界轴到足部轴的旋转；参考点始终是足部质心。

在第一阶段的理想理论与仿真中，若模型、运动状态和其他外力均精确，则

\[
\mathcal Z_t=\{\bar{\mathbf z}_t\},
\qquad
\mathscr W_{\mathrm{dyn},t}^{G}
=\{\bar{\mathbf w}_{c,t}^{G}\}.
\]

即当前合扳手是唯一值。只有引入模型、状态或未知外力的不确定性后，它才成为非单点范围；不能为了获得“集合”而人为加入误差。

例如精确仿真算得

\[
\bar{\mathbf w}_{c,t}^{G}
=\begin{bmatrix}
\bar\tau_x&\bar\tau_y&\bar\tau_z&
\bar f_x&\bar f_y&\bar f_z
\end{bmatrix}^T,
\]

那么 \(\mathscr W_{\mathrm{dyn},t}^{G}=\{\bar{\mathbf w}_{c,t}^{G}\}\) 只是把这个唯一向量写成单元素集合，以便以后与压力相容集合统一求交。若某个状态量只知道上下界，所有可能状态映射出的扳手才形成真正有宽度的集合。

### 5.4 与瞬时可行集合的区别

在 \(\mathscr W_{\mathrm{dyn},t}\) 中，\(q_t,\mathbf v_t,\dot{\mathbf v}_t\) 是当前已经发生运动的观测量。瞬时可行集合则固定当前状态

\[
x_t=(q_t,\mathbf v_t)
\]

和接触模式 \(c_t\)，把下一瞬间的广义加速度、关节力矩和接触力作为可选择变量：

\[
\dot{\mathbf v}\in\mathbb R^{n_v},
\qquad
\boldsymbol\tau\in\mathbb R^{n_a},
\qquad
\mathbf f_i\in\mathbb R^{3m_i},
\qquad
\mathbf w_i\in\mathbb R^6,
\]

其中 \(n_v\) 是浮动基机器人的广义速度维数，\(n_a\) 是执行器数，\(m_i\) 是第 \(i\) 个足当前纳入求解的离散接触点数。\(\mathbf f_i\) 堆叠这些点的三维接触力，\(\mathbf w_i\) 是它们在该足约定参考点处形成的六维合扳手。先定义全部可行变量组成的集合

\[
\boxed{
\mathcal Z_{\mathrm{inst}}(x_t,c_t)
=\left\{
\begin{aligned}
(\dot{\mathbf v},\boldsymbol\tau,
\{\mathbf f_i,\mathbf w_i\})\ \biggm|\ {}
&\mathbf M(q_t)\dot{\mathbf v}
+\mathbf h(q_t,\mathbf v_t)\\
&\quad=\mathbf S^T\boldsymbol\tau
+\sum_i\mathbf J_i^T(q_t)\mathbf f_i
+\boldsymbol\tau_{\mathrm{other}},\\
&\boldsymbol\tau\in\mathcal T_t,\\
&(\dot{\mathbf v},\mathbf f_i)
\in\mathcal C_i(x_t,c_{i,t}),\\
&\mathbf w_i=\mathbf G_i\mathbf f_i
\end{aligned}
\right\}.
}
\]

其中，\(\mathcal T_t\) 是当前状态下的驱动力矩范围；\(\mathcal C_i\) 汇总第 \(i\) 个接触的单边接触、摩擦、承载和接触运动学条件。例如固定粘着接触通常还要求

\[
\mathbf J_i\dot{\mathbf v}
+\dot{\mathbf J}_i\mathbf v_t=\mathbf 0.
\]

瞬时可行扳手集合不是凭文字列出的约束集合，而是将 \(\mathcal Z_{\mathrm{inst}}\) 投影到所需扳手变量：

\[
\mathcal Q_{\mathrm{inst}}(x_t,c_t)
=\operatorname{proj}_{\{\mathbf w_i\}}
\mathcal Z_{\mathrm{inst}}(x_t,c_t).
\]

“投影”表示只保留扳手，消去加速度、关节力矩和逐点接触力。集合中的每个扳手都必须至少对应一组满足全部约束的变量，这组变量称为该扳手的**可行性证据**。因此“固定当前运动重建扳手”不等于“遍历当前可选动作求能力”。

可以把二者理解成：

- \(\mathscr W_{\mathrm{dyn},t}\) 是回看已经发生的这一帧，求“刚才实际用了什么扳手”；
- \(\mathcal Q_{\mathrm{inst}}\) 是站在当前状态向前看，求“现在最多还能选择哪些扳手”。

### 5.5 为什么还要计算当前扳手

当前扳手不是最终的能力集合，但有三个不可替代的作用：

1. 它是局部能力集合中的当前工作点。只有知道现在位于哪里，才能计算距离边界还有多远；
2. 它与法向压力共同约束未知切向力和接触参数。没有当前完整扳手，压力只能说明法向载荷，无法检查候选摩擦参数能否解释实际水平力和偏航力矩；
3. 它把完整机器人动力学与局部足地模型连接起来。动力学给出合结果，局部模型检查这个结果能否由足底接触力合法产生。

因此第一阶段不是用当前扳手代替能力集合，而是先用当前扳手确定“当前位置和相容参数”，再计算“从这里还能到哪里”。

## 6. 压力和 IMU 对当前接触的约束

### 6.1 当前承载点和法向工作点

压力给出

\[
\bar{\boldsymbol\lambda}_t=
\begin{bmatrix}
\bar\lambda_{1,t}&\cdots&\bar\lambda_{M,t}
\end{bmatrix}^T
\in\mathbb R_{\ge0}^{M}.
\]

在无传感器误差的基础模型中

\[
\boxed{
\mathcal I_t=
\left\{j\in\{1,\ldots,M\}
\mid\bar\lambda_{j,t}>0\right\}.
}
\]

当前总法向载荷为

\[
\bar N_t=\sum_{j\in\mathcal I_t}\bar\lambda_{j,t}.
\]

法向压力还可以计算压力中心 CoP。在当前足底平面内，它是法向载荷位置的加权平均：

\[
\mathbf r_{\mathrm{CoP},t}
=\frac{1}{\bar N_t}
\sum_{j\in\mathcal I_t}
\bar\lambda_{j,t}\mathbf r_j^G,
\qquad
\bar N_t>0.
\]

CoP 只用一个位置概括法向载荷中心。两种完全不同的压力分布可能具有相同 CoP，因此局部集合计算仍保留全部 \(\bar\lambda_{j,t}\) 和点位信息，而不是只使用 CoP。

压力直接给出的法向合扳手为

\[
\bar{\mathbf w}_{n,t}
=\sum_{j\in\mathcal I_t}
\mathbf G_j^G\bar\lambda_{j,t}\mathbf n.
\]

### 6.2 给定接触参数的压力相容集合

为展示未知摩擦如何进入模型，采用 Coulomb 接触定律作为基础形式，但不假设其参数已知：

\[
\|\mathbf u_j\|_2
\le\mu_j\bar\lambda_{j,t},
\qquad
\mu_j=\mu_j(\boldsymbol\theta),
\qquad
j\in\mathcal I_t.
\]

Coulomb 约束的物理意思是：点上的切向力大小不能超过“摩擦系数乘以法向力”。法向压得越紧，能够维持静止而不滑动的切向力通常越大；当

\[
\|\mathbf u_j\|_2
=\mu_j\bar\lambda_{j,t}
\]

时，该点位于理想静摩擦边界。这个公式只规定滑动前的允许范围，并不自动给出实际 \(\mu_j\)，也不描述滑动后的速度相关摩擦、粘附或材料变形。第一阶段把它作为最基础的接触定律，以后可用更复杂模型替换。

给定候选参数 \(\boldsymbol\theta\)，定义当前压力相容扳手集合

\[
\boxed{
\mathscr W_{\mathrm{press},t}(\boldsymbol\theta)
=\left\{
\bar{\mathbf w}_{n,t}
+\sum_{j\in\mathcal I_t}
\mathbf G_j\mathbf T\mathbf u_j
\ \middle|\ 
\|\mathbf u_j\|_2
\le\mu_j\bar\lambda_{j,t}
\right\}.
}
\]

这个集合的构造可以分三步理解：

1. 压力已经固定每个点此刻的法向力 \(\bar\lambda_{j,t}\)；
2. 每个点的切向力 \(\mathbf u_j\) 尚未测得，只能在候选接触定律允许范围内变化；
3. 把所有允许的切向力分配代入合扳手公式，得到所有可能的完整六维扳手。

因此 \(\mathscr W_{\mathrm{press},t}\) 不是接触的全部能力，只是“当前法向压力不变时，完整当前扳手可能在哪里”。

若 \(\boldsymbol\theta\) 还包含局部承载上限等参数，则必须同时满足当前工作点一致性条件，例如

\[
0\le\bar\lambda_{j,t}
\le\lambda_j^{\max}(\boldsymbol\theta),
\qquad
j\in\mathcal I_t.
\]

将这些不能直接写入扳手集合的参数条件统一记为

\[
\Phi_t^{\mathrm{press}}(\boldsymbol\theta)=\mathrm{true}.
\]

当前合扳手相容集合为

\[
\boxed{
\mathscr W_{\mathrm{cur},t}(\boldsymbol\theta)
=\mathscr W_{\mathrm{dyn},t}^{G}
\cap
\mathscr W_{\mathrm{press},t}(\boldsymbol\theta).
}
\]

交集在这里表示“同时符合两条证据”。例如动力学给出的当前扳手为 \(\bar{\mathbf w}_{c,t}^{G}\)，若它落在 \(\mathscr W_{\mathrm{press},t}(\boldsymbol\theta)\) 内，则候选参数 \(\boldsymbol\theta\) 至少能够解释当前观测；若落在集合外，则不存在任何满足该候选参数的切向力分配，因而这组参数应被排除。

合扳手在理想单足支撑中可以唯一，但满足该合扳手的逐点切向力分配一般不唯一。相容分配集合为

\[
\mathscr U_{\mathrm{cur},t}(\boldsymbol\theta)
=\left\{
\begin{aligned}
\{\mathbf u_j\}_{j\in\mathcal I_t}\ \biggm|\ {}
&\bar{\mathbf w}_{n,t}
+\sum_{j\in\mathcal I_t}
\mathbf G_j^G\mathbf T\mathbf u_j
\in\mathscr W_{\mathrm{dyn},t}^{G},\\
&\|\mathbf u_j\|_2
\le\mu_j\bar\lambda_{j,t}
\end{aligned}
\right\}.
\]

原因是合扳手只有六个分量，而 \(|\mathcal I_t|\) 个接触点共有 \(2|\mathcal I_t|\) 个未知切向分量。点数较多时，不同局部分配可能产生完全相同的合力与合力矩。例如两个点的切向力同时增加一对相反分量，合力可能不变，但局部受力已经改变。\(\mathscr U_{\mathrm{cur},t}\) 保留所有满足合扳手、压力和接触约束的分配，而不是任意挑选其中一个当作真实分配。

### 6.3 IMU 的作用边界

足端 IMU 用于：

1. 给出足姿态，将法向、压力合扳手和动力学合扳手变换到统一足坐标系；
2. 判断足是否静止、发生冲击、开始滑移或绕边转动；
3. 决定当前稳定接触模型是否继续适用，并提供失效事件时刻。

当使用完整机器人整体动力学时，足端 IMU 不是求合接触扳手唯一值的必要条件；它提供局部足运动与接触状态证据。仅凭足端 IMU 和法向压力，在上部作用未知时，不能唯一恢复完整六维接触扳手。

## 7. 用当前一致性约束未知接触参数

动力学范围不能直接裁剪局部能力集合。错误写法是

\[
\mathscr W_{\mathrm{loc},t}
\leftarrow
\mathscr W_{\mathrm{loc},t}
\cap
\mathscr W_{\mathrm{dyn},t},
\]

因为它会把“全部接触能力”错误缩成“当前正在作用的扳手”。

正确做法是先排除不能解释当前动力学与压力观测的接触参数：

\[
\boxed{
\Theta_t^{\mathrm{cur}}
=\left\{
\boldsymbol\theta\in\Theta_0
\ \middle|\ 
\Phi_t^{\mathrm{press}}(\boldsymbol\theta)=\mathrm{true},
\quad
\mathscr W_{\mathrm{cur},t}(\boldsymbol\theta)
\neq\varnothing
\right\}.
}
\]

这一步是在“参数空间”中筛选，而不是在“扳手空间”中缩小能力。用公共摩擦系数举例：若 \(\mu=0.2\) 无论怎样分配切向力都无法解释当前合扳手，就删除 \(0.2\)；若 \(\mu=0.5\) 可以解释，则暂时保留。随后分别用保留下来的参数计算完整能力。

若还观测到明确的稳定、滑移、边缘转动或分离事件，应再加入相应事件一致性条件，而不是把 IMU 数值直接当作摩擦系数。

### 7.1 单个稳定时刻能得到什么

单个稳定时刻通常只能证明当前作用尚未超过接触能力。例如，在额外采用“全部活动点共享一个有效摩擦参数 \(\mu\)”这一降维参数化时，可计算解释当前数据所需的最小摩擦：

\[
\boxed{
\begin{aligned}
\mu_{\mathrm{req},t}
=\min_{\mu,\{\mathbf u_j\}}\quad&\mu\\
\text{s.t.}\quad&
\bar{\mathbf w}_{n,t}
+\sum_{j\in\mathcal I_t}
\mathbf G_j^G\mathbf T\mathbf u_j
\in\mathscr W_{\mathrm{dyn},t}^{G},\\
&\|\mathbf u_j\|_2
\le\mu\bar\lambda_{j,t}.
\end{aligned}
}
\]

若当前稳定，只能推出

\[
\mu_{\mathrm{actual}}\ge\mu_{\mathrm{req},t},
\]

不能唯一得到真实摩擦极限。公共 \(\mu\) 只是可选的最低维实现，不是理论模型默认的已知条件；若允许每点 \(\mu_j\) 独立未知，单个合扳手样本会严重欠定。

例如当前法向合力为 \(100\,\mathrm N\)，切向合力为 \(30\,\mathrm N\)，且接触稳定。忽略力矩耦合时，只能说明有效摩擦系数至少约为 \(0.3\)。真实值可能是 \(0.4\)，也可能是 \(0.8\)；只要没有加载到滑移附近，这个稳定样本就无法区分它们。接近失效的数据之所以重要，是因为它提供了能力边界，而不仅是“当前点在边界以内”。

### 7.2 边界数据怎样进一步收束

在接触点集、总法向载荷和加载方向受控的准静态试验中，设初始稳定扳手为 \(\mathbf w_{0,k}\)，实际加载路径为

\[
\mathbf w_k(\alpha)
=\mathbf w_{0,k}+\alpha\mathbf d_k,
\qquad
\alpha\ge0.
\]

定义沿该路径的径向能力

\[
\boxed{
\rho_{\mathrm{loc}}
(\mathbf w_{0,k},\mathbf d_k;
\boldsymbol\theta,N_k)
=\sup\left\{
\alpha\ge0
\ \middle|\ 
\mathbf w_{0,k}+\alpha\mathbf d_k
\in\mathscr W_{\mathrm{loc},t}
(\boldsymbol\theta;N_k)
\right\}.
}
\]

符号 \(\sup\) 表示可达到幅值的最小上界。在本文的闭合有界局部集合中，该边界通常能够实际取得，所以数值求解时可直接写成后文的最大化问题 \(\max\alpha\)。

记录

\[
\mathbf w_k^{\mathrm{last\ stable}},
\qquad
\mathbf w_k^{\mathrm{first\ failure}}.
\]

以及对应的 \(\alpha_k^{\mathrm{last}}\)、\(\alpha_k^{\mathrm{fail}}\)。仅当加载路径确实沿 \(\mathbf d_k\)、未控制分量保持在规定路径上、失败由足地接触而非作动器饱和引起时，才可写成

\[
\boxed{
\alpha_k^{\mathrm{last}}
\le
\rho_{\mathrm{loc}}
(\mathbf w_{0,k},\mathbf d_k;
\boldsymbol\theta,N_k)
\le
\alpha_k^{\mathrm{fail}}.
}
\]

固定 \(N_k\) 时，加载方向必须保持总法向力不变；若加载路径改变法向力，则应同时使用随路径变化的条件 \(N(\alpha)\)，不能与单个固定 \(N_k\) 的集合比较。

支持函数边界与径向边界不是同一量。只有加载控制器能够跟踪支持函数优化得到的边界扳手，或实验路径恰好到达对应支撑超平面的最优点时，才能直接用 \(h_{\mathrm{loc}}(\mathbf d_k)\) 与实验极值比较。一般的单方向加载应使用上述 \(\rho_{\mathrm{loc}}\)。

多次稳定与失效数据形成的数据一致参数集合

\[
\Theta_{1:t}
=\left\{
\boldsymbol\theta\in\Theta_0
\ \middle|\ 
\boldsymbol\theta
\text{ 满足全部当前扳手、压力和有效边界条件}
\right\}.
\]

这里是对参数施加多条物理一致性条件，不是把当前扳手集合与能力集合直接求交。历史不是局部集合定义所必需，但没有接近边界的数据时，许多能力参数只能得到宽泛范围或单侧界。

可以把历史理解为重复筛选：每个时刻都删除一批无法解释观测的候选参数。若第一个时刻留下 \([0.3,0.9]\)，第二个更大切向载荷的稳定时刻可能留下 \([0.5,0.9]\)，一次确认在 \(0.62\) 附近滑移的边界试验还可能把范围进一步缩小。历史不改变接触力学公式，只改变公式中允许采用的参数范围。

后文用 \(\Theta_t\) 统称当前采用的参数集合：仅使用当前时刻时取 \(\Theta_t=\Theta_t^{\mathrm{cur}}\)；使用截至时刻 \(t\) 的有效标定或历史数据时取 \(\Theta_t=\Theta_{1:t}\)。两种情况使用同一个局部集合定义，只是参数域宽度不同。

## 8. 参数化局部接触能力集合

### 8.1 将总法向载荷作为条件

若直接把“机器人能够改变的总载荷范围”放入局部集合，就会提前混入整机能力。第一阶段将总法向载荷 \(N\ge0\) 作为条件，计算一族集合

\[
N
\longmapsto
\mathscr W_{\mathrm{loc},t}(\boldsymbol\theta;N).
\]

当前工作点对应 \(N=\bar N_t\)，但模型不假定法向载荷永远不变。后续 \(\mathcal Q_{\mathrm{inst}}\) 决定机器人实际能够实现哪些左右足载荷分配和其他扳手分量。

这里固定 \(N\) 只是先回答一个条件问题：“如果总法向载荷是 \(N\)，接触界面允许哪些其余力和力矩？”分别计算 \(N=200\,\mathrm N\)、\(300\,\mathrm N\)、\(400\,\mathrm N\) 就得到三张不同的能力截面。把所有 \(N\) 的结果放在一起，法向载荷当然仍然可以变化。这样做的目的，是避免由局部接触模型擅自决定机器人能把多大法向力压到地面上。

### 8.2 候选法向载荷集合

给定 \(\boldsymbol\theta\) 和 \(N\)，定义

\[
\boxed{
\mathscr L_t(\boldsymbol\theta;N)
=\left\{
\begin{aligned}
\boldsymbol\lambda\in\mathbb R_{\ge0}^{M}
\ \biggm|\ {}
&\sum_{j\in\mathcal I_t}\lambda_j=N,\\
&0\le\lambda_j
\le\lambda_j^{\max}(\boldsymbol\theta),
\quad j\in\mathcal I_t,\\
&\lambda_j=0,
\quad j\notin\mathcal I_t
\end{aligned}
\right\}.
}
\]

该集合只描述法向力怎样分配。例如 \(N=300\,\mathrm N\) 且有三个活动点时，\((100,100,100)\) 和 \((50,120,130)\) 都可能属于 \(\mathscr L_t\)，但三点之和必须保持 \(300\,\mathrm N\)。不同分配会改变 CoP 以及绕足部质心的俯仰、横滚力矩能力。

若当前理论版本不引入局部承载上限，可删除第二行上界；此时 \(\lambda_j\le N\) 已由非负性和总载荷等式保证，集合仍然有界。是否引入 \(\lambda_j^{\max}\) 是接触物理模型选择，不应默认由传感器性能决定。

### 8.3 固定参数下的局部集合

采用 Coulomb 基础形式时

\[
\boxed{
\mathscr W_{\mathrm{loc},t}
(\boldsymbol\theta;N)
=\left\{
\begin{aligned}
\sum_{j\in\mathcal I_t}
\mathbf G_j
(\lambda_j\mathbf n+\mathbf T\mathbf u_j)
\ \biggm|\ {}
&\boldsymbol\lambda
\in\mathscr L_t(\boldsymbol\theta;N),\\
&\|\mathbf u_j\|_2
\le\mu_j(\boldsymbol\theta)\lambda_j
\end{aligned}
\right\}.
}
\]

该集合固定当前承载点、点位和公共法向，允许各点在接触定律内重新分配法向力和切向力。它不包含新接触、接触点移动、公共法向改变、整机驱动限制或未来动作改变接触后的能力。

集合定义中的“存在一组 \(\lambda_j,\mathbf u_j\)”表示：只要能找到至少一种合法的接触力分配产生某个合扳手，该合扳手就属于局部能力集合。它不表示机器人一定能够主动实现这种分配；机器人能否实现，还要在第二阶段加入关节力矩、运动学和整机动力学约束。

固定 \(N\) 时，集合内所有扳手都满足

\[
\mathbf n^T\mathbf f=N.
\]

因此 \(\mathscr W_{\mathrm{loc},t}(\boldsymbol\theta;N)\) 虽然表示为 \(\mathbb R^6\) 的子集，但必定位于一个至多五维的仿射超平面内；随 \(N\) 变化的集合族 \(\mathfrak W_{\mathrm{loc},t}(N)\) 才描述法向载荷也可变化时的六维能力。后续整机层不能把单个固定 \(N\) 截面误当成全部局部能力。

若 \(\mathscr L_t(\boldsymbol\theta;N)\) 非空，且所有相关参数有限，则接触力变量集合非空、凸、紧；线性扳手映射保持这些性质，所以 \(\mathscr W_{\mathrm{loc},t}(\boldsymbol\theta;N)\) 非空、凸、紧。除固定 \(N\) 已造成的降维外，点数不足或几何退化还可能进一步降低其维数。

这些术语的工程含义是：

- 非空：至少存在一种合法接触力分配；
- 凸：若两个扳手可行，则它们按任意比例加权得到的中间扳手也可行；
- 紧：集合闭合且有界，最大能力能够在边界上取得，而不会无限增大；
- 不满维：某些方向根本无法独立变化。例如单点接触不能任意产生三个力矩分量。

固定 \(N\) 后法向合力已经被规定，因此六个扳手分量中有一个等式约束，最多剩五个独立变化方向。这就是“位于五维仿射超平面”的直观含义。

若 \(\boldsymbol\theta\in\Theta_t^{\mathrm{cur}}\)、当前压力不违反承载上限且 \(N=\bar N_t\)，则

\[
\mathscr W_{\mathrm{cur},t}(\boldsymbol\theta)
\subseteq
\mathscr W_{\mathrm{loc},t}
(\boldsymbol\theta;\bar N_t).
\]

### 8.4 参数仍不确定时的输出

第一阶段的基本输出应保留为集合族

\[
\boxed{
\mathfrak W_{\mathrm{loc},t}(N)
=\left\{
\mathscr W_{\mathrm{loc},t}
(\boldsymbol\theta;N)
\ \middle|\ 
\boldsymbol\theta\in\Theta_t
\right\}.
}
\]

如后续决策确实需要单一集合，可再区分

\[
\mathscr W_{\mathrm{guar},t}(N)
=\bigcap_{\boldsymbol\theta\in\Theta_t}
\mathscr W_{\mathrm{loc},t}
(\boldsymbol\theta;N),
\]

\[
\mathscr W_{\mathrm{poss},t}(N)
=\bigcup_{\boldsymbol\theta\in\Theta_t}
\mathscr W_{\mathrm{loc},t}
(\boldsymbol\theta;N).
\]

前者表示对所有相容参数都保证可行的能力，后者表示至少对一个相容参数可能可行的能力。二者不能混用；并集一般也不保证凸。若 \(\Theta_t\) 没有数值界，这两个集合可能过度保守或无界，因此不能宣称已经获得可用于控制的数值集合。

这里的“保证集合”是有条件的。只有同时满足

1. 真实参数 \(\boldsymbol\theta^\star\in\Theta_t\)；
2. 选定的离散接触定律能够覆盖真实接触物理；
3. 坐标系、参考点、活动点和总载荷条件一致；

才能推出

\[
\mathscr W_{\mathrm{guar},t}(N)
\subseteq
\mathscr W_{\mathrm{loc},t}
(\boldsymbol\theta^\star;N).
\]

若 \(\Theta_t\) 只是未经覆盖性验证的神经网络置信区间，上述包含关系并不自动成立，此时“保证集合”只能称为候选鲁棒集合。稳定工作点数据也只能证明已经观测到的扳手或加载路径可行，不能单独证明尚未到达方向上的能力上界。

例如参数候选只有 \(\mu=0.4\) 和 \(\mu=0.8\)：

- 保证集合只保留两种摩擦条件下都能承受的扳手，适合强调安全保证；
- 可能集合还包含只有 \(\mu=0.8\) 时才能承受的扳手，适合描述尚未排除的能力上限，但不能直接作为安全指令。

第一阶段首先保留集合族，是为了不在尚未确定风险含义时擅自选择“保守”或“乐观”输出。

## 9. 支持函数与可解性

### 9.1 三种实际查询

局部集合位于六维空间，很难直接完整画出来。实际算法通常只需要回答三类问题：

1. 成员检验：“给定扳手 \(\mathbf w^\star\)，是否存在合法接触力分配产生它？”
2. 支持函数：“在方向 \(\mathbf d\) 上，集合投影最远能到哪里？”
3. 径向能力：“从当前扳手 \(\mathbf w_0\) 出发，沿实际加载方向 \(\mathbf d\) 还能增加多少？”

成员检验适合在线判断某个候选扳手是否可行；支持函数适合绘制集合投影和比较整体外形；径向能力最适合与实验台从当前状态逐渐加载到失效的过程比较。

### 9.2 支持函数

对固定 \(\boldsymbol\theta\)、\(N\) 和查询方向 \(\mathbf d\in\mathbb R^6\)，定义

\[
h_{\mathrm{loc},t}
(\mathbf d;\boldsymbol\theta,N)
=\max_{\mathbf w
\in\mathscr W_{\mathrm{loc},t}
(\boldsymbol\theta;N)}
\mathbf d^T\mathbf w.
\]

其中 \(\mathbf d^T\mathbf w\) 是扳手在查询方向上的标量投影。二维情况下，可以把集合想成一个平面图形：支持函数就是用一条垂直于 \(\mathbf d\) 的直线从外向内移动，第一次碰到图形时，该方向上的最远距离。六维中的含义完全相同，只是无法直接画出。

令

\[
\mathbf q_j(\mathbf d)=
\mathbf G_j^T\mathbf d
\in\mathbb R^3.
\]

给定 \(\lambda_j\) 后，切向力可解析消去：

\[
\max_{\|\mathbf u_j\|_2
\le\mu_j\lambda_j}
\mathbf q_j^T\mathbf T\mathbf u_j
=\mu_j\lambda_j
\|\mathbf T^T\mathbf q_j\|_2.
\]

因此令

\[
c_j(\mathbf d;\boldsymbol\theta)
=\mathbf d^T\mathbf G_j\mathbf n
+\mu_j(\boldsymbol\theta)
\|\mathbf T^T\mathbf G_j^T\mathbf d\|_2,
\]

即可得到线性规划

\[
\boxed{
\begin{aligned}
h_{\mathrm{loc},t}
(\mathbf d;\boldsymbol\theta,N)
=\max_{\boldsymbol\lambda}\quad&
\sum_{j\in\mathcal I_t}
c_j(\mathbf d;\boldsymbol\theta)\lambda_j\\
\text{s.t.}\quad&
\boldsymbol\lambda
\in\mathscr L_t(\boldsymbol\theta;N).
\end{aligned}
}
\]

这里称为线性规划，是因为未知量只剩 \(\lambda_j\)，目标和载荷约束对 \(\lambda_j\) 都是一次式。切向力圆盘的最优方向已经通过范数公式解析求出，因此不再作为数值优化变量。求解器返回的不仅有最大投影，还能恢复使该方向达到边界的法向和切向力分配。

成员检验和给定当前扳手的径向余量是二阶锥问题。将摩擦圆盘多边形化后可使用线性规划，但会引入近似。

“二阶锥问题”在本文中主要来自约束

\[
\|\mathbf u_j\|_2
\le\mu_j\lambda_j,
\]

它描述二维切向力圆盘。该类问题仍是凸优化问题，有成熟求解器可以得到全局最优解。若用多边形近似圆盘，约束可改为若干线性不等式，从而使用线性规划；多边形内接会偏保守，外接则可能危险高估。

### 9.3 径向能力

对给定工作点 \(\mathbf w_0\) 和加载方向 \(\mathbf d\)，径向能力可直接写成

\[
\boxed{
\begin{aligned}
\rho_{\mathrm{loc}}
(\mathbf w_0,\mathbf d;
\boldsymbol\theta,N)
=\max_{\alpha,\boldsymbol\lambda,
\{\mathbf u_j\}}\quad&\alpha\\
\text{s.t.}\quad&
\mathbf w_0+\alpha\mathbf d
=\sum_{j\in\mathcal I_t}
\mathbf G_j
(\lambda_j\mathbf n+\mathbf T\mathbf u_j),\\
&\boldsymbol\lambda
\in\mathscr L_t(\boldsymbol\theta;N),\\
&\|\mathbf u_j\|_2
\le\mu_j(\boldsymbol\theta)\lambda_j,\\
&\alpha\ge0.
\end{aligned}
}
\]

固定参数时这是二阶锥规划；它与实验的渐增加载路径直接对应。

这里的 \(\alpha\) 是沿给定加载路径增加的幅值。例如 \(\mathbf d\) 只在 \(f_x\) 分量为 1 时，\(\alpha\) 的单位是 \(\mathrm N\)，表示还能增加多少水平力；若 \(\mathbf d\) 同时包含力与力矩分量，必须先用后文的特征尺度无量纲化，否则不同单位不能直接比较。

将 \(\rho_{\mathrm{loc}}\) 解释为“从当前工作点出发的剩余裕度”之前，必须先验证

\[
\mathbf w_0
\in\mathscr W_{\mathrm{loc},t}
(\boldsymbol\theta;N).
\]

否则射线可能先从集合外进入再离开，优化结果不再是当前稳定状态的可用余量。

由于力和力矩量纲不同，采样查询方向前应给定特征尺度 \(f_0>0\)、\(\tau_0>0\)，使用无量纲扳手

\[
\widetilde{\mathbf w}
=\operatorname{diag}
(\tau_0^{-1}\mathbf I_3,
f_0^{-1}\mathbf I_3)
\mathbf w.
\]

例如可取 \(f_0=mg\)，\(\tau_0=mg\ell\)，其中 \(\ell\) 是足长或另一明确的特征长度。该缩放只用于定义方向和数值比较，不改变真实力与力矩。

## 10. 与 CWC 的关系

CWC 是 Contact Wrench Cone，即接触扳手锥。最基础的 Coulomb 模型具有比例性质：若一组接触力可行，将全部接触力同时乘以任意正数仍满足摩擦比例约束。因此所有可行扳手从原点沿射线无限延伸，几何上形成“锥”。

CWC 回答的是没有规定载荷大小时，哪些扳手方向符合单边接触和摩擦规律；它通常不提供有限的最大载荷。本文进一步固定总法向载荷 \(N\)，并可加入局部承载上限，使无界的锥变成可计算有限边界的集合截面。

对固定 \(\boldsymbol\theta\)，删除固定总载荷和局部承载上限，仅保留单边接触与 Coulomb 约束，得到接触力锥

\[
\mathscr K_t(\boldsymbol\theta)
=\left\{
(\{\lambda_j,\mathbf u_j\}_{j\in\mathcal I_t})
\ \middle|\ 
\lambda_j\ge0,
\ \|\mathbf u_j\|_2
\le\mu_j(\boldsymbol\theta)\lambda_j,
\ j\in\mathcal I_t
\right\}.
\]

令 \(\mathcal G_t\) 表示全部点力到合扳手的线性映射，则接触扳手锥为

\[
\mathscr C_t(\boldsymbol\theta)
=\mathcal G_t
(\mathscr K_t(\boldsymbol\theta)).
\]

本模型的有界集合是先在接触力变量空间施加载荷条件，再映射到扳手空间：

\[
\boxed{
\mathscr W_{\mathrm{loc},t}
(\boldsymbol\theta;N)
=\mathcal G_t
\left(
\mathscr K_t(\boldsymbol\theta)
\cap
\{\boldsymbol\lambda
\in\mathscr L_t(\boldsymbol\theta;N)\}
\right).
}
\]

因此它不是在六维扳手空间中把 \(\mathscr C_t\) 与当前扳手范围简单求交。矩形面接触的闭式 CWC 只是统一法向模型的一个特例；离散点表达允许当前承载区域不规则或间断。

“先在接触力空间约束，再映射到扳手空间”很重要，因为同一个合扳手可能由多种逐点力分配产生。某种分配超过单点上限，并不代表该合扳手一定不可行；只要还存在另一种合法分配，该合扳手仍属于集合。直接对六维合扳手做简单裁剪会丢失这层信息。

## 11. 仿真方案

仿真始终分开“环境真值通道”和“算法输入通道”。环境内部必须设置接触参数才能运行，但算法不能直接读取这些隐藏真值。

三种仿真的职责不同：

| 模块 | 主要输入 | 主要输出 | 验证问题 |
| --- | --- | --- | --- |
| 数学求解器 | 人工给定点位、参数和 \(N\) | 集合边界与力分配 | 推导和程序是否正确 |
| 数字实验台 | 法向压力、足端IMU、加载过程 | 参数条件和径向能力 | 局部接触模型是否符合接触仿真 |
| 完整X1单足支撑 | 整机模型及当前已执行运动 | 当前合接触扳手 | 能否不使用踝部六维扳手而重建当前扳手 |

它们不是三个竞争方案，而是从数学、局部物理和整机动力学三个层次验证同一条链路。

### 11.1 数学求解器验证

先脱离动力学仿真，依次测试：

1. 单点、两点和共线点的退化能力；
2. 矩形离散点与已知 CWC 边界的一致性；
3. 不规则、间断的共面点集；
4. 不同总法向载荷 \(N\)；
5. 离散加密时面积和局部承载上限的正确缩放；
6. 支持函数、成员检验和径向余量的一致性。

这里建议使用人工构造的小例子作为单元测试。例如单点位于参考点时不应产生力矩；两个关于原点对称的点施加相反切向力时应能产生纯偏航力矩。先通过这些可手算案例，再进入 MuJoCo，便于区分“公式或代码错误”和“接触仿真设置差异”。

### 11.2 单足实验台数字模型

建立

\[
\text{上部加载机构}
\longrightarrow
\text{刚性足与足端 IMU}
\longrightarrow
\text{离散足地接触}
\longrightarrow
\text{固定基座}.
\]

仿真实现中，足底离散点属于刚性足一侧，用于表达理论点载荷和输出逐点法向力；不完整地面由独立的连续共面几何构成，不能用与离散点一一对应的地面小支撑块代替。足在初始时刻已经以零速度贴合支撑面，随后平滑增加载荷，因此这里研究的是准静态接触能力而非落地碰撞。环境的支撑几何只用于生成数据，不能作为算法输入。

仿真环境记录真实接触参数、逐点三维接触力、合扳手和失效事件。拟部署算法在线只读取离散法向载荷、足端 IMU 和加载过程中的可用状态；仿真的真实合扳手作为实验台底部六维力传感器的数字对应量，只能用于离线参数收束和验证。

从规定工作点沿多组方向缓慢加载，记录最后稳定和首次失效扳手，用于：

1. 用隐藏合扳手真值离线验证当前压力与合扳手的一致性；
2. 用标定数据离线计算或收束 \(\Theta_t\)；
3. 比较理论径向边界与仿真失效区间；
4. 区分滑移、边缘转动和分离边界。

当前法向主导基线先验证压力可观测投影

\[
\mathbf w_n=[M_x,M_y,F_z]^{\mathsf T}.
\]

由参考保持阶段的正法向载荷确定承载点，并对同一批阈值化载荷同时计算总载荷和参考扳手；随后冻结集合，不能在加载逼近失效时继续用环境几何或新的失效数据改写理论边界。完整六维扳手仍作为真值记录，但在摩擦参数未独立辨识前，不把切向能力当成本试验已经验证的结论。

数字实验台验证局部接触模型，不需要假设上部加载作用等于已知踝部六维扳手。

数字实验台的数据处理顺序为：

\[
\begin{aligned}
&\text{法向压力}
\longrightarrow
\mathcal I_t,\bar{\boldsymbol\lambda}_t,\bar N_t,\\
&\text{隐藏合扳手真值}
+\text{法向压力}
\longrightarrow
\text{离线参数条件},\\
&\text{参数条件}
+\text{接触点集}
+N
\longrightarrow
\mathscr W_{\mathrm{loc},t},\\
&\text{逐渐加载到失效}
\longrightarrow
\text{理论径向能力的验证区间}.
\end{aligned}
\]

如果在参数辨识和最终验证中使用完全相同的加载数据，只能说明模型拟合了这些数据。应保留未参与参数收束的加载方向、载荷等级或支撑点集，用于真正的外部验证。

### 11.3 完整机器人单足支撑仿真

使用完整 X1 浮动基模型，仅保留一只脚接触，并确保没有未建模的手部接触、牵引绳或固定装置反力。算法读取当前整机状态和模型，通过质心动量平衡计算 \(\mathscr W_{\mathrm{dyn},t}\)；MuJoCo 合接触扳手仅作为真值比较。

基线流程为：

1. 由当前已执行运动计算 \(\dot{\mathbf h}_{G,t}\)；
2. 求 \(\mathbf w_{c,t}^G\) 并变换到足坐标系；
3. 与仿真真实合扳手比较；
4. 与法向压力联合求当前相容分配和参数条件；
5. 将识别后的接触参数送入局部集合求解器。

这一仿真只验证“消去内部踝部扳手并重建当前接触扳手”。若进一步遍历 \(\dot{\mathbf v}\) 和 \(\boldsymbol\tau\) 来求机器人能产生的全部扳手，就已经进入 \(\mathcal Q_{\mathrm{inst}}\) 及其合扳手映射，应放到第二阶段。

完整X1仿真的最直接评价量是

\[
\mathbf e_{w,t}
=\bar{\mathbf w}_{c,t}^{\mathrm{dyn}}
-\mathbf w_{c,t}^{\mathrm{gt}},
\]

其中前者由整体动量平衡计算，后者由MuJoCo接触力汇总。理想模型中二者应接近；若差异明显，应先检查参考点、坐标系、扳手排列、重力符号、角动量导数和是否存在其他外部作用，而不是立即用学习模型补偿。

## 12. 单足实验台验证

### 12.1 实验台定位

实体实验台验证局部接触物理、参数一致性和能力边界，不验证完整机器人动力学，也不验证 \(\mathcal Q_{\mathrm{inst}}\)。建议结构为

\[
\text{上部加载机构}
\longrightarrow
\text{刚性足与足端 IMU}
\longrightarrow
\text{压力阵列和足地接触}
\longrightarrow
\text{底部六维力传感器}.
\]

设备职责为：

- 压力阵列：输出当前承载点、离散法向载荷、总法向力和 CoP；
- 足端 IMU：输出足姿态并检测冲击、滑移、边缘转动和持续运动；
- 底部六维力传感器：输出统一参考点下的真实接触合扳手；
- 相机：离线标记首次宏观滑移、翻转、分离及相对位移；
- 上部加载机构：产生可重复的力与力矩组合，不作为已知踝部六维扳手来源。

底部六维力传感器是实验真值设备，不意味着最终机器人必须安装同类传感器。第一版实验台也不需要增加上部六维传感器。只有希望额外验证“足部刚体 Newton--Euler 重建”时，才需要独立测量完整上部加载扳手；该扩展不能替代完整机器人整体动力学验证。

可以把底部六维力传感器理解为实验的“答案纸”：算法用压力、IMU和已建立的接触模型作答，底部传感器只在离线阶段告诉我们当前合扳手和失效边界是否判断正确。若把底部传感器输出直接作为未来在线算法输入，就无法证明压力阵列和IMU方案本身的有效性。

### 12.2 实验顺序

1. 法向加载：验证压力阵列总法向力、CoP 与底部六维力传感器的一致性；
2. 当前相容性：联合压力和底部真实扳手求切向力相容分配及参数条件；
3. 准静态边界：在固定 \(N\) 和接触点集下沿多个方向缓慢加载；
4. 组合边界：测试 \((f_x,\tau_y)\)、\((f_y,\tau_x)\)、\((f_x,f_y,\tau_z)\) 等耦合；
5. 支撑变化：测试完整、局部、窄边及不规则共面支撑点集；
6. 留出验证：部分方向、法向载荷和支撑点集只用于检验，不参与参数收束；
7. 动态扩展：仅在加载机构带宽和同步精度满足要求后开展。

对满足准静态和接触失效条件的方向 \(\mathbf d_k\)，记录

\[
\mathbf w_k^{\mathrm{last\ stable}},
\qquad
\mathbf w_k^{\mathrm{first\ failure}},
\]

以及相对工作点的加载幅值区间 \([\alpha_k^{\mathrm{last}},\alpha_k^{\mathrm{fail}}]\)，并与理论径向能力 \(\rho_{\mathrm{loc}}\) 比较。若实验能够跟踪理论支持点，再额外验证支持函数。主要指标为当前合扳手一致性、径向边界误差、危险高估率、保守程度、失效类型识别和单次求解时间。

## 13. 第一阶段的最终输出

### 13.1 最简输入输出图

在线当前扳手重建：

\[
\boxed{
\text{完整机器人模型与当前运动}
\longrightarrow
\mathscr W_{\mathrm{dyn},t}.
}
\]

当前接触一致性分析：

\[
\boxed{
\mathscr W_{\mathrm{dyn},t}
+\text{法向压力}
+\text{足端IMU状态证据}
\longrightarrow
\mathscr W_{\mathrm{cur},t},
\mathscr U_{\mathrm{cur},t},
\Theta_t.
}
\]

局部能力计算：

\[
\boxed{
\mathcal I_t
+\Theta_t
+N
\longrightarrow
\mathfrak W_{\mathrm{loc},t}(N)
\longrightarrow
\text{成员判断、支持边界和径向余量}.
}
\]

第一阶段不应只输出一个含义模糊的“扳手集合”，而应输出：

1. 当前动力学扳手集合 \(\mathscr W_{\mathrm{dyn},t}\)，理想单足支撑时为单点；
2. 当前相容合扳手与逐点切向力分配集合 \(\mathscr W_{\mathrm{cur},t}\)、\(\mathscr U_{\mathrm{cur},t}\)；
3. 与当前数据或边界数据相容的接触参数集合 \(\Theta_t\)；
4. 以总法向载荷为条件的局部能力集合族 \(\mathfrak W_{\mathrm{loc},t}(N)\)；
5. 多方向支持边界、当前工作点的方向余量及对应接触力分配；
6. 仿真和实验中的最后稳定/首次失效边界数据。

若没有给定 \(\Theta_0\) 的数值范围，也没有足够边界数据，第一阶段仍可完成动力学扳手重建、参数化局部集合和求解器验证，但不能声称已经唯一得到真实接触能力。

## 14. 模型与学习的严格接口

模型结合学习是后续可行方向，但物理模型和学习模型承担不同职责。对单足局部集合，在线局部观测记为

\[
\mathbf y_{i,t}
=\left(
\bar{\boldsymbol\lambda}_{i,t},
\text{足端IMU},
\text{可选历史}
\right).
\]

学习器可以由 \(\mathbf y_{i,t}\) 提出参数候选 \(\widehat\Theta_{i,t}\)、优化初值、查询方向或候选边界点；整机状态和仿真接触真值可以在训练阶段作为教师信息，但不能直接变成局部集合定义的一部分。否则网络学到的将是特定机器人姿态和驱动能力，而不是可迁移的足地接触能力。

### 14.1 三种不同可信度的输出

1. **候选输出**：神经网络直接给出的参数、支持值或扳手点，尚未证明可行；
2. **模型内近似**：每个扳手点都能返回满足第 8 节局部约束，或第 5.4 节整机约束的一组优化变量作为可行性证据；
3. **真实保守集合**：除具有模型内可行性外，还要求真实参数被参数集合覆盖，并且所用物理模型对真实接触是保守有效的。

第二项只能保证“不高估所采用的数学模型”，不能仅凭求解成功就保证“不高估真实系统”。从模型结论提升为真实保证，仍需要参数覆盖证明和独立实验验证。

在固定参数、固定载荷和固定接触模式下，本文局部集合是凸集。若已验证可行点为 \(\mathbf w^{(1)},\ldots,\mathbf w^{(K)}\)，则

\[
\boxed{
\mathscr W_{\mathrm{in}}
=\operatorname{conv}
\left\{
\mathbf w^{(1)},\ldots,\mathbf w^{(K)}
\right\}
\subseteq
\mathscr W_{\mathrm{loc}}
}
\]

是该数学模型的内近似。\(\operatorname{conv}\) 表示这些点全部凸组合的集合。不同接触模式、不同公共法向或不同固定载荷截面之间不能据此共同取凸包，否则凸包可能填入任何模式都无法实现的扳手；这些结果应保留为带条件的集合族或集合并集。

相反，只查询有限个方向的支持值并写成

\[
\mathscr W_{\mathrm{out}}
=\left\{
\mathbf w
\mid
\mathbf d_k^T\mathbf w
\le h(\mathbf d_k),
\ k=1,\ldots,K
\right\}
\]

通常得到外近似。方向采样不足时，它会包含尚未证明可行的区域，不能直接作为无高估控制集合。

### 14.2 推荐的学习位置

局部层采用

\[
\mathbf y_{i,t}
\xrightarrow{\text{学习器}}
\text{参数或候选点}
\xrightarrow{\text{局部物理求解器}}
\text{可行性证据与内近似}.
\]

整机层采用

\[
(x_t,c_t,\text{局部内近似})
\xrightarrow{\text{学习器提出候选}}
\xrightarrow{\text{整机约束复核}}
\mathcal Q_{\mathrm{inst}}^{-}.
\]

学习器可以显著减少方向查询、活动集搜索和优化迭代，但最终被接受的扳手必须通过对应层的约束。若只追求概率意义上的低风险而非确定性保证，应单独给出失覆盖概率和独立测试结果，不能继续使用“必然可行”的表述。

### 14.3 训练通路与部署通路必须分开

第一阶段可以在完整机器人单足支撑仿真中使用整机状态重建当前合扳手，但该信息只适合作为当前工作点、参数一致性条件或训练教师：

\[
\left(
x_t^{\mathrm{WB}},
\bar{\boldsymbol\lambda}_t,
\text{足端IMU},
\text{边界试验}
\right)
\xrightarrow{\text{单足训练与标定}}
\text{参数标签和已验证边界}.
\]

其中 \(x_t^{\mathrm{WB}}\) 表示训练或标定阶段可用的整机状态。稳定状态重建出的单个当前扳手不能被当作完整局部能力边界标签；边界标签必须来自理论求解、逐渐加载至失效的数据或其他独立真值。

为保证以后能够迁移到双足，部署时的局部学习器只接收每只足实际可获得的局部信息：

\[
\left(
\bar{\boldsymbol\lambda}_{i,t},
\text{足端IMU}_i,
\text{可选局部历史}
\right)
\xrightarrow{\text{局部学习器与物理复核}}
\mathscr W_{\mathrm{loc},i}^{-}.
\]

若部署阶段无法获得某个变量，就不能让网络在训练时依赖它作为输入；它只能作为损失函数中的教师量。这样，双足阶段不需要先由整体动力学唯一分解左右当前扳手，也能分别形成左右足的局部候选能力。随后计算 \(\mathcal Q_{\mathrm{inst}}^{2S}\) 时，整机状态重新作为整机动力学的必要输入出现，这不是信息泄漏，而是瞬时可行集合的定义所需。

第一阶段优先采用监督学习、区间估计或物理残差学习；强化学习更适合在后续依据这些集合选择任务动作、接触调整动作和微小激励，不宜承担“宣布某扳手物理可行”的最终职责。

## 15. 双足与四个全身扳手集合的接口

### 15.1 先保留左右足扳手对

令 \(\mathbf w_L,\mathbf w_R\in\mathbb R^6\) 分别在左右足约定参考点和坐标系表达。固定当前双足接触模式 \(c_t\) 后，第 5.4 节的投影给出十二维扳手对集合

\[
\mathcal Q_{\mathrm{inst}}^{2S}(x_t,c_t)
\subseteq\mathbb R^{12}.
\]

只有满足以下一致性条件时，才能写出局部能力与瞬时能力的包含关系：

- 使用相同活动点、公共法向和接触参数；
- 使用相同左右足参考点、坐标系和扳手排列；
- 比较相同的总法向载荷截面；
- 瞬时集合中的接触定律不比局部集合更宽松。

在固定 \((N_L,N_R)\) 时才有

\[
\boxed{
\mathcal Q_{\mathrm{inst}}^{2S}
(x_t,c_t;N_L,N_R)
\subseteq
\mathscr W_{\mathrm{loc},L}
(\boldsymbol\theta_L;N_L)
\times
\mathscr W_{\mathrm{loc},R}
(\boldsymbol\theta_R;N_R).
}
\]

这里通常是严格包含，因为左侧还受到关节力矩、整机动力学和接触运动学限制。若总法向载荷也作为瞬时变量，则应对允许的载荷对集合 \(\mathcal N_t\) 写成

\[
\mathcal Q_{\mathrm{inst}}^{2S}(x_t,c_t)
\subseteq
\bigcup_{(N_L,N_R)\in\mathcal N_t}
\left[
\mathscr W_{\mathrm{loc},L}(N_L)
\times
\mathscr W_{\mathrm{loc},R}(N_R)
\right].
\]

不能把某一个固定 \(N\) 的单足截面与载荷可变化的双足瞬时集合直接比较。

### 15.2 再映射为整机合扳手

令 \(\mathcal T_{G\leftarrow L}\) 和 \(\mathcal T_{G\leftarrow R}\) 分别将左右足扳手变换到共同参考点 \(G\) 和共同坐标系。整机六维瞬时合扳手集合为

\[
\boxed{
\mathcal W_{\Sigma,\mathrm{inst}}(x_t,c_t)
=\left\{
\mathcal T_{G\leftarrow L}\mathbf w_L
+\mathcal T_{G\leftarrow R}\mathbf w_R
\ \middle|
(\mathbf w_L,\mathbf w_R)
\in\mathcal Q_{\mathrm{inst}}^{2S}(x_t,c_t)
\right\}.
}
\]

整体动量方程在双足时约束的是这个合扳手，通常不能唯一分离 \(\mathbf w_L\) 与 \(\mathbf w_R\)。左右压力阵列、完整多刚体动力学和关节力矩范围可以继续收束扳手对集合；若为了控制而用学习器选择其中一个分配，该结果是带先验的估计或决策，不是动力学唯一解。

### 15.3 与四个集合衔接

单足局部集合提供接触物理条件；\(\mathcal Q_{\mathrm{inst}}\) 或其合扳手映射 \(\mathcal W_{\Sigma,\mathrm{inst}}\) 再加入当前整机状态、驱动和运动学条件。随后才在明确的时间范围 \(H\) 内研究

\[
\mathcal W_{\mathrm{att}}^H,
\qquad
\mathcal W_{\mathrm{viab}}^H,
\qquad
\mathcal W_{\mathrm{task}}^H.
\]

它们分别表达未来可达到性、安全可维持性和任务需要。只有当这些集合采用相同时间范围、参考点、坐标系和变量定义时才能直接求交或讨论包含关系；若其中一个表示扳手轨迹、另一个只表示瞬时扳手，则必须先投影到同一空间。

未来输出的任务动作、接触调整动作和必要微小激励可以改变 \(c_t\)、\(\mathcal I_t\)、\(\mathbf n\)、\(N\) 或参数信息量，从而获得更有利的后续局部集合。当前瞬时集合固定当前接触条件；跨模式改变接触并比较未来集合属于后续时域决策问题。这样接触既是当前求解的条件，也是动作可以改变的载体，但两个时间层不能混写。

## 16. 参考文献

- Stéphane Caron, Quang-Cuong Pham, and Yoshihiko Nakamura. *Stability of Surface Contacts for Humanoid Robots*. 用于 CWC、面接触稳定边界及离散点力表示。
- Stéphane Caron, Quang-Cuong Pham, and Yoshihiko Nakamura. *Leveraging Cone Double Description for Multi-contact Stability of Humanoids with Applications to Statics and Dynamics*. 用于从局部接触扳手约束进入重力--惯性稳定条件。
- Ander Vallinas Prieto, Arvid Q. L. Keemink, Edwin H. F. van Asseldonk, and Herman van der Kooij. *Feasible Wrench Set Computation for Legged Robots*. 用于区分当前反力、局部接触能力和加入整机驱动约束后的可行扳手集合。
- Michele Orsolino et al. *The Actuation-consistent Wrench Polytope and the Feasible Wrench Polytope*. 用于后续整机驱动能力与接触能力的组合。
