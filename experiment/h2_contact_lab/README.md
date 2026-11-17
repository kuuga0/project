# H2 Plus：任务配置、训练与仿真采集

更新日期：2026-11-02。环境参数已迁入模块对应YAML，通过OmegaConf合并、Hydra构造原生Isaac Lab配置对象；任务仍由原生管理器和Gym注册管理。`env_cfg.py` 从805行收敛为约60行，删除逐项奖励／域随机化解析函数。本轮按用户9000轮从零训练的决定调整课程与奖励，已修复YAML向量类型恢复问题，30项离线测试通过，补齐管理器属性式配置分组；独立USD属性写入复现与修正检查通过，未启动Isaac Sim、训练或硬件。

## 2026-11-12肩pitch实际回放诊断

最新checkpoint为训练目录 `2026-11-12_02-54-37/model_42199.pt`；最近一次 `output/play/2026-11-12_17-08-51/exported/policy.pt` 的actor权重逐项匹配同目录 `model_38000.pt`，不是最新checkpoint。分别以当前Play配置、单环境、固定前进0.5 m/s回放20秒，排除前2秒后统计如下；两次均无回合重置，未启动优化或训练，也未修改任务参数。

| 策略 | 左肩pitch实际范围／峰峰值 | 右肩pitch实际范围／峰峰值 |
|---|---|---|
| 最新42199 | −150.0°至−96.1°／53.9° | −150.0°至−83.1°／66.9° |
| 最近回放38000 | −150.0°至−90.1°／59.9° | −150.0°至−85.8°／64.2° |

角度以URDF关节零位为参考，不是手臂相对躯干的视觉夹角。肩pitch物理限位约[−150°,105°]，最新策略左右原始目标分别最低约−222.5°／−220.3°，58.3%／53.1%的采样目标低于物理下限。因此不只是摆幅大，还存在显著负向偏置和反复向限位外请求角度；实际略低于−150°的小量误差属于求解容差。

当前配置已从arm_posture排除pitch，而arm_leg_coordination的肩pitch约±17°只是正奖励的软参考，不是角度上限；没有肩pitch专用的摆幅／速度惩罚或目标限位裁剪。这允许策略牺牲协调奖励以采用大幅动作，但尚未通过消融证明各奖励的独立因果作用。本轮仅诊断，不自动调整参数。结果仅覆盖上述单环境直行短回放，不能代替多速度／转向验证。

证据：[曲线](output/diagnostics/shoulder_pitch_20261112/pitch_comparison.png)、[统计及运行条件](output/diagnostics/shoulder_pitch_20261112/result.json)、[最新策略轨迹](output/diagnostics/shoulder_pitch_20261112/latest_checkpoint.csv)、[最近回放轨迹](output/diagnostics/shoulder_pitch_20261112/latest_viewed_export.csv)。

## 2026-11-12身体稳定、前后摆臂与肘屈伸分工（当前38项）

用户报告全身左右摆动加大、肘部仍几乎不动，希望身体稳定而手臂参与行走。代码核查发现此前只约束骨盆／躯干roll及角速度，未直接约束左右平移；arm_posture仍选择全部肩关节，包括需要前后摆臂的pitch。不能仅据用户视觉观察或离线单点输出认定肘部锁死。本次不再简单叠加roll权重，新增如下分工：

- body_lateral_position（−0.5）：骨盆和torso_link相对双踝原点中点的侧向偏移，按骨盆偏航朝向投影。2.5 cm内此新增项为0，超出后以0.08 m尺度平方处罚；不固定世界位置，前进和共同平移不改变此相对位置指标。
- body_lateral_velocity（−0.4）：同两body的link线速度转入骨盆偏航系后，比较侧向分量与指令vy。0.05 m/s内此项为0，以0.3 m/s尺度处罚多余侧向速度。两项在偏航指令绝对值达到0.1 rad/s时关闭，保留转向调整；现有roll幅度／角速度项仍按此前设置。
- arm_posture只保留肩roll／yaw，移除肩pitch零位偏差处罚；保留肩roll外摆约束，不用肩roll摆动替代前后摆臂。肘部仍不设置固定零位姿态惩罚。
- 新增arm_leg_coordination（+0.25）：把左右脚在骨盆偏航系的前后位置差，经tanh归一化构造步态相关软参考，不使用固定时钟。前迈同侧脚对应肩pitch向后，另一侧肩向前；肘参考随之反向交替。肩pitch／肘参考最大幅度分别0.3／0.2 rad（约17°／11.5°），std=0.25 rad，仅有运动指令和非零前后步差时给奖励。该项是手工设计的协调启发式，不是专家动作模仿或自然步态已验证的证据；参考只进入奖励，不直接驱动关节或改变初始角。
- elbow_motion由+0.04提高至+0.08，elbow_inactivity由−0.1提高至−0.2，保留1.5秒窗口／0.15 rad活动范围条件；新增elbow_target_limits（−0.1，std=0.35 rad）及URDF物理范围[−0.986,3.071] rad目标裁剪，避免无效超限目标被物理限位掩盖。

本轮读取最新2026-11-12_02-44-25导出策略做了单组合成零位输入探测，肘目标约0.614／−0.429 rad，在物理范围内；这不是承重回放轨迹，不能排除运行时顶限位。最近训练末100轮elbow_motion原生日志均值约0.000829、elbow_inactivity约−0.00729，仍不足以从聚合指标确认每个肘的实际运动范围。本次没有执行仿真策略回放或启动训练。URDF离线FK核对显示左右肩pitch增加0.2 rad时，同侧elbow_link向后移动约5.40 cm，支持本模型前后摆臂参考的符号选择。

数值仍在agents/rsl_rl_locomotion_rewards.yaml，绑定在env_cfg.yaml，MDP函数在mdp/rewards.py。默认零位、23维动作顺序和scale、地形配置（保留用户撤销后的plane）均保持；只增加了肘的物理限位裁剪。53项离线测试通过，新增覆盖身体侧移与roll区别、小幅平衡／前进／转向和横向指令、前后摆臂参考与静止手臂区别、停止／无前后步差不奖励，以及新增裁剪匹配URDF。没有把身体硬固定，也不据此承诺效果；用户重启并续训后，应一起观察速度跟踪、侧移、躯干roll与实际肩肘轨迹。

## 2026-11-12限制骨盆横滚（当前34项）

用户描述行走有扭胯感，明确要求限制骨盆横滚。本次增加pelvis_roll（weight=−2.0，roll_weight=1、pitch_weight=0）和pelvis_roll_velocity（weight=−0.1），SceneEntity均明确选择pelvis。复用现有投影重力倾斜函数和偏航朝向系横滚角速度函数，没有新增解析器或修改机器人关节。

pelvis_roll只增加骨盆左右侧倾分量处罚；与原flat_orientation=−0.5叠加，纯横滚幅度的总惩罚系数约由0.5提高到2.5（5倍），前后俯仰仍由原项处理。横滚角速度项抑制快速反复左右摆动，不惩罚纯世界竖直偏航转向。这里是对骨盆世界姿态的软约束，不是将腰roll关节固定为0，也不限制为某个硬角度；小幅调整仍可进行。用户所说“扭胯”没有新增运动测量，若实际问题包含水平偏航扭转，本次不能据此宣称也已解决。

数值在agents/rsl_rl_locomotion_rewards.yaml，绑定在env_cfg.yaml。51项离线测试通过，新增验证骨盆倾斜而torso_link保持直立时仍受罚，以及纯俯仰／偏航不计入新增横滚项。本轮未启动仿真／训练，没有更改默认零位、动作映射或URDF／USD；重启并续训后评估实际效果。

## 2026-11-12Play窗口启动失败记录

用户报X11 BadWindow（X_SendEvent，无效窗口句柄）。对应log/kit/play_1794416951355879085.log在图形应用启动后约14秒即进入插件清理，尚未记录任务环境构造／checkpoint加载；Replicator类别移除警告出现在清理阶段，未发现更早的Python异常。当前证据指向窗口／显示会话问题，不能据此归因于奖励或URDF配置；具体窗口失效触发原因尚未定位。可先在当前桌面会话新开终端重试；远程桌面断开／重连后应重新建立有效的图形启动会话，不盲目覆盖DISPLAY。无窗口运行可在原命令加--headless作为区分诊断；需要观察时可使用--headless --video导出视频。此项尚未运行复现或确认修复，本次未启动仿真／训练、未更改系统显示设置。

## 2026-11-12收敛肩roll摆幅并加强躯干左右稳定（当前32项）

用户报告已有行走能力，但肩关节横滚摆幅过大、躯干左右摇晃明显，要求调整后自行续训。本次只改配置与MDP奖励，并做离线验证，没有启动仿真／训练，不将用户观察写成助手的新测量结果。

| 项目 | 当前设置 |
|---|---|
| torso_orientation.params.roll_weight | 3→6，现有左右侧倾代价再提高2倍；pitch_weight仍1、总weight仍−1 |
| torso_roll_velocity.weight | −0.05→−0.15，左右角摆动速度惩罚提高3倍 |
| 新增shoulder_roll_posture | weight −0.3，左右肩roll相对关节零位±10°内新增幅度项为0，超出后以std=0.35 rad归一化平方处罚、左右平均 |
| 新增shoulder_roll_velocity | weight −0.03，使用原生joint_vel_l2，仅选择两侧shoulder_roll_joint |
| 新增shoulder_roll_target_limits | weight −0.1、std=0.35 rad；使用已有裁剪前目标检查，惩罚超出物理限位的肩roll请求 |

shoulders_elbows动作组新增原生clip，仅按URDF物理限位裁剪肩roll：左[−0.517,2.494]、右[−2.494,0.517] rad。±10°是可调软惩罚容差，不是缩小物理行程的硬限制；原有arm_posture仍另行生效。肩pitch／yaw、肘部未纳入新增肩roll项，允许前后摆臂、肘运动和正常小幅平衡调整。动作scale、23维通道排列、初始零位、躯干上下起伏惩罚−0.03及PPO配置保持。未改URDF或USD。

权重与容差在agents/rsl_rl_locomotion_rewards.yaml，绑定在env_cfg.yaml，新增通用死区姿态函数h2_plus_joint_reference_deadband_l2在mdp/rewards.py。50项离线测试通过，覆盖±10°免罚／超出后增长、仅选择shoulder roll、clip与URDF一致，以及加权躯干roll／pitch／yaw区分。本次是调参起点，需用户重启并加载checkpoint继续训练；新增裁剪可能改变旧策略的部分执行目标，不能仅用旧权重回放宣称奖励已经生效或步态改善。

## 2026-11-06足端朝向约束（当前29项）

按用户要求新增foot_heading惩罚，weight=−0.3，抑制内八字、外八字以及两脚平行但同时偏向一侧。分别比较左右ankle_pitch_link水平+x朝向与骨盆根体水平朝向，不使用双足互相平行作为唯一条件，也不固定世界朝向。相对偏航角wrap到[−pi,pi]；默认允许5度小幅偏差，超出后的角度误差除以std=0.35 rad再平方、左右平均。数值在agents/rsl_rl_locomotion_rewards.yaml，绑定在env_cfg.yaml，实现为mdp/rewards.py中的h2_plus_foot_heading_l2。

站立和摆动足均参与，不增加传感器或观察维度；整体转向且双脚随身体同向旋转时不罚。原地／行走转向中的暂时足身偏航差仍可能产生软惩罚，需训练验证，不声称硬性禁止所有偏差。只比较水平朝向，足部roll／pitch仍由原足部姿态项处理。当前URDF零位正运动学复核表明左右足坐标轴都与骨盆同向，局部+x可用作前向轴，无需额外yaw偏置；未修改URDF、USD或默认零位。

49项离线测试通过，新增覆盖内／外八字等价处罚、整体转向不罚、正负180度跨界、两脚同向歪转和纯roll／pitch区分。本轮未启动仿真或训练，参数需重启并训练更新后评估效果。

## 2026-11-05加强躯干左右摇晃惩罚（当前28项）

按用户要求，针对torso_link左右侧倾和角摆动分别增强约束。torso_orientation总权重仍−1.0，奖励YAML新增params.roll_weight=3.0、pitch_weight=1.0；函数分别加权局部投影重力y／x分量的平方，因此纯左右侧倾的代价为原来的3倍，纯前后倾代价保持，不限制偏航朝向。投影重力分量是倾斜代理，不是直接欧拉角平方。

另新增torso_roll_velocity，weight=−0.05，惩罚torso_link世界角速度转入其偏航朝向坐标系后的前向轴分量平方，抑制左右摇摆速度。使用仅偏航旋转的坐标系，避免躯干前倾时把纯世界竖直转向误计为横滚；这不是躯干横向平移速度惩罚。原上下起伏项仍−0.03，原其他奖励和动作配置保持。数值仍在agents/rsl_rl_locomotion_rewards.yaml，绑定在env_cfg.yaml，实现位于mdp/rewards.py。

48项离线测试通过，新增验证30度纯roll的原始代价0.75、同角度纯pitch代价0.25、纯yaw不计侧倾，及不同朝向／前倾状态下横滚、俯仰和纯世界竖直转向的角速度区分。测试使用真实SDK的yaw_quat函数源码和合成姿态，不是新步态运行证据。本轮未启动仿真／训练，重启并训练更新后再判断左右摇晃是否改善。

## 2026-11-05肘运动、步宽、滑移与踝roll边界调整（当前27项）

用户报告肘部几乎不动、双腿张开过大，要求进一步减弱躯干上下起伏惩罚，鼓励肘运动、足部腾空与摆动足净空，强化接触足滑移惩罚，并仅在踝roll靠近限位时约束。本次按此修改共用任务配置，数值仍在agents/rsl_rl_locomotion_rewards.yaml，绑定在env_cfg.yaml，实现放在mdp/rewards.py。

| 项目 | 之前 | 当前 |
|---|---:|---|
| torso_vertical_velocity | −0.1 | −0.03 |
| elbow_motion | +0.01，speed_scale 0.6 | +0.04，speed_scale 0.8 rad/s（峰值0.9 rad/s） |
| 新增elbow_inactivity | 无 | −0.1；运动指令下1.5秒内各肘实际角度范围不足0.15 rad（约8.6°）时罚，站立／重置／新运动窗口预热时不罚 |
| feet_slide | −0.05 | −0.2，仅当前有接触且净力超过1 N的足参与，取消历史残留力对刚离地摆动足的处罚 |
| feet_air_time | +0.2 | +0.4；单足支撑、移动／转向门控和0.4秒上限保留 |
| swing_clearance | −0.1误差惩罚 | +0.2正奖励；单足支撑时按exp(−相对踝高度误差²/0.05²)奖励接近0.06 m的摆动足，水平速度按0.3 m/s封顶归一化；无指令／摆动足不动／双支撑／腾空时不奖励 |
| 新增stance_width | 无 | −0.3；双踝link原点在骨盆偏航系的有符号侧向间距0.10—0.25 m内不罚，超宽、过窄或交叉按误差平方罚，尺度0.1 m |
| 新增ankle_roll_limits | 无 | −0.5；实际踝roll仅在两端各15%行程进入二次惩罚带，中间70%行程严格为0，达到物理边界时单关节归一化代价为1 |
| 新增ankle_roll_target_limits | 无 | −0.1；用既有裁剪前目标惩罚实现，margin_ratio=0.15、std=0.1 rad，正常区间目标不罚；另用原生clip限制目标到URDF物理范围 |

踝roll原物理限位：左约[−20°,17°]、右[−17°,20°]。本次正常免罚区约左[−14.45°,11.45°]、右[−11.45°,14.45°]；区间内没有新增踝roll零角参考约束，不限制正常调整。原骨盆／躯干直立、足部放平等其他奖励仍分别生效。近限位惩罚针对实际角度，目标惩罚针对裁剪前请求，避免物理限位挡住后仍持续给出无效大目标；两者正常区间均无惩罚。物理clip也不缩小URDF真实行程。

原feet_lateral_symmetry只检查双足是否关于骨盆左右居中，双脚对称地向两侧张开仍可能得0，不能替代步宽项。新步宽测量不是足底外缘或脚掌中心间距；当前URDF零位离线FK双踝原点间距约0.15724 m，在免罚区内。0.10—0.25 m是本次调参起点，并非验证过的最优步宽。转向／特殊地形也继承该软惩罚，效果需分别验证。

膝部与肘部窗口少动项共用H2PlusJointInactivity，替换原膝专用类名；不恢复固定屈膝／屈肘参考姿态。默认零位、23维动作排列和各关节scale保持，未改PPO预算、课程或URDF／USD。重启任务后读取新配置；现有checkpoint可加载，但动作裁剪和奖励发生变化，需要训练更新后判断步态，不能把旧权重回放当成奖励已经起效。

合理性与限制：肘活动与腾空只是辅助目标，仍可能诱导无意义摆动、过长单支撑或过分强调抬脚；各项先采用有界奖励和已有门控，并保留速度跟踪与平滑性项。净空仍是摆动足相对支撑足踝原点高度代理，不是任意地形上的真实足底净空；足滑项仍是接触期间足部水平速度代理，不能替代接触点滑动实测。

47项离线测试通过：膝／肘活动窗口及重置、原生踝roll目标裁剪与URDF限位、实际／请求进入边界带才罚、正常区间0代价、对称宽站姿与交叉姿态、当前接触滑移门控、正净空奖励及站立／支撑／腾空门控。本轮没有启动仿真或训练，未据离线检查宣称新步态改善。

## 2026-11-05已确认膝部顶限位并修复超限目标（当前23项）

用户要求继续排查后，使用最新回放导出policy（权重已核对与2026-11-05_02-04-16/model_11599.pt一致），在真实平地接触／正常重力下做了两次单环境、确定性20秒短回放，指令固定[0.5,0,0]、seed 42，不执行优化器或训练。没有改动旧checkpoint。仅为诊断直接加载现有USD，保留任务执行器；场景传感器记录默认开关未改。

修复前双膝600个控制步的目标角100%低于下限：左约−4.198至−0.713 rad，右约−12.918至−0.615 rad；真实膝角持续在−0.08725 rad附近，20秒活动范围左约0.000184、右0.000434 rad，符合“看似锁死”的表现。原始网络输出没有被限制到[−1,1]，先前单纯提高scale放大了错误方向的目标。执行器平均绝对力矩估计左281.59、右357.34 N m，右侧大量时间达到360 N m上限。这里是隐式执行器PD估计，不是关节反力传感器测量。

实际修改：

- env_cfg.yaml的actions.legs增加原生clip，仅将双膝缩放后目标裁剪到当前URDF允许的[−0.08725,2.53025] rad；初始角仍0、膝scale仍0.5、其余动作通道顺序不变。该值是物理关节限位，未规定固定屈膝角，也没有把网络原始动作裁剪掉。
- 新增knee_target_limits，weight=−0.2、std=0.5 rad，参数仍在agents/rsl_rl_locomotion_rewards.yaml。实现H2PlusJointTargetLimitPenalty按原生ManagerTermBase注册，读取原生JointAction的裁剪前目标（raw_actions×scale＋offset），对超出运行时软关节限位的距离计算左右平均log1p((超限距离/std)²)。有效目标得0，越远惩罚越大；对数增长避免极端旧动作造成过大二次惩罚。
- 原joint_pos_limits检查的是实际角度，在物理限位挡住膝盖时几乎不罚，不能替代新项。仅裁剪而不惩罚原始超限请求，也不会让策略自动学会使用有效动作区间。

同一个冻结旧策略的修复后20秒回放中，送入执行器的膝目标超限比例降到0（比较容差1e−6 rad），平均绝对力矩估计左0.63、右2.93 N m；两次均无回合重置，平均前向速度分别约0.4786／0.4788 m/s。裁剪后仍主要给伸直端目标；5秒后左右实际活动范围仅约0.0224／0.1228 rad，未形成目标步态，不能把这个结果当作协调屈膝已解决。训练需利用新惩罚改变旧策略输出，运行旧checkpoint只会应用裁剪。

原始轨迹与统计保存在output/diagnostics/knee_gait_20261105/{baseline,clipped}/trajectory.csv与result.json；comparison.json记录对比，[comparison.png](output/diagnostics/knee_gait_20261105/comparison.png)显示目标／实际角／力矩估计。每次20秒单环境诊断不能代替多指令／多回合验证。

44项离线测试通过，覆盖原生动作裁剪、保留raw_actions、限位与URDF一致，以及裁剪前目标惩罚的关节映射／默认偏置／有效目标／超限单调性；真实回放也完成了新管理器项构造和调用。本次没有启动或恢复训练，参数和日志快照仍按既有目录约定管理，课程进度不恢复的续训限制仍在。

## 2026-11-05膝关节是否锁死：直接响应测试与策略线索

用户报告膝部仍少动，要求确认是否锁住。复核主URDF、当前USD和11:43:42回放日志：双膝为启用的PhysicsRevoluteJoint，轴Y，限位约−0.08725至2.53025 rad（USD角度约−5°至145°），没有固定／零宽限位。最新回放确实绑定双膝动作和80／8执行器；USD转换产物的drive刚度／阻尼为0是资产导入target_type=none设置，运行时执行器会写入80／8，并非运行时没有驱动。

随后进行了实际PhysX诊断：只在临时内存场景加载一个当前USD机器人，关闭重力、无地面，不创建RL环境、不执行策略／优化训练；保留任务的80／8执行器，直接依次给双膝0、0.3、0.6、0 rad目标，每段120个物理步、dt=1/120 s，共480步。0.3 rad段末实际左／右为0.298668／0.298640 rad；0.6 rad段末为0.598753／0.598727 rad；回零段末为0.002825／0.002873 rad。证明当前模型双膝在无负载条件下可以转动和回零，不能据此证明落地承重时也能跟踪正常。

记录在output/diagnostics/knee_response_20261105/result.json与trajectory.csv，日志在log/knee_response_diagnostic.log。本次诊断未改主资产或训练参数。完成480步后应用关闭阶段等待过久，结果已落盘，随后仅终止本次诊断进程；没有停止用户进程或启动训练。

进一步离线检查确认最新回放导出actor与2026-11-05_02-04-16/model_11599.pt逐权重相同。给该策略一个明确标注的合成零位输入（关节与根速度0、投影重力[0,0,−1]、指令[0.5,0,0]、上步动作0），双膝原始输出约−1.703／−1.230，经0.5缩放得到目标−0.852／−0.615 rad，明显低于实际下限−0.08725 rad。如果真实回放中也持续给这类负目标，膝盖会顶在伸直端，看起来像锁死；这只是策略线索，不是整段回放实际目标轨迹的测量。policy_probe.json保存了输入说明与结果。

该checkpoint保存的训练环境已包含knee_inactivity=−0.3、knee_motion=+0.06和膝scale=0.5，因此不能简单归因于“尚未训练过新配置”。最后500轮knee_motion均值约0.0000723、knee_inactivity约−0.2346（原生回合积分／20秒口径），仍反映膝部活动目标未满足。下一步应核对真实承重回放中的膝目标角／实际角／力矩和超限目标，不宜仅凭视觉继续加大缩放或惩罚。当前joint_pos_limits惩罚实际角越限，不能直接惩罚策略持续请求不可达目标。

## 2026-11-05减弱上下起伏惩罚并加强膝部活动（当前配置）

用户报告策略已能走，但膝部仍少动、呈企鹅式步态，授权降低躯干上下运动惩罚、加强膝盖不动的惩罚并允许更大转动幅度。本次是根据用户观察进行的新参数试验，不把旧日志或离线测试当作步态改善证据。

| 参数 | 之前 | 当前 |
|---|---:|---:|
| torso_vertical_velocity.weight | −0.5 | −0.1，允许适量躯干竖直运动 |
| knee_motion.weight | +0.03 | +0.06 |
| knee_motion.params.speed_scale | 1.0 | 1.5 rad/s，运动奖励峰值从1.1移至1.6 rad/s |
| 新增knee_inactivity.weight | 无 | −0.3，近1秒膝关节活动不足惩罚 |
| actions.legs.scale中的膝关节 | 0.25 | 0.5；髋／踝仍0.25 |

knee_inactivity按原生ManagerTermBase实现，由RewardManager调用和逐环境重置，配置绑定在env_cfg.yaml，数值在agents/rsl_rl_locomotion_rewards.yaml。移动／转向命令有效且连续约1秒历史填满后，分别计算左右膝实际角度的max−min；每侧活动范围小于0.25 rad（约14.3°）时，返回归一化缺口的平方，左右平均后乘−0.3。双膝完全不动时惩罚最大；只有一侧不动且另一侧达标时约为一半。站立命令／刚重置／重新开始运动的预热期不罚，不因瞬时换向或短时支撑不动就立即处罚。保持固定弯曲角同样属于少动，不设置固定屈膝目标。窗口是近期活动范围，仍不能保证运动对推进有用，可能诱导无意义屈伸，需结合足净空、滑动、速度跟踪和动作平滑性评估；目前保留原平滑性项。

膝动作比例在env_cfg.yaml采用原生按关节正则的scale字典，没有拆分动作term或重排23维动作；同样的原始输出1对应膝目标角0.5 rad（约28.6°），此前0.25 rad（约14.3°），这不是硬限幅。默认关节角仍全0、根高度仍1.02 m，URDF原膝限位约−0.08725至2.53025 rad、执行器80／8均保持。修改不涉及几何或USD，用户重启任务后配置才生效。

43项离线测试通过，新增用原生ManagerTermBase／RewardManager reset验证活动窗口、固定弯曲仍算不动、双侧／单侧活动、站立／转向门控和选择性重置，以及原生JointAction的按关节缩放与输出排列。未启动训练／回放。22项奖励／惩罚在当前共用训练／Play配置中生效；权重尚未经过新训练验证。

可以加载原checkpoint继续优化网络，但膝关节动作比例改变后，同一旧网络输出的物理目标角会翻倍，续训初期可能退化，不能视为完全保持原控制行为的恢复；模型维度和通道顺序保持。当前课程进度仍不随checkpoint恢复，之前的续训限制仍适用。9000轮预算、保存间隔、PPO超参数与传感器采集设置未改。

## 2026-11-05USD已同步重建

用户追问USD后，已用当前assets/h2_plus.yaml的robot.spawn配置通过原生UrdfConverter重新生成output/h2_plus_usd/H2_Plus_sensorized.usd及configuration子层，包含基座、物理、机器人和传感器层；未手改USD内容。使用无窗口Kit应用仅做资产转换，没有创建RL环境、启动时间线、物理步进或训练。转换日志在log/convert_footrix_usd.log与log/kit/convert_footrix_*.log。

独立USD读取检查通过：32个RigidBody、31个RevoluteJoint、308个命名visual圆形标记，标记及其子几何没有RigidBody／Mass／Collision API，32个本体mass与新URDF逐项一致。结果保存为output/h2_plus_usd/region_conversion_check.json。这是USD结构和质量属性检查，不等于动力学或传感器实测通过。若查看器已打开旧资产，需要重新加载该USD。

## 2026-11-05FOOTRIX严格小圆域采集（当前实现）

按用户要求，删除原12 mm支持半径、归一化线性扩散和最近点兜底。每个单元仅累计其圆形范围内的PhysX接触报告冲量；圆外、单元间隙和足底平面以外的接触都不分配给单元。单元半径cell_radius_m=0.003401643 m，足底平面layout_z_m=−0.0521 m，平面容差surface_tolerance_m=0.002 m；可在sensor_cfg.yaml调整。半径和布局仍须按硬件标定，代码不声称这些图像推导尺寸等于实物尺寸。初始化拒绝非有限／非正半径、非法坐标和重叠／相切的圆域，避免一个接触被重复计入多个单元。

每物理子步结束后，RecorderManager回调只读取原始PhysX link变换，将该子步接触点转入当时的足部坐标，避免用控制步末尾姿态错配毫米级圆域；这里不读取尚未更新的IMU／接触传感器缓存。完整传感器快照与回合管理仍在record_post_step执行。一个保存样本的FOOTRIX力等于对应控制周期选中冲量之和除以控制dt，采样降频不改变这一平均时间窗。

HDF5的footrix/<left|right>/force_w_N仍为154×3世界系力，但语义改变：只包括圆内接触，不要求单元力之和等于整足聚合力。新增cell_contact_count、selected_contact_count、unmeasured_contact_count及unmeasured_force_w_N；source_contact_count保留。in_contact表示至少有一个报告接触点落入该单元且力有限，不再使用旧1 N阈值，避免微小受力被误标为无接触。对同一批有效报告冲量，单元力求和＋未测量力等于报告总力；它不保证与独立接触传感器在不同缓存／统计口径下逐帧完全一致。metadata.schema_version升级为2，并写入有效单元坐标、半径、容差、区域规则及时间窗；旧schema 1文件保留原含义，不自动转换。

主Isaac资产H2_Plus_with_footrix_no_dex_hands.urdf已把308个有占位质量／惯量的固定标记link转换成左右足link内各154个带名称的纯visual，坐标与测量布局对齐。主资产现在是32 link、31 revolute，无标记碰撞、附加质量或固定关节；原32个本体惯性、原足部碰撞、活动关节定义保持，去掉了此前共3.08 g占位质量。真实FOOTRIX硬件质量是否已包含在足部CAD惯性中仍需按装配核对，未填入假设值。旧MuJoCo源资产未改，不作为本次Isaac任务入口。force_usd_conversion仍为true；USD已按上方后续记录单独重建，下次任务启动也会重新转换。

实现边界：这是对离散PhysX接触点的严格几何筛选，不是154个独立圆面参与接触求解，也不是连续压力场积分。刚体接触点可能稀疏／约简，因此即使足底贴地，也不能据本实现保证每个真实受力单元都有非零读数。没有为补齐读数扩大范围或重新分配圆外力；需要更真实的局部压力时，另行设计接触几何或柔性传感层。

采集默认关闭，仍显式添加--record_sensors，输出output/sensors/。本次41项离线测试通过，覆盖圆内独占归属、间隙与侧面排除、未测量力、子步脚移动／旋转坐标、重置／关闭、HDF5与schema版本，以及纯视觉标记／本体结构；没有训练或GPU运行效果验证。另修复通用config_data读取标量时错误调用OmegaConf.to_container的问题，保证sensor_cfg.py可读取IMU周期等标量。

## 2026-11-04接触足水平姿态奖励（当前合计21项）

新增contact_foot_flat，权重+0.1，参数在agents/rsl_rl_locomotion_rewards.yaml，SceneEntity绑定在env_cfg.yaml，函数h2_plus_contact_foot_flat_reward在mdp/rewards.py。左右足按明确顺序绑定left／right_ankle_pitch_link，与两路接触传感器对应。

使用当前帧足部净接触合力超过1 N、连续接触超过0.05秒判定参与。足部局部+z作为足底法向，按其姿态将世界单位重力转入局部坐标，得分为exp(−水平投影平方和/0.25²)，另排除倒置足。水平足得分1，倾斜越大得分越低，不限制yaw。按符合条件的接触足取平均，因此双支撑不会直接得到单支撑两倍奖励；无接触为0，摆动足不参与。短时落脚缓冲降低对落脚瞬间的影响，但没有排除蹬地阶段。

合理性与限制：鼓励平地支撑足放平，减少仅靠脚尖／脚侧支撑的倾向；足部姿态水平不能证明完整足底接触，也不直接测量接触面积、压力均匀或地面法向。斜坡上水平不等于贴地，可变形面也未必适用；目前共用任务基线继承，特殊地形需单独调权或关闭。过高权重可能抑制正常脚跟着地／脚尖蹬地，或诱导静止支撑，故先用弱奖励；接触足平均只避免直接奖励双支撑数量，并不能排除静止获益。没有启用额外传感器、FOOTRIX记录或高度扫描，初始零位和膝肘自由运动约定保持。

38项离线测试通过，覆盖水平／偏航／侧倾／倒置、单／双支撑归一化、摆动脚忽略、腾空、新接触与力阈值判定。未启动训练／回放，奖励效果尚待实测。

## 2026-11-04新增上下起伏惩罚与膝肘运动奖励

按用户要求新增3项，当前合计20项。数值在agents/rsl_rl_locomotion_rewards.yaml，绑定在env_cfg.yaml，实现位于mdp/rewards.py。默认关节角保持全0、根高度1.02 m；不恢复膝肘固定角度／姿态偏差惩罚。

- torso_vertical_velocity，权重−0.5：torso_link世界系竖直线速度平方，直接抑制上身上下运动；不设固定世界高度，不惩罚静态高度偏差。它是上下起伏速率的代理，不直接约束峰峰幅度，也不能区分合理上下台阶和不必要弹跳；过强会压制正常步态起伏，各地形继承后需分别验证。
- knee_motion，权重+0.03；elbow_motion，权重+0.01：使用所选双关节的实际角速度，而非动作变化或偏离零位。对每关节定义x=[max(|速度|−deadband,0)/speed_scale]²，数值上截到100，得分x·exp(1−x)，再对左右关节取均值。deadband=0.1 rad/s，膝／肘speed_scale=1.0／0.6 rad/s，因此峰值速度分别1.1／0.7 rad/s；这是奖励尺度，不是硬速度限制，角速度正负同等对待。得分范围0—1，超过峰值后衰减。
- 仅平面命令超过0.05 m/s或转向命令超过0.1 rad/s时启用运动奖励；乘以实际机体系平面速度和偏航速度跟踪的指数门控，尺度分别0.5 m/s与0.5 rad/s，零命令不奖励，跟踪差时衰减。没有要求恒定关节角、特定方向或相位。权重小于速度跟踪项，保留动作变化与加速度惩罚。

合理性：躯干竖直速度惩罚与“减少踉跄弹跳”目标较直接；膝肘运动奖励只适合作为弱探索引导，“动起来”并不等于有效屈膝抬脚或合理摆臂。即使加死区、过速衰减、命令及跟踪门控，也仍可能出现小幅高频摆动，或在跟踪尚可时通过无意义运动获益；现有平滑性惩罚只是制衡，不保证排除。最终需比较有／无运动奖励的回放、关节运动幅度与频率、足部净空／滑动、躯干起伏和任务跟踪；肘部保持较静也可能是合理解。不能把新增项作为执行器正常工作的证据。改配置不改变旧checkpoint输出，需训练更新。

本次37项离线测试通过，检查竖直正负速度均受罚、静止为0、运动奖励死区／双向／峰值／过速衰减／零命令门控／差跟踪衰减／原地转向及有界性。未启动训练或Play，权重合理性尚待实测。

## 2026-11-04姿态与步态约束（零位与关节选择约定）

按用户最新要求，资产初始关节角全部为0（URDF关节坐标零位，不代表视觉上所有连杆伸直），根高度恢复为1.02 m。没有修改URDF。此前屈膝屈肘初始姿态及髋踝／根高度补偿已取消。

移除knee_posture奖励定义与权重；arm_posture仅选择肩关节，取消肘部姿态参考约束；stand_still也排除膝盖和肘部，避免零指令时仍通过通用姿态项约束它们。膝肘仍参与策略动作、关节限位与平滑性惩罚，不固定其角度。训练reset_joints原有±0.02 rad角度扰动与±0.05 rad/s速度扰动保留，因此零位是默认基准，不保证每次随机重置都精确为零；Play重置扰动为0。

当前共17项奖励／惩罚，原12项基础上保留5项新增约束：torso_orientation（−1.0，直接torso_link直立）、waist_posture（−0.2）、arm_posture（−0.05，仅肩部）、swing_clearance（−0.1，相对支撑足踝原点目标高度0.06 m）、feet_lateral_symmetry（−0.1，直行侧向居中，转向关闭）。原flat_orientation仍约束骨盆，注释已更正。腰／肩参考角0独立写在奖励YAML中；数值在agents/rsl_rl_locomotion_rewards.yaml，绑定在env_cfg.yaml，函数在mdp/rewards.py。

净空是相对踝部高度代理，不是任意地形上的真实足底净空；侧向对称允许前后交替迈步，不强制双腿同步，也不是完整步态时序对称。权重仍是实验起点，尚无新训练效果验证。初始角和动作偏置已恢复原零位基准，先前“必须因非零默认角而从零训练”的原因不再适用；奖励目标已变化，新训练与续训仍需区分。

本次只修改配置、回归检查和文档，没有启动训练／回放。用户手动训练入口不变。

## 2026-11-04最近训练日志分析

膝肘少动的定点排查：训练日志及用户最新Play日志 `log/sim/play/isaaclab_2026-11-04_23-01-03.log` 均确认双膝与双肘已解析到JointPositionAction，且执行器Kp=80、Kd=8，限位可动；23维策略输出中，双膝为零基索引6／7，双肘21／22。当前原生动作是默认角＋scale×策略输出，膝scale=0.25、肘0.2，不是每步累加角度；scale不是硬限幅，配置clip为空。静态配置和初始化日志未发现关节遗漏或零增益，但没有逐步目标角／实际角／力矩轨迹，不能据此证明运行时跟踪正常或认定策略输出接近0。候选解释包括旧奖励允许直腿／少摆臂的运动解、动作变化与加速度惩罚偏好少变化、膝部弯曲需要髋踝共同协调，以及负载下跟踪或视觉角度误判。action_rate只罚相邻动作差，不罚恒定非零关节目标；零位本身也不锁定关节。改奖励不改变已保存策略，需新训练／续训才能影响输出。下一步应对照膝肘每步原始动作、目标角、实际角和力矩，区分策略少动与执行跟踪问题；本次未启动Play或训练。

读取运行 `log/rsl_rl/h2_plus_blind_locomotion/2026-11-02_03-25-29/` 的完整TensorBoard事件与该次params快照：从零训练9000轮（0—8999）、1024环境、每轮24控制步，约2.212亿环境步，事件记录跨度约4.99小时；已保存 `model_8999.pt`。课程约第6000轮到达完整速度范围。此次只离线分析，没有启动训练／回放或修改超参数。

以下为最后1000轮均值：总回报5.171，回合长度505.21步（控制30 Hz，约16.84秒，上限20秒）；time_out历史指标0.8006、bad_orientation历史指标0.1994；加权线速度奖励1.020、偏航奖励0.0817、动作变化惩罚−0.6421、足滑惩罚−0.0131；策略平均噪声标准差1.443，价值损失0.0638，学习率约2.06e−5。后期回报仍缓慢改善，未见标量数值发散；训练完成不等于动作质量合格。

主要判断：存活能力已改善，但转向跟踪仍弱，探索噪声偏大；动作变化项是最大的持续负奖励，其绝对值约为线速度正奖励的63%。奖励侧的偏航权重为0.5，线速度为1.5，当前目标／探索配置值得优先排查。不能仅凭这些聚合曲线断言确定性回放也在抖动、打滑或转圈。

统计边界：原生RewardManager记录的是回合加权积分除以固定20秒，不是未加权瞬时奖励；CommandManager的xy/yaw误差是逐步误差累计除以10秒指令周期，直到回合重置才清零，不能把日志的0.2001／1.7861直接当真实平均m/s／rad/s。TerminationManager使用各环境最近一次结束原因的历史快照，不是独立评估成功率。角速度奖励使用世界系z，命令指标使用机体系z，两者也不完全相同。

建议先用已有8999权重在固定直行、固定转向和零指令下做确定性回放，分别观察跟踪、站立、足滑和躯干姿态；再比较较小的entropy_coef／初始噪声，以及更高的偏航奖励权重。暂不直接减小action_rate惩罚或单纯加轮数；这些是待验证建议，尚未实施。

本次生成的[曲线图](output/analysis/2026-11-02_03-25-29/training_curves.png)显示原值与100轮滑动均值；[标量摘要](output/analysis/2026-11-02_03-25-29/scalar_summary.json)保留各项首尾、极值、最后100／1000轮及每1000轮分段均值。原始日志与参数未改动。

用户随后报告该权重Play时踉跄、上半身歪斜、膝肘看起来缺少弯曲；这是用户回放观察，本次未独立运行验证。定点核查发现：该次保存配置的全部关节默认角为0，动作以默认角为偏置；膝肘均为主动关节，URDF允许弯曲，并非固定或遗漏。当前flat_orientation_l2读取机器人根体pelvis的投影重力，不能直接约束腰部以上torso_link；“躯干倾斜”注释不精确。现有行走奖励缺少持续腰部／上肢姿态约束和摆动足净空项；站立姿态项仅零指令时开启。上述配置与表现相符，但不足以单独证明因果。Play调用确定性inference policy，因此训练探索噪声不能直接解释Play抖动。建议后续优先处理非零屈膝／屈肘参考姿态与协调髋踝姿态、直接躯干直立及轻量腰臂约束，再按回放结果处理足端净空／滑动；改变默认角会同时改变策略观测和动作基准，不能当成兼容旧权重的纯显示调整。本次只记录分析，未改变训练配置。

## 2026-11-02管理器配置分组修复

用户最新启动输出显示场景创建和物理启动已通过，但CommandManager随后在 `self.cfg.debug_vis = False` 抛出 `dict has no attribute debug_vis`。本机ActionManager也执行同样的属性写入，因此仅转换commands会在后续阶段再次出错。

环境YAML现将actions、commands、rewards、terminations、events、curriculum和外层observations统一声明为 `ManagerTermsCfg` 属性式分组，内部仍由Hydra构造原生*TermCfg；scene、ObservationGroup与RecorderManager继续使用原生专用分组类。ManagerTermsCfg只是无预设字段的dataclass容器，不实现管理器逻辑，便于原生Manager解析成员后添加debug_vis等运行属性。没有把debug_vis预先当成一个配置项，避免它被解析成term；字典仍用于原生params等真正的映射字段。

奖励文件合并和数值检查会忽略分组的 `_target_`／`class_path` 元数据，不将其误当奖励项。偏航仍使用直接角速度指令；未启用的ranges.heading设为null，课程也保持heading为空，消除无效heading范围警告，不切换为朝向跟踪任务。

30项离线测试通过。新增测试保留本机CommandManager／ActionManager实际构造器和term解析流程，只替换物理环境底座与term实现，先复现旧dict报错，再验证属性式分组、debug_vis聚合及动作缓冲创建成功；六个训练／Play配置的分组声明也已检查。未启动Isaac Sim或训练，尚未据此宣称完整管理器加载或PPO运行通过。

## 2026-11-02USD颜色类型错误修复

用户启动在 `/World/DomeLight.inputs:color` 报 `expected GfVec3f, got vector<VtValue>`，是本次YAML迁移遗漏原生tuple类型恢复导致的实际配置加载错误。YAML里的 `color: [0.8, 0.8, 0.8]` 数值本身可用，无需改成Python专用YAML标签。

通用加载层新增按原生dataclass字段声明恢复tuple的处理，同时把浮点向量中的整数元素转为float，覆盖颜色、位置、旋转、尺寸等tuple字段；关节名称、环境编号和其他声明为list的字段保留list。支持实际／延迟／Optional元组注解，错误向量长度和非数值分量会提前报错。环境组装与独立资产／传感器／地形构造都使用该处理，train.py和play.py还在Hydra命令行覆盖后、gym.make之前再次归一化，避免覆盖参数重新引入列表。

28项离线测试通过，新增验证原生向量类型恢复、列表保留、覆盖后恢复、向量长度与类型错误。本机USD 0.24库的内存Stage中已直接复现同款列表写入错误，改为float tuple后DomeLight颜色属性可写；此检查只操作内存USD属性，未创建Isaac Sim应用、物理场景或训练进程。其余实际场景初始化仍需用户再次启动验证。

启动命令不变：

```bash
python train.py --task Isaac-H2Plus-Blind-Locomotion-v0 --headless
```

## 2026-11-02从零训练参数调整

用户明确本次不续训、从零开始，并已将PPO改为9000轮、每100轮保存；这两项及现有网络、学习率、驱动增益和域随机化幅度保留。按此前配置评估，实际修改如下，属于新的训练起点，尚无训练效果验证：

| 参数／条件 | 当前设置与理由 |
|---|---|
| 课程跨度 | `environment.curriculum.command_range.params.ramp_steps: 144000`。每轮24步，约第6000轮达到完整指令范围，剩余约3000轮／72000步巩固。 |
| 速度跟踪尺度 | 奖励YAML的线速度std由0.5改为0.25，偏航由0.5改为0.35。降低低速指令下静止也能获得很高跟踪奖励的风险；不是经实验确认的最优值。 |
| 站立指令 | 训练 `commands.base_velocity.rel_standing_envs: 0.05`，约5%的指令明确为站立；Play覆盖为0，继续固定0.5 m/s回放。 |
| 站立惩罚门控 | 平面指令小于0.03 m/s且偏航绝对值小于0.05 rad/s才启用；低速转向不再被误视为站立。 |
| 腾空奖励门控 | 平面指令超过0.05 m/s或偏航绝对值超过0.1 rad/s即可启用，使原地转向也能获得交替支撑奖励；权重0.2和时长上限0.4 s保留。 |
| 非足支撑软惩罚 | 新增 `non_foot_contact`，权重-1.0、接触阈值1 N。复用原生undesired_contacts；身体传感器覆盖除双足外的30个物理body，4帧历史对应一个控制周期。 |

非足接触是平地直立双足任务的软约束；原有骨盆／躯干触地和倾倒硬终止不扩大，也没有添加未经验证的根部高度硬阈值。三种当前行走地形共用此奖励，需要全身接触任务时可单独覆盖或将该项权重设0。身体传感器的记录列数随覆盖范围扩大，读取 `contact/body_contact` 时使用当次metadata的body_names，不再假设只有骨盆和躯干两列；HDF5保存当前力，奖励使用4帧历史判定。

本次没有实现checkpoint课程状态恢复：用户已选择从零训练，原有checkpoint功能保留。课程恢复仍是以后需要续训时单独处理的问题。

26项离线测试通过，新增验证课程与默认训练预算匹配、转向门控、非足传感范围排除双足及原生接触计数惩罚。测试使用合成环境，不等于实际GPU启动成功或训练收敛。手动开始新训练时省略checkpoint参数：

```bash
python train.py --task Isaac-H2Plus-Blind-Locomotion-v0 --headless
```

## YAML配置入口与模块分工

本机Isaac Lab的manager-based例子仍以Python `configclass` 组装环境，算法配置部分使用YAML。本项目保留同一套环境、Manager、MDP和Gym接口，使用已有OmegaConf／Hydra将参数放入YAML；没有另写任务调度系统。

| 修改内容 | 当前配置文件（相对h2_plus_lab） |
|---|---|
| 场景数量、间距、物理步长、回合时长 | `tasks/locomotion/env_cfg.yaml` 的 `environment.scene`、`environment.sim` 等 |
| 动作、观测、指令、终止、随机化事件、课程、传感器记录项 | 同文件的 `environment.actions/observations/commands/terminations/events/curriculum/recorders` |
| 三种地形差异和回放差异 | 同文件的 `terrains.flat/low_friction/deformable`、`play` |
| 接触与IMU挂载、更新周期、外参、FOOTRIX布局参数 | `tasks/locomotion/sensor_cfg.yaml` |
| 地形材料、几何、柔性体及备用崎岖地形 | `tasks/locomotion/terrain_cfg.yaml`；独立区域的放置仍在env YAML的地形覆盖中 |
| URDF路径、导入、初始姿态和关节执行器 | `assets/h2_plus.yaml` |
| 奖励／惩罚权重和数值参数 | `tasks/locomotion/agents/rsl_rl_locomotion_rewards.yaml`，保留原文件名 |
| PPO网络、学习率、训练轮数 | `tasks/locomotion/agents/rsl_rl_ppo_cfg.py`（算法配置，不属于本次环境参数迁移） |

Python职责：`config.py` 只负责通用配置合并／对象构造；`env_cfg.py` 保留六个注册配置类及少量组装代码；`sensor_cfg.py`、`terrain_cfg.py`、`assets/h2_plus.py` 是YAML配置的读取入口；`mdp/` 实现奖励、终止、课程和记录逻辑。原手写 `_apply_rewards_yaml`／`_apply_domain_randomization_yaml` 等逐参数分支已移除，不需要为每次新增参数编写解析函数。未使用的接口聚合类和旧地形描述包装也已收敛为YAML参数与原生对象。

配置顺序为：环境基线＋奖励数值 → 地形覆盖 → Play覆盖 → 原有Hydra命令行覆盖。所有默认路径根据代码位置计算，`${paths.project}`／`${paths.lab}` 由加载器提供；无须export路径。修改YAML后重启任务生效，不会热改正在训练的环境。

`_target_` 用于选择原生配置类，例如 `isaaclab.managers.RewardTermCfg`；`func` 指向Python MDP函数。Scene、ObservationGroup和Recorder配置允许动态增加成员，通过通用 `config_group` 组装为原生对象；其他管理器term分组使用可供属性读写的ManagerTermsCfg容器。物理步进、观测计算、奖励、事件、课程和回合导出仍由Isaac Lab的Manager完成。`checks` 是少量声明式数值约束，例如std必须大于0；它不进入环境配置。

例如在 `env_cfg.yaml` 中调整课程跨度与动作尺度：

```yaml
environment:
  actions:
    legs:
      # 保留同项已有的_target_和其他字段，只修改scale
      scale: 0.25
  curriculum:
    command_range:
      params:
        ramp_steps: 144000
```

关闭某个事件或课程项，直接把对应项设为 `null`；关闭策略噪声用 `environment.observations.policy.enable_corruption: false`。Play默认的确定性覆盖已在 `play` 中明列。**旧奖励YAML中的 `domain_randomization` 段已迁到env YAML的原生events和observations结构，旧enabled/push.enabled路径不再使用；不要再把事件参数写回奖励文件。** 关节position_range仍为rad加性偏移，velocity_range为rad/s偏移。

参数迁移时，三个训练地形和原平地Play的完整参数展开结果与迁移前逐项比对无差异；随后用户将训练改为9000轮，本轮再将课程改为144000步并执行部分奖励建议，当前值见上方从零训练参数调整。23项离线测试覆盖配置合并、类型化组装、原生configclass初始化与六任务选择、Play差异、原生重置／课程、FOOTRIX、输出路径和CLI约定；完整Isaac Sim配置加载及GPU运行仍待用户手动启动验证。

## 2026-11-02固定关节导入调整（32-body结构已确认）

`assets/h2_plus.yaml` 的 `robot.spawn.merge_fixed_joints` 已设为true，308个无碰撞FOOTRIX固定标记应在URDF→USD时合入脚踝父body，视觉几何仍由导入器保留。原始URDF仍为340个link，**导入后的物理body为32个，活动关节为31个**，已由用户最新启动日志 `log/sim/train/isaaclab_2026-11-02_02-58-27.log` 的02:58:39记录确认；用户输出也显示物理启动完成。当前失败已推进到命令管理器初始化，这不等于PPO训练已经通过。

本机UrdfConverter会同时传递固定关节合并和忽略固定link惯性限制的合并选项。现有 `force_usd_conversion=True` 已保留，下次启动自动重新转换 `output/h2_plus_usd/`，无需手动删除文件；本次没有重写既有USD或修改原始URDF。左右脚踝仍由活动关节连接，接触传感器、IMU和点力投影读取这些父body，不依赖标记作为独立物理body。

新增离线结构检查验证：根据当前URDF与合并配置，固定关节折叠后为32个body，31个活动关节没有被合并，308个标记均有visual而无collision，完整足部碰撞保留。总计22项离线测试通过；未执行Isaac导入、物理步进或训练，本次32-body结构和物理启动已由用户日志确认，视觉细节和后续完整训练仍未验收。

用户结束已报错的挂起进程后，可在本目录、已激活的project环境中手动续训：

```bash
python train.py --task Isaac-H2Plus-Blind-Locomotion-v0 --headless --ckpt log/rsl_rl/h2_plus_blind_locomotion/2026-10-29_01-27-52/model_1150.pt
```

启动时检查 `Number of bodies: 32`、`Number of joints: 31`。若仍显示340个body，则新合并配置没有实际反映到导入结果；即使body数正确，也仍需根据这次输出判断是否成功进入PPO。采集仍由 `--record_sensors` 显式开启。

## 2026-11-02修复与验证

本次在当前课程学习、域随机化和左右脚独立平面加权投影实现上修复复核发现的问题：

- 课程项改用本机原生公开接口 `command_manager.get_term("base_velocity").cfg` 更新范围，不再调用不存在的get/set配置方法；初始、半程、完成及超过完成步数的测试通过。
- 关节重置改用原生 `reset_joints_by_offset`，使零默认关节状态也能得到实际扰动。**YAML中 `environment.events.reset_joints.params.position_range` 现为加性角度偏移（rad），默认[-0.02, 0.02]，不再是[0.98, 1.02]缩放系数；`velocity_range` 为速度偏移（rad/s）。** Play和整组域随机化关闭时都使用零偏移，不会引入1 rad偏置。
- FOOTRIX四元数按N×4展开，修正逆旋转；保留当前平面加权分配和支持半径外最近点回退，新增旋转后分配及总冲量守恒测试。
- 接触回调先筛选记录环境，防止其他环境缓冲积压；重置清理对应环境接触包；记录项close释放订阅句柄，重复关闭及迟到回调安全。
- 奖励YAML测试补齐域随机化事件／观测上下文和实际辅助函数；传感器测试改为带包名加载，并接入实际FOOTRIX常量与模拟接触事件源。

21项离线测试全部通过，包括原生CommandManager接口、原生关节重置函数在零默认值上的效果、Play关闭扰动、YAML加载、FOOTRIX投影与缓冲生命周期、原生RecorderManager回合导出及HDF5读写。测试使用合成环境／接触事件，不启动Isaac Sim或训练；通过不代表实际GPU仿真已经验收，也未据此归因或解决此前CUDA故障。

## 2026-11-02用户启动失败日志定位

最新日志 `log/kit/train_1793551115184446123.log` 第7993行在本地时间2026-11-02 00:39附近首次报告 `PhysX Internal CUDA error ... Error code 700`，随后出现非法内存访问、CUDA上下文失效和多条分配失败信息。对应 `log/sim/train/isaaclab_2026-11-02_00-38-44.log` 记录机器人340个body、31个关节，处于机器人和场景初始化阶段。按训练入口执行顺序，此时尚未进入checkpoint加载与PPO更新；这是物理场景初始化失败，不能把后续分配失败直接解释成普通显存不足。

该失败运行的模型结构为：新增308个FOOTRIX固定可视标记link，当时资产转换设置 `merge_fixed_joints=False`，使标记一并成为Articulation body。[PhysX官方仓库的讨论417](https://github.com/NVIDIA-Omniverse/PhysX/discussions/417) 记录过5.6.0中大于255个link导致初始化越界的缺陷，并获维护者确认建单；这为本次340-body场景提供了具体排查线索，但不能直接证明当前Isaac Sim内置版本的故障与其完全相同。当前已开启USD导入时的固定关节合并，见上方调整说明；效果尚待实际运行验证。

用户补充的终端调用栈进一步确认失败链为 `train.py: gym.make()` → `sim.reset()` → Articulation初始化 → `ArticulationData`；错误在创建GPU重力张量时暴露，输出明确提示CUDA错误可能异步上报，不能据此把该张量语句当成根因。输入命令已使用完整checkpoint路径，失败发生在加载权重之前。终端最后的Ctrl+Z使已报错进程挂起而非退出，应在重新尝试之前结束该挂起进程。

此次仅检查既有日志、配置和上游问题记录，没有再次启动训练或仿真，没有修改模型或GPU参数；CUDA故障仍未验证解决。

## 2026-11-02此前参数评估（部分已实施）

以下保留调参前的分析依据；课程跨度、跟踪尺度、转向门控和非足触地软惩罚已按本轮用户授权调整，具体值以上方当前设置为准。课程续训状态恢复按用户从零训练的决定未实施，其他推测仍需训练验证：

- 此前默认3000轮、每轮24控制步，仅72000步，而课程跨度600000步；默认训练结束时课程最多约12%，前向／偏航指令上限约0.296。建议先对齐训练预算与课程跨度；例如48000步的课程可在默认训练约三分之二处完成，但具体速度范围仍需按学习效果调整。
- 课程使用新环境的 `common_step_counter`，现有checkpoint续训并不恢复该计数。因此续训的策略／优化器可以继续，但课程从头计算；需要明确选择恢复课程阶段还是刻意重新适应，不能把它当作完整实验状态恢复。
- 调参前初期前向指令上限0.2，而线速度跟踪std为0.5。仅看这一奖励项，机器人静止时对0.1／0.2 m/s指令仍分别得到约0.961／0.852（该项最大1），可能减弱低速行走激励。可比较更紧或随课程变化的误差尺度，不能据此断言现有策略一定原地站立。
- 调参前站立惩罚仅按平面指令速度小于0.06启用，不考虑偏航指令；默认早期前向均匀范围下约30%的指令落入该区域，非零偏航也可能被约束回默认关节姿态。腾空奖励同样只看平移速度。建议将站立条件与转向任务的需求一致化。
- 此前评估指出，若目标要求直立双足行走，仅终止骨盆／躯干接触，且没有高度约束，膝／臂支撑可能未被排除；需结合姿态回放确认是否出现这一奖励利用方式，再决定非足触地惩罚或终止项，不能泛化为后续全身接触研究必须禁用其他接触。

统一80／8驱动增益、当前PPO网络／学习率及较小的质量、质心和噪声随机化尚不能仅凭配置判为错误，应结合站立、关节跟踪误差、动作变化和各项奖励曲线判断。另据本机实际 `push_by_setting_velocity` 源码，扰动使用 `vel_w +=`，因此±0.2 m/s是水平速度增量，未扰动的竖直／角速度不会被该实现清零。上述为调参前评估；本轮仅修改配置和奖励门控，没有启动训练。

## 课程学习与域随机化（2026-10-30）

训练配置已在 `h2_plus_lab/tasks/locomotion/env_cfg.yaml` 中启用课程学习和域随机化。课程项 `command_range` 通过任务内的 `mdp/curriculums.py` 逐步放宽速度指令：从前向速度上限 `0.20 m/s`、偏航角速度绝对值 `0.20 rad/s`，在环境控制步数 `144000` 内线性过渡到 `1.00`；它只修改 `base_velocity` 命令范围，不改变动作维度、观测结构或奖励函数。平地、低摩擦双区域和柔性地面目前都使用这一命令难度课程。

训练事件包含：启动时随机化左右足材料摩擦与恢复系数、`pelvis`／`torso_link` 的质量和质心；回合重置时对关节位置／速度做加性偏移、对根部初始状态做小范围扰动；每 `8—12 s` 对根部施加一次小幅水平速度扰动。策略观测已开启轻度噪声（根部速度、投影重力、主动关节位置／速度），回放配置会关闭课程、物理随机化、周期推力、重置扰动和观测噪声，以便使用固定条件观察策略。

当前没有把 `terrain_levels_vel` 挂到这些任务：平地使用普通平面，低摩擦和柔性地面使用自定义场景，均没有可供该项管理的 `TerrainImporter` 地形生成器。`env_cfg.yaml` 的 `environment.curriculum.terrain_levels` 保留为空接口；后续为局部崎岖地形注册生成器任务时，再单独添加地形行列难度课程，避免在现有场景重置时访问空地形生成器。以上为配置层修改，本轮未启动训练或仿真验证。

## 2026-10-28目录清理

训练相关脚本为 `train.py`、`play.py`、`cli_args.py`、`list_tasks.py`；另有 `view_logs.py` 用于独立查看TensorBoard曲线。独立工具曾移至 `tools/run_h2_plus.py`，已于2026-10-29被任务内RecorderManager替代并删除；原check／smoke／collect／validate／terrain命令不再提供。已删除未接入训练且导出不完整的 `view_h2_plus_mujoco_terrains.py` 和 `h2_plus_four_terrains.xml`，不再提供该展示入口。

产物目录只保留 `log/`、`output/`；旧 `logs/`、`outputs/` 已删除。清理了旧试跑的初始权重、TensorBoard事件、重复Hydra快照、失败启动日志及旧测试运行噪声；此前还删除了字节码、空日志、失效PID、未用课程模块、空requirements及旧USD缓存。当前 `log/rsl_rl/` 的权重与记录、`output/h2_plus_usd/` 和原始机器人资产保留。两次有效早期试跑仅保留console日志与参数，迁入 `log/history/`；有独有信息的模型清单、足底布局、图片及检查摘要迁入 `output/history/`。本次未启动训练。

## 手动启动

在已激活的 `project` 环境和本目录下执行：

```bash
python train.py --task Isaac-H2Plus-Blind-Locomotion-v0 --headless
python play.py --task Isaac-H2Plus-Blind-Locomotion-Play-v0 --checkpoint /path/to/model.pt
python list_tasks.py
```

`--task` 必须明确提供；`--help` 和 `list_tasks.py` 不启动仿真。首次进入终端时使用 `conda activate project`，无需export路径或修改外部Isaac Lab。旧 `python run_h2_plus_rl.py ...` 命令已取消，旧 `--mode` 参数改由 `train.py`／`play.py` 文件区分。训练仍由用户本人启动，助手修改代码不代表获准训练。

常用覆盖示例：

```bash
python train.py --task Isaac-H2Plus-Blind-Locomotion-v0 --headless --num_envs 2048 --max_iterations 9000
python train.py --task Isaac-H2Plus-Blind-Locomotion-v0 --headless env.rewards.action_rate.weight=-0.02
```

默认PPO为9000轮，每100轮保存checkpoint；平地和低摩擦默认1024环境，柔性默认64环境，Play配置为16环境。算法参数在对应agent配置修改，也可用Hydra的 `agent.algorithm.learning_rate=...` 等覆盖。没有额外安装依赖。

## TensorBoard训练曲线

PPO配置已显式设置 `logger = "tensorboard"`。训练进入PPO后，RSL-RL自动把奖励、损失、回合长度及任务指标写入 `log/rsl_rl/<实验名>/<运行时间>/`；不需要额外开启记录参数。`--record_sensors` 仍只负责传感器轨迹，与TensorBoard曲线独立。

在另一个已激活project环境的终端中，从本目录执行：

```bash
python view_logs.py
```

然后在本机浏览器打开 `http://127.0.0.1:6006`。**默认仅选择本单元log/rsl_rl下最近写入事件数据的一次训练**，通常对应正在运行的训练；按事件文件更新时间选择，不按目录名字或目录mtime选择，忽略无事件／空事件目录。终端会打印选中的完整目录。启动后固定跟踪这一轮，曲线继续更新；新开另一轮训练时重启view_logs.py重新选择，不自动混入新旧运行。

```bash
python view_logs.py --latest       # 与默认行为相同
python view_logs.py --all          # 查看全部训练，用于对比
python view_logs.py --run log/rsl_rl/h2_plus_blind_locomotion/某次运行目录
```

已有TensorBoard服务需在其原终端按Ctrl+C结束，然后重新执行 `python view_logs.py` 并刷新网页；只重启查看服务，训练进程继续运行。需要换端口时追加 `--port 6007`，其他TensorBoard原生命令行参数也可追加。没有事件数据时会提示等待PPO开始，或用--all打开根目录。

现有环境已安装TensorBoard 2.21.0，本次没有安装依赖。已检查版本、帮助入口，并从另一个工作目录启动临时服务验证HTTP 200后关闭；没有启动仿真或训练。这里提供手动查看入口，没有由助手启动的常驻服务。本轮最近运行选择、旧目录续写、嵌套摘要、空目录与显式选择测试通过，整体32项离线测试通过，未启动训练。

## 从指定checkpoint继续训练

训练和回放均接受等价参数 `--checkpoint`、`--ckpt`，直接接收本地文件的完整路径；指定后自动启用续训，不需要 `--load_run` 或 `--resume`：

```bash
python train.py --task Isaac-H2Plus-Blind-Locomotion-v0 --headless --checkpoint /完整路径/model_100.pt
```

也接受相对于当前终端工作目录的文件路径，并在启动Isaac Sim前检查文件是否存在。显式路径直接交给训练器加载，不在运行目录内做正则匹配；新训练日志和权重仍写入本项目的 `log/`，不会覆盖输入checkpoint。不指定 `--checkpoint` 时仍可使用 `--resume` 自动查找最近运行，旧 `--load_run` 仅用于这种自动查找模式。指定路径时该参数无需提供。

## 奖励权重与数值参数调优

统一编辑 [rsl_rl_locomotion_rewards.yaml](h2_plus_lab/tasks/locomotion/agents/rsl_rl_locomotion_rewards.yaml)，只保留 `rewards` 段，当前12项奖励／惩罚包含新增非足接触惩罚，部分尺度和门控已调整。对应函数与资产选择在 `env_cfg.yaml` 的 `environment.rewards` 中声明，函数实现放在 `mdp/`。

```yaml
rewards:
  track_lin_vel_xy:
    weight: 1.5
    params:
      std: 0.5
  action_rate:
    weight: -0.01
```

权重设0可以禁用该项。新增奖励时，先在env YAML声明原生RewardTermCfg、func与绑定，再在奖励文件提供数值；只写一个未声明的奖励名会报错。命令行 `env.rewards.action_rate.weight=-0.02` 仍可临时覆盖，实际生效配置继续保存到 `log/rsl_rl/<实验名>/<运行时间>/params/env.yaml`。函数和配置类通过原生Manager检查，通用加载器检查非有限数值、权重类型以及YAML声明的正值／非负约束。

## 训练中采集传感器（RecorderManager）

传感器采集默认关闭；训练或回放显式添加 `--record_sensors` 才会启用，无需另开程序。不加开关时，入口在创建环境前清空记录器配置，不缓存或导出传感器轨迹。开启后默认记录环境0，每个控制步采样一次；当前物理步长1/120秒、decimation为4，所以记录频率为30 Hz。IMU配置的120 Hz更新周期不等于保存了完整120 Hz序列。记录项使用 `record_post_step`，位置在场景传感器更新之后、自动reset之前；不使用Lab 2.3.2中早于场景更新的物理子步回调。

配置位于 `env_cfg.yaml` 的 `environment.recorders`：`sensors.env_ids` 设置环境选择、`sensors.record_every` 设置采样步距；Manager配置控制原生回合导出。实现仅在 `mdp/recorders.py`，三种地形和Play继承同一记录项。示例（回放同样支持 `--record_sensors`）：

```bash
# 开启采集：默认环境0
python train.py --task Isaac-H2Plus-Blind-Locomotion-v0 --headless --record_sensors
# 记录前4个环境（要求num_envs至少为4）
python train.py --task Isaac-H2Plus-Blind-Locomotion-v0 --headless --record_sensors 'env.recorders.sensors.env_ids=[0,1,2,3]'
# 记录全部并行环境
python train.py --task Isaac-H2Plus-Blind-Locomotion-v0 --headless --record_sensors env.recorders.sensors.env_ids=null
# 每2个控制步记录一次；当前为15 Hz，终止帧仍额外保留
python train.py --task Isaac-H2Plus-Blind-Locomotion-v0 --headless --record_sensors env.recorders.sensors.record_every=2
# 默认关闭记录：省略开关
python train.py --task Isaac-H2Plus-Blind-Locomotion-v0 --headless
```

采集不增加策略输入，不修改奖励或传感器读数。选中的环境编号须有效且不重复，默认单环境避免把1024路轨迹全部写盘；选择全部环境时数据量和写盘开销随之增加。

每次运行生成独立目录，路径由入口设置并打印：

```text
output/sensors/train/<训练时间与唯一后缀>/
├── sensors.hdf5
└── metadata.json
```

Play使用 `output/sensors/play/`。HDF5按 `data/demo_0`、`data/demo_1` 分回合存储；不同环境的回合可以交错导出，必须按数据中的 `env_id`、`episode_id` 识别，不能把demo序号当成环境编号。

| 字段／组 | 含义 |
|---|---|
| `actions` | 本控制区间实际送入环境的动作，末态对应这一动作执行后；也是原生导出器的样本计数依据。 |
| `env_id`、`episode_id`、`episode_step` | 环境编号、该环境本次运行内的回合编号、真实控制步计数；不受RSL-RL随机初始化超时计数影响。 |
| `physics_step`、`simulation_time_s` | 本环境运行的物理步计数和仿真时间，不是墙钟时间。 |
| `robot/`、`velocity_command` | 根部位姿／速度、关节位置／速度／目标及速度指令。 |
| `imu/<名称>/` | 躯干和双足的机体坐标系加速度、角速度与finite标记；重力偏置遵循挂载配置，不额外加噪声。 |
| `contact/<名称>/` | 左右足及非足身体的世界坐标系聚合法向接触力、finite和in_contact；足部另含腾空／接触计时。|
| `footrix/<left|right>/` | 每足154个小圆域内的三维力，严格筛选足底附近报告接触点，不扩散、不兜底；形状为 `[环境数, 154, 3]`，另有finite、in_contact、各单元／选中／未测量接触计数及未测量力。单元本身无碰撞、不附加质量。|
| `reward`、`terminated`、`time_out`、`reset`、`termination_terms/` | 同一时刻的奖励和回合终止状态；终止前末帧先记录，再由环境重置。 |

`metadata.json`记录采样规则、关节顺序、动作分组维度、传感器安装外参、URDF与记录器摘要及训练日志目录。HDF5原生 `success` 属性没有被定义为行走能力标签；本任务没有success终止项，应结合终止字段分析，不能把该属性直接解释为训练成败。没有接触与非有限读数分别通过 `in_contact` 和 `finite` 表示，原值保留，不用0替换NaN。

RecorderManager在回合重置前写入并flush；正常结束、异常或Ctrl+C会先尝试导出已缓存的未结束回合，再关闭环境。最后一帧 `reset=false` 的片段可能是不完整回合或手动重置片段。强制杀进程无法保证导出尚在内存中的数据。文件在结束／停止后可用h5py读取：

```python
import h5py

with h5py.File("output/sensors/train/<本次目录>/sensors.hdf5", "r") as file:
    episode = file["data/demo_0"]
    time_s = episode["simulation_time_s"][:, 0]
    torso_accel = episode["imu/torso_imu/linear_accel_b_m_s2"][:]
    left_force = episode["contact/left_foot_contact/net_normal_force_w_N"][:]
```

验证边界：离线使用真实RecorderManager回调、EpisodeData和HDF5读写，传感器与环境为合成测试对象；验证了筛选环境、终止帧、重置分段、部分回合关闭导出、非有限标记和唯一输出路径。本次没有执行实际Isaac Sim传感器采集或训练，测试数据不作为实验结果。

## 模块结构与修改位置

采用 `isaaclab_tasks` 的配置类、MDP、agent配置和 `gym.register()` 方式。当前只有H2 Plus一种机器人，因此没有额外嵌套 `manager_based/locomotion/velocity/config/h2_plus/` 等目录。

```text
h2_contact_lab/
├── train.py                       # 直接训练入口
├── play.py                        # 直接回放与策略导出入口
├── cli_args.py                    # 共用命令行参数与Hydra输出路径
├── list_tasks.py                  # 不启动仿真的任务清单
├── view_logs.py                   # TensorBoard，默认127.0.0.1:6006
├── h2_plus_lab/
│   ├── config.py                  # 通用OmegaConf/Hydra加载
│   ├── assets/h2_plus.py          # 资产配置读取入口
│   ├── assets/h2_plus.yaml        # URDF、初始姿态与驱动
│   └── tasks/
│       ├── __init__.py            # 自动发现直属任务包
│       └── locomotion/
│           ├── __init__.py        # gym.register：任务ID与配置入口
│           ├── env_cfg.py         # 六个任务类，选择配置并组装
│           ├── env_cfg.yaml       # 环境、MDP项、训练和Play覆盖
│           ├── sensor_cfg.py      # 传感器配置读取入口
│           ├── sensor_cfg.yaml    # 接触、IMU和FOOTRIX参数
│           ├── terrain_cfg.py     # 地形配置读取入口
│           ├── terrain_cfg.yaml   # 地形几何与材料
│           ├── agents/
│           │   ├── __init__.py
│           │   ├── rsl_rl_ppo_cfg.py
│           │   └── rsl_rl_locomotion_rewards.yaml  # 奖励调优入口
│           └── mdp/
│               ├── rewards.py    # H2足部与速度跟踪奖励
│               ├── terminations.py
│               ├── recorders.py  # 训练场景传感器记录项
│               └── curriculums.py # 速度指令课程函数
├── tests/                        # 注册、记录回调及HDF5读写的离线检查
├── log/
└── output/
```

`locomotion/` 仍保留四个Python入口，参数分别放入模块对应YAML，修改数值、对象绑定和管理器项都优先编辑YAML；Python环境类仅选择地形／Play配置。MDP实现与PPO算法配置保持独立。

执行顺序为：解析参数 → AppLauncher启动Isaac Sim → 导入任务注册 → Hydra读取环境与agent配置 → `gym.make()` → RSL-RL训练／回放。新任务不需要修改 `train.py` 或 `play.py`。

## 已注册任务

| 地形 | 训练任务ID | 环境配置 |
|---|---|---|
| 平地 | `Isaac-H2Plus-Blind-Locomotion-v0` | `env_cfg.py:H2PlusFlatEnvCfg` |
| 低摩擦双区域 | `Isaac-H2Plus-Blind-Locomotion-LowFriction-v0` | `env_cfg.py:H2PlusLowFrictionEnvCfg` |
| 柔性地面 | `Isaac-H2Plus-Blind-Locomotion-Deformable-v0` | `env_cfg.py:H2PlusDeformableEnvCfg` |

每个训练任务都有在 `-v0` 前加 `-Play` 的回放任务，对应同文件中后缀 `_PLAY` 的类。原有任务ID保持不变，新增低摩擦和柔性Play注册；这些特殊地形本轮未运行。六个任务暂时共用 `H2PlusBlindPPORunnerCfg`，需要独立实验日志／超参数时可继承该类，设置独立 `experiment_name`，再更改注册中的agent入口。

## 自己增加环境并注册

修改现有动作／观测／奖励／事件，直接编辑env YAML中对应项；新增计算逻辑放到 `mdp/`，然后将YAML的func指向该函数，无需修改训练入口。新增地形任务，可在env YAML的 `terrains` 下加覆盖配置，再在 `env_cfg.py` 增加薄配置类：

```python
@configclass
class H2PlusMyEnvCfg(H2PlusLocomotionEnvCfg):
    terrain_profile: ClassVar[str] = "my_terrain"  # 对应YAML里的terrains.my_terrain
```

在 `locomotion/__init__.py` 中增加标准注册：

```python
gym.register(
    id="Isaac-H2Plus-MyTask-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:H2PlusMyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:H2PlusBlindPPORunnerCfg",
    },
)
```

随后可用 `python list_tasks.py` 查看，再手动 `python train.py --task Isaac-H2Plus-MyTask-v0 --headless`。示例任务没有预先加入代码。全新任务仍可建立独立任务包并在包内注册，父包自动发现；不需要在train.py／play.py里添加任务分支。

## 产物目录约定（2026-10-28）

用户要求使用本单元下单数形式的 `log/` 与 `output/`。默认路径按脚本位置计算，不随终端当前目录变化，无需额外export路径。

| 目录（相对h2_contact_lab） | 内容 |
|---|---|
| `log/rsl_rl/<实验名>/<运行时间>/` | TensorBoard训练记录、参数／代码快照、`model_*.pt` checkpoint；训练续跑与默认回放在这里查找权重。 |
| `log/kit/`、`log/sim/` | 训练／回放的Kit日志和Isaac Lab仿真日志。 |
| `output/hydra/` | Hydra配置快照与运行日志。 |
| `output/h2_plus_usd/` | 任务加载时生成的机器人转换USD。 |
| `output/sensors/train/`、`output/sensors/play/` | 训练／回放的HDF5传感器轨迹与metadata。 |
| `output/videos/train/`、`output/play/` | 训练视频、回放视频及导出策略 `.pt`／`.onnx`。 |
| `log/history/`、`output/history/` | 已整理的早期运行参数／日志与独有历史资料。 |

根目录的 `train.py`、`play.py` 和 `cli_args.py` 基于Isaac Lab 2.3.2入口，保留版权信息；未修改外部Isaac Lab源码。旧 `logs/`、`outputs/` 已清理，不再存在；历史资料的新位置见目录清理说明。

## 当前盲走观测与传感器

首版策略是盲走：观测包含根部状态、投影重力、23个主动关节的位置／速度、速度指令和上一时刻动作，不包含高度扫描。动作由双腿12个、腰部3个、双肩6个和双肘2个自由度组成，共23个；头部、腕部及尚未纳入策略的上肢自由度保留在模型和接口中。

RayCaster配置仍保存在 `sensor_cfg.py`，但没有挂载到首版场景。躯干 IMU 挂在 `torso_link` 原点，双足 IMU 挂在各自 `ankle_pitch_link` 并使用已给出的偏移和旋转；三路 IMU 配置为120 Hz更新、6帧历史、输出含重力的机体坐标系加速度和角速度，当前记录层保存原始模拟输出，不额外加噪声；`H2_PLUS_IMU_NOISE_CFG` 仅保留参数，未应用到新记录器。IMU不进入首版策略观测。FOOTRIX保留为每个训练／回放场景中的足底测量配置：左右脚各154个无碰撞、无独立质量的足link内visual；完整 `ankle_pitch_link` 足底碰撞仍负责全部足地接触，记录器仅累计各小圆域与平面容差内的报告接触冲量，圆外力单列。布局 JSON 中154个测量点的公共高度已对齐当前足底网格最低面，设为 `z=-0.0521 m`；该高度仍可按新网格重新标定。场景始终挂载完整足底接触传感器和标记，但点力回调只在显式添加 `--record_sensors` 时运行并写入 `footrix`，不进入首版盲走策略。单元拓扑和编号相同，右足坐标在其局部 `y` 轴上采用镜像布局，以匹配左右足网格；安装标定和硬件编号对应仍留有替换接口。

## 2026-10-28此前修复与运行记录

**此前由助手启动的训练已停止；后续由用户手动启动。** 平地盲走已实际完成环境创建、物理步进和PPO更新，但未完成训练或验证行走效果。低摩擦、柔性场景本轮仅修代码，尚未运行。

已修复：

- 删除 `DeformableBodyMaterialCfg` 不支持的 `static_friction`；补齐URDF转换驱动的零刚度／阻尼必填值，执行器仍使用80／8增益。
- Gym注册改为字符串入口，避免在 `AppLauncher` 前导入PPO与仿真运行时；现已改为根目录直接入口，任务注册在AppLauncher之后执行，CLI辅助模块移至本地。
- 无TerrainImporter的低摩擦／柔性场景按环境局部坐标检查4米场地边界；环境间距改为5米，柔性场景关闭物理复制。
- 躯干传感器取消多body过滤；训练场景的双足聚合传感器只保留净接触力、历史和腾空计时，关闭PPO未使用的成对接触过滤／接触点／摩擦力记录，避免GPU平面过滤不支持的运行警告。该历史记录形成时FOOTRIX尚未挂载；当前配置保留完整足底接触报告和154个无碰撞测量标记，点力由记录器严格筛选小圆域内的报告接触，不把圆外力分配到单元，RayCaster仍未挂载。
- 平地与低摩擦任务默认1024环境，柔性任务默认64环境；均可用 `--num_envs` 覆盖，平地play仍为16环境。

运行记录：首次有效启动日志迁至 `log/history/train_20261028_141313/console.log`；关闭多余过滤后的日志迁至 `log/history/train_20261028_141452/console.log`，后者记录PPO迭代0—7、累计196608步，随后按用户要求停止。两次试跑的参数保存在各自 `params/`；旧初始checkpoint、事件文件和失败启动日志已清理。这些是早期运行证据，不是已收敛策略。

环境判断更正：此前直接调用conda环境的Python，未执行激活脚本，因此误判为缺少torch。完整 `conda activate project` 会加载独立Isaac Sim的Python路径，本次实际使用Isaac Lab 2.3.2、Isaac Sim 5.1和PyTorch 2.7.0+cu128，在RTX 4090上进入训练；未安装新依赖。此次结构调整后直接使用已激活环境中的Isaac Lab，不再需要查找上游训练脚本，也不读取 `ISAACLAB_ROOT`；仍无需手动export。

局部崎岖地形保存在terrain YAML中，没有独立Gym任务。初始姿态、高度、驱动增益及奖励仍需根据训练表现调整；早期回合主要因姿态倾倒终止，不能据PPO运行成功认定稳定行走。

旧独立采集工具已删除，以下资产信息继续供任务配置使用。

## 资产版本如何区分

以下文件均位于 [../robot_asset/h2_plus/](../robot_asset/h2_plus/)。计数排除了XML注释；文件名中的 `no_dex_hands` 不能代替结构核对。

| 文件 | 活动结构 | 用途／差异 |
|---|---|---|
| `H2_Plus.urdf` | 98 link；75 revolute＋22 fixed | 带灵巧手、原足；保留参照。 |
| `H2_Plus_mujoco.urdf` | 98 link；75 revolute＋22 fixed | 带手、原足；可视网格使用STL的版本。 |
| `H2_Plus_mujoco_with_footrix.urdf` | 98 link；75 revolute＋22 fixed | 已换新足，仍含灵巧手；不是当前去手默认版。 |
| `H2_Plus_no_dex_hands.urdf` | 32 link；31 revolute | 去手但仍引用原足网格。 |
| `H2_Plus_with_footrix_no_dex_hands.urdf` | 32 link；31 revolute | **Isaac主入口默认**；左右足各154个纯visual圆形标记，无独立质量／碰撞，部分本体可视网格为DAE。|
| `H2_Plus_mujoco_with_footrix_no_dex_hands.urdf` | 340 link；31 revolute＋308 fixed | 旧MuJoCo展示使用的源资产；新足，新增左右脚各154个无碰撞固定FOOTRIX可视化标记，可视网格为STL。URDF保留，展示脚本及生成XML已删除。 |

原始去手改装URDF均有49个本地网格引用；当前Isaac默认URDF在双足link内另含308个无质量visual标记；旧MuJoCo源资产仍保留308个固定标记link，本次未改。源模型根link为 `pelvis`，floating joint示例位于注释中。Isaac转换器以 `fix_base=False` 请求浮动根，实际Articulation是否为浮动基座仍须运行时确认。URDF本身不生成IMU读数；Isaac任务场景另行挂载三路IMU。两份带标记URDF共享154点的图像推导布局和可替换接口；由于左右足网格的局部 `y` 轴相反，MuJoCo源资产将右脚154点的局部 `y` 坐标镜像后再挂载。MuJoCo版本只增加可视化标记，不接入Isaac记录器的接触报告投影。

`../robot_asset/footrix_design/` 保存左右新足装配、IMU零件、STL和 `L.txt`／`R.txt` 质量属性。两份文本分别使用“坐标系5／坐标系4”，包含质心处张量及输出原点处张量，引用时须核坐标、参考点、单位和负张量记数法，不能互换或把截图示例当最终参数。`h2_260601/`为简化H2 CAD，`h2_description/`为普通H2描述和网格，均不是当前H2 Plus默认资产。CAD装配、网格碰撞质量和实机安装本次未验证。

## 环境和依赖

Isaac入口沿用用户已有Ubuntu环境。此前实际启动使用Python 3.11、Isaac Lab 2.3.2、Isaac Sim 5.1和PyTorch 2.7.0+cu128，见 [ubuntu_environment.txt](ubuntu_environment.txt)。新记录器复用Isaac Lab已有的h5py/HDF5依赖，本次未安装新包。

原 `requirements.txt` 只有注释、没有依赖条目，已在目录清理中删除；依赖说明统一保存在 [ubuntu_environment.txt](ubuntu_environment.txt)，不提供Isaac Lab／Isaac Sim安装清单。现用入口无需为已删除的展示脚本安装MuJoCo；旧H2清单记录的MuJoCo版本仅作历史信息。新环境须按对应后端的匹配安装流程准备，本次未安装任何包。

## 旧独立采集入口的退出

已删除 `tools/run_h2_plus.py`，避免另建仿真后误认为采集了PPO训练过程。旧脚本还包含URDF检查、四地形USD展示、位置保持采集及JSONL校验；这些命令随脚本一起退出，不再作为现行入口。机器人资产、地形与传感器配置继续由任务管理，新记录格式为HDF5；历史JSONL字段不等同于当前记录字段，特别是旧脚本的成对摩擦力和记录层噪声没有移作训练数据。

## 已删除的MuJoCo展示与历史资料

旧展示脚本和XML未接入PPO，且只导出16个joint（源URDF为31个）、没有freejoint、未完整保留关节origin／惯性／碰撞，机器人mesh禁用碰撞，地形未写入差异摩擦，XML还含Windows绝对meshdir。因此它们不能作为动力学或成功展示证据，已删除；如以后恢复MuJoCo路线，应重新实现和核验，不能复用旧导出结论。

以下资料已统一保存在 `output/history/`：

| 目录 | 保留内容与证据边界 |
|---|---|
| `inspect_h2_QxAfbW/` | `h2_model_inventory.json`：普通H2 MJCF清单，Python 3.11.16／MuJoCo 3.3.7，加载成功、物理步数0，未验证H2 Plus对应关系。 |
| `windows_checks_20261015/`、`windows_isaac_checks_20261015/` | 仅保留 `checks.json`，属于已移除旧原型的检查，不是Isaac物理运行证据。 |
| `windows_footrix_manual_checks_20261015/` | 旧规则网格布局与检查JSON，不作为现行布局。 |
| `windows_footrix_points_checks_20261015/` | 图像估计的154点布局、PNG／SVG和检查JSON：49项通过、3项集成检查跳过；毫米坐标、右足镜像及硬件编号对应仍未验证。 |

文件中的旧绝对路径仅表示历史运行位置。训练与采集统一使用根目录 `train.py`／`play.py` 和任务配置，不能把上述旧检查结果移作现行任务的验收结论。
