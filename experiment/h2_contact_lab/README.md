# H2 Plus：Isaac Lab接入与采集、MuJoCo展示

更新日期：2026-10-25。本页区分现行代码、已保存文件和运行证据。研究目标见 [项目README](../../README.md) 与 [研究正文6.5](../../research/research_main.md)。本次仅核对文件和更新说明，没有启动仿真或重新运行检查。

## 两个入口及当前状态

| 入口 | 功能 | 当前证据 |
|---|---|---|
| [run_h2_plus.py](run_h2_plus.py) | Isaac Lab主入口：`check`、`terrain`、`smoke`、`collect`、`validate`。 | 此前记录Windows语法、URDF检查、合成记录校验和地形dry-run通过；当前目录没有H2 Plus的Isaac USD或collect结果。目标Ubuntu环境运行仍待验证。 |
| [view_h2_plus_mujoco_terrains.py](view_h2_plus_mujoco_terrains.py) | MuJoCo静态展示辅助脚本：四块地形和一个机器人，不调用 `mj_step`。 | 已保存 [h2_plus_four_terrains.xml](outputs/h2_plus_four_terrains.xml)，但静态核对发现导出关节树不完整、未配置差异摩擦，详见本页“MuJoCo展示的待修项”。不能作为已验证展示结果。 |

下文命令均从 **project根目录** 执行。文档列出的是使用入口；本次整理未执行这些命令。

## 资产版本如何区分

以下文件均位于 [../robot_asset/h2_plus/](../robot_asset/h2_plus/)。计数排除了XML注释；文件名中的 `no_dex_hands` 不能代替结构核对。

| 文件 | 活动结构 | 用途／差异 |
|---|---|---|
| `H2_Plus.urdf` | 98 link；75 revolute＋22 fixed | 带灵巧手、原足；保留参照。 |
| `H2_Plus_mujoco.urdf` | 98 link；75 revolute＋22 fixed | 带手、原足；可视网格使用STL的版本。 |
| `H2_Plus_mujoco_with_footrix.urdf` | 98 link；75 revolute＋22 fixed | 已换新足，仍含灵巧手；不是当前去手默认版。 |
| `H2_Plus_no_dex_hands.urdf` | 32 link；31 revolute | 去手但仍引用原足网格。 |
| `H2_Plus_with_footrix_no_dex_hands.urdf` | 32 link；31 revolute | **Isaac主入口默认**；新足，部分本体可视网格为DAE。 |
| `H2_Plus_mujoco_with_footrix_no_dex_hands.urdf` | 32 link；31 revolute | **MuJoCo展示默认**；新足，可视网格为STL。它与生成XML不是同一个文件。 |

两个默认URDF均有49个本地网格引用，本次路径核对均存在；源模型根link为 `pelvis`，floating joint示例位于注释中。Isaac转换器以 `fix_base=False` 请求浮动根，实际Articulation是否为浮动基座仍须运行时确认。两者均未声明IMU link/frame和安装外参，程序不生成IMU读数。

`../robot_asset/footrix_design/` 保存左右新足装配、IMU零件、STL和 `L.txt`／`R.txt` 质量属性。两份文本分别使用“坐标系5／坐标系4”，包含质心处张量及输出原点处张量，引用时须核坐标、参考点、单位和负张量记数法，不能互换或把截图示例当最终参数。`h2_260601/`为简化H2 CAD，`h2_description/`为普通H2描述和网格，均不是当前H2 Plus默认资产。CAD装配、网格碰撞质量和实机安装本次未验证。

## 环境和依赖

Isaac入口沿用用户已有Ubuntu环境。此前记录版本为Python 3.11、Isaac Lab 2.3.2、Isaac Sim 5.1.0、PyTorch 2.7.0+cu126，见 [ubuntu_environment.txt](ubuntu_environment.txt)；本次未连接Ubuntu确认这些版本。`check`、`validate`、`terrain --dry-run` 使用标准库；真正的USD导入、场景生成、smoke和collect依赖Isaac运行时。

[requirements.txt](requirements.txt) 只有“Isaac现有环境之外无额外pip依赖”的说明，不负责安装Isaac Lab／Isaac Sim，也不覆盖另一个MuJoCo展示入口。展示脚本需已有的 `mujoco` 包和可用图形窗口；旧H2清单记录的MuJoCo版本不能当作当前Windows环境版本。新环境须按对应后端的匹配安装流程准备，本次未安装任何包。

## Isaac四类地形场景：不推进物理步

```bash
# 离线检查URDF与场景计划，不启动Isaac
python experiment/h2_contact_lab/run_h2_plus.py --mode terrain --dry-run

# 在现有Ubuntu Isaac环境生成USD
python experiment/h2_contact_lab/run_h2_plus.py --mode terrain --headless --device cpu
```

`terrain`计划在同一USD中并排放置四类地形及**四个机器人初始姿态**。默认输出到 `outputs/h2_plus_terrain_scenes/<UTC时间>/`，包含 `h2_plus_four_terrain_variants.usd` 和 `terrain_manifest.json`；这些是代码约定的目标输出，当前目录尚未保存对应结果。

| 场景 | Isaac代码中的几何／物理设置 |
|---|---|
| 普通完整硬地 | 4×4 m刚性地面，静／动摩擦暂设0.8／0.7。 |
| 低摩擦与摩擦差异 | Y方向两半平面：正Y侧0.25／0.20，负Y侧0.8／0.7；左右脚与场景轴向的对应仍需查看USD。 |
| 轻度可变形铺设的外形代理 | 3×3 m、厚6 cm；当前仍是刚性碰撞体，没有压缩、剪切、倾转或恢复的可变形物理。 |
| 局部有效支撑变化 | 相邻支撑板形成2.5 cm高差，供接缝／边缘附近几何检查。 |

根部初始Z默认1.01 m，可用 `--root-height` 调整；地形间距默认6 m，`--terrain-spacing` 小于4.5 m会被拒绝。尺寸、摩擦和高差是调试参数，不是实测家居材料参数。

该模式只转换URDF、写入USD prim并导出stage，不创建Articulation运行时，不调用reset或物理步。即使导出成功，也不能证明脚已承载、静态平衡或站立能力。此模式不生成轨迹；后面的 `smoke`／`collect` 使用单一普通地面，不会自动切换到这四类地形。

## Isaac静态检查与后续采集

```bash
# 只检查URDF、关节和网格
python experiment/h2_contact_lab/run_h2_plus.py --mode check

# 后续进入物理阶段：生成USD、加载浮动基座并短步进约1秒
python experiment/h2_contact_lab/run_h2_plus.py --mode smoke --headless --device cpu --steps 240

# 采集10秒：240 Hz物理步、60 Hz记录
python experiment/h2_contact_lab/run_h2_plus.py --mode collect --headless --device cpu --duration 10 --physics-hz 240 --record-hz 60
```

当前仅检查场景时不需要运行smoke或collect。进入实际仿真后，先核浮动根、关节映射、运行版本及接触字段，再判断是否采集。位置保持增益80／8是可修改的仿真调试参数，努力限值读取URDF；零位保持可能倾倒，采集完成不等于站稳。

可选单关节位置激励：

```bash
python experiment/h2_contact_lab/run_h2_plus.py --mode collect --headless --device cpu --duration 10 --perturb-joint left_ankle_pitch_joint --perturb-amplitude 0.03 --perturb-frequency 0.5
```

转换资产默认写到 `outputs/h2_plus_usd/`。smoke／collect在 `outputs/h2_plus_runs/<UTC时间>/` 新建目录，保存 `metadata.json`；collect另写 `trajectory.jsonl`。`--output-dir PATH` 可指定尚不存在的新目录，去掉 `--headless` 可打开窗口。

## 记录字段及离线校验

`metadata.json`保存URDF、网格与采集脚本SHA-256、USD路径、运行版本、Articulation关节顺序、URDF努力上限、初始状态、位置保持参数、摩擦假设和采样配置。运行失败或中断应结合 `run_status` 和错误信息阅读，不能只看是否生成了文件。

`trajectory.jsonl`按帧保存仿真时间、关节位置目标／位置／速度、浮动根位姿／速度，以及左右足净法向合力、对地过滤法向力、对地过滤摩擦力和平均接触位置。有效接触标为 `valid=true`；无接触显式标为false，缺失值不能随意填0。

这些是Isaac Lab ContactSensor／PhysX聚合仿真量，平均位置不是逐接触点分布，不是FOOTRIX逐单元三轴读数或标定结果；当前没有IMU数据、实际电机力矩、能力标签或研究任务真值。基础位置保持轨迹也不等于完整训练数据设计。

```bash
python experiment/h2_contact_lab/run_h2_plus.py --mode validate --run-dir /path/to/h2_plus_runs/20261022_120000
```

用实际运行目录替换示例路径。validator检查运行状态、metadata／JSONL一致性、采样步距／时间戳、关节维度、有限数值、接触有效标记及资产摘要，并汇总根部／关节范围、左右足有效接触比例和力峰值。返回码0只表示记录结构检查通过，不能认证物理真实性或传感器标定。此前Windows正常／缺字段样例均为合成记录，样例已清理，不应冒充Ubuntu采集数据。

## MuJoCo展示的待修项

保存XML有四块展示地形和位于第一块地形上方的一个机器人；它与Isaac的“四机器人USD计划”不同。本次直接比对脚本、源URDF和XML，确认：

- 源URDF有31个转动关节，XML只有16个joint且无freejoint。`build_mjcf`／`_add_link`递归没有逐条转写父子关节和origin，因此源关节树与初始几何变换未被完整保留。
- 脚本只写入visual mesh，机器人geom的 `contype`／`conaffinity` 为0；没有导入源URDF的碰撞体、惯性和驱动。不能当成动力学模型。
- 地形名称含“friction difference”，但脚本和XML均未写入差异 `friction` 属性；该区域目前仅是分区外形示意。铺设物同样只是刚性外形。
- XML的 `compiler meshdir` 使用当前Windows绝对目录，移动到其他机器后不能假定可直接使用。

脚本先写XML，再加载模型和打开viewer；所以XML存在也不证明加载或显示成功。本次没有打开viewer，没有对应的H2 Plus截图证据。以上需要后续修代码并重新生成、核对模型，本次按范围仅记录问题。

脚本现有调用形式如下，**不是本次通过的验收命令**；默认会覆盖已保存的XML，后续修复验证宜指定独立输出位置：

```bash
python experiment/h2_contact_lab/view_h2_plus_mujoco_terrains.py --xml-output /path/to/h2_plus_four_terrains_review.xml
```

`--urdf`、`--spacing`、`--root-height`分别覆盖源模型、地形间距和根部高度。程序不调用 `mj_step`；即使静态显示修复，也仍需单独验证物理接触和控制。

## outputs中的历史内容

| 目录／文件 | 身份及能说明的范围 |
|---|---|
| `inspect_h2_QxAfbW/` | 保存从Linux返回的普通H2 MJCF清单：Python 3.11.16、MuJoCo 3.3.7，模型加载成功，物理步数0，`h2_plus_correspondence=not_verified`。不是当前改装H2 Plus运行报告。 |
| `windows_checks_20261015/`、`windows_isaac_checks_20261015/` | 已移除的旧H2采集原型的Windows检查；目录名含Isaac不表示实际启动过Isaac。 |
| `windows_footrix_manual_checks_20261015/` | 旧手册外形／规则网格映射方案的检查与布局；保留历史，不作为现行布局。 |
| `windows_footrix_points_checks_20261015/` | 图像估计的154点布局、PNG／SVG及旧代码检查：52项中49通过、3项MuJoCo集成检查跳过。左足方向曾获用户确认；毫米坐标为估计、右足为未验证镜像、编号不是硬件寄存器顺序。 |
| `h2_plus_four_terrains.xml` | 当前MuJoCo辅助脚本导出的静态XML，存在上述结构缺陷；不是Isaac USD、采集轨迹或成功展示证明。 |

旧 `h2_contact/`、`scripts/`、配置和测试源码已在先前收敛入口时移除；遗留日志和缓存中的文件名不是当前可运行接口。当前两个Python源文件的状态应分别判断，不把旧49项通过结果转移给新入口。

## 接续顺序

Isaac路线：在已有Ubuntu环境先生成并检查USD场景，再按实际安排做smoke→collect→validate。MuJoCo展示路线：先修关节树／origin等导出问题，再重新生成和核对静态展示。两条路线均不凭CAD、网格或文件存在推定FOOTRIX／IMU读数、物理接触或研究闭环已经完成。
