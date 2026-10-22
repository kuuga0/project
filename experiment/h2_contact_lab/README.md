# H2 Plus Isaac Lab 接入、地形场景与采集

唯一运行入口为 `run_h2_plus.py`。它提供静态资产检查、USD转换与短步进、连续状态采集、离线数据回读，以及四类地形场景生成五种模式；地形支持 `--dry-run`，不修改机器人源资产。

## 默认资产

默认使用 `../robot_asset/h2_plus/H2_Plus_with_footrix_no_dex_hands.urdf`。该文件有32个link、31个本体转动关节和逐关节努力上限，左右新足网格引用可解析。用户指定的 `H2_Plus_mujoco_with_footrix.urdf` 仍含灵巧手链，不作当前默认。

URDF根link是 `pelvis`，显式floating joint示例被注释。转换器用 `fix_base=False` 请求浮动根，运行时还会确认Articulation不是固定基座。URDF没有IMU frame及安装外参，所以程序不生成IMU读数。

## Python 环境

你已有 Ubuntu Isaac Lab 环境，直接激活后运行下面命令即可。`requirements.txt` 只留给项目额外 pip 依赖；目前没有额外包需要安装。Isaac Lab、Isaac Sim 和 PyTorch 已由现有环境提供，不要再通过本项目的 requirements 文件重装。版本说明见 `ubuntu_environment.txt`。

若要在新建环境中部署，先按对应版本的 Isaac Lab 官方流程安装 Isaac Lab／Isaac Sim 和配套 PyTorch，再使用项目代码；本项目的 requirements.txt 不负责安装仿真运行时。

## 四类地形场景生成（不推进物理步）

`terrain`模式在同一USD场景中并排放置四个H2 Plus初始姿态与地形。Windows可先运行 `python experiment/h2_contact_lab/run_h2_plus.py --mode terrain --dry-run` 检查URDF和场景配置，不启动Isaac Lab。Ubuntu生成场景时运行 `python experiment/h2_contact_lab/run_h2_plus.py --mode terrain --headless --device cpu`；脚本转换URDF、写入地形与机器人USD prim并导出场景；不调用 `reset` 或 `sim.step`。输出默认位于 `experiment/h2_contact_lab/outputs/h2_plus_terrain_scenes/<UTC时间>/`，包含 `h2_plus_four_terrain_variants.usd` 和 `terrain_manifest.json`。

| 场景 | 当前几何／物理设置 |
|---|---|
| 普通完整硬地 | 4×4 m刚性地面，静／动摩擦系数暂设0.8／0.7，作为不刻意削弱的基线。 |
| 低摩擦与摩擦差异 | 同一平面按Y方向分为两侧，正Y侧低摩擦0.25／0.20，负Y侧基线0.8／0.7；模型轴向与左右脚对应关系仍需查看USD确认。 |
| 轻度可变形铺设 | 3×3 m、厚6 cm的有限厚度外形；当前碰撞体是刚性代理，尚未配置压缩、剪切、倾转或恢复的可变形物理。 |
| 局部有效支撑变化 | 相邻完整支撑板形成2.5 cm高度差，供接缝／边缘附近的支撑变化预览。 |

四个根部初始Z坐标默认设为世界坐标1.01 m，可用 `--root-height` 调整。场景导出只证明几何与初始位姿已写入；没有物理步进，所以不证明足底实际接触、静态平衡或站立能力。摩擦和高度差是待调整的场景参数，不是实测家居材料参数。

## 静态检查与后续物理仿真／采集命令

## Windows MuJoCo 静态展示

项目另有 [view_h2_plus_mujoco_terrains.py](./view_h2_plus_mujoco_terrains.py)，用于在本机已有的 MuJoCo Python 环境中生成并展示四种并排地形和一个 H2 Plus。脚本从 [H2_Plus_mujoco_with_footrix_no_dex_hands.urdf](../robot_asset/h2_plus/H2_Plus_mujoco_with_footrix_no_dex_hands.urdf) 生成可保存的 MuJoCo 场景 XML；唯一的 H2 Plus 默认放在第一块普通硬地上，其余三种地形作为相邻展示面。不调用 `mj_step`，因此只展示初始姿态，不验证接触、平衡或动力学。

```bash
python experiment/h2_contact_lab/view_h2_plus_mujoco_terrains.py
```

默认生成 [h2_plus_four_terrains.xml](./outputs/h2_plus_four_terrains.xml)。也可以显式指定 XML 输出位置：

```bash
python experiment/h2_contact_lab/view_h2_plus_mujoco_terrains.py --xml-output /path/to/h2_plus_four_terrains.xml
```

如果本机 MuJoCo 使用的是其它 URDF，可显式指定：

```bash
python experiment/h2_contact_lab/view_h2_plus_mujoco_terrains.py --urdf /path/to/model.urdf
```

四个展示场景分别为普通硬地、摩擦差异地面、铺设物刚性代理和局部支撑高度差。当前仓库没有现成的 H2 Plus MuJoCo 截图；已有 MuJoCo 图片属于其它机器人或实验，不宜直接作为 H2 Plus 四地形展示图。

当前在Windows继续开发并运行标准库检查、离线回读校验。
以下 `smoke` 和 `collect` 会推进物理步；当前仅需生成／查看地形场景时，不运行这两条命令。
场景USD生成与结构检查完成后，若进入实际仿真／采集阶段，再在已激活的现有Ubuntu Isaac Lab环境中运行以下命令：

```bash
# 只检查URDF、关节和网格，不启动Isaac Sim
python experiment/h2_contact_lab/run_h2_plus.py --mode check

# 转换USD、加载浮动基座并短步进约1秒
python experiment/h2_contact_lab/run_h2_plus.py --mode smoke --headless --device cpu --steps 240

# 连续采集10秒：240 Hz物理步进、60 Hz记录
python experiment/h2_contact_lab/run_h2_plus.py --mode collect --headless --device cpu --duration 10 --physics-hz 240 --record-hz 60
```

## 离线回读校验

Windows阶段可用合成样例开发和检查回读器。采集结束后可在Ubuntu本地或将运行目录同步到Windows，再用同一入口的标准库模式校验，不启动Isaac Sim：

```bash
python experiment/h2_contact_lab/run_h2_plus.py --mode validate --run-dir /path/to/h2_plus_runs/20261022_120000
```

把命令里的路径替换成实际运行目录。校验运行状态、metadata／JSONL一致性、采样步距和时间戳、关节向量维度、有限数值、接触有效标记和资产摘要，并汇总根部高度／速度范围、各关节位置范围、左右足接触有效比例和法向／摩擦力峰值。有效接触必须有有限的对地法向力、摩擦力和接触位置；无接触时用valid=false明确标记。返回码0表示结构完整性通过，不认证物理真实性、站立效果或传感器标定。
可选单关节位置激励示例：

```bash
python experiment/h2_contact_lab/run_h2_plus.py --mode collect --headless --device cpu --duration 10 --perturb-joint left_ankle_pitch_joint --perturb-amplitude 0.03 --perturb-frequency 0.5
```

USD写到 `outputs/h2_plus_usd/`。每次运行在 `outputs/h2_plus_runs/<UTC时间>/` 建新目录，保存 `metadata.json`；collect模式另写 `trajectory.jsonl`。`--output-dir PATH` 可以指定新的结果目录，目录必须不存在。去掉 `--headless` 可打开窗口。

## 记录内容与边界

`metadata.json`保存URDF、每个网格和采集脚本的SHA-256、USD路径、运行版本、Articulation关节顺序、URDF逐关节努力上限、初始状态、位置保持参数、地面摩擦假设及采样配置。位置保持增益只是仿真调试参数，不是H2硬件标定值。

`trajectory.jsonl`逐帧保存仿真时间、每关节位置目标／位置／速度、浮动根位姿／速度、左右足净法向合力、对地过滤法向力、对地过滤摩擦力及平均接触位置。法向和摩擦分别记录；数据是Isaac Lab ContactSensor／PhysX仿真值，不记录实际电机力矩。平均位置不是逐接触点分布；这些字段不是FOOTRIX逐单元三轴读数或标定结果。当前URDF没有IMU外参，因此没有IMU数据字段。地面摩擦系数是普通硬地仿真基线假设，不对应实测地面。

该初始采集用于检查模型接入和连续数据链，不含环境随机化、研究任务真值、能力标签或正式训练数据设计。零位位置保持可能倾倒；倾倒过程仍记录，采集完成不代表站稳或控制器验收。

## 当前验证状态

Windows已运行 `--mode check` 和 `terrain --dry-run` 并通过，入口Python语法编译通过；离线 `validate` 对合成正常记录通过，并拒绝缺字段记录。地形USD尚未生成；Isaac Lab仿真不在当前Windows环境运行。Ubuntu场景生成、Articulation加载、PhysX步进与接触采集均待后续执行。
