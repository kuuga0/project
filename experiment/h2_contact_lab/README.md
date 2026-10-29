# H2 Plus：任务配置、训练与仿真采集

更新日期：2026-10-29。训练／回放继续使用 `train.py --task`、`play.py --task`。现已通过原生RecorderManager按 `--record_sensors` 开关采集训练场景中挂载的传感器，默认关闭；开启后默认记录环境0；旧 `tools/run_h2_plus.py` 独立仿真采集入口已删除。本轮仅完成资产、传感器配置和记录逻辑调整，没有启动仿真或训练，也未进行运行验证。

## 2026-10-28目录清理

根目录只保留四个现用Python脚本：`train.py`、`play.py`、`cli_args.py`、`list_tasks.py`。独立工具曾移至 `tools/run_h2_plus.py`，已于2026-10-29被任务内RecorderManager替代并删除；原check／smoke／collect／validate／terrain命令不再提供。已删除未接入训练且导出不完整的 `view_h2_plus_mujoco_terrains.py` 和 `h2_plus_four_terrains.xml`，不再提供该展示入口。

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
python train.py --task Isaac-H2Plus-Blind-Locomotion-v0 --headless --num_envs 2048 --max_iterations 3000
python train.py --task Isaac-H2Plus-Blind-Locomotion-v0 --headless env.rewards.action_rate.weight=-0.02
```

默认PPO为3000轮，每50轮保存checkpoint；平地和低摩擦默认1024环境，柔性默认64环境，Play配置为16环境。算法参数在对应agent配置修改，也可用Hydra的 `agent.algorithm.learning_rate=...` 等覆盖。没有额外安装依赖。

## 从指定checkpoint继续训练

训练和回放均接受等价参数 `--checkpoint`、`--ckpt`，直接接收本地文件的完整路径；指定后自动启用续训，不需要 `--load_run` 或 `--resume`：

```bash
python train.py --task Isaac-H2Plus-Blind-Locomotion-v0 --headless --checkpoint /完整路径/model_100.pt
```

也接受相对于当前终端工作目录的文件路径，并在启动Isaac Sim前检查文件是否存在。显式路径直接交给训练器加载，不在运行目录内做正则匹配；新训练日志和权重仍写入本项目的 `log/`，不会覆盖输入checkpoint。不指定 `--checkpoint` 时仍可使用 `--resume` 自动查找最近运行，旧 `--load_run` 仅用于这种自动查找模式。指定路径时该参数无需提供。

## 奖励权重与数值参数调优

统一编辑 [rsl_rl_locomotion_rewards.yaml](h2_plus_lab/tasks/locomotion/agents/rsl_rl_locomotion_rewards.yaml)。六个训练／回放配置在创建时自动读取该文件，不需要增加启动参数，修改后重新启动任务生效。

```yaml
rewards:
  track_lin_vel_xy:
    weight: 1.5
    params:
      std: 0.5
  action_rate:
    weight: -0.01
```

文件已列出当前11项奖励／惩罚权重，以及速度跟踪尺度、腾空奖励阈值和静止速度阈值，初始值与原配置一致。权重设为0可禁用该项；遗漏的项或参数沿用Python配置中的值。奖励函数、关节选择、传感器绑定等结构仍在 `env_cfg.py` 和 `mdp/` 中维护，YAML只接受已有奖励项的权重和已有数值参数。

应用顺序为Python默认值 → YAML → 具体任务类的覆盖 → Hydra命令行覆盖。例如 `env.rewards.action_rate.weight=-0.02` 仍可临时覆盖YAML。本次运行实际使用的数值会随原有训练配置保存到 `log/rsl_rl/<实验名>/<运行时间>/params/env.yaml`。非法项名、非有限数值、非正std及负时长／速度阈值会报错，避免静默使用错误参数；本文件不是PPO算法参数文件，学习率等仍在 `agents/rsl_rl_ppo_cfg.py`。

## 训练中采集传感器（RecorderManager）

传感器采集默认关闭；训练或回放显式添加 `--record_sensors` 才会启用，无需另开程序。不加开关时，入口在创建环境前清空记录器配置，不缓存或导出传感器轨迹。开启后默认记录环境0，每个控制步采样一次；当前物理步长1/120秒、decimation为4，所以记录频率为30 Hz。IMU配置的120 Hz更新周期不等于保存了完整120 Hz序列。记录项使用 `record_post_step`，位置在场景传感器更新之后、自动reset之前；不使用Lab 2.3.2中早于场景更新的物理子步回调。

配置位于 `env_cfg.py`：`H2PlusSensorRecorderTermCfg` 设置环境选择和采样步距，`H2PlusRecorderCfg` 配置原生回合导出，再由 `H2PlusLocomotionEnvCfg.recorders` 挂载。实现仅在 `mdp/recorders.py`，三种地形和Play继承同一记录项。示例（回放同样支持 `--record_sensors`）：

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
| `contact/<名称>/` | 左右足和躯干的世界坐标系聚合法向接触力、finite和in_contact；足部另含腾空／接触计时。|
| `footrix/<left|right>/` | 每足154个可视化测量点的三维力，力来自完整足底碰撞的PhysX接触报告，并按最近测量点守恒分配；形状为 `[环境数, 154, 3]`，另有finite和in_contact。154点本身无碰撞，不承担足底支撑。|
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
├── h2_plus_lab/
│   ├── assets/h2_plus.py          # URDF、初始姿态与驱动
│   └── tasks/
│       ├── __init__.py            # 自动发现直属任务包
│       └── locomotion/
│           ├── __init__.py        # gym.register：任务ID与配置入口
│           ├── env_cfg.py         # 场景、共用MDP、三种地形的训练／回放配置
│           ├── sensor_cfg.py      # 接触、IMU及保留的传感接口
│           ├── terrain_cfg.py     # 地形几何与材料
│           ├── agents/
│           │   ├── __init__.py
│           │   ├── rsl_rl_ppo_cfg.py
│           │   └── rsl_rl_locomotion_rewards.yaml  # 奖励调优入口
│           └── mdp/
│               ├── rewards.py    # H2足部与速度跟踪奖励
│               ├── terminations.py
│               └── recorders.py  # 训练场景传感器记录项
├── tests/                        # 注册、记录回调及HDF5读写的离线检查
├── log/
└── output/
```

`locomotion/` 直属脚本已从8个精简为4个：`__init__.py` 注册任务，`env_cfg.py` 集中场景、共用MDP和具体任务配置，`sensor_cfg.py` 与 `terrain_cfg.py` 分别维护传感器和地形参数。已将原场景文件及三个地形环境文件合入 `env_cfg.py`，不保留转发文件。修改共用MDP或某个具体任务，都在该文件对应类中进行；六个任务ID、配置类名和参数保持不变。奖励权重和数值阈值优先编辑 `agents/rsl_rl_locomotion_rewards.yaml`；奖励／终止函数的实现放入 `mdp/`，并在 `mdp/__init__.py` 导出。PPO网络、学习率、采样长度和保存间隔放入 `agents/rsl_rl_ppo_cfg.py`。已移除空的 `H2PlusDeferredInterfaces` 聚合层及未使用的ANYmal专用对称变换；H2足部奖励和躯干终止已分别归入对应MDP文件，FOOTRIX由完整足底接触报告和154点可视化布局共同实现，RayCaster仍只保留接口。

执行顺序为：解析参数 → AppLauncher启动Isaac Sim → 导入任务注册 → Hydra读取环境与agent配置 → `gym.make()` → RSL-RL训练／回放。新任务不需要修改 `train.py` 或 `play.py`。

## 已注册任务

| 地形 | 训练任务ID | 环境配置 |
|---|---|---|
| 平地 | `Isaac-H2Plus-Blind-Locomotion-v0` | `env_cfg.py:H2PlusFlatEnvCfg` |
| 低摩擦双区域 | `Isaac-H2Plus-Blind-Locomotion-LowFriction-v0` | `env_cfg.py:H2PlusLowFrictionEnvCfg` |
| 柔性地面 | `Isaac-H2Plus-Blind-Locomotion-Deformable-v0` | `env_cfg.py:H2PlusDeformableEnvCfg` |

每个训练任务都有在 `-v0` 前加 `-Play` 的回放任务，对应同文件中后缀 `_PLAY` 的类。原有任务ID保持不变，新增低摩擦和柔性Play注册；这些特殊地形本轮未运行。六个任务暂时共用 `H2PlusBlindPPORunnerCfg`，需要独立实验日志／超参数时可继承该类，设置独立 `experiment_name`，再更改注册中的agent入口。

## 自己增加环境并注册

例如新增低速行走任务，在 `h2_plus_lab/tasks/locomotion/env_cfg.py` 末尾新增配置类：

```python
@configclass
class H2PlusSlowEnvCfg(H2PlusFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.3)
```

然后在 `locomotion/__init__.py` 增加标准Gym注册：

```python
gym.register(
    id="Isaac-H2Plus-Slow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:H2PlusSlowEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:H2PlusBlindPPORunnerCfg",
    },
)
```

之后可用 `python list_tasks.py` 查看，再手动执行 `python train.py --task Isaac-H2Plus-Slow-v0 --headless`。上面是添加方法示例，当前没有预先注册Slow任务。

若新增完全不同的任务，在 `h2_plus_lab/tasks/新任务名/` 创建自己的 `__init__.py`、环境配置、`agents/` 和按需使用的 `mdp/`；将 `gym.register()` 放在该包的 `__init__.py` 中，父包会自动导入。注册入口使用字符串，避免在只列任务时加载仿真配置。新agent配置也只在加载任务时导入。

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

RayCaster配置仍保存在 `sensor_cfg.py`，但没有挂载到首版场景。躯干 IMU 挂在 `torso_link` 原点，双足 IMU 挂在各自 `ankle_pitch_link` 并使用已给出的偏移和旋转；三路 IMU 以120 Hz更新、保留6帧历史、输出含重力的机体坐标系加速度和角速度，当前记录层保存原始模拟输出，不额外加噪声；`H2_PLUS_IMU_NOISE_CFG` 仅保留参数，未应用到新记录器。IMU不进入首版策略观测。FOOTRIX保留为每个训练／回放场景中的足底测量配置：左右脚各154个无碰撞固定可视化标记；完整 `ankle_pitch_link` 足底碰撞仍负责全部足地接触，PhysX接触报告回调按最近测量点守恒分配接触冲量。场景始终挂载完整足底接触传感器和标记，但点力回调只在显式添加 `--record_sensors` 时运行并写入 `footrix`，不进入首版盲走策略。单元坐标来自图像估计的左脚154点布局，并暂按同一布局用于右脚；安装标定和硬件编号对应仍留有替换接口。

## 2026-10-28此前修复与运行记录

**此前由助手启动的训练已停止；后续由用户手动启动。** 平地盲走已实际完成环境创建、物理步进和PPO更新，但未完成训练或验证行走效果。低摩擦、柔性场景本轮仅修代码，尚未运行。

已修复：

- 删除 `DeformableBodyMaterialCfg` 不支持的 `static_friction`；补齐URDF转换驱动的零刚度／阻尼必填值，执行器仍使用80／8增益。
- Gym注册改为字符串入口，避免在 `AppLauncher` 前导入PPO与仿真运行时；现已改为根目录直接入口，任务注册在AppLauncher之后执行，CLI辅助模块移至本地。
- 无TerrainImporter的低摩擦／柔性场景按环境局部坐标检查4米场地边界；环境间距改为5米，柔性场景关闭物理复制。
- 躯干传感器取消多body过滤；训练场景的双足聚合传感器只保留净接触力、历史和腾空计时，关闭PPO未使用的成对接触过滤／接触点／摩擦力记录，避免GPU平面过滤不支持的运行警告。该历史记录形成时FOOTRIX尚未挂载；当前配置保留完整足底接触报告和154个无碰撞测量标记，点力由记录器进行最近点守恒分配，RayCaster仍未挂载。
- 平地与低摩擦任务默认1024环境，柔性任务默认64环境；均可用 `--num_envs` 覆盖，平地play仍为16环境。

运行记录：首次有效启动日志迁至 `log/history/train_20261028_141313/console.log`；关闭多余过滤后的日志迁至 `log/history/train_20261028_141452/console.log`，后者记录PPO迭代0—7、累计196608步，随后按用户要求停止。两次试跑的参数保存在各自 `params/`；旧初始checkpoint、事件文件和失败启动日志已清理。这些是早期运行证据，不是已收敛策略。

环境判断更正：此前直接调用conda环境的Python，未执行激活脚本，因此误判为缺少torch。完整 `conda activate project` 会加载独立Isaac Sim的Python路径，本次实际使用Isaac Lab 2.3.2、Isaac Sim 5.1和PyTorch 2.7.0+cu128，在RTX 4090上进入训练；未安装新依赖。此次结构调整后直接使用已激活环境中的Isaac Lab，不再需要查找上游训练脚本，也不读取 `ISAACLAB_ROOT`；仍无需手动export。

局部崎岖地形仍只有场景工厂入口，没有独立Gym任务。初始姿态、高度、驱动增益及奖励仍需根据训练表现调整；早期回合主要因姿态倾倒终止，不能据PPO运行成功认定稳定行走。

旧独立采集工具已删除，以下资产信息继续供任务配置使用。

## 资产版本如何区分

以下文件均位于 [../robot_asset/h2_plus/](../robot_asset/h2_plus/)。计数排除了XML注释；文件名中的 `no_dex_hands` 不能代替结构核对。

| 文件 | 活动结构 | 用途／差异 |
|---|---|---|
| `H2_Plus.urdf` | 98 link；75 revolute＋22 fixed | 带灵巧手、原足；保留参照。 |
| `H2_Plus_mujoco.urdf` | 98 link；75 revolute＋22 fixed | 带手、原足；可视网格使用STL的版本。 |
| `H2_Plus_mujoco_with_footrix.urdf` | 98 link；75 revolute＋22 fixed | 已换新足，仍含灵巧手；不是当前去手默认版。 |
| `H2_Plus_no_dex_hands.urdf` | 32 link；31 revolute | 去手但仍引用原足网格。 |
| `H2_Plus_with_footrix_no_dex_hands.urdf` | 340 link；31 revolute＋308 fixed | **Isaac主入口默认**；新足，新增左右脚各154个无碰撞固定FOOTRIX可视化标记，部分本体可视网格为DAE。|
| `H2_Plus_mujoco_with_footrix_no_dex_hands.urdf` | 32 link；31 revolute | 旧MuJoCo展示使用的源资产；新足，可视网格为STL。URDF保留，展示脚本及生成XML已删除。 |

原始去手改装URDF均有49个本地网格引用；当前Isaac默认URDF另含308个无碰撞的固定FOOTRIX可视化标记link。源模型根link为 `pelvis`，floating joint示例位于注释中。Isaac转换器以 `fix_base=False` 请求浮动根，实际Articulation是否为浮动基座仍须运行时确认。URDF本身不生成IMU读数；任务场景另行挂载三路IMU。FOOTRIX坐标和接触点分配参数由任务配置写入，后续可替换为标定布局或新的分配模型。

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
