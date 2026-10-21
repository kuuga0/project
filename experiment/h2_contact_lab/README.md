# H2 Plus Isaac Lab 接入与采集

唯一运行入口为 `run_h2_plus.py`。它提供静态资产检查、USD转换与短步进、连续状态采集三个模式；不修改机器人源资产。

## 默认资产

默认使用 `../robot_asset/h2_plus/H2_Plus_with_footrix_no_dex_hands.urdf`。该文件有32个link、31个本体转动关节和逐关节努力上限，左右新足网格引用可解析。用户指定的 `H2_Plus_mujoco_with_footrix.urdf` 仍含灵巧手链，不作当前默认。

URDF根link是 `pelvis`，显式floating joint示例被注释。转换器用 `fix_base=False` 请求浮动根，运行时还会确认Articulation不是固定基座。URDF没有IMU frame及安装外参，所以程序不生成IMU读数。

## Python 环境

你已有 Ubuntu Isaac Lab 环境，直接激活后运行下面命令即可。`requirements.txt` 只留给项目额外 pip 依赖；目前没有额外包需要安装。Isaac Lab、Isaac Sim 和 PyTorch 已由现有环境提供，不要再通过本项目的 requirements 文件重装。版本说明见 `ubuntu_environment.txt`。

若要在新建环境中部署，先按对应版本的 Isaac Lab 官方流程安装 Isaac Lab／Isaac Sim 和配套 PyTorch，再使用项目代码；本项目的 requirements.txt 不负责安装仿真运行时。
## 静态检查与运行

在项目根目录执行：

```bash
# 只检查URDF、关节和网格，不启动Isaac Sim
python experiment/h2_contact_lab/run_h2_plus.py --mode check

# 转换USD、加载浮动基座并短步进约1秒
python experiment/h2_contact_lab/run_h2_plus.py --mode smoke --headless --device cpu --steps 240

# 连续采集10秒：240 Hz物理步进、60 Hz记录
python experiment/h2_contact_lab/run_h2_plus.py --mode collect --headless --device cpu --duration 10 --physics-hz 240 --record-hz 60
```

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

工作区Python已运行 `--mode check` 并通过：32个link、31个本体转动关节、49个可解析网格引用和努力上限检查通过。当前尚未在Ubuntu运行requirements安装、USD转换、Articulation加载、PhysX接触采集或JSONL回读；这些需要在目标Ubuntu系统验证。
