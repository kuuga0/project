# H2 最小仿真基座

本轮只交付实际 MJCF 模型清单入口。使用现有 CPU MuJoCo 读取模型，不创建窗口，不调用 `mj_forward` 或 `mj_step`，不接入 Isaac Lab、控制器、训练或地形。先取得实际清单，再开展 Isaac Lab H2 资产接入。脚本不修改第三方模型，路径全部由参数传入。

## 状态与证据（2026-10-15）

| 层次 | 实际状态 |
|---|---|
| 已实现 | `scripts/inspect_h2_model.py`、标准库测试、下述 Ubuntu 命令；成功和失败均尝试生成 JSON，完整异常同时输出到标准错误。 |
| Windows 已检查 | 当前解释器为 Python 3.13.12；Python 3.11 语法检查通过；13 项测试中 10 项离线测试通过、3 项真实 MuJoCo 集成测试因缺包跳过。记录在 `outputs/windows_checks_20261015/`。限定项目范围内未找到 H2 模型，当前 Python 无 MuJoCo；未运行真实模型检查，也未安装仿真环境。 |
| 用户此前在 Ubuntu 已验证 | 在 `project` Conda 环境、Python 3.11、MuJoCo 3.3.7 下加载下述 H2 XML 并执行 `mj_forward` 成功，`nbody/njnt/nu/nq/nv = 33/32/31/38/37`。个人 Isaac Lab 源码 2.3.2、Isaac Sim 5.1.0.0、PyTorch 2.7.0+cu126、IPython 8.39.0、psutil 5.9.8；修改依赖后 `create_empty.py --headless` 已短时复测。这些是用户提供的结果，未由 Windows 重新验证。 |
| Ubuntu 待验证 | 本轮新脚本对实际 H2 的完整清单导出及 3 项真实 MuJoCo 集成测试；H2 在 Isaac Lab 中的加载、站立、控制、采数均未验证；MJCF 与目标 H2 Plus 的本体、足部及末端对应仍待人工核对。 |

用户报告 `pip check` 仍有 FastAPI 0.115.7 与 Starlette 0.49.1 的声明冲突，IOMMU 警告仍存在，本工程未改包、驱动或系统设置。Ubuntu 为独立计算机，由用户负责 Git 提交／推送、Ubuntu 拉取和执行；本轮没有自动远程通道，也未配置 SSH、隧道或 Ubuntu Codex 登录。

已知 Ubuntu 位置（仅运行示例使用，不写入通用逻辑）：

- 课题仓库：`/home/user/WYP/project`
- Isaac Lab：`/home/user/WYP/robot_deps/IsaacLab`
- Unitree 依赖：`/home/user/WYP/robot_deps/unitree_mujoco`
- H2 MJCF：`/home/user/WYP/robot_deps/unitree_mujoco/unitree_robots/h2/h2_mujoco.xml`

## Ubuntu 运行：保存标准输出、错误和 JSON

用户自行完成代码同步后，在已能使用 Conda 的 Bash 终端复制以下整组命令。`OUTPUT_ROOT` 是结果目录，可改为自己指定的位置；每次创建新子目录，避免把旧日志当作新结果。命令不安装依赖，也不修改第三方资产。

```bash
conda activate project && (
  set -eu
  cd /home/user/WYP/project/实验/h2_contact_lab
  OUTPUT_ROOT="$PWD/outputs"
  mkdir -p "$OUTPUT_ROOT"
  RUN_DIR="$(mktemp -d "$OUTPUT_ROOT/inspect_h2_$(date +%Y%m%d_%H%M%S)_XXXXXX")"

  # unittest 的正常进度也写入 stderr；以退出码和日志末尾判断通过。
  test_status=0
  python -m unittest discover -s tests -v \
    >"$RUN_DIR/tests.stdout.log" 2>"$RUN_DIR/tests.stderr.log" || test_status=$?
  printf '%s\n' "$test_status" >"$RUN_DIR/tests.exitcode.txt"

  inspect_status=0
  python scripts/inspect_h2_model.py \
    --model /home/user/WYP/robot_deps/unitree_mujoco/unitree_robots/h2/h2_mujoco.xml \
    --output "$RUN_DIR/h2_model_inventory.json" \
    >"$RUN_DIR/inspect.stdout.log" 2>"$RUN_DIR/inspect.stderr.log" || inspect_status=$?
  printf '%s\n' "$inspect_status" >"$RUN_DIR/inspect.exitcode.txt"

  printf '结果目录：%s\n测试退出码：%s\n模型检查退出码：%s\n' \
    "$RUN_DIR" "$test_status" "$inspect_status"
  if [ "$test_status" -ne 0 ]; then exit "$test_status"; fi
  exit "$inspect_status"
)
```

请回传打印出的目录内的 `h2_model_inventory.json`、`inspect.stdout.log`、`inspect.stderr.log`、两个退出码及测试日志。若没有 JSON，先回传标准错误；不要为消除报错开始全量安装。JSON 被写入命令指定的目录，不会写回 Unitree 模型目录。

正常情况下本轮脚本退出码为 `0`，且 JSON 中 `success`、`load_success` 都是 `true`、`stage` 是 `complete`。用户此前的五项计数可用于对照，但代码不将其硬编码成通过条件；若不同，应结合实际路径、摘要、版本和清单核对。Ubuntu 的集成测试应实际执行，看到 `skipped` 不算完成真实 MuJoCo 检查。

## 参数、内容与解释边界

```text
python scripts/inspect_h2_model.py --model <模型入口.xml> --output <新结果.json>
```

两项参数必填。相对路径按运行时工作目录解析；`~` 会展开，JSON 记录解析后的实际绝对路径。MJCF 的 `include` 和资产路径由 MuJoCo 的 `from_xml_path` 按模型规则处理，不复制模型或改变工作目录来补路径。

| JSON 字段 | 可检查内容 |
|---|---|
| `source`、`runtime` | 实际模型路径、入口文件字节数与 SHA-256、MuJoCo 版本、解释器与系统。摘要只覆盖入口文件，**不覆盖 include、meshes、textures 的整个依赖闭包**，不能据此证明全部资产版本相同。 |
| `counts` | 五项基本计数及几何、site、mesh、tendon、sensor、接触对、排除项、等式约束、keyframe 等计数。`nbody` 包含 world。 |
| `inventory.bodies` | 全部刚体名称、父子树、所属关节／geom／site、质量／惯量和局部位姿；world 的派生父节点为 `null`，原始 `body_parentid` 仍保留。 |
| `inventory.joints` | 类型、所在刚体、限位标志和范围、轴／锚点、位置与速度索引、默认位置、阻尼／armature／摩擦损失、直接关节驱动的执行器 ID。自由／球关节不按单自由度处理。 |
| `inventory.actuators` | 实际传动类型与目标名称／ID、控制与力／激活范围及各限制标志、gear、动力学／增益／偏置类型与参数；支持 joint、jointinparent、tendon、site/refsite、slider-crank 和 body 目标，不假定所有执行器直驱单关节。 |
| `inventory.geoms`、`sites`、`meshes` | 全量几何名称／类型、所属刚体、局部位置／四元数、尺寸、碰撞位掩码、摩擦和求解参数；mesh 关联及编译后顶点／面计数、变换。保留无名称对象的 ID，名称为 `null`。site 是标记／测量结构，不能因其存在就推定碰撞或传感器硬件。 |
| `contact_pairs`、`contact_exclusions`、`options` | 显式接触对及其参数、排除刚体对、重力／步长／全局开关。几何组号和外观不是碰撞开关；位掩码、显式 pair、排除项和全局选项需合看，清单不声称某个 geom 已实际接触。以上字段均在 `inventory` 下。 |
| `tendons`、`sensors`、`equalities` | 路径／耦合结构、传感器对象与维度／地址、等式约束原始配置。传感器清单不等于观测已采集或真实 FOOTRIX 已建模。以上字段均在 `inventory` 下。 |
| `inventory.structure_hints` | 足部／末端关键词匹配、匹配来源、刚体祖先链，以及所有非 world 叶节点。**仅为人工检查线索**；不完整覆盖不含关键词的命名，不将叶节点自动认定为手脚，也不推断 H2 Plus 匹配。可按 body 的 `geom_ids`／`site_ids` 展开原始结构。 |
| `error`、`stage`、`scope` | 捕获的异常类型、消息和完整 Python traceback；区分文件、导入、模型加载和清单提取阶段。`load_success=true` 而 `success=false` 表示加载成功但导出失败；不当作完整成功。动态运行与 H2 Plus 对应保持未验证。 |

`fields` 内保留 MuJoCo 编译后原始字段名及值。有关节／执行器 `range` 不等于启用了限位，要结合对应 `limited` 和全局选项解释；关节角度已是编译后的弧度，不能按 XML 的 degree 原值读。`geom_size` 的含义依类型变化，例如 box 是半尺寸；局部位置不是当前世界坐标，四元数顺序为 w、x、y、z。执行器的 `gear` 保留 MuJoCo 原始约定，不改造成项目扳手记录；它的六分量顺序不能直接套用项目“力矩在前、力在后”的记录格式。

接口依据：[MuJoCo 3.3.7 mjModel 与枚举](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APItypes.html)、[Python 模型加载接口](https://mujoco.readthedocs.io/en/3.3.7/python.html)、[MJCF 字段语义](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html)。本轮只据此核对 API 使用，未据网页宣称本地 H2 资产可用。

退出码 `1` 表示检查失败，仍尝试保存含完整异常的 JSON；`2` 表示参数错误、输出创建或 JSON 写入失败，查看标准错误。已有输出文件一律拒绝覆盖，包括把 `--output` 指向模型自身。目录会自动创建；输出不可写时不能保证 JSON 存在。进程被系统终止或原生库崩溃时，Python 也不能保证生成报告，所以要保留两路日志和退出码。

## 测试与下一步

`python -m unittest discover -s tests -v` 只需标准库即可执行离线部分：文件／缺包／加载／提取错误分层、摘要、CLI 失败 JSON 与日志、输出保护、严格 JSON、传动目标和结构线索。3 项集成测试需要现有 MuJoCo，使用临时合成 MJCF 检查真实 API、自由／球／标量关节、编译后限位、多种传动、mesh／碰撞／sensor 和相对 include；这些合成文件不是 H2、不会被保存成 H2 结果。

下一轮根据 Ubuntu 的实际清单核对 H2 Plus 配置，确定足部碰撞、末端结构、关节／执行器映射及资产缺口，再接 Isaac Lab。模型可加载不代表能站立、闭环控制或持续采数；这些结果需要各自的运行证据。
