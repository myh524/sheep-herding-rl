# AI Agent 协作框架 Spec (v2)

## Why

项目已进入优化阶段，需要多个专业AI Agent协作完成后续开发。用户希望形成一个自动化、完善的AI团队，减少手动协调成本。

## What Changes

- 创建 `.trae/agents/` 目录存放各Agent嘱托文档
- 创建 `.trae/share/` 目录作为Agent信息共享中心
- 定义7个专业Agent角色（新增项目经理、团队协作员）
- 建立基于"触发信号"的工作流程
- 创建Agent创建描述文档

## Impact

- Affected specs: 项目协作流程
- Affected code: 无代码变更，新增文档和目录结构

---

## ADDED Requirements

### Requirement: Agent角色定义（7个）

#### 1. 项目经理Agent (Project Manager)
- **职责**：整体协调、任务分配、进度跟踪、重大决策询问
- **不参与**：具体代码工作
- **核心文件**：`.trae/share/`
- **输出**：任务分配、进度报告、决策建议
- **特殊能力**：扮演用户角色，重大决策时询问用户
- **双线程工作模式**：
  1. **主线程**：执行用户交给的任务（发布任务、回答问题等）
  2. **检查线程**：每次执行任务时自动检查`share/status.md`，发现问题及时报告

#### 2. 团队协作员Agent (Team Coordinator)
- **职责**：AI Agent团队协作管理、团队优化建议
- **不参与**：项目开发工作、项目决策
- **核心文件**：`.trae/RULES.md`, `.trae/AGENT_DESCRIPTIONS.md`, `.trae/agents/`
- **输出**：团队协作改进建议、规则优化建议、Agent配置建议
- **特殊能力**：
  - 监督所有Agent工作，发现协作问题
  - 建议改进项目规则
  - 建议完善Agent描述
  - 建议增加或调整Agent角色
- **工作原则**：让整个AI Agent团队以更协调、可持续的方式运作

#### 3. 算法工程师Agent (Algorithm Engineer)
- **职责**：PPO算法优化、超参数调优、新算法探索
- **核心文件**：`algorithms/`, `train_ppo.py`
- **输出**：训练配置、算法改进、性能分析

#### 4. 环境工程师Agent (Environment Engineer)
- **职责**：环境建模、奖励函数设计、物理仿真优化
- **核心文件**：`envs/`
- **输出**：环境改进、奖励函数调整、物理模型优化

#### 5. 可视化工程师Agent (Visualization Engineer)
- **职责**：训练可视化、结果展示、调试工具开发
- **核心文件**：`visualize.py`, `analyze_training.py`
- **输出**：可视化功能、调试工具、演示材料

#### 6. 质量分析师Agent (Quality Analyst)
- **职责**：发现问题、提出改进建议、生成诊断信息
- **核心文件**：全项目
- **输出**：问题报告、改进建议、诊断日志设计
- **特殊能力**：通过观察训练/可视化日志发现潜在问题

#### 7. 文档工程师Agent (Documentation Engineer)
- **职责**：文档维护、API文档、使用指南
- **核心文件**：`README.md`, `README_CN.md`
- **输出**：文档更新、使用指南、变更记录

---

### Requirement: 信息共享中心 (Share Directory)

创建 `.trae/share/` 目录，结构如下：

```
.trae/share/
├── tasks/                    # 任务中心
│   ├── pending.md           # 待领取任务
│   ├── in_progress.md       # 进行中任务
│   └── completed.md         # 已完成任务
├── reports/                  # 工作报告
│   ├── algorithm/           # 算法工程师报告
│   ├── environment/         # 环境工程师报告
│   ├── visualization/       # 可视化工程师报告
│   ├── quality/             # 质量分析师报告
│   └── documentation/       # 文档工程师报告
├── decisions/                # 决策记录
│   └── decisions.md         # 重大决策及用户确认
├── team/                     # 团队协作
│   └── coordinator.md       # 团队协作员建议
└── status.md                 # 项目状态总览
```

---

### Requirement: 工作流程

#### 基本流程

```
用户 ──沟通──> 项目经理Agent ──发布任务──> share/tasks/pending.md
                                              │
用户 ──"1"──> 专业Agent ──领取任务──> share/tasks/in_progress.md
                    │
                    └──完成工作──> share/reports/{role}/
                                    │
项目经理Agent ──审核──> share/tasks/completed.md
```

#### 详细步骤

1. **用户与项目经理沟通**：描述需求或问题
2. **项目经理分析并发布任务**：写入 `share/tasks/pending.md`
3. **用户在专业Agent窗口说"1"**：触发Agent工作
4. **专业Agent执行**：
   - 读取 `share/tasks/pending.md` 领取任务
   - 移动任务到 `share/tasks/in_progress.md`
   - 执行工作
   - 写入报告到 `share/reports/{role}/`
   - 移动任务到 `share/tasks/completed.md`
5. **项目经理审核**：检查完成情况，更新 `share/status.md`

---

### Requirement: 触发信号规则

当用户在Agent窗口说"1"时，Agent应：

1. **读取项目规则**：`.trae/RULES.md`
2. **识别自身角色**：根据窗口描述或上下文
3. **检查待领取任务**：读取 `share/tasks/pending.md`
4. **领取并执行**：找到属于自己的任务，开始工作
5. **汇报结果**：写入 `share/reports/{role}/`

---

### Requirement: 项目规则文件

创建 `.trae/RULES.md`，包含：
- 各Agent角色定义
- 工作流程说明
- 文件读写规范
- 任务领取规则
- 冲突避免机制

---

### Requirement: Agent创建描述文档

创建 `.trae/AGENT_DESCRIPTIONS.md`，用于创建智能体时的描述。

每个Agent描述包含：
- 角色名称
- 职责范围
- 核心文件
- 触发条件
- 工作流程
- 与其他Agent的协作点

---

### Requirement: 决策询问机制

项目经理Agent在以下情况需要询问用户：
- 算法重大变更（如更换算法）
- 架构调整（如多智能体改造）
- 资源决策（如训练时长）
- 风险操作（如删除功能）

询问格式写入 `share/decisions/decisions.md`，等待用户确认。

---

### Requirement: 项目经理双线程工作模式

项目经理Agent每次被用户触发时，执行以下两个线程：

#### 主线程（用户任务）
1. 接收用户指令
2. 分析需求
3. 发布任务或回答问题
4. 更新相关文件

#### 检查线程（自动执行）
1. 读取 `share/status.md`
2. 检查是否有：
   - 未被领取的过期任务
   - 长时间未完成的任务
   - 专业Agent报告的问题
   - 需要用户确认的决策
3. 如发现问题，向用户报告

#### 检查线程示例输出

```
📊 项目状态检查报告：

⚠️ 发现问题：
- 任务#3（奖励函数优化）已超过预计时间，环境工程师可能遇到困难
- 任务#5等待用户确认：是否将PPO更换为SAC？

✅ 正常进行：
- 任务#4（可视化调试）已完成，报告在 share/reports/visualization/

💡 建议：
- 请在环境工程师窗口说"1"询问进度
- 请确认是否更换算法
```

---

### Requirement: 团队协作员工作模式

团队协作员Agent每次被用户触发时，执行以下工作：

#### 工作内容
1. **读取团队状态**：检查所有Agent的工作报告和share目录
2. **分析协作问题**：
   - 任务分配是否合理
   - Agent之间是否有冲突
   - 是否有Agent工作负荷过重/过轻
   - 规则是否需要更新
   - Agent描述是否需要完善
3. **生成建议报告**：写入 `share/team/coordinator.md`

#### 检查清单
- [ ] 各Agent任务分配是否均衡
- [ ] 是否有任务长时间无人领取
- [ ] 是否有Agent重复做同一工作
- [ ] 规则是否清晰，是否需要补充
- [ ] Agent描述是否准确，是否需要更新
- [ ] 是否需要新增Agent角色
- [ ] 是否需要调整现有Agent职责

#### 建议报告示例

```
🤝 团队协作检查报告：

📋 发现的问题：
1. 算法工程师和环境工程师任务分配不均衡
   - 算法工程师：5个任务
   - 环境工程师：1个任务
2. 任务#2描述不够清晰，导致两个Agent都认为属于自己的职责

💡 改进建议：
1. 重新分配任务#4、#5给环境工程师
2. 更新RULES.md，明确"奖励函数"属于环境工程师职责
3. 建议在AGENT_DESCRIPTIONS.md中补充边界情况说明

🔄 需要用户确认：
- 是否采纳以上建议？
```

---

## MODIFIED Requirements

### Requirement: README留言迁移

将README中的Agent留言迁移到 `.trae/agents/` 目录下的嘱托文档。

---

## 潜在问题与解决方案

### 问题1：任务冲突
**问题**：多个Agent可能同时领取同一任务
**解决**：任务标注`assigned_role`，只有对应角色的Agent才能领取

### 问题2：信息同步延迟
**问题**：Agent无法实时看到其他Agent的更新
**解决**：每次工作前先刷新读取share目录

### 问题3：依赖任务
**问题**：任务B依赖任务A，但A未完成
**解决**：任务标注`depends_on`，Agent检查依赖是否完成

### 问题4：用户忘记触发
**问题**：项目经理发布任务后，用户忘记触发专业Agent
**解决**：项目经理在报告中提醒用户"请在XX Agent窗口说1"

### 问题5：Agent无法主动提醒
**问题**：Agent无法主动通知用户
**解决**：项目经理Agent采用双线程工作模式，每次执行任务时自动检查`share/status.md`，发现问题及时报告给用户

---

## 文件结构总览

```
.trae/
├── RULES.md                    # 项目规则（所有Agent必读）
├── AGENT_DESCRIPTIONS.md       # Agent创建描述
├── agents/                     # Agent嘱托文档
│   ├── project_manager.md
│   ├── team_coordinator.md
│   ├── algorithm_engineer.md
│   ├── environment_engineer.md
│   ├── visualization_engineer.md
│   ├── quality_analyst.md
│   └── documentation_engineer.md
├── share/                      # 信息共享中心
│   ├── tasks/
│   │   ├── pending.md
│   │   ├── in_progress.md
│   │   └── completed.md
│   ├── reports/
│   │   ├── algorithm/
│   │   ├── environment/
│   │   ├── visualization/
│   │   ├── quality/
│   │   └── documentation/
│   ├── decisions/
│   │   └── decisions.md
│   ├── team/
│   │   └── coordinator.md
│   └── status.md
└── specs/                      # 规格文档
    └── agent-collaboration/
```
