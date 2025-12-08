# 代码审查报告 (Code Review Report)

**日期**: 2025-12-08
**审查对象**: PowerPlan4 核心模块 (`config.yaml`, `data_processor.py`, `optimization_model.py`, `validator.py`)
**审查重点**: 约束条件、边界条件、单位换算、异常处理

---

## 1. 约束条件检查 (Constraint Checks)

### 1.1 输入参数合法性
*   **发现**: `ConfigManager` 仅加载 YAML 文件，未进行 Schema 验证。
    *   **风险**: `High`。如果配置文件缺失关键字段（如 `costs.wind.capex`），程序将在运行时深层崩溃，报错信息可能不直观。
    *   **建议**: 引入 Pydantic 或 Cerberus 进行配置验证，或在 `ConfigManager` 中添加关键字段检查。
*   **发现**: `data_processor.py` 假设输入 CSV 包含 `WindPower` 和 `SolarPower` 列。
    *   **风险**: `Medium`。如果列名不匹配，会默认填充 0 并仅记录警告，可能导致模型在无数据情况下“成功”运行。
    *   **建议**: 对关键数据列缺失的情况，应抛出异常而不是静默填充 0。

### 1.2 业务规则约束
*   **发现**: 弃电率约束 (`max_curtailment_rate`) 和新能源占比约束 (`min_re_share`) 依赖于手动构建的 `linopy` 约束。
    *   **状态**: 已修复维度识别问题，逻辑数学上正确。
*   **发现**: 储能时长约束固定为 `4.0` 小时。
    *   **风险**: `Low` (符合当前业务需求)。但代码中硬编码了 `min_duration` 和 `max_duration` 的处理逻辑，灵活性略低。

### 1.3 数据完整性
*   **发现**: 典型日权重计算依赖 `snapshot_weightings`。
    *   **风险**: `Medium`。代码中存在对 `DataFrame` 和 `Series` 的类型检查分支 (`isinstance`)，表明数据结构定义不够严谨。
    *   **建议**: 在 `build_model` 阶段统一 `snapshot_weightings` 为 `Series` 类型。

## 2. 边界条件测试 (Boundary Condition Testing)

### 2.1 极值处理
*   **发现**: `p_stor > 0` 和 `total_avail_energy > 0` 的检查防止了除以零错误。
    *   **状态**: Pass。
*   **发现**: `clip(0, 1)` 用于处理标幺值 (`pu`)。
    *   **状态**: Pass。

### 2.2 空集与异常数据
*   **发现**: `generate_kmeans_typical_days` 处理某月无数据的情况是跳过。如果所有月都无数据，返回空 DataFrame。
    *   **风险**: `High`。后续 `process_input_data_typical` 若收到空 DataFrame，虽然有检查，但可能导致流程中断。
*   **发现**: `n_clusters` 大于当月可用天数时，代码使用了 `min(n_clusters, len(features))` 进行保护。
    *   **状态**: Pass。

## 3. 单位换算系统 (Unit Conversion System)

### 3.1 成本单位
*   **规则**: 配置文件使用 `元/kW`，模型内部转换为 `元/MW`。
*   **检查**:
    *   `capex`: `costs['wind']['capex'] * 1000` -> Correct.
    *   `fuel_cost`: `0.185 元/kWh` * 1000 = `185 元/MWh` -> Correct.
    *   `marginal_cost` (Opex): `(元/kW * 1000) / 8760` -> `元/MW/h` -> Correct.

### 3.2 功率单位
*   **规则**: 统一使用 `MW`。
*   **发现**: `_process_load_data` 中 `load_mw = np.array(full_year_load) / 1000.0`。
    *   **风险**: `High`。代码假设输入负荷数据单位为 `kW`。如果输入文件是 `MW`，则负荷会被错误缩小 1000 倍。
    *   **建议**: 在配置文件中明确指定输入数据的单位，或在 CSV 表头中进行标注检查。

### 3.3 验证模块单位不一致 (CRITICAL)
*   **发现**: `validator.py` 中 `stor_p_nom = capacities['Storage_Power'] / eff`。
    *   **分析**: `capacities['Storage_Power']` 来自优化结果的 `p_nom_opt` (Battery_Discharge Link)。PyPSA 中 Link 的 `p_nom` 定义在 `bus0` (电池侧)。因此它已经是电池侧容量。
    *   **错误**: 再次除以效率导致验证模型中的储能放电容量被错误放大（例如效率 0.9，容量被放大了 1.11 倍），这可能导致验证结果过于乐观。
    *   **修正**: 应直接使用 `capacities['Storage_Power']`。

### 3.4 验证模块火电成本
*   **发现**: `validator.py` 中火电 `marginal_cost` 硬编码为 `100`。
    *   **错误**: 应从配置文件读取 `fuel_cost` (e.g. 185)。这会影响验证阶段的调度优先级。

## 4. 异常处理 (Exception Handling)

*   **发现**: 优化求解失败时有重试机制。
    *   **状态**: Good。
*   **发现**: `_add_policy_constraints` 若找不到维度名称，仅打印 Warning 并返回。
    *   **风险**: `High`。如果政策约束未添加，模型会算出“无弃电约束”的最优解（弃电率极高），用户可能误以为是模型逻辑正确但物理不可行。
    *   **建议**: 关键约束添加失败应抛出 `RuntimeError` 终止程序。

---

## 修正建议汇总 (Action Plan)

1.  **修复 Validator 储能容量逻辑**: 删除多余的效率除法。
2.  **修复 Validator 火电成本**: 读取配置文件的燃料成本。
3.  **增强约束鲁棒性**: `_add_policy_constraints` 失败时抛出异常。
4.  **明确负荷单位**: 确认输入数据单位，添加注释或配置项。
