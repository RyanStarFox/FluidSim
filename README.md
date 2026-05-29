# FluidSim — FLIP/APIC/PolyPIC 3D 流体模拟对比

## 文件说明

| 文件 | 用途 | 负责人 |
|------|------|--------|
| `framework.py` | 统一 **3D** 框架（MAC网格/粒子/CG求解/透视渲染）+ FLIP 实现 | 同学 A |
| `plot_energy.py` | 动能/耗时/涡量对比绘图工具 | 阶段三使用 |
| `docs/POLYPIC_RESULTS.md` | PolyPIC 实现说明、运行记录与结果清单 | 同学 C |
| `slurm_polypic_full.sbatch` | PolyPIC 正式 GPU 任务脚本 | 同学 C |
| `slurm_polypic_smoke.sbatch` | PolyPIC 短烟测任务脚本 | 同学 C |

> **注意**：需要 Python 3.9–3.11（Taichi 1.7.x 不支持 Python 3.12+）。
> macOS 上推荐使用 `/usr/bin/python3`（系统自带的 3.9）。

## 快速开始

```bash
# 安装依赖（需要 Python 3.9–3.11，macOS 用系统自带的 python3）
/usr/bin/python3 -m pip install taichi pillow matplotlib numpy --user

# 跑 Dam Break（3D 溃坝）
/usr/bin/python3 framework.py dam_break

# 跑 Liquid Pouring（3D 倒水，带障碍物）
/usr/bin/python3 framework.py liquid_pouring 0.97

# 跑 PolyPIC（同学 C 分支）
/usr/bin/python3 framework.py dam_break polypic
/usr/bin/python3 framework.py liquid_pouring 0.97 polypic

# 两个场景都跑
/usr/bin/python3 framework.py all 0.97
/usr/bin/python3 framework.py all 0.97 polypic

# 后端说明（macOS M系列默认已自动用 Metal GPU）：
#   TI_ARCH=metal  python3 framework.py dam_break   # Apple Metal (默认)
#   TI_ARCH=vulkan python3 framework.py dam_break   # Vulkan
#   TI_ARCH=cpu    python3 framework.py dam_break   # 强制 CPU
#   TI_ARCH=cuda   python3 framework.py dam_break   # NVIDIA GPU（Linux）
```

> **性能参考（M2 Pro）**  
> | 后端 | 物理 | 渲染 | 总计 | 300帧时间 |
> |------|------|------|------|-----------|
> | CPU + numpy | 10 ms | 250 ms | 260 ms | ~78 s |
> | Metal + Taichi | 12 ms | 27 ms | 39 ms | **~12 s** |

## 框架接口（3D）

队友在 `fluid_step()` 内部实现自己的算法（APIC / PolyPIC）。**框架外的代码一行不许改。**

### 可用字段（全局）

```
u[NX+1, NY,   NZ  ]  x 方向速度（MAC x-faces）
v[NX,   NY+1, NZ  ]  y 方向速度（MAC y-faces，y 为重力方向）
w[NX,   NY,   NZ+1]  z 方向速度（MAC z-faces）
u_saved, v_saved, w_saved  速度快照（FLIP delta 用）
cell_type[NX, NY, NZ]  单元类型（FLUID=0 / SOLID=1 / AIR=2）
pressure [NX, NY, NZ]  压力场（CG 求解输出）
px, py, pz             粒子位置（MAX_PARTICLES）
pu, pv, pw             粒子速度（MAX_PARTICLES）
c0..c8                 3x3 仿射速度矩阵（APIC / PolyPIC 线性项）
q0..q17                PolyPIC 二阶多项式系数（每个速度分量 6 个）
num_particles[None]    当前激活粒子数
flip_ratio[None]       当前 PIC-FLIP 混合比
```

### 可用辅助函数

| 函数 | 说明 |
|------|------|
| `p2g_trilinear()` | 粒子 → 网格（3D trilinear 插值，8节点）|
| `save_velocities()` | 快照 u,v,w 到 u_saved,v_saved,w_saved |
| `add_gravity()` | 重力加速度（y 方向）|
| `enforce_boundary_velocity()` | Free-slip 边界（6面）|
| `compute_divergence()` | 散度 → div_field |
| `solve_pressure_cg()` | CG 求解 3D Poisson 方程（6邻居拉普拉斯）|
| `apply_pressure_gradient(dt)` | 压力梯度修正 u,v,w |
| `g2p_flip()` | 网格 → 粒子（3D PIC-FLIP 混合，清空高阶系数）|
| `p2g_polypic()` | PolyPIC 二阶多项式粒子 → 网格传输 |
| `g2p_polypic()` | PolyPIC 从网格拟合回粒子速度/线性项/二阶项 |

### 参数规格

| 参数 | 值 |
|------|-----|
| 网格分辨率 | 80 x 100 x 80（3D）|
| 格子尺寸 | DX = 1/100 = 0.01 |
| 粒子/格 | 27（3x3x3）|
| 帧数 | 300（5 秒物理时间 @ 60fps）|
| 子步 | 2/帧（dt = 1/120s）|
| 边界 | Free-slip 六面墙 |
| 场景 | Dam Break（左侧水柱塌陷）+ Liquid Pouring（顶部注水+障碍物）|
| 渲染 | 透视投影 3D 视角（1080×1080 PNG）|

## 输出结构

```
output/
  flip/
    ratio_970/
      dam_break/
        frames/        # 帧图 PNG（300 张，1080x1080 透视 3D 渲染）
        dam_break.mp4  # 合成视频（60fps）
        energy.csv     # 动能/耗时/涡量数据
        energy.png     # 自检曲线
      liquid_pouring/
        ...
  apic/                # 同学 B 的输出（同结构）
  polypic/
    ratio_970/          # 同学 C 的 PolyPIC 输出（同结构）
      polypic_energy.csv
      polypic_energy_dam_break.csv
      polypic_energy_liquid_pouring.csv
```

PolyPIC 的实现说明、最终产物和验证数据见 `docs/POLYPIC_RESULTS.md`。

## 绘图工具

```bash
# 单次自检
/usr/bin/python3 plot_energy.py single output/flip/ratio_970/dam_break/energy.csv -o check.png

# 三算法对比（阶段三）
/usr/bin/python3 plot_energy.py compare \
    output/flip/ratio_970/dam_break/energy.csv \
    output/apic/dam_break/energy.csv \
    output/polypic/ratio_970/dam_break/energy.csv \
    -s dam_break -o comparison/

# FLIP 内部混合比扫描分析
/usr/bin/python3 plot_energy.py sweep output/flip/ -s dam_break -o sweep/

# 全量报告
/usr/bin/python3 plot_energy.py all --flip output/flip/ratio_970/ --polypic output/polypic/ratio_970/ -o report/
```
