# HMP2G-DCA

Decentralized Collective Assault (DCA) 多智能体强化学习环境

## 快速开始

### 推荐：VSCode 中运行
1. 用 VSCode 打开项目文件夹
2. 按 `F5` 或点击"运行和调试" -> "运行 DCA 环境"
3. 或直接运行 `run.py` 文件

### Windows
双击运行 `run.bat`

### Linux
```bash
chmod +x run.sh
./run.sh
```

### Python 直接运行
```bash
python run.py
```

## 手动安装

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 编译 Cython 模块
```bash
python setup.py build_ext --inplace
```

3. 运行
```bash
python main.py -c DOCS/examples/dca/example_dca.jsonc
```

## 配置说明

编辑 `DOCS/examples/dca/example_dca.jsonc` 修改环境参数。

## 运行模式

### 训练模式
```bash
python main.py -c DOCS/examples/dca/train_old_dca.jsonc
```

### 测试模式
修改配置文件设置测试参数，运行评估

## 注意事项

1. **Windows 平台**: 共享内存使用管道通信替代，性能略低于 Linux
2. **GPU 支持**: 修改配置文件中的 `device` 为 `cuda`
3. **线程数**: 修改配置文件中的 `num_threads` 参数

## 常见问题

### Cython 编译失败
请确保已安装 C 编译器：
- **Windows**: 安装 Microsoft Visual C++ Build Tools 或 MinGW-w64
- **Ubuntu/Debian**: `sudo apt-get install build-essential`
- **CentOS/RHEL**: `sudo yum groupinstall 'Development Tools'`

### 跳过编译
如果 Cython 模块已编译，可以使用：
```bash
python run.py --skip-build
```

## 项目结构

```
hmp2g-dca/
├── main.py                 # 主入口
├── run.py                  # 一键运行脚本
├── config.py               # 全局配置
├── MISSION/dca/            # DCA 环境
├── ALGORITHM/conc_4hist/   # PPO 算法
├── UTIL/                   # 工具库
├── VISUALIZE/              # 可视化模块
└── DOCS/examples/dca/      # 配置文件
```
