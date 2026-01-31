#!/usr/bin/env python
"""
RSTM-CBG DCA - 自动收集数据并生成离线HTML回放
"""
import os
import sys
import subprocess
import time
import gzip
import json

def run_experiment(duration_seconds=180):
    """运行实验指定时间后停止"""
    print(f"Starting experiment ({duration_seconds} seconds)...")

    # 清除旧日志
    log_dir = "RESULT/RSTM-CBG-DCA-restore"
    if os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir, ignore_errors=True)

    # 启动实验
    cmd = [sys.executable, "main.py", "-c", "DOCS/examples/dca/rstm_cbg_dca.jsonc"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.getcwd()
    )

    print(f"Experiment started (PID: {process.pid})")

    # 等待指定时间
    for i in range(duration_seconds):
        time.sleep(1)
        if i % 15 == 0:
            remaining = duration_seconds - i
            print(f"Collecting data... {remaining} seconds remaining", end='\r')

    print("\nStopping experiment...")
    process.terminate()

    # 等待进程结束
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()

    print("Experiment stopped")

    # 等待文件关闭
    time.sleep(5)

    return True

def read_backup_data():
    """读取备份数据"""
    backup_file = "TEMP/v2d_logger/backup.dp.gz"

    if not os.path.exists(backup_file):
        print(f"错误: 备份文件不存在: {backup_file}")
        return None

    print(f"读取数据: {backup_file}")

    data_lines = []
    try:
        # 尝试正常读取
        with gzip.open(backup_file, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    data_lines.append(line)
        print(f"成功读取 {len(data_lines)} 行数据")
        return data_lines
    except Exception as e:
        print(f"警告: 读取错误 - {e}")
        # 尝试恢复读取
        try:
            with open(backup_file, 'rb') as f:
                raw_data = f.read()

            # 尝试解压部分数据
            import zlib
            decompressed = zlib.decompress(raw_data[10:], -zlib.MAX_WBITS)  # skip gzip header
            text = decompressed.decode('utf-8', errors='ignore')
            data_lines = [line + '\n' for line in text.split('\n') if line.strip()]
            print(f"恢复读取 {len(data_lines)} 行数据")
            return data_lines
        except Exception as e2:
            print(f"无法恢复数据: {e2}")
            return None

def create_standalone_html(data_lines, output_path):
    """创建包含嵌入数据的独立HTML文件"""
    if not data_lines:
        print("没有数据可以导出")
        return False

    # 限制数据量
    max_lines = 10000
    if len(data_lines) > max_lines:
        print(f"数据量太大 ({len(data_lines)} 行)，限制为 {max_lines} 行")
        data_lines = data_lines[:max_lines]

    print(f"创建HTML文件: {output_path}")

    # 将数据嵌入为JavaScript
    data_json = json.dumps(''.join(data_lines))

    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RSTM-CBG DCA 回放 - {len(data_lines)} 帧</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        #header {{
            background: rgba(0,0,0,0.3);
            color: white;
            padding: 15px 20px;
            backdrop-filter: blur(10px);
        }}
        #header h1 {{ font-size: 22px; }}
        #header p {{ font-size: 12px; opacity: 0.9; margin-top: 5px; }}
        #controls {{
            background: rgba(255,255,255,0.95);
            padding: 15px 20px;
            margin: 15px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .control-row {{
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }}
        button {{
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            background: #667eea;
            color: white;
            cursor: pointer;
            font-size: 13px;
        }}
        button:hover {{ background: #764ba2; }}
        button:disabled {{ background: #ccc; cursor: not-allowed; }}
        input[type="range"] {{ width: 400px; }}
        #frameDisplay {{ font-family: monospace; min-width: 150px; }}
        #stats {{
            display: flex;
            gap: 20px;
            font-size: 13px;
            color: #666;
        }}
        #dataDisplay {{
            flex: 1;
            background: white;
            margin: 0 15px 15px;
            border-radius: 10px;
            padding: 20px;
            overflow: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.6;
            max-height: 500px;
        }}
        .line {{ padding: 2px 5px; }}
        .line:nth-child(odd) {{ background: #f8f9fa; }}
        .line.highlight {{ background: #fff3cd; }}
        .keyword {{ color: #d63384; font-weight: bold; }}
        .string {{ color: #0d6efd; }}
        .number {{ color: #198754; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>RSTM-CBG DCA 回放</h1>
        <p>基于分层角色切换与跨层桥接博弈的多智能体协同算法 | 共 {len(data_lines)} 帧</p>
    </div>

    <div id="controls">
        <div class="control-row">
            <button onclick="togglePlay()" id="playBtn">播放</button>
            <button onclick="stepForward()">前进</button>
            <button onclick="stepBackward()">后退</button>
            <button onclick="reset()">重置</button>
            <button onclick="exportData()">导出数据</button>
        </div>
        <div class="control-row">
            <span>进度: </span>
            <input type="range" id="slider" min="0" max="{len(data_lines)-1}" value="0" oninput="seek(this.value)">
            <span id="frameDisplay">0 / {len(data_lines)}</span>
        </div>
        <div class="control-row">
            <div id="stats">
                <span>帧: <b id="currentFrame">0</b></span>
                <span>角色: <span id="roleInfo">-</span></span>
            </div>
        </div>
    </div>

    <div id="dataDisplay"></div>

    <script>
        const data = {data_json};
        const lines = data.split('\\n');
        let currentIndex = 0;
        let isPlaying = false;
        let playInterval = null;

        function displayFrame(index) {{
            if (index < 0 || index >= lines.length) return;

            const display = document.getElementById('dataDisplay');
            const line = lines[index];

            // 解析并高亮显示
            let html = `<div class="line highlight"><strong>帧 ${{index}}:</strong> `;
            html += line
                .replace(/v2dx/g, '<span class="keyword">v2dx</span>')
                .replace(/v2d_init/g, '<span class="keyword">v2d_init</span>')
                .replace(/v2d_show/g, '<span class="keyword">v2d_show</span>')
                .replace(/\\b\\d+\\.?\\d*\\b/g, '<span class="number">$&</span>')
                .replace(/'([^']*)'/g, '<span class="string">\'$1\'</span>');
            html += '</div>';

            // 显示前后几帧作为上下文
            const context = 3;
            const start = Math.max(0, index - context);
            const end = Math.min(lines.length, index + context + 1);

            for (let i = start; i < end; i++) {{
                if (i === index) continue;
                html += `<div class="line">帧 ${{i}}: ${{lines[i].substring(0, 150)}}${{lines[i].length > 150 ? '...' : ''}}</div>`;
            }}

            display.innerHTML = html;
            display.scrollTop = 0;

            // 更新显示
            document.getElementById('currentFrame').textContent = index;
            document.getElementById('slider').value = index;
            document.getElementById('frameDisplay').textContent = `${{index}} / ${{lines.length}}`;

            // 提取角色信息
            const roleMatch = line.match(/Role update #\\d+:\\s*\\{{[^}}]*\\}}/);
            if (roleMatch) {{
                document.getElementById('roleInfo').textContent = roleMatch[0];
            }}
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
            document.getElementById('playBtn').textContent = isPlaying ? '暂停' : '播放';

            if (isPlaying) {{
                playInterval = setInterval(() => {{
                    if (currentIndex < lines.length - 1) {{
                        currentIndex++;
                        displayFrame(currentIndex);
                    }} else {{
                        stopPlay();
                    }}
                }}, 100);
            }} else {{
                clearInterval(playInterval);
            }}
        }}

        function stopPlay() {{
            isPlaying = false;
            document.getElementById('playBtn').textContent = '播放';
            clearInterval(playInterval);
        }}

        function stepForward() {{
            stopPlay();
            if (currentIndex < lines.length - 1) {{
                currentIndex++;
                displayFrame(currentIndex);
            }}
        }}

        function stepBackward() {{
            stopPlay();
            if (currentIndex > 0) {{
                currentIndex--;
                displayFrame(currentIndex);
            }}
        }}

        function reset() {{
            stopPlay();
            currentIndex = 0;
            displayFrame(0);
        }}

        function seek(value) {{
            stopPlay();
            currentIndex = parseInt(value);
            displayFrame(currentIndex);
        }}

        function exportData() {{
            const blob = new Blob([data], {{type: 'text/plain'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'rstm_cbg_replay_data.txt';
            a.click();
            URL.revokeObjectURL(url);
        }}

        // 键盘快捷键
        document.addEventListener('keydown', (e) => {{
            if (e.code === 'Space') {{ e.preventDefault(); togglePlay(); }}
            else if (e.code === 'ArrowRight') stepForward();
            else if (e.code === 'ArrowLeft') stepBackward();
            else if (e.code === 'KeyR') reset();
        }});

        // 初始化
        displayFrame(0);
    </script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"HTML文件已创建: {output_path} ({file_size:.2f} MB)")
    return True

def main():
    print("=== RSTM-CBG DCA 数据收集与导出工具 ===\n")

    # 切换到正确目录
    os.chdir(r"C:\Users\TianYuZuo\Downloads\IEEE-TASE-LaTeX2e-templates-and-instructions\hmp2g-dca")

    # 运行实验收集数据
    run_experiment(duration_seconds=60)

    # 读取数据
    data_lines = read_backup_data()
    if not data_lines:
        print("错误: 无法读取数据")
        return

    # 创建HTML
    output_file = "RESULT/RSTM-CBG-DCA-offline-replay.html"
    if create_standalone_html(data_lines, output_file):
        print(f"\n=== 完成! ===")
        print(f"离线回放文件: {os.path.abspath(output_file)}")
        print(f"双击文件在浏览器中打开即可回放")

if __name__ == "__main__":
    main()
