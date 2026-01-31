#!/usr/bin/env python
"""
Export RSTM-CBG DCA Replay as Standalone HTML
This script reads the backup.dp.gz file and creates a standalone HTML file
that can be opened directly in a browser without running a server.
"""
import gzip
import os
import base64
import json

def read_backup_data(file_path):
    """Read data from backup.dp.gz file"""
    print(f"Reading data from {file_path}...")
    data_lines = []
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                data_lines.append(line)
        print(f"Read {len(data_lines)} lines of data")
        return data_lines
    except Exception as e:
        print(f"Warning: Error reading file: {e}")
        print("Trying to read partial data...")
        # Try to skip corrupted parts
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                # Filter out empty lines and keep valid v2d commands
                for line in lines:
                    if line.strip() and ('v2dx' in line or 'v2d_init' in line or 'v2d_show' in line):
                        data_lines.append(line + '\n')
            print(f"Recovered {len(data_lines)} lines of data")
            return data_lines
        except Exception as e2:
            print(f"Failed to recover data: {e2}")
            return None

def create_html_replay(data_lines, output_path, title="RSTM-CBG DCA Replay"):
    """Create standalone HTML file with embedded data"""

    # Split data into chunks to avoid extremely large files
    # Only take first N frames if data is too large
    max_frames = 5000  # Adjust this to include more/less frames
    if len(data_lines) > max_frames:
        print(f"Warning: Data has {len(data_lines)} frames, limiting to {max_frames}")
        data_lines = data_lines[:max_frames]

    # Convert data to JavaScript string
    data_str = ''.join(data_lines)
    # Escape for JavaScript
    data_escaped = json.dumps(data_str)

    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <title>{title}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0">
    <style>
        body {{
            margin: 0;
            overflow: hidden;
            background-color: #000;
            font-family: Arial, sans-serif;
        }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            background: rgba(0,0,0,0.5);
            padding: 10px;
            border-radius: 5px;
            z-index: 100;
        }}
        #controls {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            color: white;
            background: rgba(0,0,0,0.5);
            padding: 10px;
            border-radius: 5px;
            z-index: 100;
        }}
        button {{
            margin: 5px;
            padding: 5px 15px;
            font-size: 14px;
            cursor: pointer;
        }}
        #frameInfo {{
            margin-top: 10px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div id="info">
        <h2>RSTM-CBG DCA Replay</h2>
        <p>Total Frames: {len(data_lines)}</p>
        <p>Use controls below to play/pause/step through frames</p>
    </div>
    <div id="controls">
        <button onclick="togglePlay()">Play/Pause</button>
        <button onclick="stepForward()">Step Forward</button>
        <button onclick="stepBackward()">Step Backward</button>
        <button onclick="reset()">Reset</button>
        <div id="frameInfo">Frame: 0 / {len(data_lines)}</div>
    </div>

    <script type="importmap">
    {{
        "imports": {{
            "three": "https://cdn.jsdelivr.net/npm/three@0.150.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.150.0/examples/jsm/"
        }}
    }}
    </script>

    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

        // Embedded replay data
        const replayData = {data_escaped};

        // Parse data
        const lines = replayData.split('\\n');
        console.log('Loaded', lines.length, 'lines of data');

        // Scene setup
        let scene, camera, renderer, controls;
        let agents = {{}};
        let currentFrame = 0;
        let isPlaying = false;
        let playInterval = null;

        function init() {{
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x87CEEB);  // Sky blue

            // Camera
            camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 8, 8);
            camera.lookAt(0, 0, 0);

            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);

            // Controls
            controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 10, 5);
            scene.add(directionalLight);

            // Ground
            const groundGeometry = new THREE.PlaneGeometry(10, 10);
            const groundMaterial = new THREE.MeshStandardMaterial({{
                color: 0xcccccc,
                roughness: 0.8
            }});
            const ground = new THREE.Mesh(groundGeometry, groundMaterial);
            ground.rotation.x = -Math.PI / 2;
            ground.position.y = -0.01;
            scene.add(ground);

            // Grid
            const gridHelper = new THREE.GridHelper(10, 20);
            scene.add(gridHelper);

            // Parse and display first frame
            parseFrame(currentFrame);

            // Animation loop
            animate();

            // Handle window resize
            window.addEventListener('resize', onWindowResize);
        }}

        function createAgent(id, x, y, color) {{
            // Agent group
            const group = new THREE.Group();

            // Body (cone)
            const bodyGeometry = new THREE.ConeGeometry(0.15, 0.4, 8);
            const bodyMaterial = new THREE.MeshStandardMaterial({{ color: color }});
            const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
            body.rotation.x = Math.PI / 2;
            body.position.y = 0.2;
            group.add(body);

            // Direction indicator
            const dirGeometry = new THREE.BoxGeometry(0.05, 0.05, 0.2);
            const dirMaterial = new THREE.MeshStandardMaterial({{ color: 0x000000 }});
            const dir = new THREE.Mesh(dirGeometry, dirMaterial);
            dir.position.z = -0.2;
            dir.position.y = 0.2;
            group.add(dir);

            group.position.set(x, 0, y);
            scene.add(group);

            return group;
        }}

        function parseFrame(frameIndex) {{
            if (frameIndex < 0 || frameIndex >= lines.length) return;

            const line = lines[frameIndex];
            if (!line || line.trim() === '') return;

            // Simple parser for v2d format
            // This is a simplified version - you may need to adjust based on actual data format
            try {{
                // Look for agent data in the line
                // Format example: v2dx(agent_id|type|color|x|y|rotation)
                const matches = line.match(/v2dx\\([^)]+\\)/g);
                if (matches) {{
                    matches.forEach(match => {{
                        const parts = match.substring(5, match.length - 1).split('|');
                        if (parts.length >= 5) {{
                            const id = parts[0];
                            const x = parseFloat(parts[3]);
                            const y = parseFloat(parts[4]);
                            const color = parts[2] || 'blue';

                            if (!agents[id]) {{
                                agents[id] = createAgent(id, x, y, color);
                            }} else {{
                                agents[id].position.set(x, 0, y);
                            }}
                        }}
                    }});
                }}
            }} catch (e) {{
                console.error('Error parsing frame:', frameIndex, e);
            }}

            document.getElementById('frameInfo').textContent = `Frame: ${{frameIndex}} / ${{lines.length}}`;
        }}

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}

        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        // Control functions
        window.togglePlay = function() {{
            isPlaying = !isPlaying;
            if (isPlaying) {{
                playInterval = setInterval(() => {{
                    currentFrame = (currentFrame + 1) % lines.length;
                    parseFrame(currentFrame);
                }}, 100);  // 10 FPS
            }} else {{
                clearInterval(playInterval);
            }}
        }};

        window.stepForward = function() {{
            isPlaying = false;
            clearInterval(playInterval);
            currentFrame = (currentFrame + 1) % lines.length;
            parseFrame(currentFrame);
        }};

        window.stepBackward = function() {{
            isPlaying = false;
            clearInterval(playInterval);
            currentFrame = (currentFrame - 1 + lines.length) % lines.length;
            parseFrame(currentFrame);
        }};

        window.reset = function() {{
            isPlaying = false;
            clearInterval(playInterval);
            currentFrame = 0;
            parseFrame(currentFrame);
        }};

        // Initialize
        init();
    </script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"HTML replay saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

def main():
    # Paths
    backup_file = "TEMP/v2d_logger/backup.dp.gz"
    output_file = "RESULT/RSTM-CBG-DCA-replay.html"

    # Check if backup file exists
    if not os.path.exists(backup_file):
        print(f"Error: Backup file not found: {backup_file}")
        print("Please run an experiment first to generate the replay data.")
        return

    # Read backup data
    data_lines = read_backup_data(backup_file)
    if data_lines is None:
        return

    # Create HTML replay
    create_html_replay(data_lines, output_file, title="RSTM-CBG DCA Replay")

    print("\\n=== Replay Export Complete ===")
    print(f"Open the file in your browser: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main()
