"""
Dataset Visualizer API
======================
Flask backend for serving interactive exploration and annotation of multi-modal datasets.
"""

# TODO: the camera panel and other panels are not showing anything; please fix this. There is nothing is moving when I press the play button.

import os
import json
import base64
import bisect
import argparse
import numpy as np
from flask import Flask, render_template_string, jsonify, request

app = Flask(__name__)
DATASET_DIR = "./data/dual_arm_dataset"
DEFAULT_FREQ = 30.0

# --- Helper Functions ---
def get_metadata():
    with open(os.path.join(DATASET_DIR, "metadata.json"), 'r') as f:
        return json.load(f)

def save_metadata(meta):
    with open(os.path.join(DATASET_DIR, "metadata.json"), 'w') as f:
        json.dump(meta, f, indent=4)

def get_nearest(timestamps, values, query_ts):
    if len(timestamps) == 0: return None
    idx = bisect.bisect_left(timestamps, query_ts)
    if idx == 0: return values[0]
    if idx == len(timestamps): return values[-1]
    return values[idx - 1] if (query_ts - timestamps[idx - 1]) <= (timestamps[idx] - query_ts) else values[idx]

# --- API Routes ---
@app.route('/api/recordings', methods=['GET'])
def list_recordings():
    meta = get_metadata()
    recs = []
    for rec_id, data in meta.get("recordings", {}).items():
        recs.append({
            "id": rec_id, 
            "human_name": data.get("human_name", rec_id),
            "instruction": data.get("language_instruction", "")
        })
    return jsonify({"recordings": recs})

@app.route('/api/recordings/<recording_id>', methods=['DELETE'])
def delete_recording(recording_id):
    meta = get_metadata()
    if recording_id in meta["recordings"]:
        file_path = os.path.join(DATASET_DIR, meta["recordings"][recording_id]["file"])
        if os.path.exists(file_path): os.remove(file_path)
        del meta["recordings"][recording_id]
        save_metadata(meta)
        return jsonify({"status": "success"})
    return jsonify({"error": "Not found"}), 404

@app.route('/api/recordings/<recording_id>/annotate', methods=['POST'])
def annotate_recording(recording_id):
    data = request.json
    meta = get_metadata()
    if recording_id in meta["recordings"]:
        meta["recordings"][recording_id]["language_instruction"] = data.get("instruction", "")
        meta["recordings"][recording_id]["subtasks"] = data.get("subtasks", [])
        save_metadata(meta)
        return jsonify({"status": "success"})
    return jsonify({"error": "Not found"}), 404

@app.route('/api/data/<recording_id>', methods=['GET'])
def get_recording_data(recording_id):
    # Dynamic frequency requested by the frontend
    req_freq = request.args.get('freq', default=DEFAULT_FREQ, type=float)
    
    meta = get_metadata()
    rec_info = meta["recordings"].get(recording_id)
    if not rec_info: return jsonify({"error": "Not found"}), 404
        
    data = np.load(os.path.join(DATASET_DIR, rec_info["file"]), allow_pickle=True)
    streams = [k.replace('_rel_ts', '') for k in data.keys() if '_rel_ts' in k]
    
    start_times = [data[f"{s}_rel_ts"][0] for s in streams if len(data[f"{s}_rel_ts"]) > 0]
    end_times = [data[f"{s}_rel_ts"][-1] for s in streams if len(data[f"{s}_rel_ts"]) > 0]
            
    if not start_times: return jsonify({"length": 0, "frames": []})
        
    master_ts = np.arange(max(start_times), min(end_times), 1.0 / req_freq)
    
    frames = []
    for q_ts in master_ts:
        frame_data = {}
        for stream in streams:
            val = get_nearest(data[f"{stream}_rel_ts"], data[f"{stream}_values"], q_ts)
            if 'camera' in stream:
                frame_data[stream] = base64.b64encode(val).decode('utf-8')
            else:
                frame_data[stream] = val.tolist() if isinstance(val, np.ndarray) else val
        frames.append(frame_data)
        
    return jsonify({
        "length": len(frames),
        "freq": req_freq,
        "instruction": rec_info.get("language_instruction", ""),
        "subtasks": rec_info.get("subtasks", []),
        "frames": frames
    })

# --- Frontend HTML ---
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Dataset Explorer</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: sans-serif; background: #1e1e1e; color: #fff; margin: 20px; }
        .controls { display: flex; gap: 15px; margin-bottom: 20px; align-items: center; background: #2d2d2d; padding: 15px; border-radius: 8px;}
        select, button, input { padding: 8px; background: #333; color: white; border: 1px solid #555; border-radius: 4px; }
        button:hover { background: #444; cursor: pointer; }
        .workspace { display: flex; flex-direction: column; gap: 20px; }
        .row { display: flex; gap: 20px; }
        .panel { flex: 1; background: #2d2d2d; padding: 15px; border-radius: 8px; }
        img { max-width: 100%; border-radius: 4px; background: #000; }
        .playback-controls { display: flex; gap: 10px; align-items: center; margin-bottom: 15px; }
        .playback-btn { background: #4CAF50; border: none; font-weight: bold; width: 80px; }
        .playback-btn.stop { background: #f44336; }
        .chart-container { position: relative; height: 200px; width: 100%; margin-bottom: 15px;}
        
        /* The vertical scrubbing line over the chart */
        .scrubber-line { position: absolute; top: 0; bottom: 0; width: 2px; background: rgba(255, 0, 0, 0.7); pointer-events: none; z-index: 10; display: none; }
    </style>
</head>
<body>
    <div class="controls">
        <select id="rec-select"></select>
        <label>Sync Freq (Hz): <input type="number" id="freq-input" value="30" style="width: 60px;"></label>
        <button onclick="loadRec()" style="background: #2196F3;">Load Dataset</button>
        <button onclick="deleteRec()" style="background: #aa3333; margin-left: auto;">Delete Recording</button>
    </div>
    
    <div class="workspace">
        <div class="row">
            <div class="panel" style="flex: 0.8;">
                <h3>Cameras</h3>
                <div id="camera-container" style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;"></div>
            </div>
            
            <div class="panel" style="flex: 1.2;">
                <div class="playback-controls">
                    <button class="playback-btn" onclick="togglePlay()" id="play-btn">▶ Play</button>
                    <button class="playback-btn stop" onclick="stopPlay()">■ Stop</button>
                    <input type="range" id="slider" min="0" max="100" value="0" style="flex: 1;" oninput="updateView(this.value)">
                    <span id="frame-counter" style="font-family: monospace; min-width: 120px;">Frame: 0 / 0</span>
                </div>
                
                <h3>Left Arm Joints</h3>
                <div class="chart-container">
                    <canvas id="jointChart"></canvas>
                    <div id="joint-scrubber" class="scrubber-line"></div>
                </div>

                <h3>Left Arm Actions</h3>
                <div class="chart-container">
                    <canvas id="actionChart"></canvas>
                    <div id="action-scrubber" class="scrubber-line"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentData = null;
        let activeRec = null;
        let playInterval = null;
        let isPlaying = false;
        
        let jointChartInst = null;
        let actionChartInst = null;

        function fetchList() {
            fetch('/api/recordings').then(r => r.json()).then(d => {
                const sel = document.getElementById('rec-select');
                sel.innerHTML = '';
                d.recordings.forEach(rec => {
                    sel.innerHTML += `<option value="${rec.id}">${rec.human_name} [${rec.id}]</option>`;
                });
            });
        }
        
        function loadRec() {
            activeRec = document.getElementById('rec-select').value;
            const freq = document.getElementById('freq-input').value;
            if(!activeRec) return;
            
            stopPlay();
            fetch(`/api/data/${activeRec}?freq=${freq}`).then(r => r.json()).then(d => {
                currentData = d;
                document.getElementById('slider').max = d.length - 1;
                buildCharts(d);
                updateView(0);
            });
        }

        function deleteRec() {
            if(!confirm("Are you sure you want to delete this recording?")) return;
            fetch(`/api/recordings/${document.getElementById('rec-select').value}`, {method: 'DELETE'})
                .then(() => fetchList());
        }

        function togglePlay() {
            if(!currentData) return;
            const btn = document.getElementById('play-btn');
            isPlaying = !isPlaying;
            
            if(isPlaying) {
                btn.innerText = "⏸ Pause";
                const delayMs = 1000 / currentData.freq;
                playInterval = setInterval(() => {
                    let slider = document.getElementById('slider');
                    let nextVal = parseInt(slider.value) + 1;
                    if(nextVal >= currentData.length) nextVal = 0;
                    slider.value = nextVal;
                    updateView(nextVal);
                }, delayMs);
            } else {
                btn.innerText = "▶ Play";
                clearInterval(playInterval);
            }
        }
        
        function stopPlay() {
            isPlaying = false;
            document.getElementById('play-btn').innerText = "▶ Play";
            clearInterval(playInterval);
            document.getElementById('slider').value = 0;
            updateView(0);
        }

        function updateView(idx) {
            if(!currentData) return;
            const frame = currentData.frames[idx];
            document.getElementById('frame-counter').innerText = `Frame: ${idx} / ${currentData.length - 1}`;
            
            // Render Cameras
            const camContainer = document.getElementById('camera-container');
            camContainer.innerHTML = ''; 
            
            Object.keys(frame).forEach(key => {
                if(key.includes('camera')) {
                    camContainer.innerHTML += `<div><small style="color:#aaa;">${key}</small><br><img src="data:image/jpeg;base64,${frame[key]}"></div>`;
                }
            });
            
            // Update Scrubber Lines over charts
            const percentage = (idx / (currentData.length - 1)) * 100;
            const jScrubber = document.getElementById('joint-scrubber');
            const aScrubber = document.getElementById('action-scrubber');
            
            if(jointChartInst) {
                jScrubber.style.display = 'block';
                jScrubber.style.left = `calc(${percentage}% - 1px)`;
            }
            if(actionChartInst) {
                aScrubber.style.display = 'block';
                aScrubber.style.left = `calc(${percentage}% - 1px)`;
            }
        }
        
        // --- Chart.js Logic ---
        function buildCharts(data) {
            const labels = Array.from({length: data.length}, (_, i) => i);
            
            // Extract Joint trajectories (mocking 7 DoF)
            const jointDatasets = [];
            for(let dim = 0; dim < 7; dim++) {
                jointDatasets.push({
                    label: `J${dim+1}`,
                    data: data.frames.map(f => f.left_arm_joints ? f.left_arm_joints[dim] : 0),
                    borderColor: `hsl(${dim * 50}, 70%, 60%)`,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1
                });
            }
            
            // Extract Action trajectories
            const actionDatasets = [];
            for(let dim = 0; dim < 7; dim++) {
                actionDatasets.push({
                    label: `A${dim+1}`,
                    data: data.frames.map(f => f.left_actions ? f.left_actions[dim] : 0),
                    borderColor: `hsl(${dim * 50 + 180}, 70%, 60%)`,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1
                });
            }
            
            const chartOptions = {
                responsive: true, maintainAspectRatio: false,
                animation: false,
                interaction: { mode: 'index', intersect: false },
                scales: { 
                    x: { display: false }, 
                    y: { grid: {color: '#444'} } 
                },
                plugins: { legend: { position: 'right', labels: {color: '#fff', boxWidth: 10} } }
            };

            if(jointChartInst) jointChartInst.destroy();
            const ctxJ = document.getElementById('jointChart').getContext('2d');
            jointChartInst = new Chart(ctxJ, { type: 'line', data: { labels, datasets: jointDatasets }, options: chartOptions });

            if(actionChartInst) actionChartInst.destroy();
            const ctxA = document.getElementById('actionChart').getContext('2d');
            actionChartInst = new Chart(ctxA, { type: 'line', data: { labels, datasets: actionDatasets }, options: chartOptions });
        }
        
        fetchList();
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Dataset Visualizer")
    parser.add_argument("--dataset_dir", type=str, default="./data/dual_arm_dataset", help="Path to the dataset directory")
    parser.add_argument("--port", type=int, default=5000, help="Port for the Flask server")
    args = parser.parse_args()
    
    DATASET_DIR = args.dataset_dir
    
    if not os.path.exists(DATASET_DIR):
        print(f"Warning: Dataset directory {DATASET_DIR} not found.")
        
    print(f"\n🚀 Starting Visualizer Server on http://127.0.0.1:{args.port}\n")
    app.run(port=args.port, debug=False)