// --- State Variables ---
let activeTab = 'training';
let lossChart = null;
let statusInterval = null;
let currentMaxSteps = 2000;

// Baseline speed for CPU FP32 No Cache (for comparison ratio)
const BASELINE_SPEED = 15.0;

// --- Initialize App ---
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initChart();
    initControls();
    loadDatasetPreview();
    
    // Start polling training status
    pollStatus();
    statusInterval = setInterval(pollStatus, 1000);
});

// --- Tab Navigation ---
function initTabs() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            
            // Update Active button
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update Active panel
            tabPanes.forEach(pane => pane.classList.remove('active'));
            document.getElementById(`tab-${tabId}`).classList.add('active');
            
            // Update Headers text
            updateHeader(tabId);
            activeTab = tabId;
        });
    });
}

function updateHeader(tabId) {
    const title = document.getElementById('page-title');
    const subtitle = document.getElementById('page-subtitle');
    
    if (tabId === 'training') {
        title.innerText = 'Model Training & Convergence Monitor';
        subtitle.innerText = 'Train a character-level GPT on TinyStories and monitor validation loss curves.';
    } else if (tabId === 'playground') {
        title.innerText = 'Inference Playground & Benchmarks';
        subtitle.innerText = 'Run text generation interactively and profile optimization techniques.';
    } else if (tabId === 'architecture') {
        title.innerText = 'System Architecture Map';
        subtitle.innerText = 'Examine the computational flow and layers of the custom Transformer.';
    } else if (tabId === 'dataset') {
        title.innerText = 'Dataset Explorer';
        subtitle.innerText = 'Browse raw text contents of the TinyStories training corpus.';
    }
}

// --- Chart.js Setup ---
function initChart() {
    const ctx = document.getElementById('lossChart').getContext('2d');
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train Loss',
                    data: [],
                    borderColor: '#8b5cf6', // neon purple
                    backgroundColor: 'rgba(139, 92, 246, 0.05)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true,
                    pointRadius: 3
                },
                {
                    label: 'Validation Loss',
                    data: [],
                    borderColor: '#06b6d4', // neon cyan
                    backgroundColor: 'rgba(6, 182, 212, 0.05)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true,
                    pointRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Plus Jakarta Sans', weight: '600' }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Training Step',
                        color: '#94a3b8',
                        font: { family: 'Plus Jakarta Sans', size: 12, weight: '600' }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#64748b' }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Cross-Entropy Loss',
                        color: '#94a3b8',
                        font: { family: 'Plus Jakarta Sans', size: 12, weight: '600' }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#64748b' }
                }
            }
        }
    });
}

// --- Form & Button Actions ---
function initControls() {
    // Range sliders text sync
    const tempInput = document.getElementById('gen-temp');
    const tempVal = document.getElementById('val-temp');
    tempInput.addEventListener('input', () => {
        tempVal.innerText = tempInput.value;
    });

    const tokensInput = document.getElementById('gen-tokens');
    const tokensVal = document.getElementById('val-tokens');
    tokensInput.addEventListener('input', () => {
        tokensVal.innerText = tokensInput.value;
    });

    // Quantization CPU constraints toggle
    const quantCheckbox = document.getElementById('gen-quantized');
    const deviceSelect = document.getElementById('gen-device');
    
    quantCheckbox.addEventListener('change', () => {
        if (quantCheckbox.checked) {
            // Force CPU since dynamic quantization runs on CPU
            deviceSelect.value = 'cpu';
            deviceSelect.disabled = true;
        } else {
            deviceSelect.disabled = false;
        }
    });

    // Training Buttons
    document.getElementById('btn-train-start').addEventListener('click', startTraining);
    document.getElementById('btn-train-pause').addEventListener('click', pauseTraining);
    document.getElementById('btn-train-reset').addEventListener('click', resetTraining);

    // Generation Buttons
    document.getElementById('btn-generate').addEventListener('click', runGeneration);

    // Benchmark Buttons
    document.getElementById('btn-run-benchmark').addEventListener('click', runBenchmarkSuite);
}

// --- Dataset Exporter Loader ---
async function loadDatasetPreview() {
    try {
        const res = await fetch('/api/dataset/preview');
        if (res.ok) {
            const data = await res.json();
            document.getElementById('dataset-preview-text').innerText = data.preview;
        } else {
            document.getElementById('dataset-preview-text').innerText = "Dataset not downloaded. Run python download_data.py first.";
        }
    } catch (err) {
        document.getElementById('dataset-preview-text').innerText = "Error loading dataset: " + err;
    }
}

// --- Polling Training States ---
async function pollStatus() {
    try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const data = await res.json();
        
        // Update System device badge
        document.getElementById('system-device').innerText = `Device: ${data.device.toUpperCase()}`;
        
        // Update connection status
        updateConnectionBadge(data.status);
        
        // Update Training Tab Metrics
        updateMetrics(data);
        
        // Update Chart
        if (data.steps_history && data.steps_history.length > 0) {
            // Only rebuild chart if lengths match or chart is empty
            const chartLength = lossChart.data.labels.length;
            if (chartLength !== data.steps_history.length) {
                lossChart.data.labels = data.steps_history;
                lossChart.data.datasets[0].data = data.train_losses;
                lossChart.data.datasets[1].data = data.val_losses;
                lossChart.update('none'); // silent update
            }
        }
    } catch (err) {
        console.error("Polling error:", err);
        const connectionBadge = document.getElementById('connection-status');
        connectionBadge.className = 'status-badge error';
        document.getElementById('status-text').innerText = 'Offline';
    }
}

function updateConnectionBadge(status) {
    const badge = document.getElementById('connection-status');
    const text = document.getElementById('status-text');
    badge.className = `status-badge ${status}`;
    
    if (status === 'idle') {
        text.innerText = 'Connected (Idle)';
    } else if (status === 'training') {
        text.innerText = 'Training Model...';
    } else if (status === 'paused') {
        text.innerText = 'Training Paused';
    } else if (status === 'completed') {
        text.innerText = 'Training Complete';
    } else if (status === 'error') {
        text.innerText = 'Training Error';
    }
}

function updateMetrics(data) {
    // Status text
    const statusVal = document.getElementById('metric-status');
    statusVal.innerText = data.status.toUpperCase();
    statusVal.className = `metric-val text-${data.status}`;
    
    // Step
    document.getElementById('metric-step').innerText = `${data.current_step} / ${data.max_steps}`;
    currentMaxSteps = data.max_steps;
    
    // Loss
    const latestLoss = data.val_losses.length > 0 ? data.val_losses[data.val_losses.length - 1].toFixed(4) : 'N/A';
    document.getElementById('metric-loss').innerText = latestLoss;
    
    // Speed
    document.getElementById('metric-throughput').innerText = data.tokens_per_sec.toFixed(1);
    
    // Buttons state
    const startBtn = document.getElementById('btn-train-start');
    const pauseBtn = document.getElementById('btn-train-pause');
    
    if (data.status === 'training') {
        startBtn.disabled = true;
        pauseBtn.disabled = false;
    } else {
        startBtn.disabled = false;
        pauseBtn.disabled = true;
    }
}

// --- Training Actions ---
async function startTraining() {
    const steps = parseInt(document.getElementById('input-steps').value);
    const lr = parseFloat(document.getElementById('input-lr').value);
    const batch = parseInt(document.getElementById('input-batch').value);
    
    try {
        await fetch('/api/train/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                max_steps: steps,
                learning_rate: lr,
                batch_size: batch
            })
        });
        pollStatus();
    } catch (err) {
        alert("Error starting training: " + err);
    }
}

async function pauseTraining() {
    try {
        await fetch('/api/train/pause', { method: 'POST' });
        pollStatus();
    } catch (err) {
        alert("Error pausing training: " + err);
    }
}

async function resetTraining() {
    if (!confirm("Are you sure you want to reset training? This will delete the current weights and clear the logs!")) {
        return;
    }
    
    try {
        await fetch('/api/train/reset', { method: 'POST' });
        
        // Reset local chart data
        lossChart.data.labels = [];
        lossChart.data.datasets[0].data = [];
        lossChart.data.datasets[1].data = [];
        lossChart.update();
        
        pollStatus();
    } catch (err) {
        alert("Error resetting trainer: " + err);
    }
}

// --- Text Generation Streaming ---
let activeEventSource = null;

function runGeneration() {
    const prompt = document.getElementById('gen-prompt').value;
    const temp = document.getElementById('gen-temp').value;
    const maxTokens = parseInt(document.getElementById('gen-tokens').value);
    const useCache = document.getElementById('gen-use-cache').checked;
    const quantized = document.getElementById('gen-quantized').checked;
    const device = document.getElementById('gen-device').value;
    
    // Clear terminal and show loader
    const consoleOutput = document.getElementById('console-output');
    consoleOutput.innerHTML = '';
    consoleOutput.classList.remove('empty');
    
    // Close any previous stream
    if (activeEventSource) {
        activeEventSource.close();
    }
    
    // Build query params
    const query = new URLSearchParams({
        prompt: prompt,
        max_new_tokens: maxTokens,
        temperature: temp,
        use_cache: useCache,
        quantized: quantized,
        device: device
    });
    
    // Render the initial prompt in console first
    const promptSpan = document.createElement('span');
    promptSpan.style.color = '#e2e8f0'; // grayish
    promptSpan.innerText = prompt;
    consoleOutput.appendChild(promptSpan);
    
    // Initialize stream event listener
    activeEventSource = new EventSource(`/api/generate/stream?${query.toString()}`);
    
    activeEventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Init block contains static specs
        if (data.type === 'init') {
            document.getElementById('stat-size').innerText = `${data.model_size_mb.toFixed(2)} MB`;
            return;
        }
        
        // Done signal
        if (data.type === 'done') {
            activeEventSource.close();
            activeEventSource = null;
            return;
        }
        
        // Append streamed text character
        const charSpan = document.createElement('span');
        charSpan.innerText = data.char;
        consoleOutput.appendChild(charSpan);
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
        
        // Update metrics dashboard
        if (data.metrics) {
            document.getElementById('stat-ttft').innerText = `${data.metrics.ttft_ms.toFixed(1)} ms`;
            document.getElementById('stat-speed').innerText = `${data.metrics.tokens_per_sec.toFixed(1)} tokens/s`;
            
            // Calculate speedup compared to baseline
            const speed = data.metrics.tokens_per_sec;
            if (speed > 0) {
                const ratio = speed / BASELINE_SPEED;
                document.getElementById('stat-ratio').innerText = `${ratio.toFixed(2)}x`;
            } else {
                document.getElementById('stat-ratio').innerText = 'N/A';
            }
            
            // Safety close check if stream finishes
            if (data.metrics.step >= maxTokens) {
                activeEventSource.close();
                activeEventSource = null;
            }
        }
    };
    
    activeEventSource.onerror = (err) => {
        console.error("EventSource failed:", err);
        if (activeEventSource) {
            activeEventSource.close();
            activeEventSource = null;
        }
    };
}

// --- Run Benchmarks ---
async function runBenchmarkSuite() {
    const resultsBody = document.getElementById('benchmark-results');
    const runBtn = document.getElementById('btn-run-benchmark');
    
    resultsBody.innerHTML = `
        <tr>
            <td colspan="5" style="text-align: center; color: var(--accent-purple); font-weight: bold;">
                <span class="badge-dot pulse" style="display:inline-block; margin-right: 8px;"></span>
                Running Inference Benchmark Matrix (evaluating 50 tokens)...
            </td>
        </tr>
    `;
    runBtn.disabled = true;
    
    try {
        const res = await fetch('/api/benchmark');
        if (!res.ok) throw new Error("API call failed.");
        const results = await res.json();
        
        resultsBody.innerHTML = '';
        
        // Find baseline throughput (first element, which is CPU FP32 No Cache)
        let baselineThroughput = 1.0;
        const baselineRow = results.find(r => r.name.includes("No Cache") && r.name.includes("FP32 CPU"));
        if (baselineRow && !isNaN(parseFloat(baselineRow.throughput))) {
            baselineThroughput = parseFloat(baselineRow.throughput);
        }
        
        results.forEach((row) => {
            const tr = document.createElement('tr');
            
            // Style optimized rows specifically
            const isHighlyOptimized = row.name.includes("KV Cache") && (row.name.includes("INT8") || row.name.includes("GPU"));
            if (isHighlyOptimized) {
                tr.className = 'highlight-row';
            }
            
            // Calculate relative gain
            let gainBadge = 'Baseline';
            const throughput = parseFloat(row.throughput);
            if (!isNaN(throughput) && row.name !== "FP32 CPU (No Cache)") {
                const ratio = throughput / baselineThroughput;
                gainBadge = `<span class="badge-gain">${ratio.toFixed(2)}x speed</span>`;
            } else if (isNaN(throughput)) {
                gainBadge = 'Error';
            }
            
            tr.innerHTML = `
                <td><strong>${row.name}</strong></td>
                <td>${row.model_size}</td>
                <td>${row.ttft_ms} ms</td>
                <td>${row.throughput} tokens/s</td>
                <td>${gainBadge}</td>
            `;
            resultsBody.appendChild(tr);
        });
        
    } catch (err) {
        resultsBody.innerHTML = `
            <tr>
                <td colspan="5" style="text-align: center; color: var(--accent-red);">
                    Error running benchmarks: ${err.message}
                </td>
            </tr>
        `;
    } finally {
        runBtn.disabled = false;
    }
}
