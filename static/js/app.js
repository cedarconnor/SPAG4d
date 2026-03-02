// SPAG-4D Main Application JS (ES Module)
import { Viewer } from './viewer.js';

class SPAG4DApp {
    constructor() {
        this.currentFile = null;
        this.currentJobId = null;
        this.pollInterval = null;
        this.pollErrorCount = 0;
        this.viewer = null;
        this.rgbUrl = null;
        this.depthUrl = null;
        this.init();
    }

    init() {
        // DOM Elements
        this.fileInput = document.getElementById('file-input');
        this.fileLabel = document.getElementById('filename');
        this.convertBtn = document.getElementById('convert-btn');
        this.downloadPlyBtn = document.getElementById('download-ply-btn');
        this.statusText = document.getElementById('status-text');
        this.progressText = document.getElementById('progress-text');
        this.gpuStatus = document.getElementById('gpu-status');

        // Parameters
        this.depthMinInput = document.getElementById('depth-min');
        this.depthMaxInput = document.getElementById('depth-max');
        this.skyThresholdInput = document.getElementById('sky-threshold');
        this.gridJitterInput = document.getElementById('grid-jitter');
        this.outlierPruningInput = document.getElementById('outlier-pruning');
        this.globalScaleInput = document.getElementById('global-scale');
        this.sharpProjectionInput = document.getElementById('sharp-projection');
        this.sharpCubemapSizeInput = document.getElementById('sharp-cubemap-size');

        // Input preview
        this.inputPreview = document.getElementById('input-preview');
        this.inputPlaceholder = document.getElementById('input-placeholder');

        // Event Listeners
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        this.convertBtn.addEventListener('click', () => this.startConversion());
        this.downloadPlyBtn.addEventListener('click', () => this.downloadFile());

        // Reset View Button
        const resetBtn = document.getElementById('reset-view-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                if (this.viewer) this.viewer.resetView();
            });
        }

        // Upload Splat Button
        const uploadSplatBtn = document.getElementById('upload-splat-btn');
        const uploadSplatInput = document.getElementById('upload-splat-input');
        if (uploadSplatBtn && uploadSplatInput) {
            uploadSplatBtn.addEventListener('click', () => uploadSplatInput.click());
            uploadSplatInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (!file) return;
                this.ensureViewer();
                const reader = new FileReader();
                reader.onload = () => {
                    this.viewer.loadFromFile(reader.result, file.name);
                };
                reader.readAsArrayBuffer(file);
                uploadSplatInput.value = '';
            });
        }

        // Help Toggle Button
        const helpBtn = document.getElementById('help-toggle');
        const helpPanel = document.getElementById('help-panel');
        if (helpBtn && helpPanel) {
            helpBtn.addEventListener('click', () => {
                helpPanel.classList.toggle('visible');
                helpBtn.textContent = helpPanel.classList.contains('visible') ? 'Close' : '? Help';
            });
        }

        // Kill Server Button
        const killBtn = document.getElementById('kill-server-btn');
        if (killBtn) {
            killBtn.addEventListener('click', () => this.killServer());
        }

        // Tab switching (RGB / Depth)
        const tabRgb = document.getElementById('tab-rgb');
        const tabDepth = document.getElementById('tab-depth');
        if (tabRgb) tabRgb.addEventListener('click', () => this.switchInputTab('rgb'));
        if (tabDepth) tabDepth.addEventListener('click', () => this.switchInputTab('depth'));

        // Initialize viewer
        this.ensureViewer();

        // Check health
        this.checkHealth();
        setInterval(() => this.checkHealth(), 30000);

        // Preload test image
        this.preloadTestImage();
    }

    ensureViewer() {
        if (!this.viewer) {
            const splatContainer = document.getElementById('splat-viewer');
            this.viewer = new Viewer(splatContainer);
        }
    }

    switchInputTab(mode) {
        const tabRgb = document.getElementById('tab-rgb');
        const tabDepth = document.getElementById('tab-depth');

        if (mode === 'rgb' && this.rgbUrl) {
            tabRgb.classList.add('active');
            tabDepth.classList.remove('active');
            this.inputPreview.src = this.rgbUrl;
        } else if (mode === 'depth' && this.depthUrl) {
            tabDepth.classList.add('active');
            tabRgb.classList.remove('active');
            this.inputPreview.src = this.depthUrl;
        }
    }

    async preloadTestImage() {
        const testImagePath = '/TestImage/monbachtal_riverbank_primary.jpg';
        try {
            const response = await fetch(testImagePath, { method: 'HEAD' });
            if (response.ok) {
                this.rgbUrl = testImagePath;
                this.showInputPreview(testImagePath);

                const imgResponse = await fetch(testImagePath);
                const blob = await imgResponse.blob();
                this.currentFile = new File([blob], 'monbachtal_riverbank_primary.jpg', { type: 'image/jpeg' });
                this.fileLabel.textContent = 'monbachtal_riverbank_primary.jpg (demo)';
                this.convertBtn.disabled = false;
                this.setStatus('Demo image loaded - ready to convert');
            }
        } catch (e) {
            console.log('[App] No test image available');
        }
    }

    showInputPreview(url) {
        if (this.inputPreview) {
            this.inputPreview.src = url;
            this.inputPreview.style.display = 'block';
        }
        if (this.inputPlaceholder) {
            this.inputPlaceholder.style.display = 'none';
        }
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        this.currentFile = file;
        this.fileLabel.textContent = file.name;
        this.convertBtn.disabled = false;
        this.downloadPlyBtn.disabled = true;

        // Show in input preview
        const url = URL.createObjectURL(file);
        this.rgbUrl = url;
        this.showInputPreview(url);

        // Disable depth tab
        const tabDepth = document.getElementById('tab-depth');
        if (tabDepth) tabDepth.disabled = true;

        this.setStatus('Ready to convert');
    }

    async startConversion() {
        if (!this.currentFile) return;

        this.convertBtn.disabled = true;
        this.setStatus('Uploading...', 'Preparing');

        // Disable depth tab during conversion
        const tabDepth = document.getElementById('tab-depth');
        if (tabDepth) tabDepth.disabled = true;

        const formData = new FormData();
        formData.append('file', this.currentFile);

        const params = new URLSearchParams({
            depth_min: this.depthMinInput.value,
            depth_max: this.depthMaxInput.value,
            sky_threshold: this.skyThresholdInput.value,
            grid_jitter: this.gridJitterInput.value,
            outlier_pruning: this.outlierPruningInput.value,
            global_scale: this.globalScaleInput.value,
            sharp_projection: this.sharpProjectionInput.value,
            sharp_cubemap_size: this.sharpCubemapSizeInput.value,
        });

        try {
            const response = await fetch(`/api/convert?${params}`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }

            const result = await response.json();
            this.currentJobId = result.job_id;
            this.setStatus('Processing...', `Queue position: ${result.queue_position}`);
            this.startPolling();
        } catch (error) {
            this.setStatus(`Error: ${error.message}`);
            this.convertBtn.disabled = false;
        }
    }

    startPolling() {
        if (this.pollInterval) clearInterval(this.pollInterval);
        this.pollInterval = setInterval(() => this.checkJobStatus(), 1000);
    }

    async checkJobStatus() {
        if (!this.currentJobId) return;

        try {
            const response = await fetch(`/api/status/${this.currentJobId}`);
            if (!response.ok) throw new Error(`Status check failed: ${response.status}`);

            const status = await response.json();
            this.pollErrorCount = 0;

            if (status.status === 'queued') {
                this.setStatus('Waiting...', `Queue position: ${status.queue_position}`);
            } else if (status.status === 'processing') {
                this.setStatus('Processing...', 'GPU active');
            } else if (status.status === 'complete') {
                clearInterval(this.pollInterval);
                this.pollInterval = null;

                this.setStatus('Complete!',
                    `${status.splat_count.toLocaleString()} splats · ${status.file_size_mb} MB · ${status.processing_time}s`
                );

                this.downloadPlyBtn.disabled = false;

                // Load PLY into viewer
                if (status.ply_url) {
                    this.ensureViewer();
                    this.viewer.loadScene(status.ply_url);
                }

                // Enable depth tab
                if (status.depth_preview_url) {
                    this.depthUrl = status.depth_preview_url;
                    const tabDepth = document.getElementById('tab-depth');
                    if (tabDepth) tabDepth.disabled = false;
                }

                this.convertBtn.disabled = false;
            } else if (status.status === 'error') {
                clearInterval(this.pollInterval);
                this.pollInterval = null;
                this.setStatus(`Error: ${status.error}`);
                this.convertBtn.disabled = false;
            }
        } catch (error) {
            console.error('Polling error:', error);
            this.pollErrorCount++;
            if (this.pollErrorCount >= 5) {
                clearInterval(this.pollInterval);
                this.pollInterval = null;
                this.setStatus('Connection error - please try again');
                this.convertBtn.disabled = false;
                this.pollErrorCount = 0;
            }
        }
    }

    downloadFile() {
        if (!this.currentJobId) return;
        const url = `/api/download/${this.currentJobId}`;
        const link = document.createElement('a');
        link.href = url;
        link.download = `spag4d_${this.currentJobId.slice(0, 8)}.ply`;
        link.click();
    }

    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const health = await response.json();

            if (health.gpu_available) {
                this.gpuStatus.className = 'gpu-status';
            } else if (health.active_jobs > 0) {
                this.gpuStatus.className = 'gpu-status busy';
            } else {
                this.gpuStatus.className = 'gpu-status error';
            }
        } catch (error) {
            this.gpuStatus.className = 'gpu-status error';
        }
    }

    async killServer() {
        if (!confirm('Shut down the SPAG-4D server?')) return;
        this.setStatus('Server shutting down...');
        try {
            await fetch('/api/shutdown', { method: 'POST' });
        } catch (e) {
            // Expected — server dies before response completes
        }
    }

    setStatus(text, progress = '') {
        this.statusText.textContent = text;
        this.progressText.textContent = progress;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new SPAG4DApp();
});
