// SPAG-4D Main Application JS

class SPAG4DApp {
    constructor() {
        this.currentFile = null;
        this.currentJobId = null;
        this.pollInterval = null;

        this.panoViewer = null;
        this.splatViewer = null;

        this.init();
    }

    init() {
        // DOM Elements
        this.fileInput = document.getElementById('file-input');
        this.fileLabel = document.getElementById('filename');
        this.convertBtn = document.getElementById('convert-btn');
        this.downloadPlyBtn = document.getElementById('download-ply-btn');
        this.downloadSplatBtn = document.getElementById('download-splat-btn');
        this.statusText = document.getElementById('status-text');
        this.progressText = document.getElementById('progress-text');
        this.gpuStatus = document.getElementById('gpu-status');

        // Video Elements
        this.downloadZipBtn = document.getElementById('download-zip-btn');
        this.videoParams = document.getElementById('video-params');
        this.fpsInput = document.getElementById('fps');
        this.videoStartInput = document.getElementById('video-start');
        this.videoDurationInput = document.getElementById('video-duration');
        this.temporalAlphaInput = document.getElementById('temporal-alpha');
        this.stabilizeVideoInput = document.getElementById('stabilize-video');
        this.videoInfo = document.getElementById('video-info');
        this.videoMetadata = document.getElementById('video-metadata');

        // Parameters
        this.strideSelect = document.getElementById('stride');
        this.scaleFactorInput = document.getElementById('scale-factor');
        this.thicknessInput = document.getElementById('thickness');
        this.globalScaleInput = document.getElementById('global-scale');
        this.depthMinInput = document.getElementById('depth-min');
        this.depthMinInput = document.getElementById('depth-min');
        this.depthMaxInput = document.getElementById('depth-max');

        // SHARP Elements
        this.sharpRefineInput = document.getElementById('sharp-refine');
        this.scaleBlendInput = document.getElementById('scale-blend');
        this.opacityBlendInput = document.getElementById('opacity-blend');
        this.sharpBlendGroup = document.getElementById('sharp-blend-group');
        this.sharpOpacityGroup = document.getElementById('sharp-opacity-group');
        this.sharpProjectionInput = document.getElementById('sharp-projection');
        this.sharpProjectionGroup = document.getElementById('sharp-projection-group');
        this.sharpCubemapSizeInput = document.getElementById('sharp-cubemap-size');
        this.sharpResolutionGroup = document.getElementById('sharp-resolution-group');
        this.colorBlendInput = document.getElementById('color-blend');
        this.sharpColorGroup = document.getElementById('sharp-color-group');
        this.skyThresholdInput = document.getElementById('sky-threshold');
        this.skyDomeInput = document.getElementById('sky-dome');

        // Depth Model Elements
        this.depthModelSelect = document.getElementById('depth-model');
        this.guidedFilterInput = document.getElementById('guided-filter');
        this.guidedStrengthInput = document.getElementById('guided-strength');
        this.guidedStrengthGroup = document.getElementById('guided-strength-group');

        // DA3 specific elements
        this.da3ProjectionInput = document.getElementById('da3-projection');
        this.da3ProjectionGroup = document.getElementById('da3-projection-group');

        if (this.guidedFilterInput) {
            this.guidedFilterInput.addEventListener('change', () => this.updateGuidedStrengthVisibility());
            this.updateGuidedStrengthVisibility();
        }

        if (this.depthModelSelect) {
            this.depthModelSelect.addEventListener('change', () => this.updateDepthModelParamsVisibility());
            this.updateDepthModelParamsVisibility();
        }

        if (this.sharpRefineInput) {
            this.sharpRefineInput.addEventListener('change', () => this.updateSharpVisibility());

            // Initialize visibility based on default state
            this.updateSharpVisibility();
        }




        // Event Listeners
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        this.convertBtn.addEventListener('click', () => this.startConversion());
        this.downloadPlyBtn.addEventListener('click', () => this.downloadFile('ply'));
        this.downloadSplatBtn.addEventListener('click', () => this.downloadFile('splat'));
        this.downloadZipBtn.addEventListener('click', () => this.downloadZip());

        // Reset View Button
        const resetBtn = document.getElementById('reset-view-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                if (this.splatViewer) this.splatViewer.resetView();
            });
        }

        // Orbit View Button
        const orbitBtn = document.getElementById('orbit-view-btn');
        if (orbitBtn) {
            orbitBtn.addEventListener('click', () => {
                if (this.splatViewer) this.splatViewer.setOutsideView();
            });
        }

        // Help Toggle Button
        const helpBtn = document.getElementById('help-toggle');
        const helpPanel = document.getElementById('help-panel');
        if (helpBtn && helpPanel) {
            helpBtn.addEventListener('click', () => {
                helpPanel.classList.toggle('visible');
                helpBtn.textContent = helpPanel.classList.contains('visible') ? '✕ Close' : '? Help';
            });
        }

        // Flip Y Toggle
        const flipYToggle = document.getElementById('flip-y-toggle');
        if (flipYToggle) {
            flipYToggle.addEventListener('change', () => {
                if (this.splatViewer) {
                    this.splatViewer.setFlipY(flipYToggle.checked);
                }
            });
        }

        // Initialize viewers
        this.initViewers();
        this.setupTabs();

        // Check health
        this.checkHealth();
        setInterval(() => this.checkHealth(), 30000);

        // Preload test image if available
        this.preloadTestImage();
    }

    updateSharpVisibility() {
        if (!this.sharpRefineInput) return;

        const checked = this.sharpRefineInput.checked;
        if (this.sharpBlendGroup) {
            this.sharpBlendGroup.style.display = checked ? 'flex' : 'none';
            this.sharpBlendGroup.style.opacity = checked ? '1' : '0.5';
        }
        if (this.sharpOpacityGroup) {
            this.sharpOpacityGroup.style.display = checked ? 'flex' : 'none';
            this.sharpOpacityGroup.style.opacity = checked ? '1' : '0.5';
        }
        if (this.sharpProjectionGroup) {
            this.sharpProjectionGroup.style.display = checked ? 'flex' : 'none';
            this.sharpProjectionGroup.style.opacity = checked ? '1' : '0.5';
        }
        if (this.sharpResolutionGroup) {
            this.sharpResolutionGroup.style.display = checked ? 'flex' : 'none';
            this.sharpResolutionGroup.style.opacity = checked ? '1' : '0.5';
        }
        if (this.sharpColorGroup) {
            this.sharpColorGroup.style.display = checked ? 'flex' : 'none';
            this.sharpColorGroup.style.opacity = checked ? '1' : '0.5';
        }
    }

    updateDepthModelParamsVisibility() {
        if (!this.depthModelSelect) return;

        const isDa3 = this.depthModelSelect.value === 'da3';
        if (this.da3ProjectionGroup) {
            this.da3ProjectionGroup.style.display = isDa3 ? 'flex' : 'none';
        }
    }

    updateGuidedStrengthVisibility() {
        if (!this.guidedFilterInput) return;

        const isGuided = this.guidedFilterInput.checked;
        if (this.guidedStrengthGroup) {
            this.guidedStrengthGroup.style.display = isGuided ? 'flex' : 'none';
        }
    }


    setupTabs() {
        const tabRgb = document.getElementById('tab-rgb');
        const tabDepth = document.getElementById('tab-depth');

        if (tabRgb && tabDepth) {
            tabRgb.addEventListener('click', () => this.switchTab('rgb'));
            tabDepth.addEventListener('click', () => this.switchTab('depth'));
        }

        // Quality selector
        const qualitySelect = document.getElementById('splat-quality');
        if (qualitySelect) {
            qualitySelect.addEventListener('change', () => this.handleQualityChange());
        }

        // Projection toggle
        const projBtn = document.getElementById('pano-projection-btn');
        if (projBtn) {
            projBtn.addEventListener('click', () => {
                if (!this.panoViewer) return;

                if (this.panoViewer.projection === 'sphere') {
                    this.panoViewer.setProjection('flat');
                    projBtn.textContent = 'Flat';
                } else {
                    this.panoViewer.setProjection('sphere');
                    projBtn.textContent = '360°';
                }
            });
        }
    }

    handleQualityChange() {
        if (!this.currentJobId || !this.splatViewer) return;
        const quality = document.getElementById('splat-quality').value;
        const url = quality === 'preview'
            ? `/api/preview/${this.currentJobId}`
            : `/api/download/${this.currentJobId}?format=splat`;

        console.log(`[App] Loading ${quality} splat: ${url}`);
        this.splatViewer.loadSplat(url);
    }

    switchTab(mode) {
        const tabRgb = document.getElementById('tab-rgb');
        const tabDepth = document.getElementById('tab-depth');

        if (mode === 'rgb' && this.rgbUrl && this.panoViewer) {
            tabRgb.classList.add('active');
            tabDepth.classList.remove('active');
            this.panoViewer.loadImage(this.rgbUrl);

        } else if (mode === 'depth' && this.depthUrl && this.panoViewer) {
            tabDepth.classList.add('active');
            tabRgb.classList.remove('active');
            this.panoViewer.loadImage(this.depthUrl);
        }
    }

    async preloadTestImage() {
        const testImagePath = '/TestImage/monbachtal_riverbank_primary.jpg';
        try {
            const response = await fetch(testImagePath, { method: 'HEAD' });
            if (response.ok) {
                // Load test image into pano viewer
                if (this.panoViewer) {
                    console.log('[App] Preloading test image');
                    this.rgbUrl = testImagePath; // Save as RGB URL
                    this.panoViewer.loadImage(testImagePath);
                }

                // Fetch and set as current file for conversion
                const imgResponse = await fetch(testImagePath);
                const blob = await imgResponse.blob();
                const file = new File([blob], 'monbachtal_riverbank_primary.jpg', { type: 'image/jpeg' });

                this.currentFile = file;
                this.fileLabel.textContent = 'monbachtal_riverbank_primary.jpg (demo)';
                this.convertBtn.disabled = false;
                this.setStatus('Demo image loaded - ready to convert');
            }
        } catch (e) {
            console.log('[App] No test image available for preload');
        }
    }

    initViewers() {
        const panoContainer = document.getElementById('pano-viewer');
        const splatContainer = document.getElementById('splat-viewer');

        // Initialize 360 viewer
        if (typeof PanoViewer !== 'undefined') {
            this.panoViewer = new PanoViewer(panoContainer);
        }

        // Initialize splat viewer
        if (typeof SplatViewer !== 'undefined') {
            this.splatViewer = new SplatViewer(splatContainer);
        }
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        console.log('[App] File selected:', file ? file.name : 'none');
        if (!file) return;

        this.currentFile = file;
        this.fileLabel.textContent = file.name;
        this.convertBtn.disabled = false;
        this.downloadPlyBtn.disabled = true;
        this.downloadSplatBtn.disabled = true;
        this.downloadZipBtn.style.display = 'none';

        // Check if video
        this.isVideo = file.type.startsWith('video/');
        if (this.isVideo) {
            console.log('[App] Video detected');
            this.videoParams.style.display = 'flex';
            this.videoInfo.style.display = 'flex';
            this.downloadSplatBtn.style.display = 'none';
            this.downloadPlyBtn.style.display = 'none';
            this.downloadZipBtn.style.display = 'flex';
            this.downloadZipBtn.disabled = true;
            this.extractVideoMetadata(file);
        } else {
            this.videoParams.style.display = 'none';
            this.videoInfo.style.display = 'none';
            this.downloadSplatBtn.style.display = 'flex';
            this.downloadPlyBtn.style.display = 'flex';
            this.downloadZipBtn.style.display = 'none';
        }

        // Load into 360 viewer

        // Load into 360 viewer
        if (this.panoViewer) {
            const url = URL.createObjectURL(file);
            console.log('[App] Loading pano from blob URL:', url);
            this.rgbUrl = url;
            this.switchTab('rgb');

            // Disable depth tab
            const tabDepth = document.getElementById('tab-depth');
            if (tabDepth) tabDepth.disabled = true;
        } else {
            console.warn('[App] PanoViewer not initialized');
        }

        this.setStatus('Ready to convert');
    }

    async startConversion() {
        if (!this.currentFile) return;

        this.convertBtn.disabled = true;
        this.setStatus('Uploading...', 'Preparing');

        // Clear old subtext
        const subtexts = document.querySelectorAll('.status-subtext');
        subtexts.forEach(el => el.remove());

        // Disable depth tab during conversion
        const tabDepth = document.getElementById('tab-depth');
        if (tabDepth) tabDepth.disabled = true;

        // Disable quality select
        const qualitySelect = document.getElementById('splat-quality');
        if (qualitySelect) qualitySelect.disabled = true;

        // Prepare form data
        const formData = new FormData();
        formData.append('file', this.currentFile);

        // Add parameters as query string
        const params = new URLSearchParams({
            stride: this.strideSelect.value,
            scale_factor: this.scaleFactorInput.value,
            thickness: this.thicknessInput.value,
            global_scale: this.globalScaleInput.value,
            depth_min: this.depthMinInput.value,
            depth_max: this.depthMaxInput.value,
            sharp_refine: this.sharpRefineInput ? this.sharpRefineInput.checked : false,
            sharp_projection: this.sharpProjectionInput ? this.sharpProjectionInput.value : 'cubemap',
            scale_blend: this.scaleBlendInput ? this.scaleBlendInput.value : 0.5,
            opacity_blend: this.opacityBlendInput ? this.opacityBlendInput.value : 1.0,
            sharp_cubemap_size: this.sharpCubemapSizeInput ? this.sharpCubemapSizeInput.value : 1536,
            sky_threshold: this.skyThresholdInput ? this.skyThresholdInput.value : 80.0,
            color_blend: this.colorBlendInput ? this.colorBlendInput.value : 0.5,
            sky_dome: this.skyDomeInput ? this.skyDomeInput.checked : true,
            depth_model: this.depthModelSelect ? this.depthModelSelect.value : 'panda',
            da3_projection: this.da3ProjectionInput ? this.da3ProjectionInput.value : 'equirectangular',
            guided_filter: this.guidedFilterInput ? this.guidedFilterInput.checked : true,
            guided_strength: this.guidedStrengthInput ? this.guidedStrengthInput.value : 0.25
        });

        try {
            let url = '/api/convert';
            if (this.isVideo) {
                url = '/api/convert_video';
                params.append('fps', this.fpsInput.value);
                params.append('start_time', this.videoStartInput.value);
                params.append('temporal_alpha', this.temporalAlphaInput.value);
                params.append('temporal_alpha', this.temporalAlphaInput.value);
                params.append('stabilize_video', this.stabilizeVideoInput.checked);
                params.append('scale_blend', this.scaleBlendInput ? this.scaleBlendInput.value : 0.5);
                params.append('opacity_blend', this.opacityBlendInput ? this.opacityBlendInput.value : 1.0);
                params.append('sharp_cubemap_size', this.sharpCubemapSizeInput ? this.sharpCubemapSizeInput.value : 1536);
                params.append('sky_threshold', this.skyThresholdInput ? this.skyThresholdInput.value : 80.0);
                if (this.videoDurationInput.value) {
                    params.append('duration', this.videoDurationInput.value);
                }
            }

            const response = await fetch(`${url}?${params}`, {
                method: 'POST',
                body: formData
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
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }

        this.pollInterval = setInterval(() => this.checkJobStatus(), 1000);
    }

    async checkJobStatus() {
        if (!this.currentJobId) return;

        try {
            const response = await fetch(`/api/status/${this.currentJobId}`);

            // Handle non-OK responses
            if (!response.ok) {
                throw new Error(`Status check failed: ${response.status}`);
            }

            const status = await response.json();

            // Reset error count on success
            this.pollErrorCount = 0;

            if (status.status === 'queued') {
                this.setStatus('Waiting...', `Queue position: ${status.queue_position}`);

            } else if (status.status === 'processing') {
                if (status.is_video) {
                    this.setStatus('Processing Video...', `Frame ${status.current_frame} / ${status.total_frames}`);
                } else {
                    this.setStatus('Processing...', 'GPU active');
                }

            } else if (status.status === 'complete') {
                clearInterval(this.pollInterval);
                this.pollInterval = null;

                this.setStatus('Complete!',
                    this.isVideo
                        ? `${status.total_frames} frames • ${status.total_frames} splats`
                        : `${status.splat_count.toLocaleString()} splats • ${status.file_size_mb} MB • ${status.processing_time}s`
                );

                // Additional status info for SHARP
                if (status.params && status.params.sharp_refine) {
                    const blend = status.params.scale_blend;
                    const sharpInfo = document.createElement('div');
                    sharpInfo.className = 'status-subtext';
                    sharpInfo.style.fontSize = '0.85em';
                    sharpInfo.style.color = '#4CAF50';
                    sharpInfo.textContent = `✨ SHARP Active (Blend: ${blend})`;
                    this.statusText.parentElement.appendChild(sharpInfo);

                    // Auto-remove after 5s or next job? 
                    // Better to just clear it on start.
                }

                if (this.isVideo) {
                    this.downloadZipBtn.disabled = false;
                    this.zipUrl = status.zip_url;

                    if (status.preview_manifest_url && this.splatViewer) {
                        this.splatViewer.loadVideo(status.preview_manifest_url);
                    }
                } else {
                    this.downloadPlyBtn.disabled = false;
                    this.downloadSplatBtn.disabled = false;
                }

                // Enable quality select
                const qualitySelect = document.getElementById('splat-quality');
                if (qualitySelect) {
                    qualitySelect.disabled = false;
                    qualitySelect.value = 'preview';
                }

                // Load preview into splat viewer
                if (this.splatViewer && status.preview_url) {
                    this.splatViewer.loadSplat(status.preview_url);
                }

                // Enable depth tab
                if (status.depth_preview_url) {
                    this.depthUrl = status.depth_preview_url;
                    const tabDepth = document.getElementById('tab-depth');
                    if (tabDepth) tabDepth.disabled = false;
                }

                // Re-enable convert button so user can re-convert with different params
                this.convertBtn.disabled = false;

            } else if (status.status === 'error') {
                clearInterval(this.pollInterval);
                this.pollInterval = null;

                this.setStatus(`Error: ${status.error}`);
                this.convertBtn.disabled = false;
            }

        } catch (error) {
            // Handle polling errors - stop polling and show error after multiple failures
            console.error('Polling error:', error);
            this.pollErrorCount = (this.pollErrorCount || 0) + 1;

            if (this.pollErrorCount >= 5) {
                clearInterval(this.pollInterval);
                this.pollInterval = null;
                this.setStatus('Connection error - please try again');
                this.convertBtn.disabled = false;
                this.pollErrorCount = 0;
            }
        }
    }

    downloadFile(format) {
        if (!this.currentJobId) return;

        const url = `/api/download/${this.currentJobId}?format=${format}`;
        const link = document.createElement('a');
        link.href = url;
        link.download = `spag4d_${this.currentJobId.slice(0, 8)}.${format}`;
        link.click();
    }

    downloadZip() {
        if (!this.zipUrl) return;
        const link = document.createElement('a');
        link.href = this.zipUrl;
        link.download = `spag4d_video_${this.currentJobId.slice(0, 8)}.zip`;
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

    setStatus(text, progress = '') {
        this.statusText.textContent = text;
        this.progressText.textContent = progress;
    }

    extractVideoMetadata(file) {
        if (!this.videoMetadata) return;
        this.videoMetadata.textContent = 'Analyzing...';

        const video = document.createElement('video');
        video.preload = 'metadata';
        video.onloadedmetadata = () => {
            const width = video.videoWidth;
            const height = video.videoHeight;
            const duration = video.duration;
            let fps = 30; // Default fallback

            // Attempt to detect FPS via MediaStreamTrack settings
            try {
                // captureStream() is not standard but widely supported in Chrome/Firefox
                if (video.captureStream) {
                    const stream = video.captureStream();
                    const tracks = stream.getVideoTracks();
                    if (tracks.length > 0) {
                        const settings = tracks[0].getSettings();
                        if (settings.frameRate) {
                            fps = settings.frameRate;
                            console.log('[App] Detected FPS:', fps);
                            if (this.fpsInput) this.fpsInput.value = Math.round(fps);
                        }
                    }
                }
            } catch (e) {
                console.warn('[App] Could not detect FPS via captureStream', e);
            }

            // Reset time inputs
            if (this.videoStartInput) {
                this.videoStartInput.value = "0.0";
                this.videoStartInput.max = duration;
            }
            if (this.videoDurationInput) {
                this.videoDurationInput.value = "";
                this.videoDurationInput.placeholder = "All";
                this.videoDurationInput.max = duration;
            }

            const durationStr = duration < 60
                ? `${duration.toFixed(1)}s`
                : `${Math.floor(duration / 60)}m ${(duration % 60).toFixed(0)}s`;

            this.videoMetadata.innerHTML = `
                <span>${width}x${height}</span> • 
                <span>${durationStr}</span> • 
                <span>${Math.round(fps)} FPS</span>
             `;

            // Cleanup
            window.URL.revokeObjectURL(video.src);
        };

        video.onerror = () => {
            this.videoMetadata.textContent = 'Could not read metadata';
        };

        video.src = URL.createObjectURL(file);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new SPAG4DApp();
});
