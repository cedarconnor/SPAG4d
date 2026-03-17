// GaussianSplats3D Viewer wrapper
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

export class Viewer {
    constructor(container) {
        this.container = container;
        this.viewer = null;
    }

    async loadScene(url) {
        this.dispose();
        this._lastUrl = url;

        // Hide placeholder
        const placeholder = this.container.querySelector('.viewer-placeholder');
        if (placeholder) placeholder.style.display = 'none';

        this.viewer = new GaussianSplats3D.Viewer({
            rootElement: this.container,
            cameraUp: [0, 1, 0],
            initialCameraLookAt: [0, 0, -1],
            initialCameraPosition: [0, 0, 0],
            sharedMemoryForWorkers: false,
            dynamicScene: false,
        });

        // Detect format from URL extension; default to PLY for API URLs without extension
        const FORMAT_MAP = { ply: GaussianSplats3D.SceneFormat.Ply,
                             splat: GaussianSplats3D.SceneFormat.Splat,
                             ksplat: GaussianSplats3D.SceneFormat.KSplat,
                             spz: GaussianSplats3D.SceneFormat.Spz };
        const ext = url.split('?')[0].split('.').pop()?.toLowerCase();
        const format = FORMAT_MAP[ext] ?? GaussianSplats3D.SceneFormat.Ply;

        await this.viewer.addSplatScene(url, {
            splatAlphaRemovalThreshold: 1,
            showLoadingUI: true,
            progressiveLoad: true,
            format,
        });

        this.viewer.start();
    }

    async loadFromFile(arrayBuffer, filename) {
        const blob = new Blob([arrayBuffer]);
        const url = URL.createObjectURL(blob);
        try {
            await this.loadScene(url);
        } finally {
            // Don't revoke immediately — viewer needs it during load
            setTimeout(() => URL.revokeObjectURL(url), 30000);
        }
    }

    resetView() {
        if (!this.viewer) return;
        const camera = this.viewer.camera;
        const controls = this.viewer.controls;
        if (!camera) return;

        // Use scene bounding box center as target if available, else origin
        let cx = 0, cy = 0, cz = 0;
        const splatMesh = this.viewer.splatMesh;
        if (splatMesh && typeof splatMesh.getCenter === 'function') {
            const center = splatMesh.getCenter();
            cx = center.x; cy = center.y; cz = center.z;
        }

        camera.position.set(cx, cy, cz);
        camera.up.set(0, 1, 0);
        if (controls) {
            controls.target.set(cx, cy, cz - 1);
            controls.update();
        }
    }

    dispose() {
        if (this.viewer) {
            try {
                this.viewer.dispose();
            } catch (e) {
                console.warn('Viewer dispose error:', e);
            }
            this.viewer = null;
        }
        // Clean up any canvases left behind
        const canvases = this.container.querySelectorAll('canvas');
        canvases.forEach(c => c.remove());
    }
}
