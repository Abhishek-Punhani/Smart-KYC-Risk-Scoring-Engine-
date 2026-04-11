from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

from job_manager import JobManager
from pipeline import score_single_customer


def create_app() -> Flask:
    app = Flask(__name__)

    workspace_root = Path(__file__).resolve().parent.parent
    default_dataset = workspace_root / "kyc_industry_dataset.csv"
    runs_dir = Path(__file__).resolve().parent / "runs"

    manager = JobManager(base_dir=runs_dir)

    @app.get("/api/health")
    def health() -> Any:
        return jsonify({"status": "ok", "service": "kyc-backend"})

    @app.get("/api/runs")
    def list_runs() -> Any:
        return jsonify({"runs": manager.list_runs()})

    @app.post("/api/runs")
    def start_run() -> Any:
        payload = request.get_json(silent=True) or {}
        config = payload.get("config", {})

        dataset_path = payload.get("dataset_path")
        if dataset_path:
            dpath = Path(dataset_path)
        else:
            dpath = default_dataset

        if not dpath.exists():
            return jsonify({"error": f"Dataset not found: {dpath}"}), 400

        run_id = manager.create_run(dpath, config=config)
        return jsonify({"run_id": run_id, "status": "queued"}), 202

    @app.get("/api/runs/<run_id>")
    def get_run(run_id: str) -> Any:
        run = manager.get(run_id)
        if run is None:
            return jsonify({"error": "Run not found"}), 404
        return jsonify(run)

    @app.get("/api/runs/<run_id>/summary")
    def get_run_summary(run_id: str) -> Any:
        run = manager.get(run_id)
        if run is None:
            return jsonify({"error": "Run not found"}), 404
        return jsonify(
            {"run_id": run_id, "status": run["status"], "summary": run.get("summary")}
        )

    @app.get("/api/runs/<run_id>/artifacts")
    def list_artifacts(run_id: str) -> Any:
        run = manager.get(run_id)
        if run is None:
            return jsonify({"error": "Run not found"}), 404

        run_dir = Path(run["run_dir"])
        if not run_dir.exists():
            return jsonify({"files": []})

        files = []
        for f in sorted(run_dir.glob("*")):
            if f.is_file():
                files.append(
                    {
                        "name": f.name,
                        "size_bytes": f.stat().st_size,
                        "url": f"/api/runs/{run_id}/artifacts/{f.name}",
                    }
                )

        return jsonify({"run_id": run_id, "files": files})

    @app.get("/api/runs/<run_id>/artifacts/<path:filename>")
    def get_artifact(run_id: str, filename: str) -> Any:
        run = manager.get(run_id)
        if run is None:
            return jsonify({"error": "Run not found"}), 404
        run_dir = Path(run["run_dir"])
        file_path = run_dir / filename
        if not file_path.exists() or not file_path.is_file():
            return jsonify({"error": "Artifact not found"}), 404
        return send_from_directory(run_dir, filename, as_attachment=True)

    @app.post("/api/score")
    def score_customer() -> Any:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "JSON body required"}), 400

        try:
            result = score_single_customer(payload)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
