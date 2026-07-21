import glob
import subprocess
import sys
from pathlib import Path
from flask import Flask, redirect, render_template, request, url_for

ROOT = Path(__file__).resolve().parent
MODEL_CONFIG = ROOT / "configs" / "synthetic_demo.json"
PROMPT_PATH = ROOT / "prompt.txt"
OUTPUT_PATTERN = ROOT / "predictions_web_*.txt"
FALLBACK_PREFIX = ROOT / "predictions_web_fallback"

app = Flask(__name__)


def run_prediction(prompt_text: str) -> dict:
    # Try to import heavy dependencies first; if they are missing, use a lightweight fallback generator
    try:
        import mesh_tensorflow  # noqa: F401
        can_run_backend = True
    except Exception:
        can_run_backend = False

    if can_run_backend:
        PROMPT_PATH.write_text(prompt_text, encoding="utf-8")
        command = [
            sys.executable,
            str(ROOT / "main.py"),
            "--model",
            str(MODEL_CONFIG),
            "--predict",
            "--prompt",
            str(PROMPT_PATH),
            "--sacred_id",
            "web"
        ]
        result = subprocess.run(command, capture_output=True, text=True, cwd=str(ROOT))

        prediction_files = sorted(glob.glob(str(OUTPUT_PATTERN)), key=lambda path: Path(path).stat().st_mtime)
        output_text = None
        if prediction_files:
            output_text = Path(prediction_files[-1]).read_text(encoding="utf-8")

        # If the backend returned an error or produced no output file, fall back to the lightweight generator
        if result.returncode != 0 or output_text is None:
            # create fallback output
            fallback_msg = "[Fallback after backend failure] Backend returned non-zero or produced no output.\n" + result.stderr
            try:
                raw_path = ROOT / "data" / "raw" / "input.txt"
                sample = raw_path.read_text(encoding="utf-8", errors="replace")[:800] if raw_path.exists() else None
            except Exception:
                sample = None

            generated = [prompt_text.strip(), ""]
            if sample:
                generated.append("Source excerpt:\n" + sample)
            generated.append("")
            generated.append(fallback_msg)
            output_text = "\n\n".join(generated)
            import time
            fname = f"{FALLBACK_PREFIX}_{int(time.time())}.txt"
            Path(fname).write_text(output_text, encoding="utf-8")

            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_text": output_text,
                "output_file": fname,
                "fallback": True,
            }

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "output_text": output_text,
            "output_file": prediction_files[-1] if prediction_files else None,
            "fallback": False,
        }

    # Fallback generator: no heavy dependencies required
    try:
        # prefer using a short excerpt from local Gutenberg raw file if available
        raw_path = ROOT / "data" / "raw" / "input.txt"
        if raw_path.exists():
            sample = raw_path.read_text(encoding="utf-8", errors="replace")[:1000]
        else:
            sample = None
    except Exception:
        sample = None

    # very small deterministic "generation": echo prompt and append a short canned continuation
    generated = []
    generated.append(prompt_text.strip())
    generated.append("")
    if sample:
        generated.append("Source excerpt:\n" + sample)
    generated.append("")
    generated.append("[Fallback generation] This is a lightweight local demo response. To enable real model outputs, install Mesh TensorFlow and the required dependencies, and provide a model checkpoint.")
    output_text = "\n\n".join(generated)

    # write a fallback output file so the UI can show it
    import time
    fname = f"{FALLBACK_PREFIX}_{int(time.time())}.txt"
    Path(fname).write_text(output_text, encoding="utf-8")

    return {
        "returncode": 0,
        "stdout": "Fallback generation used (mesh_tensorflow not installed).",
        "stderr": "",
        "output_text": output_text,
        "output_file": fname,
        "fallback": True,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)


@app.route("/generate", methods=["POST"])
def generate():
    prompt_text = request.form.get("prompt", "")
    if not prompt_text.strip():
        prompt_text = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley."

    result = run_prediction(prompt_text)
    return render_template("index.html", prompt=prompt_text, result=result)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
