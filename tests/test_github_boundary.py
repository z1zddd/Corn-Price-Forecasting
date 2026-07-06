from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

IGNORED_PARTS = {
    ".git",
    ".mplconfig",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
    ".venv",
    "experiments",
    "outputs",
}

FORBIDDEN_SUFFIXES = {
    ".docx",
    ".joblib",
    ".pkl",
    ".pt",
    ".pth",
    ".tar",
    ".gz",
    ".zip",
}

FORBIDDEN_NAME_ENDINGS = (
    "_outputs",
)


def is_ignored(path: Path) -> bool:
    parts = set(path.relative_to(REPO_ROOT).parts)
    return bool(parts & IGNORED_PARTS)


def test_repository_contains_no_large_or_generated_artifacts():
    offenders: list[str] = []
    for path in REPO_ROOT.rglob("*"):
        if is_ignored(path):
            continue
        if any(part.endswith(FORBIDDEN_NAME_ENDINGS) for part in path.parts):
            offenders.append(str(path.relative_to(REPO_ROOT)))
            continue
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_github_packaging_files_exist():
    assert (REPO_ROOT / "LICENSE").exists()
    assert (REPO_ROOT / ".github" / "workflows" / "tests.yml").exists()
    assert (REPO_ROOT / "scripts" / "run_corn_smoke.py").exists()
    assert (REPO_ROOT / "scripts" / "clean_outputs.py").exists()

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "trees = [" in pyproject
    assert "tree = [" in pyproject


def test_docs_cover_agent_workflow_commands_and_model_families():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    workflow = (REPO_ROOT / "docs" / "agent-workflow.md").read_text(encoding="utf-8")
    architecture = (REPO_ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    configuration = (REPO_ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")
    metrics = (REPO_ROOT / "docs" / "metrics.md").read_text(encoding="utf-8")
    combined_model_docs = "\n".join([readme, architecture, configuration])

    for command in ["diagnose", "build-config", "auto-window", "run", "run-lookbacks", "interpret"]:
        assert command in workflow
        assert command in readme
    for model_name in [
        "regression_mse_sign",
        "regression_mae_sign",
        "regression_huber_sign",
        "dual_head_mse_bce",
        "focal_logistic",
        "lstm",
        "gru",
        "transformer",
        "patchtst",
        "itransformer",
        "dlinear",
    ]:
        assert model_name in combined_model_docs
    assert "R2_health" in metrics
    assert "rolling_dir_acc.png" in readme
    assert "rolling_sharpe.png" in readme