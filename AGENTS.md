# Agent instructions

Before opening a pull request, run the check in `.github/workflows/pr-check.yml`. Use Python 3.11, the version CI uses, in a virtual environment kept outside the repository (`.venv/` is not gitignored here):

```bash
python -m pip install -r requirements.txt
python -m pytest scripts -q
```

`scripts` holds both the ETL tests and the contract tests. Run the whole directory before a pull request, as CI does.

The pre-commit hook scans staged changes with gitleaks (`gitleaks.toml`). Enable it once per clone as shown in [README.md → 로컬 훅](README.md#로컬-훅).
