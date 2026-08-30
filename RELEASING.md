# Releasing vovo-mlx (maintainer notes)

## One-time setup

1. Create a PyPI account at https://pypi.org/account/register/ (and enable 2FA — required for uploads).
2. Create an API token at https://pypi.org/manage/account/token/ — scope "entire account" the first time
   (the project does not exist yet), then replace it with a project-scoped token after the first release.
3. Put it where `uv publish` finds it: `export UV_PUBLISH_TOKEN=pypi-…` (or pass `--token`). The username
   for token auth is always `__token__`.

Optional dry run against TestPyPI (separate account at https://test.pypi.org): add `--publish-url
https://test.pypi.org/legacy/` to the publish step and install with
`pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple vovo-mlx`.

## Every release

```
# 1. version: bump `version` in pyproject.toml and `__version__` in vovo_mlx/__init__.py (keep them equal)
# 2. tests (with the Swift parity dumps if available)
pytest
# 3. build sdist + wheel into dist/
rm -rf dist && uv build
python -m zipfile -l dist/*.whl | grep en_US.txt     # the lexicon must be in the wheel
# 4. smoke-test the wheel in a throwaway venv
uv venv /tmp/vovo-wheel --python 3.12 && uv pip install -p /tmp/vovo-wheel/bin/python dist/*.whl \
  && /tmp/vovo-wheel/bin/vovo-mlx say "Release check." -o /tmp/release.wav
# 5. publish
uv publish            # uses UV_PUBLISH_TOKEN
# 6. tag
git tag v0.1.0 && git push --tags
```

`pip install vovo-mlx` works within a minute of publishing. A version can never be re-uploaded: fix,
bump, publish again.

## Weights

The weights live on the Hub, not in the package. To update them (from the Vovo repo, logged in with `hf`):

```
vovo export --ckpt checkpoints/<run>/step_N.safetensors --out exports/vovo/model.safetensors
cp <vocoder>.safetensors exports/vovo/vocoder.safetensors
hf upload franckverrot/vovo exports/vovo . --commit-message "…"
```

The package does not pin a weights revision; pass `revision=` to `from_pretrained` if you need one.
