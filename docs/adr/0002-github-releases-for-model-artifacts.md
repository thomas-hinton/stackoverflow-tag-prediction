# Store trained Taggers as GitHub Release assets

Trained Taggers are not committed to git (the Transformer tagger is around 250 MB). They are published as assets of a GitHub Release, together with their metrics file, by the training workflow. One Release is one versioned pair of Taggers tied to a commit, and the Docker image is built from it. This keeps the whole lifecycle (code, CI, training, artifacts, container image) inside GitHub, which is a goal of the project.

## Considered Options

- **Hugging Face Hub**: natural home for transformer weights, but adds a second platform and account to the lifecycle.
- **Git LFS**: limited free storage and bandwidth quotas on GitHub, and model versions would not be linked to their metrics.
- **DVC with remote storage**: powerful, but it needs a separate storage backend and is more tooling than a two-model project warrants.
