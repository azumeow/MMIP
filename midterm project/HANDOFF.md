# Codex handoff — MailGuard / MMIP midterm

## User-approved requirements

- Due **2026-10-26 08:00 Asia/Taipei**.
- Local computer/classroom demonstration; no public deployment requested.
- React frontend; Traditional Chinese and English interface/input; Simplified Chinese can be converted to Traditional.
- Email / arbitrary text input. Three main classes: normal, spam, scam. Show highest class probability and all main probabilities.
- Two genuinely trained models: TF-IDF + Logistic Regression and fine-tuned BERT. Do not substitute keyword heuristics for either model.
- If scam: highlight suspicious original phrases, explain cues, show one of six scam categories and its confidence. Be honest about rule-derived explanations and uncalibrated subtype confidence.
- Six categories exactly: 投資詐騙、網路購物與消費詐騙、情感與社交信任詐騙、政府與公家機關冒充、求職與申貸詐騙、科技與帳號盜用。
- Source lives in `MMIP/midterm project`. Keep large raw data and weights out of Git; user can place portable packages on the local Desktop.
- User PC: RTX 3060 Ti. Preferred training machine: **Tesla V100 PCIe 32 GB, 6 cores, 60 GiB RAM**.
- User's existing local SSH config defines the alias `Lab`. No jump host is configured. User can run `ssh Lab` from their PC. Connection details and credentials remain on the user's PC; do not request/upload private keys or copy SSH configuration into the repository.

## Current implementation

- `frontend/`: React 19 + TypeScript + Vite, bilingual responsive UI, example inputs, LR/BERT selection and comparison, probability bars, uncertainty/disagreement messages, six subtype labels, transparent evidence highlights, metrics modal, error/loading states, no stored input. Python code-point offsets are rendered correctly even with emoji.
- `backend/app.py`: FastAPI, validated input (3–12,000 characters), model health, `/api/analyze`, `/api/metrics`, trained-artifact loading, partial model failures, 503 if no requested model exists. Serves built frontend on a single local origin. No fake predictions. No database or third-party inference API.
- `ml/prepare_data.py`: public corpus download, provenance, normalization, Simplified→Traditional conversion, quarantine ambiguous UCI spam, synthetic bilingual templates, near-duplicate/template grouping, group-disjoint splits, manifest/audit queue.
- `ml/train_lr.py`: character+word TF-IDF LR; held-out validation sigmoid calibration; separate six-type LR; independent test report.
- `ml/train_bert.py`, `bert_model.py`: pinned real `google-bert/bert-base-multilingual-cased`; shared encoder with main/subtype heads, full GPU fine-tune or explicitly partial CPU fine-tune; early selection by validation macro-F1; validation temperature scaling; safetensors outputs; test report.
- `ml/explain.py`: independent bilingual transparent rules; offsets refer to original input. Rules do not override classification.
- Setup/start/Windows/V100/export scripts, dataset card, requirements, Git ignores and API/preprocessing tests.

## Verified so far (update this section when resuming)

- React TypeScript compilation and final production Vite build passed, including correction of an initial CSS empty-import warning.
- Chromium/Playwright loaded the real page at 1440 px desktop and 390 px mobile widths with no horizontal overflow. Screenshots exist only in the cloud workspace `/workspace/.mmip-env/mailguard-desktop.png` and `mailguard-mobile.png` unless copied into the handoff package later.
- The cloud FastAPI server started on 127.0.0.1:8000. It is only a process in the old environment and does NOT migrate.
- Python 3.11.17, PyTorch 2.5.1+cpu, Transformers 4.57.6, scikit-learn 1.9.1, OpenCC, FastAPI and test dependencies were installed in `/workspace/.mmip-env/venv`; this virtualenv is NOT portable and is NOT in the package.
- Official UCI SMS ZIP downloaded, SHA256 `1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3`.
- **No LR or BERT model has been trained yet. No final model metrics exist.** Missing models must stay visibly unavailable.
- Data preparation has not completed because public phishing download is blocked in the cloud. No processed train/test files yet.
- The test suite was written; consult `reports/handoff_validation.json` for its actual executed outcome. Do not claim skipped integration tests passed.
- **V100 script has not been executed or verified on the user's machine.** A direct cloud TCP connection to the user's supplied SSH endpoint returned Connection refused. Merely copying SSH config does not confer local network access.

## Cloud network issue / local transfer

Metadata and tokenizer downloads from huggingface.co work. Large data/weights redirect to **us.aws.cdn.hf.co** and received proxy tunnel 403 in the cloud. A draft allowlist addition was saved, but draft persistence does not prove runtime application. Continue using TLS/artifact verification; do not disable SSL or checksum checks. These restrictions may not apply when running locally/on V100.

Original cloud custom-domain additions: huggingface.co, cdn-lfs.huggingface.co, cdn-lfs-us-1.hf.co, cas-bridge.xethub.hf.co, cas-server.xethub.hf.co, datasets-server.huggingface.co, archive.ics.uci.edu, api.github.com, us.aws.cdn.hf.co. No Hugging Face token is required for this public BERT model. Do not add secret requirements without a demonstrated need.

## Next actions for local/remote Codex

1. Inspect the actual project and this file. Preserve existing user files; do not create a new Git worktree unless requested. The user authorizes completing the project, not deploying it publicly.
2. Work on the V100 via an actual locally initiated Remote–SSH / terminal session if available. Verify the remote identity, Python, disk, GPU, NVIDIA driver and CUDA before installing/training. The local VS Code/Codex process must genuinely have remote terminal access; do not infer it from a config file.
3. Run setup scripts or equivalent reviewed commands. Python 3.11+ is required; resolve missing interpreter normally. Check `pip check` and CUDA availability. V100 script uses torch 2.5.1 cu121 and fp16, not bf16.
4. Complete public corpus acquisition and preparation; inspect actual JSON format (expected list of `{text,label}`). Review output manifest/counts and at least the audit sample; fix genuine label ambiguities in a versioned annotation file rather than silently treating weak labels as gold.
5. Improve real Traditional Chinese / six-category data if suitable licensed sources are available. Current Chinese supervision is synthetic and MUST remain disclosed. Do not claim all data manually annotated or real.
6. Train LR and full BERT on the same prepared split. `bash scripts/train_v100.sh` orchestrates the intended GPU workflow, but has not been tested on V100. Keep test data out of model selection. Capture real metrics and inspect per-language and per-source results.
7. Add/verify representative three-class, six-type, bilingual, Simplified, long-input, benign-warning, disagreement and failure-state tests. Run API tests, UI end-to-end tests with actual trained models, production build and view desktop/mobile screenshots. Do not rely on build-only checks.
8. Investigate model errors and calibrate rather than forcing example results with keyword overrides. Report limitations and actual outcomes honestly. Both trained models are required for final completion.
9. Finalize report, model/data cards, screenshots, Windows/local startup and portable source+artifacts exports. Check repository status; do not commit raw datasets, virtualenvs or large model files. Do not claim old cloud processes or filesystem paths exist on the user's PC.

## Useful commands

```bash
# Inside the copied project directory (not the MMIP repository root):
python -m ml.prepare_data
python -m ml.train_lr
python -m ml.train_bert --device cuda --epochs 3 --batch-size 16 --max-length 192
python -m pytest -q
npm --prefix frontend ci
npm --prefix frontend run build
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --no-access-log
python -m scripts.package_project --include-models --include-data
```

This is a progress handoff, not a completed project submission. Continue implementation autonomously within the approved scope; do not restart from scratch or claim unexecuted training passed.
