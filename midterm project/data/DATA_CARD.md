# Dataset and annotation card — development version

此文件記錄**計畫與程式實作**。實際完成下載／前處理的數量與 checksum，以產生的 `reports/dataset_manifest.json` 為準；不能把目標筆數當作已取得筆數。

| Source | Purpose | Provenance / license | Label treatment |
|---|---|---|---|
| UCI SMS Spam Collection | English ham / spam | https://archive.ics.uci.edu/dataset/228/sms+spam+collection ; UCI dataset page declares CC BY 4.0. Almeida, Hidalgo, Yamakami, 2011. | Original ham→normal; spam→spam, but spam with explicit scam-type cues is quarantined for review. Binary spam is not automatically gold scam. |
| ealvaradob/phishing-dataset, texts subset | English phishing / benign | https://huggingface.co/datasets/ealvaradob/phishing-dataset ; pinned revision `55caf9af01ae498d68ff44886ff69a4b165eb54b`; compilation declares Apache-2.0. Upstream email/SMS sources and licenses must also be consulted before redistribution. | Source phishing label→scam; source benign→normal (possible promotion/label noise). Six-category labels, where available, are explicitly weak rule labels. No remote dataset Python is executed. |
| `seed_templates.json` | Traditional Chinese / English, six-type coverage | Project-authored synthetic examples, not collected real user messages or human-audited corpus | Every record marked `synthetic_template_label`. Translations and numeric variants share a template family. Included normal counterexamples mention fraud warnings, investment risk, and legitimate loan/job messages. |

The source defines phishing as malicious and benign otherwise. Its benign class is not guaranteed to exclude all promotional mail. Source labels must be audited before treating three-class performance as authoritative.

## Six categories

1. Investment (`investment`)
2. Shopping / consumer (`shopping`)
3. Relationship / social trust (`relationship`)
4. Government impersonation (`government`)
5. Job / loan (`job_loan`)
6. Technology / account theft (`account`)

Scam takes priority over marketing. A message mentioning police, investment, passwords, URLs or urgency is **not automatically a scam**. Legitimate warnings, ordinary invoices and account notices require context. Unknown or ambiguous examples belong in the review queue.

## Reproducibility and leakage controls

- Seed: 20261023. Public phishing sampled up to 1,600 per binary class; normal SMS capped at 3,000. Templates default to 20 variants per language/family. These are configurable targets.
- NFKC, HTML unescape/tag removal, OpenCC `s2twp`, zero-width removal, whitespace normalization. No stemming/stop-word removal for BERT.
- Exclude blank/very short/over-12,000-character records; remove normalized conflicting labels and exact duplicates.
- Union identical numeric/URL-masked fingerprints, bilingual template families, and character 4-gram hash cosine similarity ≥0.94 before group-stratified splitting.
- Class/language-stratified grouped 60/20/20 approximation. No group or fingerprint overlap allowed across train/validation/test.
- Validation is used for calibration and BERT epoch selection. The test split is evaluated only after selection.
- Data annotation methods and subtype weak labels must remain explicit. An audit sample is generated; it is **not** automatically a completed human review.

## Outstanding data work

Real Traditional Chinese data and independently adjudicated six-category labels are still needed. Do not claim that generated translations, synthetic templates, or heuristic subtypes are manually verified real-world examples. Collection alone is not sufficient; document ambiguous cases, label corrections, source balance and duplicate decisions.

Do not execute links or attachments in the corpus. Original datasets can contain personal information; do not publish raw messages in UI screenshots or logs. The website does not retain user input.
