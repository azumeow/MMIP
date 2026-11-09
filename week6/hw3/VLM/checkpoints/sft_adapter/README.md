# InternVL3.5-14B 圖片描述 SFT＋QLoRA adapter

這是語言部分的 LoRA adapter，不含完整 VLM 基礎權重。

基礎模型：OpenGVLab/InternVL3_5-14B
Revision：a1e37197b393ce9eec9df700fef65c11f4a6ffbd

載入時先依 ../../experiment.py 的 load_model() 建立相同 NF4 基礎 VLM，
再用 PeftModel.from_pretrained(model.language_model, adapter_path) 掛載。
請勿將此資料夾當成完整 VLM 直接載入或自動掛到外層 VLM。

選模依據、資料雜湊與設定見 training_manifest.json；實際訓練紀錄見
../../results/sft_training.json，完整說明見 ../../report.md。

adapter_config.json 的基礎模型與 revision 已指向實際使用的公開模型，
取代原始巢狀 Qwen config 所保留的上游開發者本機路徑；權重未改動。
