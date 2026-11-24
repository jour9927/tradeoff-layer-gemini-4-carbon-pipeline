# Tradeoff Layer — Gemini 4 Carbon 無人值守全自動長短片產線 (`v3`)

本倉庫完整逆向並復刻 YouTube 頻道 **[@TradeoffLayer](https://www.youtube.com/@TradeoffLayer)** 的長片（`1920x1080 @ 60fps`、`~8` 分鐘深度商業懸疑敘事）與對應前段截斷推流 Shorts（`1080x1920 @ 60fps`）全自動無人值守產線，融合**即時公開資訊爬蟲與無框微字角標／資訊欄雙重來源引用**、**逐句專屬分鏡 AI 產稿（85～100+ 鏡／集）**、**本人聲音 PyTorch 零樣本克隆配音（F5-TTS 短句戲劇停頓 + Whisper 詞級對齊）**、**`Z-Image-Turbo` 純平塗漫畫場景與原生品牌文字渲染 + `BiRefNet` 神經網路無毛邊去背**、**`sfx_engine` 逐動作 Foley 音效**與 **18 套多圖層 Skia 60fps 動畫合成引擎**。

---

## v3 核心架構（對標原頻道百萬觀看成片）

1. **公開資訊深度爬蟲 + 乾淨來源引用（`research_scraper.py`）**
   - 自動爬取 Hacker News Algolia API、arXiv 論文庫、Wikipedia 完整章節全文與技術媒體報導，過濾頻道已發佈主題並自動評分選題。
   - 建立結構化來源註冊表 `[S1]..[Sn]`：畫面徹底摒棄破壞美感的常駐黑底字幕條，改於量化數據鏡頭右下角浮現 `2.4s` 極簡無框微字來源註腳（淡入淡出），並於 YouTube `description.txt` 與 `citations_audit.json` 附上完整原始連結與時間軸。
2. **8～9 分鐘、90+ 鏡逐句分鏡劇本 + 原生前段截斷推流 Short（`script_generator.py`）**
   - 對齊 `@TradeoffLayer` 真實篇幅與剪輯密度（每 `3.8s～5.0s` 切換一個量身打造的視覺動作鏡頭，全片 `90+` 鏡、`1,250～1,450` 字）。
   - 1:1 還原原頻道 Shorts 推流機制（`hook_prefix_excerpt`）：直接擷取長片前 `~25` 鏡（約 `120～135s` 黃金開場懸念與第一幕衝突），沿用同一套精修語音母帶並將所有分鏡**原生重排版為滿版 `9:16` 直式無字幕乾淨畫面**，斷在最抓人的轉折問句導流回母長片。
   - 支援 `GEMINI_API_KEY`、`OPENAI_API_KEY`，或零 API Key 直接於 96GB RTX PRO 6000 執行 `Qwen2.5-14B-Instruct` 自動分鏡。
3. **本人聲音 PyTorch 克隆配音 + 乾淨留白與逐動作 Foley 音效（`voice_engine.py` & `sfx_engine.py`）**
   - 以頻道高訊噪比（SNR > 61 dB）母帶驅動 PyTorch **F5-TTS**，採短句切分與問句留白（停頓底噪 `<-66 dB`，無混濁合成背景底音）。
   - `sfx_engine.py` 依每個鏡頭動作關鍵幀自動混入真實物理質感 Foley 音效（`whoosh_soft`、`pop_clean`、`tick_counter`、`stamp_heavy`、`shutter_flash`、`paper_slide`、`chime_reveal`）。
4. **`Tongyi-MAI/Z-Image-Turbo` 純平塗漫畫生圖 + `ZhengPeng7/BiRefNet` 神經去背（`visual_engine.py`）**
   - 全面淘汰舊版 SDXL，改用 `Z-Image-Turbo`（BF16）生成統一粗黑線純平塗漫畫場景（含原生精準品牌字樣 `NVIDIA`、`SK hynix`、`SAMSUNG`、`TSMC`）與固定當集多表情／手勢角色立繪。
   - 透過 `ZhengPeng7/BiRefNet` 1024px 神經網路分割產生零白邊 RGBA 角色與道具圖層，並自動合成對標原頻道的斜角漫畫大字縮圖（`thumbnail_1280x720.png`）。
5. **18 套多圖層分件與極簡資訊圖表 60fps 動畫引擎（`skia_animator.js` & `video_composer.py`）**
   - 內建 18 種 `@TradeoffLayer` 招牌鏡頭語言：影印機放晶片蓋蓋掃描（`copier_blueprint_action`）、辦公桌推合約／支票（`desk_deal_transfer`）、液壓機下壓（`hydraulic_press_crush`）、攝影棚快門閃光變焦（`studio_shutter_flash`）、斜角漫畫雙格對決（`split_screen_rivalry`）、無塵室放大鏡檢視（`cleanroom_microscope_zoom`）、工廠輸送帶與擺盪吊牌（`oven_conveyor_bottleneck`）、奶油底 100 圓點矩陣（`waffle_100_dots`）、升降石柱晶片對決（`pedestal_rivalry`）、深海藍紅叉蓋章（`red_cross_stamp`）、木紋工作台 12 層 HBM 堆疊與 TSV 鑽孔（`workbench_hbm_stack`）、`MR-MUF` 液態環氧樹脂 vs `TC-NCF` 熱壓膜對比（`thermal_mrmuf_vs_tcncf`）、窄管滴水 vs 並聯水渠（`rain_pipe_analogy`）、`CoWoS` 矽中介層封包流（`cowos_silicon_bridge`）、`3x` 晶圓面積排擠（`wafer_3x_cannibalization`）與時間軸演進（`timeline_paradox_arc`）。
6. **主腳本本機優先 + 異地算力彈性調度（`orchestrator.py` & `scripts/remote_client.py`）**
   - 所有主程式與產物優先保存於本機 `alan-hub`（內建保護鎖禁止本機渲染），重活自動派發至專用 **G4 · RTX PRO 6000 (`Untitled10.ipynb`)**，離線時可切換至 **Tailscale SSH `b850`（`alan-eagle@b850-ubuntu`）**。

---

## 快速使用方式

### 1. 無人值守全自動產出（自動爬蟲選題 → 90 鏡分鏡產稿 → 克隆配音 → Z-Image + BiRefNet 產圖 → 60fps 長片 + 推流 Short）
```bash
PYTHONPATH=src python3 -m tradeoff_pipeline.orchestrator run --worker auto
```

### 2. 指定主題無人值守產出
```bash
PYTHONPATH=src python3 -m tradeoff_pipeline.orchestrator run \
  --topic "Why Every AI Chip Needs SK Hynix? HBM Memory Explained" \
  --queries "High Bandwidth Memory" "SK Hynix HBM" "TSMC CoWoS packaging" \
  --worker auto
```

### 3. 同步產線至 `b850`（Tailscale SSH）
```bash
python3 scripts/remote_client.py sync-b850
```
