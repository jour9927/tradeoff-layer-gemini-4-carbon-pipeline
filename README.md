# Tradeoff Layer — Gemini 4 Carbon 無人值守全自動長短片產線

本倉庫完整逆向並復刻 YouTube 頻道 **[@TradeoffLayer](https://www.youtube.com/@TradeoffLayer)** 的長片（`1920x1080 @ 60fps`）與導流 Shorts（`1080x1920 @ 60fps`）全自動無人值守產線，融合**即時公開資訊爬蟲與畫面／資訊欄雙重來源引用**、**頻道敘事結構 AI 產稿**、**本人聲音 PyTorch 零樣本克隆配音（F5-TTS + Whisper 詞級時間軸對齊）**、**PyTorch 擴散模型（SDXL-Turbo）賽璐璐扁平向量產圖**與**三模態 60fps 動態合成引擎**。

---

## 核心特性

1. **公開資訊爬蟲 + 雙重來源引用（`research_scraper.py`）**
   - 自動爬取 Hacker News Algolia API、arXiv 論文庫、Wikipedia REST API 與技術媒體原文，過濾頻道已發佈主題並依 `@TradeoffLayer` 選題指標（反直覺悖論、量化數據、工程／商業權衡）自動評分選題。
   - 建立結構化來源註冊表 `[S1]..[Sn]`，同時在**影片畫面左下／頂部動態渲染 `SOURCE [S1]` 引用標籤**，並於生成的 YouTube 資訊欄附上完整標題、日期與原始連結，另輸出 `citations_audit.json` 供稽核。
2. **100% 頻道敘事基因復刻 AI 產稿 + 導流 Shorts 連動（`script_generator.py`）**
   - 嚴格遵循 `@TradeoffLayer` 劇本結構：零廢話反直覺開場 Hook → 生活化物理類比（如 *buckets catching rain*、*cake factory*）→ 數據化底層機制拆解 → 終章 **The Tradeoff**（為何最大護城河同時也是最大風險）。
   - 每支長片自動同步產出對應的 **60–75 秒直式推流 Short**，結尾自然導流回母長片。
   - 支援 `GEMINI_API_KEY`、`OPENAI_API_KEY` 或無需任何外部 API Key 的遠端自託管 `Qwen2.5-Instruct` 推理。
3. **本人聲音 PyTorch 克隆配音 + 詞級時間軸對齊（`voice_engine.py`）**
   - 從授權下載之頻道高純度人聲（SNR > 61 dB）提取四種語氣聲紋母帶（`hook`、`narrative`、`analytical`、`short_punchy`）。
   - 使用 PyTorch **F5-TTS** 流匹配擴散模型進行本人聲音克隆，內建美金金額（`$6.3B`）、倍率（`5x`、`100x`）與半導體縮寫（`GPU`、`TPU`、`ASML`、`HBM3e`）口語正規化，並經 EBU R128 `-14.0 LUFS` 廣播級母帶處理與 `faster-whisper` 詞級時間軸強制對齊。
4. **全彩扁平賽璐璐角色／道具去背 + Google Skia 60fps 真 2D 骨骼與物理動畫引擎（`visual_engine.py`, `skia_animator.js` & `video_composer.py`）**
   - 於 RTX PRO 6000 執行 PyTorch **SDXL-Base 1.0（FP16 原生高解析 + HSV 彩度品管）** 產出明亮粉彩扁平賽璐璐場景、全彩角色上半身去背精靈（RGBA Cutout）與硬體道具去背素材。
   - 內建 `skia_animator.js`（`@napi-rs/canvas` 原生 RGBA 串流直入 FFmpeg），以 **8 大參數化 60fps 2D 動畫編排** 1:1 還原 `@TradeoffLayer` 動畫語言（零靜態投影片字卡）：
     1. **`character_walk_handoff`**：2-Bone IK 火柴人雙腿行走步態＋全彩角色上半身走入鏡位、雙手遞交發光晶片／GPU 道具＋2D 虛擬攝影機推軌（對標 `yB9SC4HbH5o`）。
     2. **`rain_buckets_analogy`**：生活物理類比模擬——窄管阻塞滴水 vs 平行寬管注水、玻璃量杯內正弦波液位動態上升（對標 `2d883UK7p9E`）。
     3. **`chip_3d_flip_stack`**：木紋工作台精密鑷子夾持矽晶圓 3D 翻轉（`scaleY(cos)`）、12 層 DRAM 彈簧物理堆疊於 2.5D Interposer 並鑽入貫穿式金色 TSV 電梯發射資料光點（對標 `2d883UK7p9E`）。
     4. **`whiteboard_systolic_sim`**：角色持教鞭指向圓角白板，模擬窄匯流排飢餓暫停（`||`）切換至 HBM 對角線 Systolic Array 波前脈衝（對標 `yB9SC4HbH5o`）。
     5. **`dark_cross_section_meter`**：霧黑 `#0B0B0E` 貝茲曲線光球計價表連續跳表＋地底／跨海／封裝剖面線條漸進繪製與封包奔馳（對標 `0vZCvW7IzmA`）。
     6. **`dark_gpu_partition_split`**：單顆 GPU 權重矩陣溢位閃紅後，以彈簧物理分裂為 4 顆 GPU 並發射黃色 `NCCL Sync` 同步雷射（對標 `0vZCvW7IzmA`）。
     7. **`map_supply_chain_arcs`**：暖米色向量地圖＋飄動旗幟＋晶片模組沿虛線拋物線跨國飛行（對標 `KTY7TnGIVNY`）。
     8. **`wafer_conveyor_zoom_out`**：3 片晶圓壓縮為 1 座 HBM 塔＋攝影機 `2.0x -> 0.75x` 大範圍拉遠展現產能排擠（對標 `cER6uZO6gvA`）。
   - **Shorts 逐詞高亮字幕**：精確還原頻道 Shorts 下方逐詞漸進變白（已唸 `#FFFFFF` Bold ／ 未唸 `#71717A` Bold）動態字幕（對標 `bmFUEo6A_1U`）。
5. **主腳本本機優先 + 異地算力彈性調度（`orchestrator.py` & `scripts/remote_client.py`）**
   - 所有主程式與階段產物一律優先寫入並同步回本機（`alan-hub`），內建硬體保護鎖嚴禁於 `alan-hub` 執行影片渲染。
   - 優先調度臨時高速 **RTX PRO 6000** 節點執行配音、產圖與 60fps 並行渲染；當臨時節點離線時，自動無縫切換至 **Tailscale SSH 可達之 `b850`（`alan-eagle@b850-ubuntu`）** 永久節點執行。

---

## 目錄結構

```text
.
├── assets/voice_profile/          # 本人高純度多語氣聲紋參考母帶 (>56–64 dB SNR)
├── config/                        # 遠端工作節點與 SSH 連線設定
├── scripts/
│   ├── colab_rtx6000_worker_daemon.py  # RTX PRO 6000 隔離 HTTP 工作守護進程 (:19531)
│   ├── remote_client.py                # RTX PRO 6000 與 b850 Tailscale SSH 統一調度客戶端
│   ├── reverse_engineer_channel.py     # 頻道全量長短片規格、語速與視覺關鍵幀逆向分析工具
│   └── extract_voice_profile.py        # 純淨語音切片與 SNR 檢測擷取工具
└── src/tradeoff_pipeline/
    ├── research_scraper.py        # 公開資訊自動選題爬蟲與 [S1..Sn] 來源引用建構器
    ├── script_generator.py        # @TradeoffLayer 劇本與推流 Shorts 生成引擎
    ├── voice_engine.py            # PyTorch F5-TTS 本人聲音克隆 + Whisper 詞級對齊
    ├── visual_engine.py           # PyTorch SDXL-Turbo 賽璐璐向量場景產圖引擎
    ├── video_composer.py          # 60fps 1080p 長片 + 1080x1920 Shorts 多進程合成器
    └── orchestrator.py            # 無人值守端到端總控 CLI
```

---

## 快速使用方式

### 1. 無人值守全自動產出（自動爬蟲選題 → 產稿 → 克隆配音 → 產圖 → 60fps 長片 + Short）
```bash
PYTHONPATH=src python3 -m tradeoff_pipeline.orchestrator run --worker auto
```

### 2. 指定主題無人值守產出
```bash
PYTHONPATH=src python3 -m tradeoff_pipeline.orchestrator run \
  --topic "Why Every AI Chip Needs SK Hynix HBM Memory?" \
  --queries "HBM memory" "SK Hynix HBM3E" "TSMC CoWoS HBM" \
  --worker auto
```

### 3. 預覽爬蟲即時候選選題（對應傳統選題模式）
```bash
PYTHONPATH=src python3 -m tradeoff_pipeline.orchestrator discover --count 5
```

### 4. 同步產線至 `b850`（Tailscale SSH）
```bash
python3 scripts/remote_client.py sync-b850
```
