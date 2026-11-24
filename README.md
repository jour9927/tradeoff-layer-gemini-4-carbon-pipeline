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
4. **AI 賽璐璐向量產圖 + 三模態 60fps 動態合成引擎（`visual_engine.py` & `video_composer.py`）**
   - 完整復刻 `@TradeoffLayer` 三大標誌視覺模式：
     - **`dark_metric_card`**：霧黑 `#0B0B0E` 底板、`#1A1A1F` 數據卡、動態貝茲曲線連接線與紅綠對比計價表（對標 `0vZCvW7IzmA`）。
     - **`vector_whiteboard_scene`**：PyTorch SDXL-Turbo 扁平賽璐璐建築／機房背景 + 四色頂條圓角白板圖表 + `@TradeoffLayer` 標誌性 **2D 賽璐璐上半身角色配黑色火柴人手腳與語音同步嘴型**（對標 200 萬觀看 `yB9SC4HbH5o`）。
     - **`editorial_illustration`**：全幅賽璐璐敘事插畫搭配 60fps Ken Burns 運鏡、懸浮指標卡與章節進度條（對標 `2d883UK7p9E`）。
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
