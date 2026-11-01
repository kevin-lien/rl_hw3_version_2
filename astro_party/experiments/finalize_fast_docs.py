"""Refresh local experiment documentation without changing training sources."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
readme = root / 'README_TW.md'
text = readme.read_text(encoding='utf-8')
text = text.replace('submission_reproduce/', 'fast_cover_reproduce/').replace('runs/0njzp6y9', 'runs/ghme0u0w')
text += '''

## 高速掩體實驗：速度獎勵、同向繞行與追撞

此實驗已完成 40,960 步 PPO，獨立 checkpoint 為 `models/model_astro_party_fast_cover.zip`。原正式模型 `models/model_astro_party.zip` 保留，未被較弱的實驗覆蓋。

```powershell
python train_astro_party.py --controller fast-cover
python watch_astro_party.py --model models/model_astro_party_fast_cover.zip --opponent sniper --seed 0
```

`train_astro_party_env.py` 的 fast_cover reward 增加速度 bonus：`0.0015 * min(有效速度 / 260, 1) * 情境權重`。對方仍是船時，中央行星遮蔽下權重 1、暴露時 0.25；對方變小人時只獎勵朝對方移動的速度，approach coefficient 提高至 6；自身出艙時不發速度 bonus。每步最大 bonus 小於時間成本，避免單靠繞圈取得正回報。

控制器估計對方繞中央行星的角速度，優先同向快速移動。有一發且冷卻與前一發飛行窗口結束即可探頭；敵人出艙時優先追撞，仍保留原本的 60 步避彈搜尋。高速移動、瞄準、近距離防禦會互相競爭，並不保證每幀同向或所有缺彈時間都能躲好。

固定種子 65000–65099，每個版本對各 baseline 各 100 場：

| 版本 | Rusher 勝率 | Sniper 勝率 | 對 Sniper 平均船速 | Sniper 平手 |
| --- | --- | --- | --- | --- |
| 舊掩體 | 86% | 76% | 128.5 px/s | 13% |
| 高速、無速度 bonus | 72% | 77% | 168.1 px/s | 1% |
| 高速、有速度 bonus | 72% | 80% | 166.2 px/s | 1% |

速度提升主要來自新導航；此單一訓練 seed 的對照沒有顯示速度 bonus 額外提高船速。對 Sniper 退避期間遮蔽從 32% 提升至 43%，但真正零彈期間遮蔽僅約 3%，尚未實現穩定的缺彈藏身。撞擊獲勝比例增加，但對方出艙後的完成率沒有改善。這版未達兩個 baseline 門檻，因此不取代正式模型。完整比較見 `FAST_COVER_EXPERIMENT_TW.md`、`results/fast_cover_comparison.json`。

已驗證的安全與重現性：實際訓練 checkpoint 在 180 個避彈介入狀態都提高模型預測的最小子彈距離（中位改善 11.44 px；這不是存活保證）；固定零彈、無自身子彈的近距離小人情境在 46 步／0.92 秒完成追撞。原 reward profile 與舊版 1,000 步回歸一致。最新訓練檔在乾淨平面繳交目錄執行裸命令，40,960 步後所有 policy tensors 與正式模型完全相同（最大差 0）。驗證環境為目前鎖定的 Python 3.11 套件版本。

W&B 已完成並確認雲端資料，新增可視化欄位為 `motion/ship_speed_mean`、`motion/velocity_reward_mean`、`motion/covered_mean`、`motion/hunt_mode_mean`，橫軸選 `global_step`。

- [高速有速度獎勵訓練](https://forge.coreweave.com/wandb/kevinboyulien-national-taiwan-university/astro-party-hw3/runs/xdfwkau6)
- [相同設定、零速度獎勵對照](https://forge.coreweave.com/wandb/kevinboyulien-national-taiwan-university/astro-party-hw3/runs/4r0t3h22)
- [同種子實際 checkpoint 比較](https://forge.coreweave.com/wandb/kevinboyulien-national-taiwan-university/astro-party-hw3/runs/x4hdfhjo)
- [最新裸命令正式模型重現](https://forge.coreweave.com/wandb/kevinboyulien-national-taiwan-university/astro-party-hw3/runs/ghme0u0w)

關閉速度獎勵的對照命令：

```powershell
python train_astro_party.py --controller fast-cover --velocity-coef 0 --save-path models/model_astro_party_fast_cover_novelo
```

裸命令 `python train_astro_party.py` 仍重現正式通過版本；`--controller fast-cover` 才選這次實驗。W&B 預設記錄訓練數值與設定，不自動上傳模型或報告。
'''
readme.write_text(text, encoding='utf-8')

plan = root.parent / 'My_Plan.md'
with plan.open('a', encoding='utf-8') as stream:
    stream.write('''

### 高速掩體實驗完成與決策

1. 有／無速度 bonus 兩組皆完成 10,000 示範、120 輪 BC、40,960 步 PPO；相同 seed、初始 BC 權重逐 tensor 相等。W&B run 為 xdfwkau6 / 4r0t3h22，完整雲端曲線已讀回驗證。
2. 固定種子 65000–65099，每個模型對每個 baseline 各 100 場。舊掩體 Rusher/Sniper 86%/76%；高速無 bonus 72%/77%；高速有 bonus 72%/80%。Sniper 船速 128.5→166.2，平手 13→1 場，但零彈遮蔽僅約 3%，尚未達成穩定躲藏。速度 bonus 不比零 bonus 更快；不能將單 seed 的三場勝率差解讀為普遍有效。
3. 保留原正式 checkpoint（公開評估 95%/91%）；新版本獨立儲存。撞擊勝場增加不等於出艙後完成率提升，避免用局部行為指標取代最終勝率。
4. 實際訓練模型的 180 個避彈介入狀態皆提高預測最小子彈距離，中位改善 11.44 px；受控無彈追撞 46 步成功。這些是機制驗證，不能當完整勝率保證。
5. 最新兩份訓練原始碼在 fast_cover_reproduce/ 中完全從零用 python train_astro_party.py 重訓，run ghme0u0w；40,960 步後全部 policy tensors 與正式 reference 相同，最大差 0。原 reward 1,000 步回歸一致。報告更新為 11 頁，含對照圖與負面結果；繳交包仍選正式通過模型。
''')

print('README and My Plan updated')
