# Results: doubao-seed-2.0-pro

This page shows the full evaluation results for **doubao-seed-2.0-pro** on the 200-sample DuReader evaluation set.

The full markdown with all samples is also available in the repository at: [`results/dureader/doubao-seed-2.0-pro.md`](https://github.com/leideng/DuReader-OpenAI/blob/main/results/dureader/doubao-seed-2.0-pro.md)

## First Sample

| Index | Question | Gold Answers | Model Response | F1 | Precision | Recall | ROUGE-L |
|-------|----------|--------------|----------------|----|-----------|--------|---------|
| 1 | 热诚传说结局 | - 《热诚传说》动画的结局是百合。 | 关于动画版《热诚传说X》的结局：史雷（男主）沉睡醒来已是多年后，公主等旧人已逝，官方最终盖章了百合感情线，不少观众对此结局感到失望。<br>游戏版若触发隐藏结局，条件是通关4神殿、德泽尔在队时打死显主，且途中在大劣势下击败遇到的狮子。 | 0.2143 | 0.1200 | 1.0000 | 0.1786 |

## Summary

| Metric | Average |
|--------|---------|
| F1 | **0.3646** |
| Precision | 0.3446 |
| Recall | 0.5407 |
| ROUGE-L | 0.2641 |

- Number of samples: 200
- Evaluation date: 2026-04-13

## Notes

- This is the raw evaluation with no filtering
- All 200 samples from `data/dureader.jsonl` are included
- Chinese character tokenization fix applied (F1 scores are accurate)
