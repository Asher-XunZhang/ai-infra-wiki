# SGLang PP 共识文档图片来源

读取日期：2026-09-14（Asia/Shanghai）。编号 01～05 是原作者公开帖中的技术原图，保留原始字节，没有 AI 重绘、裁剪或数据修改。编号 06～08 为整理者依据固定源码制作的原创教学动画及静态步骤图。各图的技术解释和证据边界见 [学习文档](../../sglang/SGLang%20PP%20共识机制源码学习文档.md)。

博客：Shangming Cai，2026-01-15，[Pipeline Parallelism in SGLang: Scaling to Million-Token Contexts and Beyond](https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/)。

PR：stepinto，2026-09-11（北京时间），[[WIP][PP+PD] Fix #38206 Reduce bootstrap consensus latency](https://github.com/sgl-project/sglang/pull/38959)；读取时未合并，所读代码 head `8242c4e1b014ad823330100bc8f0546bd712378b`。截图的 POC 未单独标注完整 commit，不把它等同于该 head 的复现实验。

## 官方文章图 1：固定 chunk 气泡

- 本地文件：[01-fixed-chunk-bubbles.jpg](01-fixed-chunk-bubbles.jpg)
- 原始地址：[原图](https://www.lmsys.org/images/blog/chunked_pipeline/pp_bubbles_before.jpg)
- 页面来源：[LMSYS 作者文章](https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/)
- 尺寸：1660 × 912 px；大小：126364 bytes
- SHA-256：`126b07721698a10b86f439f2f79929d727604902528c589005362c223542bf64`

## 官方文章图 3：动态 chunk 气泡

- 本地文件：[02-dynamic-chunk-bubbles.jpg](02-dynamic-chunk-bubbles.jpg)
- 原始地址：[原图](https://www.lmsys.org/images/blog/chunked_pipeline/pp_bubbles_after.jpg)
- 页面来源：[LMSYS 作者文章](https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/)
- 尺寸：1703 × 850 px；大小：137670 bytes
- SHA-256：`3c552c79746e0bf70038590fe9a62404b176074073839cc432d9d9b2fc9d2c7e`

## 官方文章图 2：固定 chunk profile

- 本地文件：[03-fixed-chunk-profile.png](03-fixed-chunk-profile.png)
- 原始地址：[原图](https://www.lmsys.org/images/blog/chunked_pipeline/profile_before.png)
- 页面来源：[LMSYS 作者文章](https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/)
- 尺寸：1999 × 683 px；大小：774069 bytes
- SHA-256：`d11cd1bb4bcda926157b529a6d6fda553b2f1cb08b37d836875e7933d1e76b48`

## 官方文章图 6：动态 chunk profile

- 本地文件：[04-dynamic-chunk-profile.png](04-dynamic-chunk-profile.png)
- 原始地址：[原图](https://www.lmsys.org/images/blog/chunked_pipeline/profile_after.png)
- 页面来源：[LMSYS 作者文章](https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/)
- 尺寸：1999 × 515 px；大小：631675 bytes
- SHA-256：`0b47dc644d43e51286e4cfc447fae253a6a9b8f449575cae2395272c49a0c6cb`

## PR #38959 作者实验截图

- 本地文件：[05-bootstrap-consensus-store-benchmark.png](05-bootstrap-consensus-store-benchmark.png)
- 原始地址：[原图](https://github.com/user-attachments/assets/89badd5a-d7ae-4663-a6b6-dbbd4d7a3873)
- 页面来源：[PR #38959](https://github.com/sgl-project/sglang/pull/38959)
- 尺寸：498 × 994 px；大小：206372 bytes
- SHA-256：`aa9ea8c19ce316e85c7e59c19454e91c2fceea0008d30541aabefd3a5485a5b4`


## 原创动画与静态步骤图（06～08）

制作日期：2026-09-14。依据本地开源基线 `72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a`，用程序绘制文字、状态卡片和运动标记；没有修改或重绘上方五张作者原图，也没有使用模型生成技术事实。播放速度只为便于阅读，不是 runtime profile。

| 文件 | 本文中的作用 | 固定源码依据 |
| --- | --- | --- |
| [06-rid-consensus.gif](06-rid-consensus.gif) / [静态步骤](06-rid-consensus-steps.png) | 三站正向求 good 交集、bad 并集；不演回传与出队 | `scheduler_pp_mixin.py::_pp_pd_get_bootstrapped_ids` / `_route_aborts_to_bad`；正文 [S10]/[S11] |
| [07-admission-gates.gif](07-admission-gates.gif) / [静态步骤](07-admission-gates-steps.png) | 单个 Prefill stage 从资格、资源检查到后续调度 | `prefill.py::PrefillBootstrapQueue.finalize_bootstrap` / `pop_bootstrapped`；正文 [S10]/[S16] |
| [08-abort-deferred-release.gif](08-abort-deferred-release.gif) / [静态步骤](08-abort-deferred-release-steps.png) | 满足延迟释放条件，并在后端判安全后归还资源的一条路径 | `decode.py::DecodeTransferQueue.resolve_deferred_releases` / `_do_release`；正文 [S19]/[S23] |

源码路径相对于 `python/sglang/srt/`，Scheduler 文件位于 `managers/`，Prefill/Decode 文件位于 `disaggregation/`。精确固定版本链接、条件分支和比喻边界均见[主文档](../../sglang/SGLang%20PP%20共识机制源码学习文档.md)。动画不构成硬件、传输退休或端到端一致性的实验验证。

**重新生成：** [render_animations.py](render_animations.py) 依赖 Pillow ≥ 10；在装有中文字体的环境执行 `python render_animations.py`。默认探测 macOS STHeiti 或 Linux Noto Sans CJK，也可用 `--font /path/to/CJK-font.ttc` 指定。输出写回脚本所在目录；字体或 Pillow 版本变化可能改变像素和文件哈希。

[animation-manifest.json](animation-manifest.json) 记录 GIF 帧数、每轮总时长、关键状态起始时间、尺寸，以及 GIF/静态图的 SHA-256。GIF 为无限循环；不支持播放或希望减少动态内容的阅读器可直接打开静态版本。细字可打开原文件放大查看。
