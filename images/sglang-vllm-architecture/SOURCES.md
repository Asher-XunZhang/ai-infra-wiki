# SGLang 与 vLLM 架构图源

- 对应正文：[架构差异源码学习文档](<../../大模型推理/推理基础/SGLang 与 vLLM V1 架构差异源码学习文档.md>)。
- 读取与保存日期：2026-11-04。
- 前两张为作者讲义指定整页的栅格化版本，未裁剪、改写或重绘；第三张保留固定 Git 提交的原始 PNG 字节。正文 Mermaid 为整理者重绘，依据独立标注的固定源码，不替代原图证据。

| 本地文件 | 原始来源 | 作者与版本 | 处理方式 |
| --- | --- | --- | --- |
| [01-sglang-author-overview.png](01-sglang-author-overview.png) | [SGLang CMU 讲义，第 6 页](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-25-sglang-72edc5043338f59db34d47e5b96ac870.pdf) | Ying Sheng；CMU 2025 春季课程，具体讲授日未在所读页确认 | PDF 第 6 页；Poppler 渲染最长边 1800 像素 |
| [02-vllm-v1-author-loop.png](02-vllm-v1-author-loop.png) | [vLLM CMU 讲义，第 83 页](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-22-vLLM_woosuk_kwon-1f34697dbb1a1fb5b798daf6eff14b67.pdf) | Woosuk Kwon；CMU 2025 春季课程，具体讲授日未在所读页确认 | PDF 第 83 页；Poppler 渲染最长边 1800 像素 |
| [03-vllm-v1-tp4.png](03-vllm-v1-tp4.png) | [vLLM V1 TP=4 架构图](https://raw.githubusercontent.com/vllm-project/vllm/9a56d4fec6b0a30a0a989f1cd06cd3c2f7325619/docs/assets/design/arch_overview/v1_process_architecture_tp4.png) | vLLM 官方；commit `9a56d4fec6b0a30a0a989f1cd06cd3c2f7325619` | 下载原始 PNG，不修改 |

## SHA-256

| 文件 | 保存图像 SHA-256 | 原始下载文件 SHA-256 |
| --- | --- | --- |
| `01-sglang-author-overview.png` | `ad0ab95a80e013ed7fd16072943967a3a939c18994920238a9ba99957a85412a` | `050931d462eb36395da420de477b53acbde9036eda6f88d3ae191f3f97432b97` |
| `02-vllm-v1-author-loop.png` | `a968f59305d932be8370ce6f5707bdfb65d80c101abc51df65308468d7f8107d` | `e6f4580ec31f63178d571cf80bc110bfad4f44b10c16a9265956a03a0e3a1869` |
| `03-vllm-v1-tp4.png` | `8602fecbabf3b5fb4ceeb831f608dc178d14ec4a686e2f2cff58b1af78f8951d` | `8602fecbabf3b5fb4ceeb831f608dc178d14ec4a686e2f2cff58b1af78f8951d` |

## 使用边界

- 历史讲义用于解释设计思路，不作为 2026 年默认进程数、功能矩阵或性能排名的证据。
- vLLM TP=4 原图与正文 TP=2 教学拓扑分别标明卡数，不进行同条件性能比较。
- 图片内的模块框不自动等于进程；控制权、消息载荷及省略的通路以正文图意解读为准。

## 离线对照台的表达方式参考

- 资料：用户提供的 `inference-atlas-offline.zip`，内部 `inference-atlas-offline/index.html`；页面题名《推理引擎对照台 · 本地离线版》。未提供公开发布地址，不在文档中保存作者电脑路径。
- 页面自署：DeepSeek Harness 插件 `@local/dsh-inference-atlas`；构建与本次读取日期均为 2026-11-04。
- 包内声明：SGLang `v0.5.23`；vLLM 展示版本 `v0.33.0`，tag 为 `v0.33.0rc3`。这些声明不替换正文的两个固定 commit。
- 用途：完整审阅 6 章、24 个步骤、模块注释、通信表与 4 张场景表；只选取三张代表画面保留，说明问题对齐、局部高亮和状态比较的方法。正文技术细节经独立源码复核，不照搬包内简化。
- 处理：离线 Chrome 中逐项选择相应章节/步骤，启用 reduced motion 固定画面，对原页面 `.ia-stage` 元素截图；没有改写图中文字。viewport 为 1440 × 1100，截图保留原元素尺寸。
- 许可：[随包许可文本](ATLAS-LICENSE.txt)。Atlas 应用与内嵌 React / React DOM 的 MIT 通知保留于该文件。
- ZIP SHA-256：`a0aa033efedfd78f31dcc8a282af83f2892ea95f427e57e4700afb891e3e7541`。
- HTML SHA-256：`d47bdb9e5d1362ccf34c42a857865f2017056192ce68acc75ab10d7e2a00ddb1`（与随包 README 声明一致）。

| 保存文件 | 原页面位置 | 图意与使用边界 | SHA-256 |
| --- | --- | --- | --- |
| [04-atlas-process.png](04-atlas-process.png) | 进程骨架，步骤 4/4 | 固定两列、同一回程问题、局部高亮与分叉点 | `4e142afd854f23d595b007b777709ab9f70bdb6d542aca8684bfc00afaab144a` |
| [05-atlas-prefix.png](05-atlas-prefix.png) | 前缀缓存，步骤 2/4 | 同 token 刻度的形象对照；10/8 含粒度假设，不作为命中率证据 | `c2f191a217bf9d7d1b16828e273d54cd725d7adb49368469458d34484c2b672a` |
| [06-atlas-pressure.png](06-atlas-pressure.png) | 同题异路，步骤 2/4 | 相同压力问题与状态路径；92% 为示意，不是源码阈值 | `7ca272cc0b83018f770e59fa4b9ed05c9c7261693622b53a0451b97c417d6f6b` |

原页面链接可作为其资料来源线索，例如 [SGLang release](https://github.com/sgl-project/sglang/releases/tag/v0.5.23) 与 [vLLM release](https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3)；它还引用了 tag 源码及 `latest` 文档。本文没有据此认定这几类来源具有同一个实现边界。
