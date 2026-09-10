# SGLang 架构图来源与维护档案

这些图片是 **2026-09-10 根据固定源码绘制的整理图**，用于宏观学习，不是第三方原图、运行截图或性能证据。内容来源与范围见[架构导读](../../sglang/source-study/architecture/README.md)及各篇源码锚点。

| 项目 | 内容 |
| --- | --- |
| 源码目录 | SGLang 仓库根目录 `.`；路径均为仓内相对路径 |
| 分支 | `codex/sglang-source-study-20260909` |
| commit | `72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a` |
| 读取日期 | 2026-09-10 |
| 工作区与操作 | 学习源码 worktree 干净；只做静态复核与文档渲染，不执行模型 / GPU / 网络实验 |
| 图数 | 12 份 SVG 首图；12 篇正文内共 38 个 Mermaid 图块 |
| 工具 | Mermaid 10.9.3 和 Chromium；M08 的 SVG 使用原生 SVG 重新安排空间布局 |
| 原始依据 | 各章文内 Mermaid 与源码锚点；M08 另参考该章本地页号 / 槽位映射表 |
| 背景 | 固定白底，避免深色 Markdown 主题下线条和文字失去对比 |

## 图片清单

| 图片 | 对应章节 | 生成关系 | viewBox |
| --- | --- | --- | --- |
| [01-system-overview.svg](01-system-overview.svg) | [M01](<../../sglang/source-study/architecture/01-SGLang全景与四种架构视图.md>) | 从本篇首个 Mermaid 图导出；补白底和固定 viewBox 尺寸 | `-8 -8 662.55615234375 614` |
| [02-startup-assembly.svg](02-startup-assembly.svg) | [M02](<../../sglang/source-study/architecture/02-启动装配与运行时边界.md>) | 从本篇首个 Mermaid 图导出；补白底和固定 viewBox 尺寸 | `-8 -8 346.03125 655` |
| [03-request-lifecycle.svg](03-request-lifecycle.svg) | [M03](<../../sglang/source-study/architecture/03-请求对象与生命周期全景.md>) | 从本篇首个 Mermaid 图导出；补白底和固定 viewBox 尺寸 | `-8 -8 592.676513671875 657` |
| [04-scheduler-overview.svg](04-scheduler-overview.svg) | [M04](<../../sglang/source-study/architecture/04-调度器架构与时间模型.md>) | 从本篇首个 Mermaid 图导出；补白底和固定 viewBox 尺寸 | `-8 -8 596.125 533` |
| [05-cache-ownership.svg](05-cache-ownership.svg) | [M05](<../../sglang/source-study/architecture/05-缓存架构与资源所有权.md>) | 从本篇首个 Mermaid 图导出；补白底和固定 viewBox 尺寸 | `-8 -8 422 660` |
| [06-execution-layers.svg](06-execution-layers.svg) | [M06](<../../sglang/source-study/architecture/06-模型执行与算子分层.md>) | 从本篇首个 Mermaid 图导出；补白底和固定 viewBox 尺寸 | `-8 -8 711.109375 567` |
| [07-parallel-topologies.svg](07-parallel-topologies.svg) | [M07](<../../sglang/source-study/architecture/07-并行部署与通信拓扑.md>) | 从本篇首个 Mermaid 图导出；补白底和固定 viewBox 尺寸 | `-7.5 -8 656.046875 365` |
| [08-pd-three-paths.svg](08-pd-three-paths.svg) | [M08](<../../sglang/source-study/architecture/08-PD与Encoder分离的三条通路.md>) | 整理者调整布局的 SVG，显式标出源 / 目标缓冲区；与文内 Mermaid 共同表达三条通路 | `0 0 1080 820` |
| [09-generation-features.svg](09-generation-features.svg) | [M09](<../../sglang/source-study/architecture/09-高级生成特性插入位置.md>) | 从本篇首个 Mermaid 图导出；补白底和固定 viewBox 尺寸 | `-8 -8 912.875 446` |
| [10-model-assembly.svg](10-model-assembly.svg) | [M10](<../../sglang/source-study/architecture/10-模型装配状态形态与输入输出.md>) | 从本篇首个 Mermaid 图导出；补白底和固定 viewBox 尺寸 | `-8 -8 544.63671875 477` |
| [11-serving-observability.svg](11-serving-observability.svg) | [M11](<../../sglang/source-study/architecture/11-网关服务治理与性能诊断全景.md>) | 从本篇首个 Mermaid 图导出；补白底和固定 viewBox 尺寸 | `-8 -8 718.59375 446` |
| [12-dsl-runtime.svg](12-dsl-runtime.svg) | [M12](<../../sglang/source-study/architecture/12-DSL与Diffusion及插件边界.md>) | 从本篇首个 Mermaid 图导出；补白底和固定 viewBox 尺寸 | `-8 -8 559.09375 409` |

## 完整性记录

下面分别记录 SVG 字节和文内首个 Mermaid 源图的 SHA-256。源图以 UTF-8 编码，取代码围栏内部内容，不含首尾围栏与额外换行。M08 两种图按职责与数据关系核对，不声称 SVG 是 Mermaid 的机械导出。

| 编号 | SVG SHA-256 | Mermaid 源图 SHA-256 |
| --- | --- | --- |
| M01 | `653de12348231d3dbd4a91fb53283465bfa5675f6f503e432de9ac0458f57383` | `53304757ef48ad1071310c025e8a10d63005334b0c8fe510ff3cbc3205755c6e` |
| M02 | `699682b1cc142377b47d92352723ad3ec91806a57f0aa39f016ca77f4c415cb7` | `f1a1bba1be8afcac6fcacbcbf1a07415e29b8acb42ffc60031d39a5f3af363ed` |
| M03 | `62b6d674f51076fb6c209615b5434886b83614fd9a1fc4350c5b16e34fc55d34` | `cdbb1a9f233290255052d8e956a13acbec32a731b25bbf08d4566f5051dcd5eb` |
| M04 | `e898a4c0dd8e95dedf544c54fd59f20fc1ce2665bc442ab13bb392de5870e139` | `b06132b186b79d284afadbbd2b4222dbe9f4af65bbe9764cde7c90874ef0a493` |
| M05 | `da2642ddcea5286e25c4764084c00d5c20460ffc8673453fe7ee9efb810ab25c` | `988c29f8ff28620b52f902ca196d57148e69ab1b7f1226b86cd223f8bdf9d99f` |
| M06 | `adc1306d6e729a61847f0c2703f5c982fe952318fc5e907fa0401530d7cda623` | `bc56013ca558be2e2725864e6442954ebb212c4354ade8f8dbec1c1a254d4210` |
| M07 | `3d51f75b2c8bac4202a919982ab1bf8278906e68c7ddd0370aa295733fa2e08f` | `30d3aa341d6ecf408a311e1fa2cf3e78c717227b15896041ffae4f0c82a861b2` |
| M08 | `d10d3310a0d8bbaddcbebfd878330d63dcc494a85c7746191db97c982f6f657e` | `aee7732e5e3094694b84ab60e2fbb0d01aa70390191fd8a5c0306da979d64ab4` |
| M09 | `00a2fcd8f5b482ccf8a68ac35ada3ccc985ffbe6622a899abbf259bf2989d806` | `69e42666ca5a3bfe91b971663dfea6a137511a659e2130f0b407fedcd434e60a` |
| M10 | `24417dbce72f9c5563aa6823126f36147ee7b74fe186c11ca4c40aab28e9f170` | `23e7269302c43fec35f4718d015059a3e6d5022b6f542a773e465881d3327bf7` |
| M11 | `240f0ef30d2485e58be17ded6cee17e395d7e9d709970d6099256842482bcb8b` | `e55da03b1a7cec65ea89dabce447688e8484658d4f18178e1c83e4756a813fce` |
| M12 | `60e20f0a26e9c1a42e32e0d98d677308a374540cd10c06d68420e4ba8b5d9318` | `2e0e3f82f972093d730191a82e7f4cfa18dec3de99933deb4d568d8aaa136dc2` |

## 更新与验证要求

先修改文档内的 Mermaid 与图意解读，再导出对应 SVG；M08 还需同步维护 SVG 的实例外框、源 / 目标位置和三类通路。不要只改一个图片而让正文、源图与实际源码出现分歧。

检查包括：38 个 Mermaid 图实际渲染、12 份 SVG 的 XML 与本地依赖、首图可读性、正文相对图片路径、源码文件 / 符号及固定 commit、旧正文与旧图保留。移动端复杂宽图通过原尺寸链接查看细节，不能把缩小后的预览当成所有文字都清晰可读。图片渲染不代表 SGLang 运行行为、精度或性能得到验证。

## 本次验证记录（2026-09-10）

| 检查 | 结果 |
| --- | --- |
| 新图语法与渲染 | 38 / 38 个 Mermaid 图成功生成 SVG |
| 图文预览 | 12 篇导读均完成本地 HTML 预览，图片加载无失败；逐张查看首图，M01 / M05 / M08 另做桌面与 375 px 窄屏预览 |
| SVG 文本边界 | 12 份最终 SVG 的文字边界检查无超出画布；复杂宽图提供原尺寸查看入口 |
| 链接与锚点 | 系列及受影响导航的 2058 处本地链接、46 处本地页内锚点检查通过；新增 63 处固定源码引用可解析到本地文件 / 目录及有效行 |
| 内容保留 | 75 篇正文移除新增导航后与原稿逐字一致；193 个既有 Mermaid 图和原图文件保留 |
| 变更范围 | 247 份未涉及文件 hash 保持一致；学习源码 worktree 的分支、commit 和干净状态保持一致 |

预览使用本地 Chromium 与 Markdown 渲染，并以导出 SVG 呈现机制图；未声称逐一测试所有 Markdown 客户端，也未执行任何 SGLang 运行实验。
