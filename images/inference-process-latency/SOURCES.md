# 推理引擎进程边界与延迟拆解：图源记录

- 读取日期：2026-11-13。
- 公众号文章：[LLM推理引擎解剖:vLLM与SGLang的五层进程切分](https://mp.weixin.qq.com/s/pbzBiFfBR-KpVBpxYGSk4Q)，AI圈的9527，2026-11-13 07:45（Asia/Shanghai）。
- 英文来源：[Anatomy of an LLM Serving Engine](https://hiraditya.github.io/posts/anatomy-of-an-llm-serving-engine/)，Aditya Kumar，2026-10-28 发布，2026-11-01 更新；页面标注 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。
- 对应正文：[SGLang 与 vLLM V1 架构差异源码学习文档](<../../大模型推理/推理基础/SGLang 与 vLLM V1 架构差异源码学习文档.md>)，第 0.5、2.1、6.3、8.5–8.7、9.5 节；文章已整合到该对比文档，不另设重复正文。
- 完整正文仅有 1 张图片，无 SVG 或背景技术图；图位于双卡进程切分章节。本文保留公众号渲染图，未改字、裁剪或重绘。

| 文件 | 内容与使用边界 | 尺寸 | SHA-256 |
| --- | --- | --- | --- |
| [01-process-boundaries.png](01-process-boundaries.png) | SGLang 0.5.22 与 vLLM 0.32 的双卡进程图；是架构示意，未覆盖全部返回通路、collective 和辅助进程，不表示性能或实测 Trace | 1080 × 353 | `36de6e3dc32a80fd9a7f21c140b5e534c65e44dd5b7d84c482d4e96078830a89` |

[原始图片下载地址](https://mmbiz.qpic.cn/mmbiz_png/MJRkdzPjdxQtrO1UqvW29kFY54RhRlicwuyVBmt8n9pcpZBwgqaRfFjqPbTTias6ItnNqAHwkrrrU3xcsYyZynXYDtHV6lK3DFHIcVjYGFnDo/640?from=appmsg)。下载地址仅用于追溯；学习文档使用本目录的相对路径。

图片为白底宽图，已查看完整画面；正文另用文字和表格解释控制权、数据所在位置及省略的边界。小屏可打开本地图放大，不依赖缩小后的图中文字获取结论。
