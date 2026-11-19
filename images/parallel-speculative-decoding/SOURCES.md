# 并行投机解码图片来源

- 来源文章：[并行投机解码(DFlash/DSpark)的快速理解与vLLM实测](https://zhuanlan.zhihu.com/p/2069029506447417522)，作者 kaiyuan。
- 读取日期：2026-11-19。机制部分包含 5 张独立配图，已逐图阅读；重复放大图不重复收录。公式以文本保留，不保存公式图片；部署与实测部分不在本次整理范围。
- 前两张保存为原始 GIF，未转换成静态封面；其余保留原图。未生成或重绘图片。
- DFlash、DSpark 图按原文归因于对应项目/论文，红色编号属于文章标注；本文没有把标注图称为未经修改的论文原图。
- 因果可见范围动图表示 Attention 依赖，不是实测执行时间线。

| 文件 | 用途 | 原始资源 | SHA-256 |
| --- | --- | --- | --- |
| [01-causal-attention.gif](01-causal-attention.gif) | 因果可见范围动图 | [原始资源](https://pica.zhimg.com/v2-3c4004b1224ba678a666439bba10c0b2_1440w.gif) | `04e0cff431edd166c72386aa4ac6e2aa5f671d8a35ed5917a3d04be2cbf89e91` |
| [02-full-attention.gif](02-full-attention.gif) | 双向可见范围动图 | [原始资源](https://pic3.zhimg.com/v2-42f08c73002c474757b0296db7f79364_1440w.gif) | `cfdf98ab835aa15179af253584441532f215c49cd752386e16d6187cfbbfafd4` |
| [03-draft-verify.jpg](03-draft-verify.jpg) | 通用草稿与验证图 | [原始资源](https://picx.zhimg.com/v2-9fcac545fc958ab9da404aa16eeea06b_r.jpg) | `ef57a8b69f31e0be25f3832d8749daebbb775f8ff1faa4dfc704a7a8f189757a` |
| [04-dflash-kv-injection.jpg](04-dflash-kv-injection.jpg) | DFlash 特征注入图，带原文编号 | [原始资源](https://pic1.zhimg.com/v2-1984c1ea72ad5bc54692ddf3a1eba8dc_r.jpg) | `4c45972788b6b5caf357934dca6d482517f0dab562108bab8f6f3e421c92f9f8` |
| [05-dspark-pipeline.jpg](05-dspark-pipeline.jpg) | DSpark 架构图，带原文编号 | [原始资源](https://pic2.zhimg.com/v2-4417356e8d3ef8fd243dec707c35fa79_r.jpg) | `a3d6d84e89137666a218c17aee7c0d4b8e99ec290ffa2b1d065dfe2241313e85` |

正文：[DFlash 与 DSpark 并行投机解码](<../../大模型推理/高级生成/DFlash 与 DSpark 并行投机解码学习文档.md>)。
