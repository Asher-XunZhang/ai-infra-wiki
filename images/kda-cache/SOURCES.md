# KDA Cache 图片来源与阅读记录

读取日期：2026-09-17。关联文档：[KDA Cache 检查点、前缀复用与投机回滚](<../../llm-inference/model-architecture/KDA Cache 检查点、前缀复用与投机回滚学习文档.md>)。

## 原文与查重

- 原文：[图解KDA Cache](https://mp.weixin.qq.com/s/baxG8LbBmuLLhpGrO1jusA)，作者 franky1011，发布时间 2026-09-16 14:27:21（Asia/Shanghai）。
- 按文章短 ID 与标题查重，未发现同一网页的既有整理。与混合缓存专题重叠的机制通过链接复用，本文进一步展开恢复与提交边界。
- 使用完整桌面浏览器 User-Agent 与 `scene=1` 取得正文；以 HTML parser 提取 `js_content`，并用 lxml 复查结构。已检查开头、末尾及六个参考链接文本，不把页面推荐或脚本纳入正文。
- 正文有 38 个 `pre` 文本块、0 个 `img`、38 个相同的 45×12 三圆点装饰 SVG；无技术 CSS 背景图。原文技术图为文本排版，故不将装饰圆点作为学习配图。
- 正文的机制以原创 Mermaid 重新组织，明确标注为整理者解释模型；两张补充图来自下列一手资料，不冒充微信原图。
- 抓取 HTML 为临时核读材料，不入仓；下列 SHA-256 记录本次读取版本，不代表网页永远不变。

HTML SHA-256：`cb3a30de794dbfe9cb9836ff38eddfd9f8cdd9fa1165bfef6ad13ebe3649641f`。

正文提取文本 SHA-256：`9cf5eb5a5c050fb257a7cbdf777462d972184a8536d217125a243f9ae2e9e5e0`。

## 一手技术原图

| 文件 | 来源与原图 | 用途 | SHA-256 |
| --- | --- | --- | --- |
| [01-internal-checkpoints.svg](01-internal-checkpoints.svg) | vLLM，2026-09-13；[原图](https://vllm.ai/blog-assets/figures/2026-09-13-kimi-k3-performance-optimization/internal-kda-checkpoints.svg) | Prefill 内部导出检查点，见学习文档第 5 节 | `ef7797171bac6cff3047ea8a93b0fb523fa9490ed7146b91b5db62a305823ba8` |
| [02-replayssm-state-commit.svg](02-replayssm-state-commit.svg) | SGLang，2026-07-27；[原图](https://www.lmsys.org/images/blog/kimi-k3-day0-support/fig-replayssm-kda.svg) | 验证输入缓冲与接受前缀的状态提交，见学习文档第 6 节 | `d1e984bc290fdd825713c891745efc96a644058d828154d2504cde3f27201e53` |

以上 SVG 保留下载字节，未重绘或修改。已检查 XML、脚本与外部资源依赖，使用浏览器渲染核对文字和布局。学习文档逐图提供原创图意解读；图中的执行路径仅对应各自文章描述。

## 核读边界

正文与五篇项目/作者网页做交叉核对；未逐行审查其中链接的 PR，未进行 GPU 复现。尤其不能把关闭 prefix caching 的 vLLM 总榜收益归因于内部检查点。原文引用的技术报告保留为延伸阅读，不列为已全文核读来源。
