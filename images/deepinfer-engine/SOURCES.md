# DeepInfer 图文来源

文章：[DeepInfer框架分析笔记](https://mp.weixin.qq.com/s/-f6g4J3FMqR59YN2E-Ofrg)，作者 kaiyuan，公众号 InfraTech，发布于 2026-10-26 19:30（Asia/Shanghai），读取于 2026-10-27。

正文共 9 张图片，保留第 2–8 张技术图的原始下载字节，未重绘。第 1 张为关注引导动图、第 9 张为二维码，不作为技术资料收录。图中截图小字较密，学习文档另用文字、表格和 Mermaid 拆解，点击文件可查看原尺寸。

原图权利归原权利人；公众号文章未发现明确的图片再许可声明。本表记录来源，不把上游代码的 Apache-2.0 自动视为全部文章配图的许可。

| 本地文件 | 正文图片序号 | 尺寸 | SHA-256 | 原始下载地址 |
| --- | --- | --- | --- | --- |
| [01-model-overview.png](01-model-overview.png) | 2 | 720 × 621 | `71f3eb06cbf30e82bb20878fa79ae184a7cd0fc8ea6e9d79706fbfba20628304` | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/uIP3tuXZx8CKRfBOcmiaxX7Ad39QzQYOToygzrzkXdia4JD9XZmzDtn9n8LdicIpQfhDOrb6zKEmWfibpxjbzGbUlw5iboiaK46Xuay2r4I6bQzhU/640?wx_fmt=png&from=appmsg) |
| [02-serving-architecture.png](02-serving-architecture.png) | 3 | 720 × 372 | `dc9afd93863322ed471250b7d23a6b4f4715bf72ff61ea018d7a93f44d651245` | [原图](https://mmbiz.qpic.cn/mmbiz_png/uIP3tuXZx8DYFkOezxvRIoSw6DOQSQqTiauoSsjMzU8ksrq9uL1sbO2GjwmWWp6oXxdw8dJMOAhQXbRYxDcauNkokoW8iaWmcDf80RtA2ibfFU/640?wx_fmt=png&from=appmsg) |
| [03-engine-step.png](03-engine-step.png) | 4 | 720 × 377 | `45a5fb05aac8a65237ae9fb4cf24b2baeaaa540d61c0a9e93a7275bf59ed82f3` | [原图](https://mmbiz.qpic.cn/mmbiz_png/uIP3tuXZx8DrYXZV2iaqcLbF5Qr82oLuEa4avGMAtib2eLs5ZUlVK90rlM95YgaoNv4VTDiasZTiblbscR7STgfvfmfBodEvYalZqa6TnTyKS6w/640?wx_fmt=png&from=appmsg) |
| [04-layer-components.png](04-layer-components.png) | 5 | 720 × 157 | `d447e195839c9c5a734465733d4e1833c66b80a14fedd489b952141d088320ea` | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/uIP3tuXZx8CKl8TcUibriavLWQmNicfXzAF35OIIYibxW4hxMpzKL55d57o7I1yibFk7XuD7js6M8zrQFSzOZGasYjq60qrbPLC3z5zpQfiaDq5Jk/640?wx_fmt=png&from=appmsg) |
| [05-architecture-comparison.png](05-architecture-comparison.png) | 6 | 720 × 347 | `88ff0b19f16ed94f11f2abb3d1dc0bbb46331442bc693ed925ae7f97486a5734` | [原图](https://mmbiz.qpic.cn/mmbiz_png/uIP3tuXZx8DR9uheBfdTEWpkTtFEpCbJ13uB7o1Jctc8eAe5Ym4uiaiac0xaCbdWx3sWzrich7NzCX2Vfq0X4svtnmqib77N7gq0vEFQXRstAuM/640?wx_fmt=png&from=appmsg) |
| [06-prefill-decode-pareto.png](06-prefill-decode-pareto.png) | 7 | 720 × 264 | `a4c69fd43dcac6d31b3dc576d9bc52491e67e6b25ffa34a9ca0c88fe16e332cf` | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/uIP3tuXZx8CFRT42bicctaNlQNbO1SqYq3KxEZqjEKrQqDaH8ZpMNGH7dhKYTA1BkTHiaoK9CCW5CWXMMFPqr4ickWKicEhiasTWJVpKuf6WokaA/640?wx_fmt=png&from=appmsg) |
| [07-wide-ep-pareto.png](07-wide-ep-pareto.png) | 8 | 720 × 502 | `af5d31cc6381bda2f0e6304f943b2833908c7769726eb2a289d6648ecb036347` | [原图](https://mmbiz.qpic.cn/mmbiz_png/uIP3tuXZx8AelhiaWlicJr8k8u5u9Ahu7mrdSYmUic0lZly6wqD8dGtOmHYPLPiajmndicBeU0gE2JpeBPRnT7EWtfTWiaSbQvs5bX6JgYDDxibAlM/640?wx_fmt=png&from=appmsg) |

## 图的来源层级与使用边界

- 01、05：文章注明来自 [InfraTech 3D 模型页面](https://calvinxky.github.io/InfraTech/3d_models/)。本次阅读了文章中的截图，未验证该互动页面；层数、模型布局仍以固定源码与模型配置为准。
- 02：文章转载的官方生产架构图，可对照 [固定提交的架构原图](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/images/architecture_overview.png)。这是完整生产系统图，不代表所有组件均已公开。
- 03、04：文章中的引擎时序和模型组件示意。时序已经与 Worker 源码复核；组件图是简化图，不能当作逐层准确执行顺序。
- 06：文章的 Prefill/Decode 曲线拼图。图内标注 context=131,072；Prefill 为 EP4/DP4、384+32 experts、Decoder Bounded Replay 140 tokens；Decode 为 EP64/DP64、384+64 experts、nextn=3 fixed、逐 token 接受率 75%，折算 τ=3.25。上游 README 说明 Prefill 用 4 张、Decode 用 64 张 SM100，均在单一 scale-up 域内。
- 07：官方 [Wide-EP 对照图](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/images/wide_ep_decode_pareto.png) 的文章转载：EP64/DP64（384+64 experts）与 EP8/DP8（384+32 experts）同时改变并行规模和冗余专家数，不能单独归因于 EP。context=131,072、nextn=3 fixed、逐 token 接受率 75%；点上数字为每个 DP rank 的请求数。

正文与图意解读：[DeepInfer 专用推理引擎与生产系统边界](<../../大模型推理/分布式服务/DeepInfer 专用推理引擎与生产系统边界学习文档.md>)。
