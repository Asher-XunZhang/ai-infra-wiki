# 三类推理引擎比较图源

- 来源：[DeepInfer 和 vLLM、SGLang 的对比](https://mp.weixin.qq.com/s/t9lemS4pN4UfIBjCFAcj1w)
- 作者 / 公众号：AI灰鲸；发布时间：2026-10-25 08:09:49（Asia/Shanghai）；读取日期：2026-10-27。
- 从文章正文提取全部 8 张有信息量的图片，逐张查看，原样保存；未生成或重绘。版权归原作者，保留图不表示拥有可转授权的许可。
- 前 6 张为机制示意，后 2 张为文章作者的比较表；不是一手实验数据，也不是无版本条件的官方兼容矩阵。
- 所有图片均为不透明底色。前 6 张为 1080 × 720 PNG，后 2 张为竖版 JPEG；后两张的 HTML `data-type` 标作 png，但文件字节为 JPEG，以实际格式保存。长表在窄屏需放大，正文另用 Markdown 表说明关键边界。
- 对应正文：[交付范围与机制比较](<../../大模型推理/分布式服务/DeepInfer、vLLM 与 SGLang 的交付范围与机制比较学习文档.md>)；图意解读为整理者撰写。

| 本地文件 | 原文图意 | 图片来源 | SHA-256 |
| --- | --- | --- | --- |
| [01-request-stages.png](01-request-stages.png) | 一次请求的计算流程示意 | [原图](https://mmbiz.qpic.cn/mmbiz_png/DsDpfuXjiaoANsm60BjwyMGBJTvF62lBVgT0mU6WEmwibPRSFc4weia0eiaYvfHYMN0bFKdyjNN6pePibFQEj4NicFLHrFhUNnyfxuFI6vnoMlSL8/640?wx_fmt=png&from=appmsg) | `5e04afbc7705b0c920740bcfac1555957edba9b864818f0e0a4511fb8734394f` |
| [02-paging-prefix-reuse.png](02-paging-prefix-reuse.png) | 上半图解释 vLLM 的分页管理，下半图解释 SGLang 的前缀树复用 | [原图](https://mmbiz.qpic.cn/mmbiz_png/DsDpfuXjiaoB8JgpVfvLWuoWu5via6Icg3ibT1Pq47P2bsUgNZvGFD56rQkfXmZ7vVFI2mNpnq8PCV2ibliaNJiaGGQsicfpkVR0vhRTciamQibyKLe8/640?wx_fmt=png&from=appmsg) | `6799f54f8f0d63fd6a80b59b46db5cf21006557ac2ce3527f0e47834ffead30a` |
| [03-serving-boundary.png](03-serving-boundary.png) | DeepInfer 的开源核心与完整服务的集成边界 | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/DsDpfuXjiaoBj0WkoCvDbjQIBGongNkyXA5cZWUEOWzafdG5yHs45IoVU6sz4sPia79BkLyH4KjU5sQdPKusFw8mFueapgHL8dtuXvJs0Xu5c/640?wx_fmt=png&from=appmsg) | `18c7590b83aafcbad79e4c471509d3ce1e4f02a1fffb8f679628f94565954645` |
| [04-attention-dp-moe-ep.png](04-attention-dp-moe-ep.png) | DP 与 EP 的组合示意 | [原图](https://mmbiz.qpic.cn/mmbiz_png/DsDpfuXjiaoBfCbmz19gV9azSxzUuvicib3Kgib5f5JoTxWY1xJp8icia1gwrcLZBxT62gDNl2b2LOnICL1IyqsfniaHWDibGPAfkCPvD37BVVQiaoqk/640?wx_fmt=png&from=appmsg) | `24a50f186999d0f7413efd573a6fe8f2127ab70a96e229783637bdbd9eccf6ec` |
| [05-verification-budget.png](05-verification-budget.png) | 两条请求分别验证 4 个和 2 个草稿 token，仅为预算分配示意 | [原图](https://mmbiz.qpic.cn/mmbiz_png/DsDpfuXjiaoDv8JAvomQt8T4bvUol3Bfiammmnqk6gicAf4icaVUeZdoU76M1aGUNEgMz8mUhWQfl0k8ZAgWVaUHthicjjHpXurCCtE7Z3dT5mQM/640?wx_fmt=png&from=appmsg) | `ae2d2a8478cbe17d269cbccf3cb0a573b7e4b557e5512427238c49beba01cd30` |
| [06-bounded-replay.png](06-bounded-replay.png) | 有限后缀重放示意 | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/DsDpfuXjiaoDKVNvibFicIwsxy9ibZZB4lBgmImD9rRJBHLRpttNfooDJDrcP51XuGTYNRu3mgcwyv3QgIOoF0cicg1z1f3uLhbbutIKDqz8xusQ/640?wx_fmt=png&from=appmsg) | `3e99c08b29f21c0c3852725d07b2086b391a4c63a1c606a8f9cc99e1d819f89c` |
| [07-delivery-comparison.jpg](07-delivery-comparison.jpg) | 表 1：定位与交付范围 | [原图](https://mmbiz.qpic.cn/mmbiz_jpg/DsDpfuXjiaoDsefEaFzJpWqXjQocoacllrFCTBf4oggKEmSsN3SoaoyMRsdThsOT4HECh9C2p2D9jyS9haD2A13dU8PQ8MGPMcbiaCpibEqBRA/640?wx_fmt=jpeg) | `897283338e5630929ee7742b5503336b392ecc93b3bbcbd148154e4f2677f4e5` |
| [08-mechanism-comparison.jpg](08-mechanism-comparison.jpg) | 表 2：缓存与执行机制 | [原图](https://mmbiz.qpic.cn/mmbiz_jpg/DsDpfuXjiaoA65rLAQIiaLde8Yb01X30H60IkCxOt0gjNkAmSMwQBuULPgiaymJqWASAwDclzjAe8XCcW6dI2KK5icPRcvzkCctP6hWkSNLYWNQ/640?wx_fmt=jpeg) | `cc3f4e0399859ce696fe66a9e6643c93896aa7126d05141b03303f250a194b60` |
