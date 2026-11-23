# SGLang：前缀缓存与分层存储

[专题学习入口](<../README.md>) · [Wiki 首页](<../../README.md>)

从逻辑前缀到物理 KV，再追踪 Host 和外部存储的回载与释放。

各篇文章的来源、源码版本和验证范围见文首说明。阅读顺序可按自己的问题调整。

## 先修知识

先确认以下基础；已经掌握的部分可以跳过：

- [推理基础](<../../大模型推理/推理基础/README.md>)
- [请求、调度与执行主链](<../运行时/README.md>)

## 映射与共享回收主线

先读 [KV 映射与共享回收](<SGLang KV 映射与共享回收学习文档.md>)，配合 [交互课程](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/kv-memory/) 观察尾页续写、整页命中、共享保护与驱逐。先修为 KV 计算与普通请求运行时；随后按问题进入下方前缀缓存、分层存储或混合状态资料。

## 阅读顺序

先用物理页的交互例子建立直觉，再读 KV Pool 主文，把匹配、地址和可读数据连起来。技术演进、分层 I/O 和混合状态按问题选读。

| 顺序 | 资料 | 阅读定位 | 学习重点 |
| --- | --- | --- | --- |
| 演进选读 | [SGLang RadixAttention 与 HiCache KV Cache 技术主线](<SGLang RadixAttention 与 HiCache KV Cache 技术主线学习文档.md>) | 参考 | 沿技术演进认识 Radix、HiCache、HiSparse 与 ShadowRadix；具体实现按各自版本读。 |
| 主线 | [SGLang KV Pool、请求视图与 HiCache 工程](<SGLang KV Pool、请求视图与 HiCache 工程学习文档.md>) | 主线 | 用一个前缀例子连接索引、请求映射、物理位置与数据可读条件。 |
| 4 | [Mooncake 与 SGLang HiCache](<Mooncake 与 SGLang HiCache 学习文档.md>) | 选读：接入 | 先看外部存储接在哪里，再下钻源码。 |
| 5 | [HiCache 前缀命中源码](<HiCache 前缀命中源码学习文档.md>) | 深入 | 核对页对齐、树匹配与设备可用前缀。 |
| 6 | [HiCache 下 SGLang L1、L2、L3 与上传回载源码](<HiCache 下 SGLang L1、L2、L3 与上传回载源码学习文档.md>) | 深入 | 沿 D2H、L3 和 H2D 跟踪事件与资源所有权。 |
| 7 | [SGLang Unified Radix Cache](<SGLang Unified Radix Cache 学习文档.md>) | 选读：混合状态 | 在普通前缀缓存基础上理解多种状态组件；配合模型结构专题阅读。 |

## 对应源码课程

需要系统读代码时，从下面的课程入口衔接。课程与专题可能采用不同固定提交，阅读前分别核对文首基线。

- [缓存架构与资源所有权](<../source-study/architecture/05-缓存架构与资源所有权.md>)
- [请求视图与分配器](<../source-study/04-kv-cache/01-请求视图物理槽位与分配器.md>)
- [Unified 与混合状态组件](<../source-study/04-kv-cache/04-UnifiedRadix与混合状态组件.md>)

## 自测与下一步

能分清前缀索引、请求映射、物理池、传输控制和完成事件；命中不等于已经可计算。

按目标继续：

- [v0.5.20：SWA 分叉缓存与外部 Linker](<../版本研究/SGLang v0.5.20 缓存调度与 Serving 协同学习文档.md>)：版本案例；先读 Unified Radix Cache，再区分分叉恢复条件与远端内存的管理权。
- [Agent 暂停与恢复时的 KV 保留与状态所有权](<../../大模型推理/KV缓存/Agent 暂停与恢复时的 KV 保留与状态所有权学习文档.md>)：跨引擎深入；区分会话软引用、请求锁、分层恢复和已有 Hint 协议的执行边界。
- [分离部署与状态交接](<../分离部署/README.md>)
- [模型适配与特殊执行机制](<../模型适配/README.md>)
- [缓存身份与生命周期](<../../vllm/KV缓存/README.md>)

## 本目录收录范围

收录 SGLang 的缓存索引、分配器、物理池、复用、分层存储和混合状态缓存；跨服务角色的请求与 KV 交接归 disaggregation。

## 容量与回载交互

先操作[显存账本](../../pages/sglang/memory-budget/index.html)，再按需进入 [HiCache 完整回载](../../pages/sglang/hicache-loadback/index.html)。两课固定 72d5c5bb，与普通分页机制课的 279339f113 分别保留基线；教学预算、执行依赖与实际验证范围单独说明。
