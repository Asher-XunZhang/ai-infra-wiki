# SGLang：前缀缓存与分层存储

[专题学习入口](<../README.md>) · [Wiki 首页](<../../README.md>)

从逻辑前缀到物理 KV，再追踪 Host 和外部存储的回载与释放。

本文是学习导航，分类与阅读顺序由整理者归纳。技术资料继续保留各自的来源、版本和验证边界；导航不表示已完成运行或性能复现。

## 先修知识

先确认以下基础；已经掌握的部分可以跳过：

- [推理基础](<../../llm-inference/foundations/README.md>)
- [请求、调度与执行主链](<../runtime/README.md>)

## 映射与共享回收主线

先读 [KV 映射与共享回收](<SGLang KV 映射与共享回收学习文档.md>)，配合 [交互课程](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/kv-memory/) 观察尾页续写、整页命中、共享保护与驱逐。先修为 KV 计算与普通请求运行时；随后按问题进入下方前缀缓存、分层存储或混合状态资料。

## 阅读顺序

首轮读前三篇即可。Mooncake 与 HiCache 源码是存储支线；Unified 是混合状态支线，建议配合通用模型结构专题阅读。

| 顺序 | 资料 | 阅读定位 | 学习重点 |
| --- | --- | --- | --- |
| 1 | [SGLang RadixAttention 与 HiCache KV Cache 技术主线](<SGLang RadixAttention 与 HiCache KV Cache 技术主线学习文档.md>) | 起步 | 用技术主线建立 Radix 与分层缓存的整体地图。 |
| 2 | [SGLang RadixAttention 前缀缓存命中定义](<SGLang RadixAttention 前缀缓存命中定义学习文档.md>) | 主线 | 明确 token 匹配与物理 KV 复用的条件。 |
| 3 | [SGLang KV Pool、请求视图与 HiCache 工程](<SGLang KV Pool、请求视图与 HiCache 工程学习文档.md>) | 主线 | 拆清物理池、请求视图和缓存生命周期控制。 |
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

- [v0.5.20：SWA 分叉缓存与外部 Linker](<../version-studies/SGLang v0.5.20 缓存调度与 Serving 协同学习文档.md>)：版本案例；先读 Unified Radix Cache，再区分分叉恢复条件与远端内存的管理权。
- [分离部署与状态交接](<../disaggregation/README.md>)
- [模型适配与特殊执行机制](<../model-support/README.md>)
- [缓存身份与生命周期](<../../vllm/kv-cache/README.md>)

## 本目录收录范围

收录 SGLang 的缓存索引、分配器、物理池、复用、分层存储和混合状态缓存；跨服务角色的请求与 KV 交接归 disaggregation。

后续文章先加入阅读表，标明先修、定位和学习重点。当形成多篇相互关联且需要独立导航的子主题时，再建立子目录；跨主题内容保留一份正文，由相关 README 交叉引用。
