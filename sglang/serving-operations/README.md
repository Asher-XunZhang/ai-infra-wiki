# SGLang：服务部署与运行治理

[专题入口](../README.md) · [Wiki 首页](../../README.md)

本目录研究实例外的 API、Gateway、部署与运维边界，解释从可访问到可接单，再到排空退出的证据。实例内调度和执行归 [runtime](../runtime/README.md)，P/D 状态交接归 [disaggregation](../disaggregation/README.md)。

## 先修与推荐路线

先理解[请求生命周期](../runtime/README.md)、[并行实例](../parallelism/README.md)和[通信与回收](../parallelism/SGLang%20通信与传输机制学习文档.md)。

1. [服务生命周期与请求治理](SGLang%20服务生命周期与请求治理学习文档.md) · [交互课程](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/serving-operations/)：观察启动探活、路由准入、超时取消与退出。
2. [Gateway 注册、路由与缓存亲和](../source-study/10-serving-operations/01-ModelGateway注册路由与缓存亲和.md)：深入实例名单与选择依据。
3. [健康检查、超时、限流与优雅退出](../source-study/10-serving-operations/03-健康检查超时限流与优雅退出.md)：深入不同层的计时、额度和失败处理。
4. [Metrics、日志与 Trace](../source-study/10-serving-operations/04-Metrics日志与Trace关联.md)：把一个现象与具体请求、阶段和原因关联。

## 按问题选读

- [HTTP、gRPC 与 Rust 边界](../source-study/10-serving-operations/02-HTTPgRPC与Rust服务边界.md)：不同入口不能套用同一健康或取消结论。
- [权重更新、暂停恢复与 RL 接口](../source-study/10-serving-operations/05-权重更新暂停恢复与RL接口.md)：运行中的服务变更与状态协调。

资料保留独立固定提交；阅读导航不表示已经验证所有部署组合。

## 自测与下一步

能区分端口可访问、服务状态 Up、探活活动、请求完成、取消完成和进程退出；能指出队列与额度属于哪一层。

下一步进入[性能分析](../performance-engineering/README.md)，将指标与等待位置关联；或进入[分离部署](../disaggregation/README.md)，补跨角色的状态交接与资源退役。
