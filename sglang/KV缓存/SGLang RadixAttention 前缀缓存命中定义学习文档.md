# SGLang RadixAttention 前缀缓存命中定义学习文档

<!-- article-redirect: SGLang KV Pool、请求视图与 HiCache 工程学习文档.md#prefix-match-to-readable-kv -->

前缀命中的解释已整合到[从公共前缀走到可读数据](<SGLang KV Pool、请求视图与 HiCache 工程学习文档.md#prefix-match-to-readable-kv>)，包括压缩树、节点分裂、原图、child_key 的资料边界和页对齐例子。

原资料：Lychee & Ethan，[《KV Cache 前缀匹配的设计分野》](https://mp.weixin.qq.com/s/GciJebFszheSqU-s3yZPMQ)，读取于 2026-07-17。原稿未固定源码版本；新版正文将它与固定源码的分页约束分开说明。
