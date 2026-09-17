# RDMA KV 传输图片与来源记录

对应文档：[学习文档](<../../llm-inference/distributed-serving/P-D 分离的 RDMA、IB 与 GPU 可见性学习文档.md>)。读取日期：2026-09-14。

- 原文：[P/D 分离中的 RDMA 传输：基于 Wireshark 的 IB 与 GPUDirect 分析](https://mp.weixin.qq.com/s/kK6jmNGU7RrKlpnrKp10KA)
- 作者：AI码酱；发布时间：2026-09-13T23:51:24+08:00。
- 已定位并完整提取 `#js_content`，标准库 parser 与 lxml 独立检查正文、末尾和图片数量；排除验证页。正文结构化文本 7817 字符，HTML 3601440 字节。
- 原始 HTML SHA-256：`c01b823291ecdf247e3a33296a1098746e787745d637a41473b200f8cdf1ff82`。
- 结构化正文 SHA-256：`ad92504501d3ba5558f7cc7864e088c4195a10ff09781873cad52e24ed4fdfd1`。
- 原始整页 HTML 与抓取脚本只保留在临时目录；仓内保留原创笔记与用于解读的关键原图。
- 正文有 23 张图片：第 1 张为封面，排除；第 2–3 张为抓包截图，保留；第 4–23 张为原文图 01–20，全部保留。
- 原图明确包含未闭环证据说明。本文未取得 PCAP、容器或运行日志，图片不是本次实测结果。
- 图片逐张检查了含义、尺寸和白色背景可读性；保持下载字节不变，未 AI 重绘。窄屏下复杂图需点击放大，正文提供文字解释。

| 文件 | 原始来源 | 像素尺寸 | SHA-256 |
| --- | --- | --- | --- |
| [capture-port-a.png](capture-port-a.png) | [原文第 2 张](https://mmbiz.qpic.cn/mmbiz_png/zfX4RBQuwDzFscnQjhccAszehLFXKPiaSukqaFWl73gRrA2ghndTuA8EyVdY0cFSria7rqibF4QPutyjZhHYTjOaE2UooUA0HPo7ia2whp1ibXOU/640?wx_fmt=png&from=appmsg) | 1080×578 | `06154f26a98b653ac0d5e7e81ba8e55a260c1213863f636daae6d04cfcb77866` |
| [capture-port-b.png](capture-port-b.png) | [原文第 3 张](https://mmbiz.qpic.cn/mmbiz_png/zfX4RBQuwDwJMibEYgSSec3piaPQc09UnkcJHicBlxgMXTV6AKFcj36xDcicocytUs3gjEjZ0mQy9DTUHRicKjFeBmD4t8Ge84VaXP83MTUAPCL0/640?wx_fmt=png&from=appmsg) | 1080×562 | `e21c915bbd4d95a57064330d7b41b51f97ba62ac4118e78de5dc84abc81ec1f2` |
| [01-kv-pull-lifecycle.png](01-kv-pull-lifecycle.png) | [原文第 4 张](https://mmbiz.qpic.cn/mmbiz_png/zfX4RBQuwDytNHglgGibAHQ8kbNQaZDvb7rLYLUVE1rpZHWKtiawWr0uK2pePx37TyILu8I8gy8rS0fNzd85EOeY7YuPhcWKqETOCGVic5ME1c/640?wx_fmt=png&from=appmsg) | 1080×773 | `32d9ad76ccf70a9f112eaf905da6c9b5813d2b5064175fc00680a75e73087b46` |
| [02-dual-end-capture.png](02-dual-end-capture.png) | [原文第 5 张](https://mmbiz.qpic.cn/mmbiz_png/zfX4RBQuwDweYEHzGwkfygTV0tRy31ia95Dsk5oV78WKviaRlxewA2OQNSFKwFyA7JrMaTxPtXwBagDoGbDWbgo9DcziczoBibHFD1RoiaYk0t7Q/640?wx_fmt=png&from=appmsg) | 1080×684 | `316fea37ca1913a36e552eddc2293f33bdf31eb0f65163cfd606f51347c14ce5` |
| [03-read-correlation.png](03-read-correlation.png) | [原文第 6 张](https://mmbiz.qpic.cn/sz_mmbiz_png/zfX4RBQuwDxuGwzKekTiciaNvGicX9qv3NeETy7bibzbUGbkYibvmuNYEbTOMPk21vxQyn79rJmYND6VssqWXoHeIhBNF9efM7lmibiaTCNUHFiaHUk/640?wx_fmt=png&from=appmsg) | 1080×725 | `aa7c108af44cdec9c81bc684317a796cfd2783bbebd4ab86c0b3230b62b7f9c6` |
| [04-read-packet-sequence.png](04-read-packet-sequence.png) | [原文第 7 张](https://mmbiz.qpic.cn/mmbiz_png/zfX4RBQuwDzGr77gs0PELmTuOYicvT29VMMMbdKibATnxD5icYvALekJSpriculY06TCMZNPXYQNcibGgtrDzYF8Q2edWa5cwKBbhUMEOxsXzrjQ/640?wx_fmt=png&from=appmsg) | 1080×534 | `968546a68f604905670682a667999f6a7cc147edefff127cbd90ecf307c0db50` |
| [05-packet-fields.png](05-packet-fields.png) | [原文第 8 张](https://mmbiz.qpic.cn/mmbiz_png/zfX4RBQuwDzkSB50YSuzxjECd2tgu2DyK528soKqAdzrzoKY6Dzhgv2DAONkfWl1TBaaWkofwxQiaxd6tibSticElGq3Nm7tklwUQ3ZibYTdacE/640?wx_fmt=png&from=appmsg) | 1080×597 | `3291a9956f91a2593a5c469cab84c641c8aa91edb5f8d299085c9a75f5d8dd65` |
| [06-pull-direction.png](06-pull-direction.png) | [原文第 9 张](https://mmbiz.qpic.cn/mmbiz_png/zfX4RBQuwDxMPUKOnqJwZdxGkmF2su6LLpvngn2lAplB6Xavghs56CibbxhZMrUZj0KXhzBic5agpBCupDastcBSIia3k3VGcibOhqRkdVWzLR0/640?wx_fmt=png&from=appmsg) | 1080×465 | `33688b24e2ced88775a1056924d185fd3b52ba061c43791fd2aa6db28faaf5a3` |
| [07-kv-block-layout.png](07-kv-block-layout.png) | [原文第 10 张](https://mmbiz.qpic.cn/mmbiz_png/zfX4RBQuwDzOSMJMflvdaNFmMg5tJ4jgw4MNIzpx7vA2LzJ0vBXDf9LTFUzJZTzV3J2A8FUZtEej0PedeuDmbGnETqnWujaaS3uTDwib4nRs/640?wx_fmt=png&from=appmsg) | 1080×507 | `7934189a37748073183f4b2117523311aee327263569c6696437b650fda61cd1` |
| [08-request-address-mapping.png](08-request-address-mapping.png) | [原文第 11 张](https://mmbiz.qpic.cn/mmbiz_png/zfX4RBQuwDy3l3Et28luetib0qBjAFXZpfWWs56bQVGk1YuABb0Fic99xCsdomtTlm5ibSNQm8WuRd1IQILVsPo6UqYfmRScmicVgj7Oadu3z7M/640?wx_fmt=png&from=appmsg) | 1080×771 | `910aaec59d4d72ec9e8f1e0d8e9e718a24263da15bcb253433be706862f0b618` |
| [09-gpu-memory-mapping.png](09-gpu-memory-mapping.png) | [原文第 12 张](https://mmbiz.qpic.cn/sz_mmbiz_png/zfX4RBQuwDyYwbMzTAbz67e6EtSoHicJnKce9HJWtibAEp2RkWUgfP59XV9qsArBMNcx5BoVu8h5Qr2NYLP6d4nJ984ZdhDl9SCVLrIdPByaY/640?wx_fmt=png&from=appmsg) | 1080×477 | `240893592c497985c2109dd7196b420e518e0789cb80197e61562f49be369cfe` |
| [10-mr-permissions.png](10-mr-permissions.png) | [原文第 13 张](https://mmbiz.qpic.cn/sz_mmbiz_png/zfX4RBQuwDxu4r4F8mklEdocGIicXN4tbks3WaYsLSs3Qt3JleDJYtBfcarjf4WXuiaNYOP6U5z9sWq5kJJ1pW905IibWiaaU22TFJfXAT1AXxo/640?wx_fmt=png&from=appmsg) | 1080×450 | `78a1d2abb1a4e2ef1d31f248801d6d601a38e0797f75cd334f6383c1183f76c6` |
| [11-qp-connection.png](11-qp-connection.png) | [原文第 14 张](https://mmbiz.qpic.cn/sz_mmbiz_png/zfX4RBQuwDy2RQJsekbCmdLGY6as7oD9RIvG4e9zRa6SyJBK8LHDlia89gmssdicZnBu3hO3kxDgeMGwmG8vINVAbA6zsl4Yp8VJJvibF3bUz0/640?wx_fmt=png&from=appmsg) | 1080×486 | `b3d2240395bfcf1d4c6e4d50c6f006f0218603e87e13ef28281c8ecdf4a3f13e` |
| [12-rdma-operations.png](12-rdma-operations.png) | [原文第 15 张](https://mmbiz.qpic.cn/mmbiz_png/zfX4RBQuwDzpoBuqSYlL5jibPkZpBJQt9aZ8HvNh0qVrDkgoU7icEBRWU4S82Gr6bEOQYibMUWQKcIFn4kZJXUePU288G4rElJOjCMMCpibLIbc/640?wx_fmt=png&from=appmsg) | 1080×732 | `b0644e3d38d6bf6e6fec5bac484992ccabecd9f205c96d3da168c752dc98caad` |
| [13-transfer-submission.png](13-transfer-submission.png) | [原文第 16 张](https://mmbiz.qpic.cn/mmbiz_png/zfX4RBQuwDxiczSj3ZjhBfOPFRzGicDvX8vUq05YI21yQjlJ2GX9xg2HvzEeao26iaAdq3v5tl5Tr6qMDVeSWyqzh7aERr4fzzMPQEgia6DC1MM/640?wx_fmt=png&from=appmsg) | 1080×548 | `686a568f72ebf22d5ea53d7bde9a4931cac5d96e935cff65bac1ecaee6f742f1` |
| [14-doorbell-completion.png](14-doorbell-completion.png) | [原文第 17 张](https://mmbiz.qpic.cn/mmbiz_png/zfX4RBQuwDyrG0icFKyA3cAZZpW6nIctwYgUBdibQVtEC6LrmxxUtIMWF9hyy96PFaUicYMK0LqCuqg6JImibSqgBmgV0GGibWp8QoyJNVKxdDJA/640?wx_fmt=png&from=appmsg) | 1080×611 | `00c99fce5e949a035838ca5f198b9b7d3e6c8581209d0f84e41948c473c72140` |
| [15-fabric-flow-control.png](15-fabric-flow-control.png) | [原文第 18 张](https://mmbiz.qpic.cn/mmbiz_png/zfX4RBQuwDzNibIavLBNCGlM4PgCul7N2vXKuVFMibFicy2HiaGmibFumtgASsbbIztEADYgl7GLLibPbukibYjyk51D7l6j6y2Ce5ibbjuD1x71B8A/640?wx_fmt=png&from=appmsg) | 1080×465 | `261647b92ead68cb9ed43df3d2b099693f23ab4276af95f49c07ae96d9206dd8` |
| [16-gpu-data-placement.png](16-gpu-data-placement.png) | [原文第 19 张](https://mmbiz.qpic.cn/sz_mmbiz_png/zfX4RBQuwDwicgAhSK5nttTMUeXhtCb54FwZqwOiaMvOzG9JnHgsnRptVdhaLUowMmJoicLXcy0S2X3M53iceCdzJNomNiaDHHhk59ATClzv4pz4/640?wx_fmt=png&from=appmsg) | 1080×419 | `0382ddb3eb047bb276b2b557fb009a61ac4d3593fce56ca7cf8dc38460b87eb1` |
| [17-layout-evidence.png](17-layout-evidence.png) | [原文第 20 张](https://mmbiz.qpic.cn/sz_mmbiz_png/zfX4RBQuwDzo3b1wB37H3NvXHfAHZW0AVqLxS6z7ypghdsWyzpU2kZnsdKkJuDAWQPcpLO3E7koia3B36udGY5IPTE2hPkXDOkuibIQHkzsa4/640?wx_fmt=png&from=appmsg) | 1080×590 | `b434edd848de295b9e9164e009022db39191210f8f40be66e46b441ca33e1478` |
| [18-request-retirement.png](18-request-retirement.png) | [原文第 21 张](https://mmbiz.qpic.cn/sz_mmbiz_png/zfX4RBQuwDyLm8hOjprKF79J9nK6A7iadBx7SGGtiaoAFNS6eAVykRxUYAwvI2BHMCbnM212nyxxIiafdSAcqCDT1V8aibfIibgKmXwGojmqTL3I/640?wx_fmt=png&from=appmsg) | 1080×513 | `104708ef55876613dc1083b3e14f75b0fe89a5fe5ab38b39499e876c237804b7` |
| [19-gpu-visibility.png](19-gpu-visibility.png) | [原文第 22 张](https://mmbiz.qpic.cn/mmbiz_png/zfX4RBQuwDw3tu24nHRKHPkbX1ndFSL6icsKqVIGXOKt4DPHTkjeKgibtTj4Q7OmG8OC7beXrwq9AhVKyyAzoHickgYyKMb8v1RWTy5NnpGFIQ/640?wx_fmt=png&from=appmsg) | 1080×533 | `8576fba7d1faf5af61e53475857d0323de6addd4058135eda793099c6c445c98` |
| [20-failure-map.png](20-failure-map.png) | [原文第 23 张](https://mmbiz.qpic.cn/sz_mmbiz_png/zfX4RBQuwDwvpWQZ4N9pdM3iaJTB86cDQ5TDItN7vamrGmg91krP2QWEicibPz0xPMUvAogEZC2xrLAIjibDLeC1zkYHR0w1znYoYe4rHddq1JU/640?wx_fmt=png&from=appmsg) | 1080×575 | `78205706efc4a1eafd73e03586b9fd1fd7919830992db7c94c4769946289b301` |

## 本次验证范围

四篇新文档共 8 张 Mermaid 已用 Mermaid 11 在无头 Chrome 中解析并渲染成功；165 处本地引用可解析；30 张采用图片通过解码与 SHA-256 检查；全仓图像路径扫描未发现顶层 `images/` 以外的图片；`git diff --check` 通过。临时渲染结果未放入仓库。以上为文档验证，不是 RDMA/GPU 运行验证。
