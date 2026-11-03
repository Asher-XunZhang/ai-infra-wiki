# Attention 变体机制图来源

## NSA 三路注意力原图

- 论文：Yuan et al., *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*。
- 固定版本：[arXiv 2502.11089 v2](https://arxiv.org/html/2502.11089v2)，2025-02-27；读取日期：2026-11-02。
- 图号：Figure 2。左侧表示压缩、选择与滑窗三路 Attention 及门控汇合；右侧表示三种访问模式。图中的块、门与输出不是性能测量。
- 本地文件：[01-nsa-three-branches.png](01-nsa-three-branches.png)，2038 × 579，130618 bytes；保留原始 PNG，未重绘或裁剪。白底可读；窄屏阅读时应结合正文的逐步解释。
- [原图地址](https://arxiv.org/html/2502.11089v2/fig2.png)。版权归原作者。
- SHA-256：`11a82ab6b15f61e528072602f2b93da456e72bbe57f6ba3d0848a119ada708b5`。
- 采用位置：[KV Cache 容量优化技术地图 §7](<../../大模型推理/KV缓存/KV Cache 容量优化技术地图学习文档.md#7-dsa先选位置再做稀疏-attention>)。正文中的 8 token 选块及门控数值是整理者教学算例，不是论文实验或训练得到的压缩结果。
