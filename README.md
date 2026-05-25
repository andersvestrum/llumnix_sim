<h2 align="center">Beyond Binary Priorities: Multi-Tier SLA Scheduling for Large Language Model Serving</h2>

<p align="center">
  <strong>Project repository for extending Llumnix's binary priority model to arbitrary multi-tier SLA scheduling, evaluated in the Vidur LLM inference simulator.</strong>
</p>

<p align="center">
  Anders Vestrum, Arya Raeesi, Hanna Rød<br>
  UC Berkeley, EECS Department
</p>

<p align="center">
  <a href="./docs/Multi-Tier_SLA_Scheduling_for_Large_Language_Model_Serving.pdf">Final Paper</a>
  |
  <a href="https://arxiv.org/abs/2406.03243">Llumnix Paper</a>
  |
  <a href="https://github.com/microsoft/vidur">Vidur Simulator</a>
</p>

## Overview

Modern LLM serving deployments must simultaneously satisfy heterogeneous service-level objectives (SLOs) across a diverse population of user tiers, ranging from latency-critical interactive API calls to background batch processing. Llumnix introduced a dynamic, migration-capable multi-instance scheduler that achieves load balancing, defragmentation, prioritization, and auto-scaling through a unified **"freeness"** metric.

This repository focuses on a narrower and more diagnostic question:

**What is the right number of priority tiers for a migration-capable LLM serving system, and how should isolation headroom be allocated across them?**

Llumnix's priority model is restricted to two levels (high and normal) controlled by a single fixed headroom value, an abstraction too coarse to express the richer SLA classes common in production deployments. Providers commonly differentiate across three-to-five SLA tiers (e.g., platinum, gold, silver, standard, free-tier) with distinct latency targets. We extend Llumnix to support an arbitrary number of tiers and characterize the tradeoff that finer priority granularity introduces.

This is a scheduling-research project built on top of the Vidur simulator and the Llumnix design, not the upstream training or serving repositories themselves.

## Core Idea

We generalize Llumnix's single scalar headroom to `K` priority tiers, where each tier `p ∈ {0, …, K−1}` receives a dedicated headroom budget allocated by exponential decay. We then sweep `K` from 1 to 10 and measure the effect on per-tier latency, aggregate latency, and cost-efficiency.

The central abstraction is the **freeness** of a replica:

| Quantity | Definition | What it measures |
| --- | --- | --- |
| `F = (M − Σ_r V(r)) / B` | capacity `M` minus total virtual usage, normalized by batch size `B` | available capacity; `F < 0` triggers migration |
| `H_p = M · h_max · e^(−λp)` | per-tier headroom budget (`h_max = 0.20`) | KV-cache reserved to isolate tier `p` |
| `F_full` | freeness including priority headroom | drives dispatch and migration targeting |
| `F_normal` | freeness excluding headroom | drives auto-scaling, avoiding spurious scale-out |

Tier 0 is the highest priority (critical/interactive) and tier `K−1` is the lowest (background/batch), consistent with Llumnix's convention. Each tier's full budget `H_p` is charged whenever *any* request of that priority is present, so a single critical request can effectively block normal-priority dispatch to an overloaded replica.

## Project Goals

- Extend Llumnix's binary high/normal priority model to support up to 10 SLA tiers with per-tier headroom and priority-aware dispatch ordering.
- Implement full live migration of running requests inside the Vidur simulator, faithful to Llumnix's multi-stage KV-cache transfer.
- Identify the priority-granularity sweet spot that maximizes cost-efficiency without collapsing tail latency.
- Characterize how priority effectiveness interacts with system load and with realistic workload distributions (uniform, Gaussian, enterprise).
- Provide a reproducible, GPU-free evaluation framework for SLA-aware multi-instance LLM scheduling.

## Audit Pipeline

The final paper centers on the following pipeline:

1. Generate synthetic request streams with a length distribution calibrated to realistic LLM API traffic (right-skewed, ~65% short conversational turns).
2. Assign each request a priority tier by sampling from one of three distributions — uniform, Gaussian, or enterprise — via the `PrioritySampler`.
3. Dispatch requests through the multi-tier `LlumnixGlobalScheduler` (priority-ordered, freest-replica selection) to per-instance `LlumletReplicaScheduler` instances.
4. Periodically evaluate load imbalance and run multi-stage live migration to rebalance overloaded replicas; emit auto-scaling recommendations from cluster-average normal-priority freeness.
5. Sweep priority-tier count `K` from 1 to 10, across two request-volume scales (10K and 15K requests).
6. Aggregate per-tier and cluster-wide metrics (TTFT, TBT, end-to-end latency percentiles, prefill/decode speedups, cost-per-latency) and compare against four baseline schedulers.

## Why This Matters

Migration-capable schedulers like Llumnix are appealing because they promise editable, isolatable quality-of-service across heterogeneous workloads without overprovisioning. If that promise extends to fine-grained SLA classes, providers could offer differentiated latency guarantees to many tiers from a single shared cluster.

But isolation has a cost: every active tier consumes headroom, reducing effective batching capacity. This project tests directly where the line is — how many priority tiers a freeness-based scheduler can support before the overhead of reserved headroom outweighs the SLA-differentiation benefit.

## Results

The evaluation sweeps the cross product of priority-tier count, workload distribution, request volume, and scheduler. The headline findings reported in the final paper:

**Four priority tiers is consistently optimal.** At `K = 4`, all three workload distributions achieve peak end-to-end P99 speedup, near-peak E2E mean speedup, and peak cost-per-latency improvement. Four tiers provides enough granularity to separate critical, high, standard, and background traffic without fragmenting queues so finely that load-balancing heuristics lose effectiveness. Beyond `K = 5`, benefits plateau and overhead from headroom fragmentation begins to dominate.

**Prefill mean speedup exceeds the original paper.** Across all conditions, our system achieves a prefill mean speedup of 5.0–8.3× over the INFaaS+vLLM baseline, substantially exceeding Llumnix's reported ≤2.2×. We attribute this to migration-driven load balancing that prevents the "convoy effect" where bursty prefill requests pile up on a single overloaded instance.

**Cost-per-latency improvements of 46–68%** (10K requests) and 24–53% (15K requests) compare favorably to Llumnix's reported 16–36%, attributable to consolidating low-priority requests and freeing capacity for high-priority workloads.

**No tail-latency collapse up to 10 tiers.** Aggregate P50/P90/P99 across all schedulers remains broadly stable as `K` increases from 1 to 10, confirming baseline parity: the multi-tier scheduler does not degrade aggregate performance, with prioritization overhead concentrated in the prefill phase while decode latency stays stable.

**Load-dependence.** Priority benefits are most pronounced at moderate load, where freeness variance across replicas is high and migration opportunities are abundant. Near saturation, average freeness approaches zero and migration loses leverage — suggesting priority scheduling should be paired with proactive auto-scaling.

## Acknowledgements

We thank the authors of the Vidur and Llumnix open-source projects for making their frameworks publicly available. This work used computing resources provided by Berkeley Research Computing through the Compton Spectrometer and Imager (COSI) mission (NASA Small Explorers (SMEX) Program).

## References

- Sun, B., et al. Llumnix: Dynamic scheduling for large language model serving. In *Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)*, 2024.
- Agrawal, A., Kedia, N., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B. S., Tumanov, A., and Ramjee, R. Vidur: A large-scale simulation framework for LLM inference. In *Proceedings of Machine Learning and Systems (MLSys)*, 2024. [arXiv:2405.05465](https://arxiv.org/abs/2405.05465).
- Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with PagedAttention (vLLM). In *Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP)*, 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).
- Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., and Chun, B.-G. Orca: A distributed serving system for transformer-based generative models. In *Proceedings of the 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI)*, 2022.
- Agrawal, A., Kedia, N., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B. S., Tumanov, A., and Ramjee, R. Taming throughput-latency tradeoff in LLM inference with Sarathi-Serve. In *Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)*, 2024. [arXiv:2403.02310](https://arxiv.org/abs/2403.02310).
- Romero, F., Li, Q., Yadwadkar, N. J., and Kozyrakis, C. INFaaS: Automated model-less inference serving. In *USENIX Annual Technical Conference (ATC)*, 2021.
- Li, Z., Zheng, L., Zhong, Y., Liu, V., Sheng, Y., Jin, X., Huang, Y., Chen, Z., Zhang, H., Gonzalez, J. E., and Stoica, I. AlpaServe: Statistical multiplexing with model parallelism for deep learning serving. In *Proceedings of the 17th USENIX Symposium on Operating Systems Design and Implementation (OSDI)*, 2023. [arXiv:2302.11665](https://arxiv.org/abs/2302.11665).

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
