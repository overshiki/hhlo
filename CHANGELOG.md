# Revision history for hhlo

## 0.1.0.0 -- 2026-04-22

* Initial release.
* Type-safe EDSL for StableHLO with 50+ ops.
* CPU execution via PJRT CPU plugin.
* GPU execution via PJRT CUDA plugin with device enumeration and selection.
* Multi-GPU concurrent inference scaling via `executeReplicas`.
* 115 CPU tests + 6 GPU integration tests.
* 29 executable examples including ResNet-18, AlexNet, Transformer, and UNet.
