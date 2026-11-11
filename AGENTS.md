# video2text — agent instructions

## 存储位置（2026-11-05，后续修订）

用户已改为按用途安排存储。本仓库源码、`.git` 和必要运行环境放持久盘 `/opt/tiger/video2text`。
音视频、模型权重和转写材料继续放 HDFS；本项目默认材料目录为
`/mnt/hdfs/byte_data_seed_wl/projects/artifacts/video2text/`。
仓库的 `transcripts` 和 `artifacts` 通过链接访问 HDFS；凭据和必要运行状态留在持久盘。
旧 HDFS 仓库路径是指向本机代码的兼容链接，不作为新仓库位置。

## 用户的转写偏好

- 用户明确要求：以后转写都在沙箱外使用 GPU。通过执行工具的
  `sandbox_permissions="require_escalated"` 运行，遵守工具的权限审批机制。
- 转写时显式指定 `--device cuda --compute-type float16`；直接调用
  `WhisperModel` 时使用 `device="cuda", compute_type="float16"`。
- 不要因为沙箱内 `nvidia-smi` 失败或 CUDA 设备数为 0 就切换到 CPU；
  应在沙箱外检查 GPU。如果沙箱外仍不能使用 GPU，排查并告知用户，
  不要擅自降级为 CPU 转写。
- 2026-10-22 已验证：这台机器有 8 张 H20；沙箱内看不到 GPU 设备，
  沙箱外能识别全部 8 张卡，Whisper large-v3 的 CUDA FP16 推理成功。

项目使用方法与环境注意事项见 [CLAUDE.md](CLAUDE.md)。
