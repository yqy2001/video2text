# video2text — agent instructions

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
