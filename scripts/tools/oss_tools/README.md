# OSS Tools

本目录封装本地 `rclone`，用于访问 OSS。

## 文件

- `client.py`：Python API，提供 `OssClient.upload_file()`、`download_file()`、`exists()`、`delete_file()`。
- `cli.py`：命令行入口，使用 `python -m scripts.tools.oss_tools.cli ...`。
- `bin/rclone`：`rclone v1.71.0` Linux amd64 二进制，随工具目录保存。
- `rclone.conf`：本机 OSS 配置文件，已被 `.gitignore` 忽略；不要提交或写入文档。

## 配置优先级

`OssClient()` 默认按以下顺序找依赖：

1. 显式传入的 `rclone_bin` / `rclone_config`。
2. 环境变量 `OSS_TOOLS_RCLONE_BIN` / `OSS_TOOLS_RCLONE_CONFIG`。
3. 本目录的 `bin/rclone` / `rclone.conf`。

## 示例

```bash
python -m scripts.tools.oss_tools.cli exists qz_oss2:public/path/file.txt
python -m scripts.tools.oss_tools.cli upload local.txt qz_oss2:public/path/local.txt
python -m scripts.tools.oss_tools.cli download qz_oss2:public/path/local.txt /tmp/local.txt
python -m scripts.tools.oss_tools.cli delete qz_oss2:public/path/local.txt --missing-ok
```
