#!/bin/bash
# 完全清空环境，只保留最小必要变量
# 解决 exec tool 继承 OpenClaw 环境导致网络变慢的问题
cd "$(dirname "$0")"

export HOME="${HOME:-/home/fandy}"
export USER="${USER:-fandy}"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

exec env -i \
    HOME="$HOME" \
    USER="$USER" \
    PATH="$PATH" \
    SHELL="/bin/bash" \
    bash "$@"
