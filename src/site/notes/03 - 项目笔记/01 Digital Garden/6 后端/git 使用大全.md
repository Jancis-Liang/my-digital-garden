---
{"dg-publish":true,"permalink":"/03 - 项目笔记/01 Digital Garden/6 后端/git 使用大全/","noteIcon":""}
---


# 历史背景

Git 是一款免费开源的分布式版本管理系统，由 Linus Torvalds 于 2005 年用了 10 天时间以 C 语言开发（amazing~），最初用于 Linux 内核开发的版本控制

[官方介绍](https://git-scm.com/book/zh/v2/%E8%B5%B7%E6%AD%A5-Git-%E7%AE%80%E5%8F%B2)

# 仓库架构图

![](https://jancis-1361410855.cos.ap-beijing.myqcloud.com/ObsidianImage/20250827181736285.png)

## 名词解释

Workspace: 工作区
Index/Stage: 暂存区
Repository: 仓库区(或本地仓库)
Remote: 远程仓库

# git 的鉴权方式

## ssh 与 https 鉴权方式

> [!info]

> [! Info]
> 有时候防火墙会限制 ssh 22 端口，可以通过使用 https 的端口使用 ssh，[配置文档](https://docs.github.com/en/authentication/troubleshooting-ssh/using-ssh-over-the-https-port)

git push 时出现以下窗口，证明走了 HTTP 的 PAT 认证方式。

![](https://jancis-1361410855.cos.ap-beijing.myqcloud.com/ObsidianImage/20250827181501879.png)

![](https://jancis-1361410855.cos.ap-beijing.myqcloud.com/ObsidianImage/20250827180910921.png)

> [!tip]
> 如果仓库的格式是 https://github.com/.. ，则通过 HTTP 的 Personal Access Token 进行鉴权，如果是 git@github.com:你的用户名/仓库名.git 格式的，则通过 ssh 鉴权。切记！

## ssh 鉴权

> [!info]
> 本机生成 SSH 密钥对，将公钥粘贴到 github、gitlab 等的 SSH Keys 设置中

```BASH
# 1. **生成SSH密钥对**（如已存在可跳过）

ssh-keygen -t ed25519 -C "your_email@example.com"

# 按Enter使用默认路径C:\Users\用户名\.ssh\id_ed25519

# 2. 将**公钥**添加到剪贴板（Windows）

get-content %userprofile%\.ssh\id_ed25519.pub | clip

# 3. 将剪贴板内容粘贴到GitHub/GitLab的SSH Keys设置中

# (https://github.com/settings/keys)

# 4. 测试连接

ssh -T git@github.com

# 5. 将远程仓库URL改为SSH格式

git remote set-url origin git@github.com:用户名/仓库名.git
```

## HTTPS+PAT 方式

> [!info]
> 使用 Windows 凭据管理器存储凭证，git 自动调用凭据管理器获取用户名密码。每次 github、gitlab 改密码，这边还得跟着改，麻烦。推荐 SSH

```BASH
# 配置Windows凭据管理器存储凭证

git config --global credential.helper wincred

# 或者使用缓存方式（临时存储1小时）

git config --global credential.helper "cache --timeout=3600"

# 确保使用HTTPS方式克隆（如果已克隆需先更新URL）

git remote set-url origin https://github.com/用户名/仓库名.git
```

注意事项：

- ⚠️ 不要在.gitconfig 中直接写密码，Git 设计上**不支持明文存储密码**
- **SSH 方式更安全**，私钥应严格保密，建议设置密码保护（passphrase）
- 首次使用 HTTPS 方式时仍需输入一次密码，之后会自动存储
- 企业环境可能需要配置公司统一认证系统（如 GitLab SSO）
- 修改远程 URL 后，所有协作者都需要更新本地仓库配置
- 🔍 **SSH 密钥默认位置**：`C:\Users\<用户名>\.ssh\`
- 🔍 Windows 凭据存储位置：控制面板 > 用户账户 > 凭据管理器 > 普通凭据（搜索"git:"）
- ⚠️ 如果测试 SSH 连接时提示"Permission denied (publickey)"，说明公钥未正确添加到远程平台
- ⚠️ 凭证管理器中的 Git 凭据可能标记为"git:[https://github.com"或类似格式](https://github.com%22%E6%88%96%E7%B1%BB%E4%BC%BC%E6%A0%BC%E5%BC%8F)
- ⚠️ 某些公司网络可能需要额外配置 SSH 代理或使用特定端口

原理说明： Git 提供两种主要认证方式：

1. SSH 认证：使用**非对称加密**，本地私钥签名验证，无需每次传输密码
2. HTTPS 凭证存储：通过 credential.helper 调用系统安全存储（Windows 凭据管理器）缓存凭证

### 验证 Git 免密配置主要通过三个维度：

- SSH 密钥验证：检查是否存在私钥文件(.ssh 目录)，并通过 ssh -T 测试实际连接
- 凭证存储验证：检查 credential.helper 配置及系统凭据管理器中的存储记录
- 实际操作验证：执行 git push --dry-run 进行模拟推送测试

github 配置详见：[[03 - 项目笔记/01 Digital Garden/6 后端/github#鉴权机制\|03 - 项目笔记/01 Digital Garden/6 后端/github#鉴权机制]]

# 主要命令

https://git-scm.com/docs

https://jason-effi-lab.notion.site/Obsidian-20698ac9981180229066ff67342e8232?p=24198ac998118031ab4df603ea094964&pm=c

# 衍生工具

## 常用命令

### 配置

```

```

# 官方指导书

![[03 - 项目笔记/01 Digital Garden/6 后端/progit.epub]]
