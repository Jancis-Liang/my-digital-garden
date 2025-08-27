---
{"dg-publish":true,"permalink":"/6 后端/git 使用大全/","noteIcon":""}
---


# 历史背景

Git是一款免费开源的分布式版本管理系统，由Linus Torvalds于2005年用了10天时间以C语言开发（amazing~），最初用于Linux内核开发的版本控制

[官方介绍](https://git-scm.com/book/zh/v2/%E8%B5%B7%E6%AD%A5-Git-%E7%AE%80%E5%8F%B2)

# 仓库架构图

![Pasted image 20250527191042.png](/img/user/6%20%E5%90%8E%E7%AB%AF/Pasted%20image%2020250527191042.png)
## 名词解释

Workspace: 工作区
Index/Stage: 暂存区
Repository: 仓库区(或本地仓库)
Remote:  远程仓库

# git的鉴权方式

git push时出现以下窗口，证明走了HTTP的PAT认证方式。

![6 后端/Pasted image 20250819002044.png](/img/user/6%20%E5%90%8E%E7%AB%AF/Pasted%20image%2020250819002044.png)

![6 后端/Pasted image 20250819002118.png](/img/user/6%20%E5%90%8E%E7%AB%AF/Pasted%20image%2020250819002118.png)

> [!tip]
> 如果仓库的格式是 https://github.com/.. ，则通过HTTP的Personal Access Token进行鉴权，如果是 git@github.com:你的用户名/仓库名.git 格式的，则通过ssh鉴权。切记！

## ssh鉴权

> [!info]
> 本机生成SSH密钥对，将公钥粘贴到github、gitlab等的SSH Keys设置中

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

## HTTPS+PAT方式

> [!info]
> 使用Windows凭据管理器存储凭证，git自动调用凭据管理器获取用户名密码。每次github、gitlab改密码，这边还得跟着改，麻烦。推荐SSH

```BASH
# 配置Windows凭据管理器存储凭证

git config --global credential.helper wincred

# 或者使用缓存方式（临时存储1小时）

git config --global credential.helper "cache --timeout=3600"

# 确保使用HTTPS方式克隆（如果已克隆需先更新URL）

git remote set-url origin https://github.com/用户名/仓库名.git
```

注意事项：
- ⚠️ 不要在.gitconfig中直接写密码，Git设计上**不支持明文存储密码**
- **SSH方式更安全**，私钥应严格保密，建议设置密码保护（passphrase）
- 首次使用HTTPS方式时仍需输入一次密码，之后会自动存储
- 企业环境可能需要配置公司统一认证系统（如GitLab SSO）
- 修改远程URL后，所有协作者都需要更新本地仓库配置
- 🔍 **SSH密钥默认位置**：`C:\Users\<用户名>\.ssh\`
- 🔍 Windows凭据存储位置：控制面板 > 用户账户 > 凭据管理器 > 普通凭据（搜索"git:"）
- ⚠️ 如果测试SSH连接时提示"Permission denied (publickey)"，说明公钥未正确添加到远程平台
- ⚠️ 凭证管理器中的Git凭据可能标记为"git:[https://github.com"或类似格式](https://github.com%22%E6%88%96%E7%B1%BB%E4%BC%BC%E6%A0%BC%E5%BC%8F)
- ⚠️ 某些公司网络可能需要额外配置SSH代理或使用特定端口

原理说明： Git提供两种主要认证方式：
1. SSH认证：使用**非对称加密**，本地私钥签名验证，无需每次传输密码
2. HTTPS凭证存储：通过credential.helper调用系统安全存储（Windows凭据管理器）缓存凭证

### 验证Git免密配置主要通过三个维度：

- SSH密钥验证：检查是否存在私钥文件(.ssh目录)，并通过ssh -T测试实际连接
- 凭证存储验证：检查credential.helper配置及系统凭据管理器中的存储记录
- 实际操作验证：执行git push --dry-run进行模拟推送测试

github配置详见：[[6 后端/github#鉴权机制\|github#鉴权机制]]



# 主要命令

## 


https://jason-effi-lab.notion.site/Obsidian-20698ac9981180229066ff67342e8232?p=24198ac998118031ab4df603ea094964&pm=c

# 衍生工具





## 常用命令

### 配置

```

```


# 官方指导书
![[6 后端/progit.epub]]