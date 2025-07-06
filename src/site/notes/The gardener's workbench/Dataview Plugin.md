---
{"dg-publish":true,"permalink":"/The gardener's workbench/Dataview Plugin/","noteIcon":""}
---



# 初次尝试

| File                                                                   | 创建日期                     |
| ---------------------------------------------------------------------- | ------------------------ |
| [[test\|test]]                                                      | 10:40 PM - July 06, 2025 |
| [[AI/Prompts\|Prompts]]                                             | 8:18 PM - July 06, 2025  |
| [[AI/AI音乐\|AI音乐]]                                                   | 8:17 PM - July 06, 2025  |
| [[The gardener's workbench/Obsidian笔记增加挂件美化\|Obsidian笔记增加挂件美化]]     | 6:08 PM - July 06, 2025  |
| [[The gardener's workbench/Dataview Plugin\|Dataview Plugin]]       | 5:00 PM - July 06, 2025  |
| [[AI/Stable diffusion/Stable diffusion原理解析\|Stable diffusion原理解析]]  | 4:36 PM - July 06, 2025  |
| [[后端/Python\|Python]]                                               | 1:09 PM - July 06, 2025  |
| [[后端/后端\|后端]]                                                       | 1:09 PM - July 06, 2025  |
| [[The gardener's workbench/Digital Garden插件应用\|Digital Garden插件应用]] | 12:59 PM - July 06, 2025 |
| [[AI/AI视频\|AI视频]]                                                   | 12:51 PM - July 06, 2025 |
| [[AI/大语言模型\|大语言模型]]                                                 | 12:50 PM - July 06, 2025 |
| [[投资理财/投资理财\|投资理财]]                                                 | 12:33 PM - July 06, 2025 |
| [[心理学/心理学\|心理学]]                                                    | 12:32 PM - July 06, 2025 |
| [[神秘学/神秘学\|神秘学]]                                                    | 12:32 PM - July 06, 2025 |
| [[读书笔记/哲学思考\|哲学思考]]                                                 | 12:32 PM - July 06, 2025 |
| [[读书笔记/读书笔记\|读书笔记]]                                                 | 12:32 PM - July 06, 2025 |
| [[AI/AIGC\|AIGC]]                                                   | 12:06 PM - July 06, 2025 |
| [[Welcome to Jancis's Space\|Welcome to Jancis's Space]]            | 11:57 AM - July 06, 2025 |

{ .block-language-dataview}


该代码可以罗列出每一篇笔记中的所有属性，方便新入手时直观理解。


# 属性
## 属性的数据类型
- 属性数据以Key Value形式定义；
- 数据类型支持：文本，列表，数字，复选框，日期，日期和时间

## 属性的存储方式
- 以yaml形式存储在文件的顶部 ，格式如下
```yaml
---
标题: 新的希望 # 文本类型
年份: 1977
喜爱: true
演员:  # 列表类型
  - 马克·哈米尔
  - 汉森·福特
  - 凯丽·费雪
链接: "[[链接]]" 
链接列表: 
	- "[[链接]]" 
	- "[[链接2]]"
年份: 1977 
圆周率: 3.14
是否喜欢: true 
是否回复: false 
是否可持续: # 这里没填任何值，将被视为false
日期: 2020-08-21 
时间: 2020-08-21T10:30:00
---
```

# 添加元数据到笔记中
- 添加到Properties文档属性中，元数据针对整个文件
- 添加到Inline Fields 行内字段中，元数据针对文件中某一段落

# 添加到整篇笔记的文档属性中
obsidian 1.4 版本以前，你只能手动在笔记最开头输入 --- 来添加文档属性，现在官方对这块区域做了优化，添加了几种更加方便的方式，并且加入了渲染来降低使用门槛，避免了手动输入时产生的各种语法错误。

- 快捷键法：ctrl + ;
- 打开命令面板（ctrl + p），搜索添加文档属性 (英文为 add file property)；
- 标签页标题栏的竖着的三个点：选择增加文档属性 (英文为 add file property)；
- 笔记开头输入 --- ；

# 添加某一区域的属性
- 语法格式： ```
key::value```

读完这本的感受：[feel:: 震撼]；
读完这本的感受：(feelisthewordafter:: test)；
