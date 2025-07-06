---
{"dg-publish":true,"permalink":"/The gardener's workbench/Dataview Plugin/","noteIcon":""}
---



# 初次尝试

该代码可以罗列出每一篇笔记中的所有属性，方便新入手时直观理解。

```code
TABLE this
WHERE file = this.file
```


列出最近三天创建的文档

```code
Table file.ctime as "创建日期"
Where date(today) - file.ctime <= dur(3 day)
Sort file.ctime desc
```





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
