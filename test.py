import sys
import markdown
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineProfile, QWebEnginePage


def display_markdown(filename):
    app = QApplication(sys.argv)

    # 创建QWebEngineView
    view = QWebEngineView()

    # 创建QWebEngineProfile
    profile = QWebEngineProfile.defaultProfile()

    # 创建QWebEnginePage并将其设置为view的页面
    page = QWebEnginePage(profile, view)
    view.setPage(page)

    # 读取Markdown文件内容
    with open(filename, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    # 将Markdown转换为HTML
    html = markdown.markdown(markdown_text)

    # 加载HTML内容
    view.setHtml(html)

    # 显示窗口
    view.show()

    sys.exit(app.exec_())
# 替换为你的Markdown文件路径
markdown_file = "E:/Blog/mark down学习.md"
display_markdown(markdown_file)
