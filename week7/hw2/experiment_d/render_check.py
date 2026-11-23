"""用 Chrome 無介面模式驗證審核頁可渲染及通過案例預設收合。"""
import subprocess
from html.parser import HTMLParser
from classifier.common import ROOT


class Cards(HTMLParser):
    """以 HTML parser 檢查，不依賴屬性排列或引號格式。"""
    def __init__(self):
        super().__init__();self.total=0;self.hidden=0

    def handle_starttag(self,tag,attributes):
        if tag=='article':
            self.total+=1
            attrs=dict(attributes)
            if attrs.get('class')=='ok' and 'hidden' in attrs:self.hidden+=1


if __name__=='__main__':
    output=ROOT/'hw2/results/experiment_d/test'
    rendered=subprocess.run(['C:/Program Files/Google/Chrome/Application/chrome.exe',
        '--headless','--disable-gpu','--dump-dom',(output/'report.html').as_uri()],
        capture_output=True,timeout=30,check=True)
    parser=Cards();parser.feed(rendered.stdout.decode('utf-8'))
    assert parser.total==210 and parser.hidden==169
    (output/'report_render_check.txt').write_text(
        'Headless Chrome: 210 articles rendered; 169 passing cases hidden by default; 41 cases shown for review.\n',encoding='utf-8')
    print('Chrome rendering passed')
