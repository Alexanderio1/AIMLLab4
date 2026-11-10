"""Build the Russian 49-question defense guide with embedded Cyrillic fonts."""
from __future__ import annotations

import argparse
import hashlib
import html
import re
from pathlib import Path

from reportlab import __version__ as reportlab_version
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    PageBreak, Preformatted, LongTable, Table, TableStyle, KeepTogether,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
INK = colors.HexColor('#193654')
BLUE = colors.HexColor('#176e9d')
GRAY = colors.HexColor('#536374')
LIGHT = colors.HexColor('#eef3f8')
BORDER = colors.HexColor('#ced8e2')
PAGE_W, PAGE_H = A4
LEFT = RIGHT = 19 * mm
TOP = 19 * mm
BOTTOM = 19 * mm
WIDTH = PAGE_W - LEFT - RIGHT


def register_fonts():
    fonts = {
        'Main': 'DejaVuSans.ttf', 'Main-Bold': 'DejaVuSans-Bold.ttf',
        'Main-Italic': 'DejaVuSans-Oblique.ttf', 'Main-BoldItalic': 'DejaVuSans-BoldOblique.ttf',
        'Mono': 'DejaVuSansMono.ttf', 'Mono-Bold': 'DejaVuSansMono-Bold.ttf',
    }
    for name, filename in fonts.items():
        path = FONT_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f'{path}; install fonts-dejavu-core or change FONT_DIR')
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily('Main', normal='Main', bold='Main-Bold', italic='Main-Italic', boldItalic='Main-BoldItalic')
    pdfmetrics.registerFontFamily('Mono', normal='Mono', bold='Mono-Bold', italic='Mono', boldItalic='Mono-Bold')


def inline(text):
    """Render only the Markdown inline forms used in the reviewed source."""
    pattern = re.compile(r'`[^`]+`|\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*')
    result = []
    pos = 0
    for match in pattern.finditer(text):
        result.append(html.escape(text[pos:match.start()]))
        token = match.group()
        if token.startswith('`'):
            result.append('<font name="Mono" size="8.5">' + html.escape(token[1:-1]) + '</font>')
        elif token.startswith('**'):
            result.append('<b>' + inline(token[2:-2]) + '</b>')
        else:
            label, href = re.match(r'\[([^\]]+)\]\(([^)]+)\)', token).groups()
            result.append('<link href="' + html.escape(href, quote=True) + '" color="#176e9d"><u>' + inline(label) + '</u></link>')
        pos = match.end()
    result.append(html.escape(text[pos:]))
    return ''.join(result)


class GuideDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(filename, pagesize=A4, leftMargin=LEFT, rightMargin=RIGHT,
                         topMargin=TOP, bottomMargin=BOTTOM,
                         title='Ответы к защите лабораторной № 4. MLOps. Вариант 17',
                         author='', subject='29 практических и 20 теоретических вопросов; проект CCPP')
        frame = Frame(LEFT, BOTTOM, WIDTH, PAGE_H - TOP - BOTTOM, leftPadding=0,
                      rightPadding=0, topPadding=0, bottomPadding=0, id='main')
        self.addPageTemplates(PageTemplate(id='guide', frames=[frame], onPage=self.page_chrome))

    def page_chrome(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(BORDER)
            canvas.setLineWidth(.5)
            canvas.line(LEFT, PAGE_H - 13 * mm, PAGE_W - RIGHT, PAGE_H - 13 * mm)
            canvas.setFont('Main', 8)
            canvas.setFillColor(GRAY)
            canvas.drawString(LEFT, PAGE_H - 10.5 * mm, 'MLOps · CCPP · Вариант 17')
            canvas.drawRightString(PAGE_W - RIGHT, PAGE_H - 10.5 * mm, 'Ответы к защите')
        canvas.setFillColor(GRAY)
        canvas.setFont('Main', 8)
        canvas.drawString(LEFT, 11 * mm, 'Практика: П1–П29   ·   Теория: Т1–Т20')
        canvas.drawRightString(PAGE_W - RIGHT, 11 * mm, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        info = getattr(flowable, '_toc_info', None)
        if info:
            level, title, key = info
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, level=level, closed=(level == 0))
            self.notify('TOCEntry', (level, html.escape(title), self.page, key))


def styles():
    base = dict(fontName='Main', fontSize=10.0, leading=14.5, textColor=colors.HexColor('#18232d'),
                spaceAfter=7, allowWidows=0, allowOrphans=0, splitLongWords=True)
    return {
        'body': ParagraphStyle('body', **base),
        'list': ParagraphStyle('list', **{**base, 'leftIndent': 12, 'firstLineIndent': -8, 'spaceAfter': 4}),
        'h1': ParagraphStyle('h1', fontName='Main-Bold', fontSize=17, leading=23,
                             textColor=INK, spaceAfter=15, keepWithNext=True),
        'h2': ParagraphStyle('h2', fontName='Main-Bold', fontSize=12, leading=17,
                             textColor=INK, spaceBefore=13, spaceAfter=8, keepWithNext=True),
        'table': ParagraphStyle('table', fontName='Main', fontSize=8.6, leading=12.3,
                                textColor=colors.HexColor('#18232d'), spaceAfter=0, splitLongWords=True),
        'tablehead': ParagraphStyle('tablehead', fontName='Main-Bold', fontSize=8.8, leading=12,
                                    textColor=colors.white, spaceAfter=0),
        'code': ParagraphStyle('code', fontName='Mono', fontSize=8.0, leading=10.4,
                               textColor=colors.HexColor('#153044'), backColor=LIGHT,
                               borderColor=BORDER, borderWidth=.4, borderPadding=7,
                               leftIndent=7, rightIndent=7, spaceBefore=4, spaceAfter=11),
    }


def heading(text, level, sty):
    p = Paragraph(inline(text), sty)
    key = 's' + hashlib.sha256(text.encode()).hexdigest()[:18]
    p._toc_info = (level, text, key)
    return p


def make_table(lines, st):
    rows = [[value.strip() for value in line.strip().strip('|').split('|')] for line in lines]
    rows.pop(1)  # Markdown separator
    n = len(rows[0])
    proportions = {2: [.31, .69], 3: [.24, .39, .37], 4: [.22, .18, .40, .20]}.get(n, [1 / n] * n)
    content = [[Paragraph(inline(value), st['tablehead'] if i == 0 else st['table']) for value in row] for i, row in enumerate(rows)]
    t = LongTable(content, colWidths=[WIDTH * p for p in proportions], repeatRows=1,
                  hAlign='LEFT', spaceBefore=5, spaceAfter=12)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), INK),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f6f9')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LINEBELOW', (0, 0), (-1, 0), .7, INK),
        ('LINEBELOW', (0, 1), (-1, -1), .3, BORDER),
    ]))
    return t


def parse_markdown(text, st):
    lines = text.splitlines()
    story = [heading('Вводные сведения и результаты проекта', 0, st['h1'])]
    index = 0
    first_title = True
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith('```'):
            index += 1
            block = []
            while index < len(lines) and not lines[index].startswith('```'):
                block.append(lines[index])
                index += 1
            if index == len(lines):
                raise ValueError('Unclosed code fence')
            text_block = '\n'.join(block)
            pre = Preformatted(text_block, st['code'], maxLineLength=94, splitChars=' ', newLineChars='')
            # Small examples stay together; longer examples can continue to a new page.
            if len(block) <= 13 and max((len(x) for x in block), default=0) < 140:
                story.append(KeepTogether([pre]))
            else:
                story.append(pre)
            index += 1
            continue
        if line.startswith('# '):
            if first_title:
                first_title = False
            else:
                story.append(PageBreak())
                story.append(heading(line[2:].strip(), 0, st['h1']))
            index += 1
            continue
        if line.startswith('## '):
            story.append(heading(line[3:].strip(), 1, st['h2']))
            index += 1
            continue
        if line.startswith('|'):
            block = []
            while index < len(lines) and lines[index].startswith('|'):
                block.append(lines[index])
                index += 1
            story.append(make_table(block, st))
            continue
        list_match = re.match(r'^(-|\d+\.)\s+(.+)', line)
        if list_match:
            marker, text_part = list_match.groups()
            index += 1
            while index < len(lines) and lines[index].strip() and not re.match(r'^(-|\d+\.)\s+', lines[index]) and not lines[index].startswith(('#', '```', '|')):
                text_part += ' ' + lines[index].strip()
                index += 1
            story.append(Paragraph(('• ' if marker == '-' else marker + ' ') + inline(text_part), st['list']))
            continue
        block = [line.strip()]
        index += 1
        while index < len(lines) and lines[index].strip() and not lines[index].startswith(('#', '```', '|')) and not re.match(r'^(-|\d+\.)\s+', lines[index]):
            block.append(lines[index].strip())
            index += 1
        story.append(Paragraph(inline(' '.join(block)), st['body']))
    return story


def build(source, output):
    register_fonts()
    text = source.read_text()
    for prefix, count in [('П', 29), ('Т', 20)]:
        numbers = [int(x) for x in re.findall(r'^## ' + prefix + r'(\d+)\.', text, re.M)]
        if numbers != list(range(1, count + 1)):
            raise ValueError(f'Missing or reordered questions: {prefix}')
    st = styles()
    story = []
    story.append(Spacer(1, 30 * mm))
    story.append(Paragraph('ЛАБОРАТОРНАЯ РАБОТА № 4', ParagraphStyle('label', fontName='Main-Bold', fontSize=10, leading=15, textColor=BLUE)))
    story.append(Spacer(1, 9 * mm))
    story.append(Paragraph('Ответы к защите', ParagraphStyle('cover', fontName='Main-Bold', fontSize=30, leading=38, textColor=INK)))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph('Основы MLOps', ParagraphStyle('subtitle', fontName='Main', fontSize=19, leading=26, textColor=INK)))
    story.append(Spacer(1, 13 * mm))
    story.append(Paragraph('Вариант 17<br/>Прогноз выработки электростанции<br/>Combined Cycle Power Plant', ParagraphStyle('topic', fontName='Main', fontSize=14, leading=21, textColor=GRAY)))
    story.append(Spacer(1, 15 * mm))
    count_style = ParagraphStyle('counts', fontName='Main-Bold', fontSize=12, leading=20, textColor=colors.white, alignment=TA_CENTER)
    counts = Table([[Paragraph('29 вопросов<br/>по отчёту', count_style), Paragraph('20 вопросов<br/>по теории', count_style)]], colWidths=[WIDTH / 2] * 2)
    counts.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), INK), ('TOPPADDING', (0, 0), (-1, -1), 14), ('BOTTOMPADDING', (0, 0), (-1, -1), 14)]))
    story.append(counts)
    story.append(Spacer(1, 14 * mm))
    story.append(Paragraph('Команды Ubuntu, объяснения кода, фактические метрики, MLflow Model Registry, API, Docker, Compose и откат.', st['body']))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(inline('[Alexanderio1/AIMLLab4](https://github.com/Alexanderio1/AIMLLab4)'), st['body']))
    story.append(Paragraph('Подготовлено 11 ноября 2026 года', st['body']))
    story.append(PageBreak())
    story.append(Paragraph('Содержание', st['h1']))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle('toc0', fontName='Main-Bold', fontSize=9.2, leading=13, textColor=INK, spaceBefore=8, leftIndent=0, firstLineIndent=0),
        ParagraphStyle('toc1', fontName='Main', fontSize=8.6, leading=11.6, textColor=GRAY, spaceBefore=4, leftIndent=12, firstLineIndent=0),
    ]
    story.append(toc)
    story.append(PageBreak())
    story.extend(parse_markdown(text, st))
    doc = GuideDoc(str(output))
    doc.multiBuild(story)
    print(f'Built {output} ({output.stat().st_size:,} bytes), ReportLab {reportlab_version}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'answers_lab4.md')
    parser.add_argument('--output', type=Path, default=ROOT / 'answers_lab4.pdf')
    arguments = parser.parse_args()
    build(arguments.source, arguments.output)
