"""解析 RUM「页面性能详情」导出的 xlsx，统计慢渲染/慢加载。

用法: python parse_export.py <导出文件.xlsx>
输出: data rows / slow_render / slow_load（均为 >3000ms 统计，首屏耗时=B列、资源加载耗时=C列）
仅用标准库（zipfile + ElementTree），无需安装 openpyxl。
"""
import sys, zipfile
import xml.etree.ElementTree as ET

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
T = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def text(c):
    is_el = c.find('m:is', NS)
    if is_el is not None:
        return ''.join(x.text or '' for x in is_el.iter(T + 't'))
    v = c.find('m:v', NS)
    return v.text if v is not None else ''


def num(c):
    try:
        return float(text(c))
    except (TypeError, ValueError):
        return None


def main(path):
    z = zipfile.ZipFile(path)
    sheet = next(n for n in z.namelist() if n.startswith('xl/worksheets/sheet'))
    rows = ET.fromstring(z.read(sheet)).findall('.//m:sheetData/m:row', NS)
    headers = [text(c) for c in rows[0].findall('m:c', NS)]
    if len(headers) < 3 or '首屏耗时' not in headers[1] or '资源加载耗时' not in headers[2]:
        print('ERROR unexpected headers:', headers)
        sys.exit(1)
    n = sr = sl = 0
    for r in rows[1:]:
        cs = r.findall('m:c', NS)
        if len(cs) < 3:
            continue
        b, c2 = num(cs[1]), num(cs[2])
        if b is None:
            continue
        n += 1
        if b > 3000:
            sr += 1
        if c2 is not None and c2 > 3000:
            sl += 1
    print(f'total={n} slow_render={sr} slow_load={sl}')


if __name__ == '__main__':
    main(sys.argv[1])
