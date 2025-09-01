# -*- coding: utf-8 -*-
"""
清洗微博 CSV 中的发布时间列。
- 输入：/Users/zhangbowen/Downloads/MA Thesis BWZ/test twi:weibo/reddit_weibo/all/weiboDATAdebug/weibo cln vfin/微博搜索关键词采集.csv
- 目标：将“发布时间”清洗为标准时间戳（YYYY-MM-DD HH:MM:SS），结合每行“当前时间”推断年份及相对时间（今天/昨天/xx分钟前/xx小时前）。
- 输出：与输入同目录，文件名追加 _cleaned.csv。

命令行用法示例：
python clean_weibo_publish_time.py \
  --input "/Users/zhangbowen/Downloads/MA Thesis BWZ/test twi:weibo/reddit_weibo/all/weiboDATAdebug/weibo cln vfin/微博搜索关键词采集.csv"
"""

import argparse
import csv
import os
import re
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


def _normalize_time_str(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # 去除不可见字符与首尾空白
    s = s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    s = re.sub(r"\s+", " ", s).strip()
    # 统一 今天/昨天 与时间之间的空格
    s = re.sub(r"^(今天|昨天)(\d{1,2}:\d{2})$", r"\1 \2", s)
    # 去除“年/月/日”前后的多余空格
    s = s.replace("年 ", "年").replace(" 月", "月").replace("日 ", "日").replace(" 月 ", "月")
    # 常见中文标点替换
    s = s.replace("：", ":")
    return s


def _infer_year(month: int, base_dt: datetime) -> int:
    """根据抓取行的当前时间，推断只有月-日格式的年份。
    规则：
    - 默认取 base_dt.year；
    - 若出现跨年（例如当前 1-2 月而发布时间为 11-12 月），则回退一年；
    - 使用“间隔>=6个月”作为跨年阈值以增强鲁棒性。
    """
    year = base_dt.year
    if month > base_dt.month and (month - base_dt.month) >= 6:
        year -= 1
    return year


def _parse_relative_terms(s: str, base_dt: datetime) -> Optional[datetime]:
    # 今天 / 昨天
    m = re.match(r"^(今天|昨天)(?:\s*(\d{1,2}):(\d{2}))?$", s)
    if m:
        day_word = m.group(1)
        hh = int(m.group(2)) if m.group(2) else 0
        mm = int(m.group(3)) if m.group(3) else 0
        delta = 0 if day_word == "今天" else 1
        d = (base_dt - timedelta(days=delta)).replace(hour=hh, minute=mm, second=0, microsecond=0)
        return d

    # x分钟前
    m = re.match(r"^(\d{1,3})分钟前$", s)
    if m:
        mins = int(m.group(1))
        return (base_dt - timedelta(minutes=mins)).replace(second=0, microsecond=0)

    # x小时前
    m = re.match(r"^(\d{1,3})小时前$", s)
    if m:
        hrs = int(m.group(1))
        return (base_dt - timedelta(hours=hrs)).replace(second=0, microsecond=0)

    return None


def _parse_cn_datetime(s: str, base_dt: datetime) -> Optional[datetime]:
    """解析常见中文日期格式：
    - YYYY年M月D日[ HH:MM]
    - M月D日[ HH:MM]（年份从 base_dt 推断）
    - 今天/昨天、x分钟前、x小时前
    """
    s = _normalize_time_str(s)
    if not s:
        return None

    # 相对时间
    rel = _parse_relative_terms(s, base_dt)
    if rel is not None:
        return rel

    # 完整年月日
    m = re.match(r"^(\d{4})年(\d{1,2})月(\d{1,2})日(?:\s*(\d{1,2}):(\d{2}))?$", s)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        hh = int(m.group(4)) if m.group(4) else 0
        mm = int(m.group(5)) if m.group(5) else 0
        try:
            return datetime(year, month, day, hh, mm)
        except ValueError:
            return None

    # 仅月日
    m = re.match(r"^(\d{1,2})月(\d{1,2})日(?:\s*(\d{1,2}):(\d{2}))?$", s)
    if m:
        month = int(m.group(1))
        day = int(m.group(2))
        hh = int(m.group(3)) if m.group(3) else 0
        mm = int(m.group(4)) if m.group(4) else 0
        year = _infer_year(month, base_dt)
        try:
            return datetime(year, month, day, hh, mm)
        except ValueError:
            return None

    # 回退：尝试直接由 pandas 解析（能解析如 2025-06-16 21:57）
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            # 转换为 python datetime
            return dt.to_pydatetime()
    except Exception:
        pass

    return None


def clean_publish_time(df: pd.DataFrame) -> pd.DataFrame:
    pub_col = "发布时间"
    now_col = "当前时间"
    if pub_col not in df.columns:
        raise KeyError(f"CSV 中不存在列：{pub_col}")
    if now_col not in df.columns:
        raise KeyError(f"CSV 中不存在列：{now_col}")

    # 行级当前时间解析（用于推断相对时间与年份）
    df["_当前时间_dt"] = pd.to_datetime(df[now_col], errors="coerce")

    cleaned_values = []
    success_flags = []

    for idx, row in df.iterrows():
        base_dt = row["_当前时间_dt"]
        if pd.isna(base_dt):
            base_dt = pd.Timestamp.now()
        base_dt = pd.to_datetime(base_dt).to_pydatetime()

        raw = row.get(pub_col, None)
        dt = _parse_cn_datetime(raw, base_dt)
        if dt is None:
            cleaned_values.append(pd.NaT)
            success_flags.append(False)
        else:
            cleaned_values.append(pd.Timestamp(dt))
            success_flags.append(True)

    df["发布时间_clean_dt"] = cleaned_values
    df["发布时间_clean"] = df["发布时间_clean_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["发布时间_clean_success"] = success_flags

    return df


def main():
    parser = argparse.ArgumentParser(description="清洗微博 CSV 中的发布时间列")
    parser.add_argument(
        "--input",
        default="/Users/zhangbowen/Downloads/MA Thesis BWZ/test twi:weibo/reddit_weibo/all/weiboDATAdebug/weibo cln vfin/微博搜索关键词采集.csv",
        help="输入 CSV 路径（包含“发布时间”和“当前时间”列）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 CSV 路径（默认：与输入同目录，文件名追加 _cleaned.csv）",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV 编码（默认 utf-8-sig）",
    )

    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"未找到输入文件：{input_path}")

    # 生成默认输出路径
    if args.output:
        output_path = args.output
    else:
        d, b = os.path.split(input_path)
        name, ext = os.path.splitext(b)
        output_path = os.path.join(d, f"{name}_cleaned{ext}")

    # 读取（考虑到文本字段中含换行，使用 python 引擎）
    df = pd.read_csv(
        input_path,
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
        encoding=args.encoding,
        dtype=str,
        keep_default_na=False,
    )

    # 清洗
    df_clean = clean_publish_time(df)

    # 统计
    total = len(df_clean)
    success = int(df_clean["发布时间_clean_success"].sum())
    failed = total - success

    # 保存
    df_clean.to_csv(output_path, index=False, encoding=args.encoding)

    print("清洗完成：")
    print(f"  输入：{input_path}")
    print(f"  输出：{output_path}")
    print(f"  总行数：{total}")
    print(f"  解析成功：{success}")
    print(f"  解析失败：{failed}")


if __name__ == "__main__":
    main()