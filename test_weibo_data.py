#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

try:
    df = pd.read_csv('../微博搜索关键词采集.csv')
    print(f'数据行数: {len(df)}')
    print(f'列名: {list(df.columns)}')
    print(f'前5行:')
    print(df.head())
    
    if '发布时间' in df.columns:
        print(f'\n时间列示例:')
        print(df['发布时间'].head())
        
        # 尝试转换时间
        df['date'] = pd.to_datetime(df['发布时间'], errors='coerce')
        print(f'\n转换后的时间:')
        print(df['date'].head())
        print(f'时间范围: {df["date"].min()} 到 {df["date"].max()}')
    
except Exception as e:
    print(f'错误: {e}')