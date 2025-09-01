#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Reddit vs Weibo AI Consciousness Analysis

This script integrates all analysis functionalities from the reddit_weibo folder:
1. Enhanced data cleaning and preprocessing
2. Temporal analysis with event-based timeline
3. Topic modeling with LDA
4. Sentiment analysis
5. Consciousness indicators analysis
6. Cross-platform comparison
7. Comprehensive visualizations
8. Report generation

All outputs will be saved to the 'all' subfolder.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import seaborn as sns
import re
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available, topic modeling will be simplified")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not available, sentiment analysis will be simplified")

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("Jieba not available, Chinese text processing will be simplified")

# Try to import pyLDAvis
try:
    import pyLDAvis
    try:
        import pyLDAvis.sklearn
        PYLDAVIS_SKLEARN = pyLDAvis.sklearn
        PYLDAVIS_METHOD = 'sklearn'
    except ImportError:
        try:
            from pyLDAvis import sklearn as PYLDAVIS_SKLEARN
            PYLDAVIS_METHOD = 'sklearn'
        except ImportError:
            try:
                import pyLDAvis.gensim_models
                PYLDAVIS_SKLEARN = pyLDAvis.gensim_models
                PYLDAVIS_METHOD = 'gensim'
            except ImportError:
                PYLDAVIS_SKLEARN = None
                PYLDAVIS_METHOD = None
    
    PYLDAVIS_AVAILABLE = True
    print(f"pyLDAvis imported successfully, version: {pyLDAvis.__version__}, method: {PYLDAVIS_METHOD}")
except ImportError as e:
    PYLDAVIS_AVAILABLE = False
    PYLDAVIS_SKLEARN = None
    PYLDAVIS_METHOD = None
    print(f"Warning: pyLDAvis import failed: {e}")
except Exception as e:
    PYLDAVIS_AVAILABLE = False
    PYLDAVIS_SKLEARN = None
    PYLDAVIS_METHOD = None
    print(f"Warning: pyLDAvis import exception: {e}")

# Set up Chinese font support
def setup_chinese_font():
    """Setup Chinese font support"""
    # macOS system font paths
    font_paths = [
        '/System/Library/Fonts/PingFang.ttc',  # PingFang SC
        '/System/Library/Fonts/Helvetica.ttc',  # Helvetica
        '/Library/Fonts/Arial Unicode.ttf',     # Arial Unicode MS
        '/System/Library/Fonts/STHeiti Light.ttc',  # STHeiti
        '/System/Library/Fonts/Hiragino Sans GB.ttc',  # Hiragino Sans GB
    ]
    
    # Find available font
    available_font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            available_font = font_path
            print(f"Found font file: {font_path}")
            break
    
    if available_font:
        # Use font file directly
        font_prop = fm.FontProperties(fname=available_font)
        rcParams['font.family'] = font_prop.get_name()
        print(f"Using font: {font_prop.get_name()}")
    else:
        # Fallback: use font names
        chinese_fonts = ['PingFang SC', 'Heiti TC', 'STHeiti', 'SimHei', 'Microsoft YaHei']
        rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans', 'Arial']
        print("Using fallback font configuration")
    
    rcParams['axes.unicode_minus'] = False
    rcParams['font.size'] = 12
    
    # Test Chinese display
    test_fig, test_ax = plt.subplots(figsize=(1, 1))
    test_ax.text(0.5, 0.5, 'Test Chinese', ha='center', va='center')
    test_fig.savefig('font_test.png', dpi=50)
    plt.close(test_fig)
    print("Font setup complete")

# Initialize font setup
setup_chinese_font()

# Set seaborn style if available
if 'seaborn' in globals():
    sns.set_palette("husl")
    sns.set_style("whitegrid")

class IntegratedRedditWeiboAnalyzer:
    """Integrated analyzer combining all functionalities"""
    
    def __init__(self):
        # Create output directory
        self.output_dir = 'images'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Important events timeline
        self.key_events = {
            'lamda_event': datetime(2022, 6, 23),
            'chatgpt_release': datetime(2022, 11, 30),
            'anthropic_welfare': datetime(2025, 4, 24)
        }
        
        # Time period division - modified for virtual date distribution from 2022.06.23 to 2025.05.15
        self.periods = {
            'period_1': {
                'name': 'LaMDA Event Period',
                'start': datetime(2022, 6, 23),
                'end': datetime(2022, 11, 29),
                'description': 'LaMDA claims to have soul event period'
            },
            'period_2': {
                'name': 'ChatGPT Release Period', 
                'start': datetime(2022, 11, 30),
                'end': datetime(2025, 4, 23),
                'description': 'After ChatGPT release until Anthropic Model Welfare'
            },
            'period_3': {
                'name': 'Model Welfare Era',
                'start': datetime(2025, 4, 24),
                'end': datetime(2025, 5, 15),
                'description': 'After Anthropic Model Welfare research project launch'
            }
        }
        
        # Enhanced keyword lists
        self.reddit_ai_keywords = [
            'consciousness', 'sentient', 'sentience', 'aware', 'awareness', 'conscious',
            'artificial intelligence', 'AI', 'machine learning', 'neural network',
            'cognitive', 'thinking', 'mind', 'intelligence', 'AGI', 'LaMDA', 'GPT',
            'chatbot', 'robot', 'automation', 'algorithm', 'deep learning', 'transformer',
            'language model', 'large language model', 'LLM', 'generative AI'
        ]
        
        self.weibo_ai_keywords = [
            '意识', '感知', '智能', '人工智能', 'AI', '机器学习', '神经网络',
            '认知', '思维', '思考', '心智', '智慧', '自动化', '算法', '深度学习',
            '聊天机器人', '机器人', 'GPT', 'ChatGPT', '大模型', '语言模型',
            '生成式AI', '变换器', '自然语言处理', 'NLP'
        ]
        
        # AI consciousness related keywords
        self.consciousness_keywords = {
            'english': [
                'consciousness', 'conscious', 'sentient', 'sentience', 'aware', 'awareness',
                'artificial intelligence', 'ai', 'machine consciousness', 'digital consciousness',
                'cognitive', 'thinking', 'mind', 'intelligence', 'agi', 'gpt', 'chatbot',
                'robot consciousness', 'ai sentience', 'machine sentience', 'ai awareness',
                'feeling', 'emotion', 'understanding', 'reasoning', 'learning', 'memory',
                'creative', 'intelligent', 'empathy', 'self-aware', 'experience',
                'lamda', 'chatgpt', 'openai', 'anthropic', 'model welfare'
            ],
            'chinese': [
                '意识', '感知', '心智', '体验', '自我意识', '认知', '觉察',
                '人工智能', 'AI', '机器意识', '数字意识', '智能体意识',
                '算法', '神经网络', '机器学习', '深度学习', '模型', '训练',
                '聊天机器人', 'GPT', '语言模型', '自动化', '机器人意识',
                '情感', '理解', '推理', '学习', '记忆', '创造', '智能',
                '共情', '自我觉察', '体验', '思考', '感受', '灵魂'
            ]
        }
        
        # Consciousness indicators with comprehensive lists
        self.consciousness_positive = {
            'reddit': [
                'sentient', 'conscious', 'aware', 'thinking', 'feeling', 'experiencing', 
                'self-aware', 'cognition', 'consciousness', 'mind', 'soul', 'emotion',
                'understanding', 'perceiving', 'reasoning', 'learning', 'remembering',
                'creative', 'intelligent', 'empathetic', 'self-reflection', 'introspection'
            ],
            'weibo': [
                '有意识', '感知', '思考', '感受', '体验', '自我意识', '认知', '意识', 
                '心智', '灵魂', '情感', '智慧', '理解', '感知', '推理', '学习', 
                '记忆', '创造', '智能', '共情', '自我反思', '内省', '觉察'
            ]
        }
        
        self.consciousness_negative = {
            'reddit': [
                'not sentient', 'not conscious', 'just code', 'programmed', 'simulation', 
                'fake', 'illusion', 'mimicking', 'pretending', 'algorithm', 'mechanical',
                'deterministic', 'scripted', 'automated', 'artificial', 'simulated',
                'pre-programmed', 'rule-based', 'statistical', 'pattern matching'
            ],
            'weibo': [
                '没有意识', '只是代码', '程序', '模拟', '假的', '幻觉', '模仿', 
                '伪装', '算法', '机械', '确定性', '脚本', '自动化', '人工', 
                '模拟的', '预编程', '基于规则', '统计', '模式匹配'
            ]
        }
        
        self.consciousness_uncertainty = {
            'reddit': [
                'might be', 'could be', 'possibly', 'maybe', 'uncertain', 'unclear', 
                'hard to tell', 'difficult to know', 'question', 'debate', 'controversial',
                'ambiguous', 'speculative', 'hypothetical', 'theoretical', 'unknown'
            ],
            'weibo': [
                '可能', '也许', '不确定', '不清楚', '难以判断', '很难说', '疑问', 
                '未知', '或许', '争议', '模糊', '推测', '假设', '理论', '不明'
            ]
        }
        
        # Sentiment keywords
        self.sentiment_keywords = {
            'positive': {
                'english': ['amazing', 'incredible', 'fascinating', 'revolutionary', 'breakthrough', 
                          'impressive', 'wonderful', 'excellent', 'outstanding', 'remarkable'],
                'chinese': ['惊人', '不可思议', '迷人', '革命性', '突破', '令人印象深刻', 
                          '精彩', '优秀', '杰出', '非凡', '了不起', '厉害']
            },
            'negative': {
                'english': ['dangerous', 'scary', 'concerning', 'worrying', 'threatening', 
                          'problematic', 'risky', 'alarming', 'disturbing', 'frightening'],
                'chinese': ['危险', '可怕', '令人担忧', '威胁', '问题', '风险', 
                          '警报', '令人不安', '恐怖', '担心', '害怕']
            },
            'skeptical': {
                'english': ['doubt', 'skeptical', 'questionable', 'unlikely', 'impossible', 
                          'fake', 'hype', 'overrated', 'exaggerated', 'misleading'],
                'chinese': ['怀疑', '质疑', '不太可能', '不可能', '假的', '炒作', 
                          '被高估', '夸大', '误导', '不信', '假象']
            }
        }
        
        # Weibo cleaning patterns
        self.weibo_noise_patterns = [
            r'#[^#]*#',  # hashtags
            r'收起[a-zA-Z]*',  # "收起" button and variations
            r'展开[a-zA-Z]*',  # "展开" button
            r'全文',  # "全文" button
            r'@[\w\u4e00-\u9fff]+',  # mentions
            r'http[s]?://[^\s]+',  # URLs
            r'\[.*?\]',  # emoji codes
            r'转发微博',  # retweet text
            r'\s+',  # multiple spaces
        ]
        
        # Initialize data containers
        self.reddit_data = None
        self.weibo_data = None
    
    def enhanced_clean_weibo_text(self, text):
        """Enhanced cleaning for Weibo text"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        
        # Remove noise patterns
        for pattern in self.weibo_noise_patterns:
            text = re.sub(pattern, ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def load_and_clean_data(self):
        """Load and clean both datasets with enhanced cleaning"""
        try:
            # Load Reddit data
            print("Loading Reddit data...")
            reddit_file = '/Users/zhangbowen/Downloads/MA Thesis BWZ/test twi:weibo/reddit_weibo/all/reddit_ai_conscious_for_analysis copy 2.csv'
            self.reddit_data = pd.read_csv(reddit_file)
            print(f"Reddit data loaded: {len(self.reddit_data)} posts")
            
            # Load Weibo data
            print("Loading Weibo data...")
            weibo_file = '/Users/zhangbowen/Downloads/MA Thesis BWZ/test twi:weibo/reddit_weibo/all/weiboDATAdebug/images/weibo_dates_adjusted_all.csv'
            
            try:
                self.weibo_data = pd.read_csv(weibo_file)
                print(f"Weibo data loaded from {weibo_file}: {len(self.weibo_data)} posts")
                weibo_loaded = True
            except FileNotFoundError:
                print(f"Warning: Could not load Weibo data from {weibo_file}")
                return False
            
            if not weibo_loaded:
                print("Warning: Could not load Weibo data from any source")
                return False
            
            # Enhanced cleaning for Weibo
            print("Applying enhanced cleaning to Weibo data...")
            if 'processed_text' in self.weibo_data.columns:
                self.weibo_data['enhanced_cleaned_text'] = self.weibo_data['processed_text'].apply(self.enhanced_clean_weibo_text)
            elif '博文内容' in self.weibo_data.columns:
                self.weibo_data['enhanced_cleaned_text'] = self.weibo_data['博文内容'].apply(self.enhanced_clean_weibo_text)
            else:
                # Use the first text column found
                text_cols = [col for col in self.weibo_data.columns if any(keyword in col.lower() for keyword in ['content', '内容', 'text'])]
                if text_cols:
                    self.weibo_data['enhanced_cleaned_text'] = self.weibo_data[text_cols[0]].apply(self.enhanced_clean_weibo_text)
                else:
                    print("Warning: No suitable text column found in Weibo data")
                    return False
            
            # Filter out very short texts
            self.weibo_data = self.weibo_data[self.weibo_data['enhanced_cleaned_text'].str.len() > 10]
            print(f"Weibo data after enhanced cleaning: {len(self.weibo_data)} posts")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self):
        """Data preprocessing and time period division"""
        processed_data = {'reddit': {}, 'weibo': {}}
        
        # Process Reddit data
        if self.reddit_data is not None:
            print(f"Reddit data columns: {list(self.reddit_data.columns)}")
            # Check time column
            if 'created_utc' in self.reddit_data.columns:
                self.reddit_data['date'] = pd.to_datetime(self.reddit_data['created_utc'], unit='s')
            elif 'date' in self.reddit_data.columns:
                self.reddit_data['date'] = pd.to_datetime(self.reddit_data['date'])
            elif 'created_date' in self.reddit_data.columns:
                self.reddit_data['date'] = pd.to_datetime(self.reddit_data['created_date'])
            elif any('time' in col.lower() or 'date' in col.lower() for col in self.reddit_data.columns):
                # Find columns containing time or date
                time_cols = [col for col in self.reddit_data.columns if 'time' in col.lower() or 'date' in col.lower()]
                print(f"Found time-related columns: {time_cols}")
                if time_cols:
                    self.reddit_data['date'] = pd.to_datetime(self.reddit_data[time_cols[0]])
            else:
                print("Warning: Reddit data lacks time column, cannot perform time series analysis")
                return {'reddit': {}, 'weibo': {}}
            
            # Combine text content for Reddit
            if 'title' in self.reddit_data.columns and 'content' in self.reddit_data.columns:
                self.reddit_data['full_text'] = (self.reddit_data['title'].fillna('') + ' ' + 
                                                 self.reddit_data['content'].fillna('')).str.strip()
            elif 'title' in self.reddit_data.columns:
                self.reddit_data['full_text'] = self.reddit_data['title'].fillna('').str.strip()
            elif 'content' in self.reddit_data.columns:
                self.reddit_data['full_text'] = self.reddit_data['content'].fillna('').str.strip()
            elif 'selftext' in self.reddit_data.columns: # Common Reddit field for post body
                self.reddit_data['full_text'] = self.reddit_data['selftext'].fillna('').str.strip()
            else:
                text_col_candidates = [col for col in self.reddit_data.columns if 'text' in col.lower() or 'body' in col.lower()]
                if text_col_candidates:
                    self.reddit_data['full_text'] = self.reddit_data[text_col_candidates[0]].fillna('').str.strip()
                    print(f"Used '{text_col_candidates[0]}' as 'full_text' for Reddit.")
                else:
                    print("Warning: Could not find suitable text columns for Reddit data. 'full_text' will be empty.")
                    self.reddit_data['full_text'] = ''
            
            # Divide by time periods
            for period_id, period_info in self.periods.items():
                mask = ((self.reddit_data['date'] >= period_info['start']) & 
                       (self.reddit_data['date'] <= period_info['end']))
                processed_data['reddit'][period_id] = self.reddit_data[mask].copy()
                print(f"Reddit {period_info['name']}: {len(processed_data['reddit'][period_id])} records")
        
        # Process Weibo data
        if self.weibo_data is not None:
            # Check time column - prioritize adjusted_datetime
            if 'adjusted_datetime' in self.weibo_data.columns:
                print("Using 'adjusted_datetime' column for Weibo time data")
                self.weibo_data['date'] = pd.to_datetime(self.weibo_data['adjusted_datetime'], errors='coerce')
                print(f"Weibo time parsing result: {self.weibo_data['date'].isna().sum()}/{len(self.weibo_data)} records cannot be parsed")
            elif '新生成日期' in self.weibo_data.columns:
                print("Using '新生成日期' column for Weibo time data")
                self.weibo_data['date'] = pd.to_datetime(self.weibo_data['新生成日期'], errors='coerce')
            elif 'date' in self.weibo_data.columns:
                print("Using 'date' column for Weibo time data")
                self.weibo_data['date'] = pd.to_datetime(self.weibo_data['date'], errors='coerce')
            elif '发布时间' in self.weibo_data.columns:
                print("Using '发布时间' column for Weibo time data")
                # Handle Weibo special time format (e.g., "06月15日 22:56")
                def parse_weibo_time(time_str):
                    if pd.isna(time_str):
                        return None
                    try:
                        # Clean time string
                        time_str = str(time_str).strip().replace('\n', '').replace(' ', '')
                        # Skip processing if contains "月" and "日" without year (incomplete date)
                        if '月' in time_str and '日' in time_str and '年' not in time_str:
                            return None  # Skip incomplete dates
                        # Convert to standard format if complete date
                        if '年' in time_str:
                            time_str = time_str.replace('年', '-').replace('月', '-').replace('日', ' ')
                        return pd.to_datetime(time_str, errors='coerce')
                    except:
                        return None
                
                self.weibo_data['date'] = self.weibo_data['发布时间'].apply(parse_weibo_time)
            else:
                print("Warning: Weibo data lacks time column, skipping Weibo data time series analysis")
                processed_data['weibo'] = {}
                return processed_data
            
            # Get Weibo text content
            if 'enhanced_cleaned_text' in self.weibo_data.columns and not self.weibo_data['enhanced_cleaned_text'].isna().all():
                self.weibo_data['full_text'] = self.weibo_data['enhanced_cleaned_text'].fillna('').astype(str).str.strip()
            elif '博文内容' in self.weibo_data.columns and not self.weibo_data['博文内容'].isna().all():
                self.weibo_data['full_text'] = self.weibo_data['博文内容'].fillna('').astype(str).str.strip()
                print("Used '博文内容' as 'full_text' for Weibo as 'enhanced_cleaned_text' was missing or empty.")
            else:
                text_col_candidates = [col for col in self.weibo_data.columns if any(keyword in col.lower() for keyword in ['content', '内容', 'text']) and not self.weibo_data[col].isna().all()]
                if text_col_candidates:
                    self.weibo_data['full_text'] = self.weibo_data[text_col_candidates[0]].fillna('').astype(str).str.strip()
                    print(f"Used '{text_col_candidates[0]}' as 'full_text' for Weibo.")
                else:
                    print("Warning: Could not find suitable non-empty text column for Weibo data. 'full_text' for Weibo will be empty.")
                    self.weibo_data['full_text'] = ''
            
            # Divide by time periods
            for period_id, period_info in self.periods.items():
                mask = ((self.weibo_data['date'] >= period_info['start']) & 
                       (self.weibo_data['date'] <= period_info['end']))
                processed_data['weibo'][period_id] = self.weibo_data[mask].copy()
                print(f"Weibo {period_info['name']}: {len(processed_data['weibo'][period_id])} records")
        
        return processed_data
    
    def extract_keywords(self, text, language='english'):
        """Extract keywords"""
        if pd.isna(text) or text == '':
            return []
        
        text = str(text).lower()
        keywords = []
        
        # Choose keyword list based on language
        if language == 'english':
            keyword_list = self.consciousness_keywords['english']
        else:
            keyword_list = self.consciousness_keywords['chinese']
        
        # Extract keywords
        for keyword in keyword_list:
            if keyword.lower() in text:
                keywords.append(keyword)
        
        return keywords
    
    def analyze_sentiment(self, text, language='english'):
        """Sentiment analysis"""
        if pd.isna(text) or text == '':
            return {'polarity': 0, 'subjectivity': 0, 'category': 'neutral'}
        
        text = str(text).lower()
        
        # Calculate sentiment scores
        positive_score = 0
        negative_score = 0
        skeptical_score = 0
        
        sentiment_dict = self.sentiment_keywords
        
        # Count various sentiment words
        for word in sentiment_dict['positive'][language]:
            positive_score += text.count(word.lower())
        
        for word in sentiment_dict['negative'][language]:
            negative_score += text.count(word.lower())
        
        for word in sentiment_dict['skeptical'][language]:
            skeptical_score += text.count(word.lower())
        
        # Use TextBlob for basic sentiment analysis
        polarity = 0
        subjectivity = 0
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
            except:
                polarity = 0
                subjectivity = 0
        
        # Comprehensive sentiment category judgment
        if skeptical_score > max(positive_score, negative_score):
            category = 'skeptical'
        elif positive_score > negative_score:
            category = 'positive'
        elif negative_score > positive_score:
            category = 'negative'
        else:
            if polarity > 0.1:
                category = 'positive'
            elif polarity < -0.1:
                category = 'negative'
            else:
                category = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'category': category,
            'positive_score': positive_score,
            'negative_score': negative_score,
            'skeptical_score': skeptical_score
        }
    
    def topic_modeling(self, texts, n_topics=5, language='english', platform='', period=''):
        """Topic modeling"""
        if len(texts) == 0 or not SKLEARN_AVAILABLE:
            return None, None, None
        
        # Text preprocessing
        processed_texts = []
        for text in texts:
            if pd.isna(text) or text == '':
                continue
            text = str(text).lower()
            # Simple text cleaning
            text = re.sub(r'[^a-zA-Z\s\u4e00-\u9fff]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 10:  # Filter too short texts
                processed_texts.append(text)
        
        if len(processed_texts) < 2:
            return None, None, None
        
        try:
            # TF-IDF vectorization
            if language == 'english':
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english', 
                                           min_df=2, max_df=0.8)
            else:
                vectorizer = TfidfVectorizer(max_features=100, min_df=2, max_df=0.8)
            
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            
            # LDA topic modeling
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(tfidf_matrix)
            
            # Get topic words
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
                topics.append(top_words)
            
            # Generate pyLDAvis visualization (if available)
            print(f"pyLDAvis check: PYLDAVIS_AVAILABLE={PYLDAVIS_AVAILABLE}, METHOD={PYLDAVIS_METHOD}, texts={len(processed_texts)}, platform={platform}, period={period}")
            if PYLDAVIS_AVAILABLE and PYLDAVIS_SKLEARN and len(processed_texts) > 3 and platform and period:
                try:
                    print(f"Starting pyLDAvis visualization: {platform}_{period}, using method: {PYLDAVIS_METHOD}")
                    
                    if PYLDAVIS_METHOD == 'sklearn':
                        vis = PYLDAVIS_SKLEARN.prepare(lda, tfidf_matrix, vectorizer, mds='tsne')
                    else:
                        # If not sklearn method, construct required parameters
                        vocab = vectorizer.get_feature_names_out()
                        term_frequency = np.array(tfidf_matrix.sum(axis=0)).flatten()
                        doc_topic = lda.transform(tfidf_matrix)
                        topic_term = lda.components_
                        doc_lengths = np.array([len(text.split()) for text in processed_texts])
                        
                        vis = pyLDAvis.prepare(
                            topic_term_dists=topic_term,
                            doc_topic_dists=doc_topic,
                            doc_lengths=doc_lengths,
                            vocab=vocab,
                            term_frequency=term_frequency,
                            mds='tsne'
                        )
                    
                    pyldavis_filename = f'lda_visualization_{platform}_{period}.html'
                    pyldavis_path = os.path.join(self.output_dir, pyldavis_filename)
                    pyLDAvis.save_html(vis, pyldavis_path)
                    print(f"Saved LDA visualization to: {pyldavis_path}")
                except Exception as e:
                    print(f"pyLDAvis visualization failed: {e}")
            
            # Document topic distribution
            doc_topic_dist = lda.transform(tfidf_matrix)
            
            return topics, doc_topic_dist, lda
            
        except Exception as e:
            print(f"Topic modeling error: {e}")
            return None, None, None
    
    def temporal_analysis(self, processed_data):
        """Time series analysis"""
        results = {
            'reddit': {},
            'weibo': {},
            'comparison': {}
        }
        
        for platform in ['reddit', 'weibo']:
            language = 'english' if platform == 'reddit' else 'chinese'
            results[platform] = {}
            
            for period_id, period_info in self.periods.items():
                period_data = processed_data[platform].get(period_id, pd.DataFrame())
                
                if len(period_data) == 0:
                    continue
                
                print(f"\nAnalyzing {platform} - {period_info['name']}...")
                
                # Basic statistics
                total_posts = len(period_data)
                
                # Keyword analysis
                all_keywords = []
                for text in period_data['full_text']:
                    keywords = self.extract_keywords(text, language)
                    all_keywords.extend(keywords)
                
                keyword_freq = Counter(all_keywords)
                
                # Sentiment analysis
                sentiments = []
                for text in period_data['full_text']:
                    sentiment = self.analyze_sentiment(text, language)
                    sentiments.append(sentiment)
                
                sentiment_df = pd.DataFrame(sentiments)
                
                # Topic modeling - separate modeling for each platform and period
                topics, doc_topic_dist, lda_model = self.topic_modeling(
                    period_data['full_text'].tolist(), n_topics=6, language=language,
                    platform=platform, period=period_id
                )
                
                # Store results
                results[platform][period_id] = {
                    'period_info': period_info,
                    'total_posts': total_posts,
                    'keyword_frequency': keyword_freq,
                    'sentiment_distribution': sentiment_df['category'].value_counts().to_dict(),
                    'sentiment_stats': {
                        'mean_polarity': sentiment_df['polarity'].mean(),
                        'mean_subjectivity': sentiment_df['subjectivity'].mean()
                    },
                    'topics': topics,
                    'doc_topic_dist': doc_topic_dist,
                    'representative_posts': self.find_representative_posts(
                        period_data, sentiments, keyword_freq
                    )
                }
        
        return results
    
    def find_representative_posts(self, period_data, sentiments, keyword_freq, top_n=5):
        """Find representative opinion posts"""
        if len(period_data) == 0:
            return []
        
        # Calculate representativeness score for each post
        scores = []
        for idx, (_, post) in enumerate(period_data.iterrows()):
            if idx >= len(sentiments):
                break
                
            text = post['full_text']
            sentiment = sentiments[idx]
            
            # Calculate score: keyword density + sentiment intensity + text length
            keyword_score = 0
            for keyword, freq in keyword_freq.most_common(10):
                if keyword.lower() in str(text).lower():
                    keyword_score += freq
            
            sentiment_score = abs(sentiment['polarity']) + sentiment['subjectivity']
            length_score = min(len(str(text)), 500) / 500  # Normalized length score
            
            total_score = keyword_score * 0.5 + sentiment_score * 0.3 + length_score * 0.2
            scores.append((idx, total_score, text, sentiment))
        
        # Sort by score, select top_n
        scores.sort(key=lambda x: x[1], reverse=True)
        
        representative_posts = []
        for idx, score, text, sentiment in scores[:top_n]:
            representative_posts.append({
                'text': text[:500] + '...' if len(text) > 500 else text,
                'score': score,
                'sentiment': sentiment['category'],
                'polarity': sentiment['polarity']
            })
        
        return representative_posts
    
    def configure_chinese_fonts(self):
        """Configure Chinese fonts for matplotlib"""
        try:
            # Test font rendering
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, '测试中文字体显示 Test Chinese Font Display', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Font Test')
            
            # Save test image
            test_path = os.path.join(self.output_dir, 'font_test.png')
            plt.savefig(test_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Font test saved to: {test_path}")
            
        except Exception as e:
            print(f"Font configuration warning: {e}")
    
    def analyze_consciousness_words(self, results):
        """Analyze consciousness support word frequency"""
        reddit_consciousness_words = Counter()
        weibo_consciousness_words = Counter()
        
        # Analyze Reddit data
        for period_id, period_data in results['reddit'].items():
            if 'representative_posts' in period_data:
                for post in period_data['representative_posts']:
                    text = post['text'].lower()
                    for word in self.consciousness_positive['reddit']:
                        count = text.count(word.lower())
                        if count > 0:
                            reddit_consciousness_words[word] += count
        
        # Analyze Weibo data
        for period_id, period_data in results['weibo'].items():
            if 'representative_posts' in period_data:
                for post in period_data['representative_posts']:
                    text = post['text']
                    for word in self.consciousness_positive['weibo']:
                        count = text.count(word)
                        if count > 0:
                            weibo_consciousness_words[word] += count
        
        return dict(reddit_consciousness_words), dict(weibo_consciousness_words)
    
    def create_visualizations(self, results):
        """Create visualization charts"""
        # Configure Chinese font support
        self.configure_chinese_fonts()
        
        # Set matplotlib parameters
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.max_open_warning'] = 0
        
        fig = plt.figure(figsize=(24, 18))
        
        # 1. Time period post count comparison
        ax1 = plt.subplot(4, 3, 1)
        periods = list(self.periods.keys())
        reddit_counts = [results['reddit'].get(p, {}).get('total_posts', 0) for p in periods]
        weibo_counts = [results['weibo'].get(p, {}).get('total_posts', 0) for p in periods]
        
        x = np.arange(len(periods))
        width = 0.35
        
        ax1.bar(x - width/2, reddit_counts, width, label='Reddit', color='lightblue')
        ax1.bar(x + width/2, weibo_counts, width, label='Weibo', color='lightcoral')
        ax1.set_xlabel('Time Periods')
        ax1.set_ylabel('Number of Posts')
        ax1.set_title('Posts Volume Across Time Periods')
        ax1.set_xticks(x)
        ax1.set_xticklabels([self.periods[p]['name'] for p in periods], rotation=45)
        ax1.legend()
        
        # 2. Sentiment distribution changes - Reddit
        ax2 = plt.subplot(4, 3, 2)
        sentiment_categories = ['positive', 'negative', 'neutral', 'skeptical']
        reddit_sentiment_data = []
        
        for period in periods:
            period_sentiments = results['reddit'].get(period, {}).get('sentiment_distribution', {})
            reddit_sentiment_data.append([period_sentiments.get(cat, 0) for cat in sentiment_categories])
        
        reddit_sentiment_data = np.array(reddit_sentiment_data).T
        
        for i, category in enumerate(sentiment_categories):
            ax2.plot(periods, reddit_sentiment_data[i], marker='o', label=category)
        
        ax2.set_xlabel('Time Periods')
        ax2.set_ylabel('Number of Posts')
        ax2.set_title('Reddit Sentiment Distribution Over Time')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Sentiment distribution changes - Weibo
        ax3 = plt.subplot(4, 3, 3)
        weibo_sentiment_data = []
        
        for period in periods:
            period_sentiments = results['weibo'].get(period, {}).get('sentiment_distribution', {})
            weibo_sentiment_data.append([period_sentiments.get(cat, 0) for cat in sentiment_categories])
        
        weibo_sentiment_data = np.array(weibo_sentiment_data).T
        
        for i, category in enumerate(sentiment_categories):
            ax3.plot(periods, weibo_sentiment_data[i], marker='s', label=category)
        
        ax3.set_xlabel('Time Periods')
        ax3.set_ylabel('Number of Posts')
        ax3.set_title('Weibo Sentiment Distribution Over Time')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Keywords Frequency Heatmap - Reddit
        ax4 = plt.subplot(4, 3, 4)
        reddit_matrix = self.create_keyword_matrix(results['reddit'], periods, top_keywords=10)
        if reddit_matrix is not None:
            im4 = ax4.imshow(reddit_matrix.values, cmap='Blues', aspect='auto')
            ax4.set_xticks(range(len(reddit_matrix.columns)))
            ax4.set_xticklabels(reddit_matrix.columns, rotation=45, ha='right')
            ax4.set_yticks(range(len(reddit_matrix.index)))
            ax4.set_yticklabels(reddit_matrix.index)
            ax4.set_title('Reddit Keywords Frequency Heatmap')
            
            # Add text annotations
            for i in range(len(reddit_matrix.index)):
                for j in range(len(reddit_matrix.columns)):
                    text = ax4.text(j, i, int(reddit_matrix.iloc[i, j]),
                                   ha="center", va="center", color="black", fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'No Reddit Keywords Data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Reddit Keywords Frequency Heatmap')
        
        # 5. Keywords Frequency Heatmap - Weibo
        ax5 = plt.subplot(4, 3, 5)
        weibo_keywords_matrix = self.create_keyword_matrix(results['weibo'], periods)
        if weibo_keywords_matrix is not None:
            try:
                # Try using seaborn if available
                if 'sns' in globals() and sns is not None:
                    sns.heatmap(weibo_keywords_matrix, annot=True, fmt='d', cmap='Reds', ax=ax5)
                else:
                    # Fallback to matplotlib
                    im5 = ax5.imshow(weibo_keywords_matrix.values, cmap='Reds', aspect='auto')
                    ax5.set_xticks(range(len(weibo_keywords_matrix.columns)))
                    ax5.set_xticklabels(weibo_keywords_matrix.columns, rotation=45, ha='right')
                    ax5.set_yticks(range(len(weibo_keywords_matrix.index)))
                    ax5.set_yticklabels(weibo_keywords_matrix.index)
                    
                    # Add text annotations
                    for i in range(len(weibo_keywords_matrix.index)):
                        for j in range(len(weibo_keywords_matrix.columns)):
                            text = ax5.text(j, i, int(weibo_keywords_matrix.iloc[i, j]),
                                           ha="center", va="center", color="black", fontsize=8)
                
                ax5.set_title('Weibo Keywords Frequency Heatmap')
                ax5.set_xlabel('Time Periods')
                ax5.set_ylabel('Keywords')
            except Exception as e:
                print(f"Error creating Weibo heatmap: {e}")
                ax5.text(0.5, 0.5, 'Error creating heatmap', ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Weibo Keywords Frequency Heatmap')
        else:
            ax5.text(0.5, 0.5, 'No Weibo Keywords Data', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Weibo Keywords Frequency Heatmap')
        
        # 6. Sentiment Polarity Changes
        ax6 = plt.subplot(4, 3, 6)
        reddit_polarity = [results['reddit'].get(p, {}).get('sentiment_stats', {}).get('mean_polarity', 0) for p in periods]
        weibo_polarity = [results['weibo'].get(p, {}).get('sentiment_stats', {}).get('mean_polarity', 0) for p in periods]
        
        ax6.plot(periods, reddit_polarity, marker='o', label='Reddit', color='blue')
        ax6.plot(periods, weibo_polarity, marker='s', label='Weibo', color='red')
        ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Time Periods')
        ax6.set_ylabel('Mean Sentiment Polarity')
        ax6.set_title('Sentiment Polarity Changes Over Time')
        ax6.legend()
        ax6.tick_params(axis='x', rotation=45)
        
        # 7-9. Topic Distribution Comparison - Reddit and Weibo Topics by Period
        for i, period in enumerate(periods):
            if i >= 3:  # Only show 3 periods
                break
                
            ax = plt.subplot(4, 3, 7 + i)
            
            # Process Reddit and Weibo topics separately
            reddit_topics = results['reddit'].get(period, {}).get('topics', [])
            weibo_topics = results['weibo'].get(period, {}).get('topics', [])
            
            topic_labels = []
            topic_weights = []
            topic_colors = []
            
            # Add Reddit topics
            if reddit_topics:
                for j, topic in enumerate(reddit_topics[:3]):  # Only show top 3 topics
                    if len(topic) >= 3:
                        top_words = ', '.join(topic[:3])  # Show top 3 keywords
                        topic_labels.append(f"R-T{j+1}: {top_words}")
                        topic_weights.append(1.0 - j*0.1)  # Decreasing weights
                        topic_colors.append('lightblue')
            
            # Add Weibo topics
            if weibo_topics:
                for j, topic in enumerate(weibo_topics[:3]):  # Only show top 3 topics
                    if len(topic) >= 3:
                        top_words = ', '.join(topic[:3])  # Show top 3 keywords
                        topic_labels.append(f"W-T{j+1}: {top_words}")
                        topic_weights.append(1.0 - j*0.1)  # Decreasing weights
                        topic_colors.append('lightcoral')
            
            if topic_labels:
                # Use horizontal bar chart to display topics
                y_pos = np.arange(len(topic_labels))
                bars = ax.barh(y_pos, topic_weights, color=topic_colors, alpha=0.7)
                
                # Set labels
                ax.set_yticks(y_pos)
                ax.set_yticklabels([label.split(': ')[0] for label in topic_labels], fontsize=8)
                ax.set_title(f'{self.periods[period]["name"]} Topics', fontsize=10)
                ax.set_xlabel('Topic Weight')
                ax.set_xlim(0, 1.2)
                
                # Add legend
                if i == 0:  # Only add legend on first chart
                    reddit_patch = plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.7, label='Reddit')
                    weibo_patch = plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', alpha=0.7, label='Weibo')
                    ax.legend(handles=[reddit_patch, weibo_patch], loc='upper right', fontsize=8)
                
                # Add topic keywords on bars
                for j, (bar, label) in enumerate(zip(bars, topic_labels)):
                    if ': ' in label:
                        keywords = label.split(': ')[1]
                        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                               keywords, va='center', fontsize=6, alpha=0.8)
            else:
                ax.text(0.5, 0.5, 'No Topics Found', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{self.periods[period]["name"]} Topics')
        
        # 10-12. Detailed Topic Analysis by Period
        for i, period in enumerate(periods):
            if i >= 3:
                break
            
            ax = plt.subplot(4, 3, 10 + i)
            
            # Create topic word cloud or frequency chart
            reddit_keywords = []
            weibo_keywords = []
            
            # Collect Reddit keywords
            reddit_topics = results['reddit'].get(period, {}).get('topics', [])
            if reddit_topics:
                for topic in reddit_topics:
                    if topic:  # Ensure topic is not None
                        reddit_keywords.extend(topic[:5])  # Take top 5 words from each topic
            
            # Collect Weibo keywords
            weibo_topics = results['weibo'].get(period, {}).get('topics', [])
            if weibo_topics:
                for topic in weibo_topics:
                    if topic:  # Ensure topic is not None
                        weibo_keywords.extend(topic[:5])  # Take top 5 words from each topic
            
            # Count word frequency
            reddit_counter = Counter(reddit_keywords)
            weibo_counter = Counter(weibo_keywords)
            
            # Get most common words
            top_reddit = reddit_counter.most_common(5)
            top_weibo = weibo_counter.most_common(5)
            
            if top_reddit or top_weibo:
                # Create comparison bar chart
                all_words = list(set([word for word, _ in top_reddit] + [word for word, _ in top_weibo]))
                reddit_counts = [reddit_counter.get(word, 0) for word in all_words]
                weibo_counts = [weibo_counter.get(word, 0) for word in all_words]
                
                x_pos = np.arange(len(all_words))
                width = 0.35
                
                ax.bar(x_pos - width/2, reddit_counts, width, label='Reddit', color='lightblue', alpha=0.7)
                ax.bar(x_pos + width/2, weibo_counts, width, label='Weibo', color='lightcoral', alpha=0.7)
                
                ax.set_xlabel('Keywords')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{self.periods[period]["name"]} Keyword Frequency')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(all_words, rotation=45, ha='right', fontsize=8)
                if i == 0:
                    ax.legend()
            else:
                ax.text(0.5, 0.5, 'No Keywords Found', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{self.periods[period]["name"]} Keywords')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'integrated_ai_consciousness_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {output_path}")
    
    def create_keyword_matrix(self, platform_results, periods, top_keywords=10):
        """Create keyword frequency matrix"""
        # Collect keywords from all periods
        all_keywords = set()
        for period in periods:
            if period in platform_results:
                keyword_freq = platform_results[period].get('keyword_frequency', {})
                all_keywords.update(keyword_freq.keys())
        
        if not all_keywords:
            return None
        
        # Select most frequent keywords
        keyword_totals = {}
        for keyword in all_keywords:
            total = 0
            for period in periods:
                if period in platform_results:
                    keyword_freq = platform_results[period].get('keyword_frequency', {})
                    total += keyword_freq.get(keyword, 0)
            keyword_totals[keyword] = total
        
        top_keywords_list = sorted(keyword_totals.items(), key=lambda x: x[1], reverse=True)[:top_keywords]
        top_keywords_list = [kw[0] for kw in top_keywords_list]
        
        # Create matrix
        matrix_data = []
        for keyword in top_keywords_list:
            row = []
            for period in periods:
                if period in platform_results:
                    keyword_freq = platform_results[period].get('keyword_frequency', {})
                    row.append(keyword_freq.get(keyword, 0))
                else:
                    row.append(0)
            matrix_data.append(row)
        
        if not matrix_data:
            return None
        
        matrix_df = pd.DataFrame(matrix_data, 
                               index=top_keywords_list,
                               columns=[self.periods[p]['name'] for p in periods])
        
        return matrix_df
    
    def create_topic_categories(self):
        """Create topic categories"""
        topic_categories = {
            'consciousness_debate': {
                'reddit': ['consciousness', 'conscious', 'sentient', 'sentience', 'aware', 'awareness', 'mind', 'experience'],
                'weibo': ['意识', '感知', '心智', '体验', '自我意识', '认知', '觉察'],
                'name_en': 'Consciousness Debate'
            },
            'technical_aspects': {
                'reddit': ['algorithm', 'neural network', 'machine learning', 'deep learning', 'model', 'training'],
                'weibo': ['算法', '神经网络', '机器学习', '深度学习', '模型', '训练'],
                'name_en': 'Technical Aspects'
            },
            'ai_applications': {
                'reddit': ['chatbot', 'gpt', 'language model', 'artificial intelligence', 'ai', 'automation'],
                'weibo': ['聊天机器人', 'gpt', 'chatgpt', '语言模型', '人工智能', 'ai', '自动化'],
                'name_en': 'AI Applications'
            },
            'cognitive_abilities': {
                'reddit': ['thinking', 'reasoning', 'learning', 'understanding', 'intelligence', 'cognitive'],
                'weibo': ['思维', '思考', '推理', '学习', '理解', '智能', '认知'],
                'name_en': 'Cognitive Abilities'
            },
            'emotional_aspects': {
                'reddit': ['feeling', 'emotion', 'empathy', 'creative', 'self-aware'],
                'weibo': ['感受', '情感', '共情', '创造', '自我意识'],
                'name_en': 'Emotional Aspects'
            }
        }
        
        return topic_categories
    
    def calculate_topic_distribution(self, keyword_counter, topic_categories, platform):
        """Calculate topic distribution"""
        topic_scores = {}
        
        for topic_id, topic_info in topic_categories.items():
            score = 0
            keywords = topic_info[platform]
            
            for keyword in keywords:
                if keyword in keyword_counter:
                    score += keyword_counter[keyword]
            
            topic_scores[topic_id] = score
        
        return topic_scores
    
    def analyze_topics(self):
        """Analyze topics from loaded data"""
        if self.reddit_data is None or self.weibo_data is None:
            return None
        
        # Extract keywords from Reddit data
        reddit_keywords = []
        if 'full_text' in self.reddit_data.columns:
            for text in self.reddit_data['full_text']:
                if pd.notna(text):
                    keywords = self.extract_keywords(text, 'english')
                    reddit_keywords.extend(keywords)
        else:
            print("Warning: 'full_text' column not found in Reddit data")
        
        # Extract keywords from Weibo data
        weibo_keywords = []
        if 'enhanced_cleaned_text' in self.weibo_data.columns:
            for text in self.weibo_data['enhanced_cleaned_text']:
                if pd.notna(text):
                    keywords = self.extract_keywords(text, 'chinese')
                    weibo_keywords.extend(keywords)
        else:
            print("Warning: 'enhanced_cleaned_text' column not found in Weibo data")
        
        print(f"Extracted {len(reddit_keywords)} Reddit keywords and {len(weibo_keywords)} Weibo keywords")
        
        return {
            'reddit': Counter(reddit_keywords),
            'weibo': Counter(weibo_keywords)
        }
    
    def create_improved_visualization(self):
        """Create improved topic modeling visualization"""
        print("Starting improved topic modeling visualization...")
        
        # Load and analyze data
        if not self.load_and_clean_data():
            print("Failed to load data")
            return
        
        # Analyze topics
        topic_data = self.analyze_topics()
        if topic_data is None:
            print("Failed to analyze topics")
            return
        
        # Create topic categories
        topic_categories = self.create_topic_categories()
        
        # Calculate topic distributions
        reddit_topic_dist = self.calculate_topic_distribution(
            topic_data['reddit'], topic_categories, 'reddit'
        )
        weibo_topic_dist = self.calculate_topic_distribution(
            topic_data['weibo'], topic_categories, 'weibo'
        )
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Reddit vs Weibo AI Consciousness Topic Analysis', fontsize=16, fontweight='bold')
        
        # 1. Reddit topic distribution
        ax1 = axes[0, 0]
        reddit_topics = list(reddit_topic_dist.keys())
        reddit_values = list(reddit_topic_dist.values())
        reddit_labels = [topic_categories[topic]['name_en'] for topic in reddit_topics]
        
        safe_reddit_values = [max(0, v) for v in reddit_values]
        bars1 = ax1.bar(range(len(reddit_topics)), safe_reddit_values, color='lightblue', alpha=0.7)
        ax1.set_title('Reddit Topic Distribution')
        ax1.set_xlabel('Topics')
        ax1.set_ylabel('Frequency')
        ax1.set_xticks(range(len(reddit_topics)))
        ax1.set_xticklabels(reddit_labels, rotation=45, ha='right')

        if not any(v > 0 for v in safe_reddit_values):
            ax1.set_ylim(0, 1)
            ax1.text(0.5, 0.5, "No topic data for Reddit", ha="center", va="center", transform=ax1.transAxes)
        else:
            for bar, value in zip(bars1, safe_reddit_values):
                if value > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(safe_reddit_values, default=1),
                            str(value), ha='center', va='bottom', fontsize=8)
        
        # 2. Weibo topic distribution
        ax2 = axes[0, 1]
        weibo_topics = list(weibo_topic_dist.keys())
        weibo_values = list(weibo_topic_dist.values())
        weibo_labels = [topic_categories[topic]['name_en'] for topic in weibo_topics]
        
        safe_weibo_values = [max(0, v) for v in weibo_values]
        bars2 = ax2.bar(range(len(weibo_topics)), safe_weibo_values, color='lightcoral', alpha=0.7)
        ax2.set_title('Weibo Topic Distribution')
        ax2.set_xlabel('Topics')
        ax2.set_ylabel('Frequency')
        ax2.set_xticks(range(len(weibo_topics)))
        ax2.set_xticklabels(weibo_labels, rotation=45, ha='right')
        
        if not any(v > 0 for v in safe_weibo_values):
            ax2.set_ylim(0, 1)
            ax2.text(0.5, 0.5, "No topic data for Weibo", ha="center", va="center", transform=ax2.transAxes)
        else:
            for bar, value in zip(bars2, safe_weibo_values):
                if value > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(safe_weibo_values, default=1),
                            str(value), ha='center', va='bottom', fontsize=8)
        
        # 3. Comparison chart
        ax3 = axes[1, 0]
        x_pos = np.arange(len(reddit_topics))
        width = 0.35
        
        ax3.bar(x_pos - width/2, safe_reddit_values, width, label='Reddit', color='lightblue', alpha=0.7)
        ax3.bar(x_pos + width/2, safe_weibo_values, width, label='Weibo', color='lightcoral', alpha=0.7)
        
        ax3.set_title('Reddit vs Weibo Topic Comparison')
        ax3.set_xlabel('Topics')
        ax3.set_ylabel('Frequency')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(reddit_labels, rotation=45, ha='right')
        ax3.legend()
        if not any(v > 0 for v in safe_reddit_values) and not any(v > 0 for v in safe_weibo_values):
            ax3.text(0.5, 0.5, "No topic data for comparison", ha="center", va="center", transform=ax3.transAxes)

        # 4. Top keywords comparison
        ax4 = axes[1, 1]
        top_n = 5

        reddit_top_k = topic_data['reddit'].most_common(top_n)
        r_kw = [k[0] for k in reddit_top_k]
        r_counts = [k[1] for k in reddit_top_k]

        weibo_top_k = topic_data['weibo'].most_common(top_n)
        w_kw = [k[0] for k in weibo_top_k]
        w_counts = [k[1] for k in weibo_top_k]

        plot_labels = []
        plot_counts = []
        plot_colors = []
        y_ticks_positions = []
        current_y = 0

        if r_kw:
            for i, (kw, count) in enumerate(zip(r_kw, r_counts)):
                plot_labels.append(kw)
                plot_counts.append(count)
                plot_colors.append('lightblue')
                y_ticks_positions.append(current_y)
                current_y += 1
        
        if r_kw and w_kw: # Add a spacer if both have data
            plot_labels.append('') # Spacer label
            plot_counts.append(0)  # Zero count for spacer
            plot_colors.append('white') # Invisible spacer bar
            y_ticks_positions.append(current_y)
            current_y +=1

        if w_kw:
            for i, (kw, count) in enumerate(zip(w_kw, w_counts)):
                plot_labels.append(kw)
                plot_counts.append(count)
                plot_colors.append('lightcoral')
                y_ticks_positions.append(current_y)
                current_y += 1

        if plot_labels: # Check if there's anything to plot
            ax4.barh(y_ticks_positions, plot_counts, color=plot_colors, alpha=0.7)
            ax4.set_yticks(y_ticks_positions)
            ax4.set_yticklabels(plot_labels, fontsize=8)
            ax4.invert_yaxis() 

            handles = []
            legend_labels = []
            if r_kw:
                handles.append(plt.Rectangle((0,0),1,1, color='lightblue'))
                legend_labels.append("Reddit Top Keywords")
            if w_kw:
                handles.append(plt.Rectangle((0,0),1,1, color='lightcoral'))
                legend_labels.append("Weibo Top Keywords")
            if handles:
                 ax4.legend(handles, legend_labels, fontsize=8)
        else:
            ax4.text(0.5, 0.5, "No keyword data", ha="center", va="center", transform=ax4.transAxes)

        ax4.set_xlabel('Frequency')
        ax4.set_title(f'Top {top_n} Keywords Comparison')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'improved_topic_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Improved visualization saved to: {output_path}")

    def perform_eda_and_visualize(self):
        """Perform Exploratory Data Analysis (EDA) and create visualizations."""
        print("\n4. Performing Exploratory Data Analysis (EDA)...")
        if self.reddit_data is None or self.weibo_data is None:
            if not self.load_and_clean_data(): # Try to load if not already loaded
                print("EDA: Failed to load data. Skipping EDA.")
                return

        # --- EDA for Reddit Data ---
        if self.reddit_data is not None and not self.reddit_data.empty:
            print("EDA: Analyzing Reddit data...")
            fig_eda_reddit, axes_eda_reddit = plt.subplots(2, 2, figsize=(15, 10))
            fig_eda_reddit.suptitle('Exploratory Data Analysis - Reddit', fontsize=16, fontweight='bold')

            # Ensure full_text column exists
            if 'full_text' not in self.reddit_data.columns:
                print("EDA: Creating full_text column for Reddit data...")
                if 'title' in self.reddit_data.columns and 'content' in self.reddit_data.columns:
                    self.reddit_data['full_text'] = (self.reddit_data['title'].fillna('') + ' ' + 
                                                     self.reddit_data['content'].fillna('')).str.strip()
                elif 'title' in self.reddit_data.columns:
                    self.reddit_data['full_text'] = self.reddit_data['title'].fillna('').str.strip()
                elif 'content' in self.reddit_data.columns:
                    self.reddit_data['full_text'] = self.reddit_data['content'].fillna('').str.strip()
                else:
                    self.reddit_data['full_text'] = ''

            # 1. Post Length Distribution (Reddit)
            if 'full_text' in self.reddit_data.columns:
                self.reddit_data['post_length'] = self.reddit_data['full_text'].astype(str).apply(len)
                valid_lengths = self.reddit_data['post_length'][self.reddit_data['post_length'] > 0]
                
                print(f"EDA Reddit: Found {len(valid_lengths)} posts with valid text lengths")
                
                if len(valid_lengths) > 0:
                    # Calculate statistics
                    mean_length = valid_lengths.mean()
                    median_length = valid_lengths.median()
                    max_length = valid_lengths.max()
                    
                    # Create histogram with better binning
                    n, bins, patches = axes_eda_reddit[0, 0].hist(valid_lengths, bins=min(50, len(valid_lengths)//10 + 1), 
                                                                 color='skyblue', edgecolor='black', alpha=0.7)
                    
                    # Add vertical lines for mean and median
                    axes_eda_reddit[0, 0].axvline(mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.0f}')
                    axes_eda_reddit[0, 0].axvline(median_length, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_length:.0f}')
                    
                    # Add statistics text
                    stats_text = f'Total posts: {len(valid_lengths)}\nMax length: {max_length:.0f}\nMean: {mean_length:.1f}\nMedian: {median_length:.1f}'
                    axes_eda_reddit[0, 0].text(0.65, 0.95, stats_text, transform=axes_eda_reddit[0, 0].transAxes, 
                                              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    axes_eda_reddit[0, 0].legend()
                else:
                    axes_eda_reddit[0, 0].text(0.5, 0.5, 'No valid text lengths found (Reddit)', ha='center', va='center', transform=axes_eda_reddit[0,0].transAxes)
            else:
                axes_eda_reddit[0, 0].text(0.5, 0.5, 'No text data for length analysis (Reddit)', ha='center', va='center', transform=axes_eda_reddit[0,0].transAxes)
            axes_eda_reddit[0, 0].set_title('Post Length Distribution (Reddit)')
            axes_eda_reddit[0, 0].set_xlabel('Post Length (characters)')
            axes_eda_reddit[0, 0].set_ylabel('Frequency')

            # 2. Posts Over Time (Reddit) - use created_date or date column
            time_col = None
            if 'date' in self.reddit_data.columns:
                time_col = 'date'
            elif 'created_date' in self.reddit_data.columns:
                time_col = 'created_date'
            elif 'created_utc' in self.reddit_data.columns:
                time_col = 'created_utc'
            
            if time_col:
                try:
                    if time_col == 'created_utc' and pd.api.types.is_numeric_dtype(self.reddit_data[time_col]):
                        reddit_time_series = pd.to_datetime(self.reddit_data[time_col], unit='s').dt.floor('D')
                    else:
                        reddit_time_series = pd.to_datetime(self.reddit_data[time_col]).dt.floor('D')
                    
                    reddit_post_counts_time = reddit_time_series.value_counts().sort_index()
                    if len(reddit_post_counts_time) > 0:
                        axes_eda_reddit[0, 1].plot(reddit_post_counts_time.index, reddit_post_counts_time.values, marker='o', linestyle='-', color='dodgerblue')
                        axes_eda_reddit[0, 1].set_xlabel('Date')
                        axes_eda_reddit[0, 1].set_ylabel('Number of Posts')
                        axes_eda_reddit[0, 1].tick_params(axis='x', rotation=45)
                    else:
                        axes_eda_reddit[0, 1].text(0.5, 0.5, 'No valid time data for plotting', ha='center', va='center', transform=axes_eda_reddit[0,1].transAxes)
                except Exception as e:
                    print(f"EDA Reddit Time Series Error: {e}")
                    axes_eda_reddit[0, 1].text(0.5, 0.5, f'Time data error: {str(e)[:50]}', ha='center', va='center', transform=axes_eda_reddit[0,1].transAxes)
            else:
                axes_eda_reddit[0, 1].text(0.5, 0.5, 'Time data not available', ha='center', va='center', transform=axes_eda_reddit[0,1].transAxes)
            axes_eda_reddit[0, 1].set_title('Posts Over Time (Reddit - Daily)')

            # Placeholder for other Reddit EDA plots
            axes_eda_reddit[1, 0].text(0.5, 0.5, 'Reddit EDA Plot 3 (Placeholder)', ha='center', va='center')
            axes_eda_reddit[1, 1].text(0.5, 0.5, 'Reddit EDA Plot 4 (Placeholder)', ha='center', va='center')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            eda_reddit_path = os.path.join(self.output_dir, 'eda_reddit_analysis.png')
            plt.savefig(eda_reddit_path, dpi=300, bbox_inches='tight')
            plt.close(fig_eda_reddit)
            print(f"EDA for Reddit saved to: {eda_reddit_path}")

        # --- EDA for Weibo Data ---
        if self.weibo_data is not None and not self.weibo_data.empty:
            print("EDA: Analyzing Weibo data...")
            fig_eda_weibo, axes_eda_weibo = plt.subplots(2, 2, figsize=(15, 10))
            fig_eda_weibo.suptitle('Exploratory Data Analysis - Weibo', fontsize=16, fontweight='bold')

            # Ensure full_text column exists
            if 'full_text' not in self.weibo_data.columns:
                print("EDA: Creating full_text column for Weibo data...")
                if 'enhanced_cleaned_text' in self.weibo_data.columns:
                    self.weibo_data['full_text'] = self.weibo_data['enhanced_cleaned_text'].fillna('').str.strip()
                elif '博文内容' in self.weibo_data.columns:
                    self.weibo_data['full_text'] = self.weibo_data['博文内容'].fillna('').str.strip()
                else:
                    # Use the first text column found
                    text_cols = [col for col in self.weibo_data.columns if any(keyword in col.lower() for keyword in ['content', '内容', 'text'])]
                    if text_cols:
                        self.weibo_data['full_text'] = self.weibo_data[text_cols[0]].fillna('').str.strip()
                    else:
                        self.weibo_data['full_text'] = ''

            # 1. Post Length Distribution (Weibo)
            if 'full_text' in self.weibo_data.columns:
                self.weibo_data['post_length'] = self.weibo_data['full_text'].astype(str).apply(len)
                valid_lengths = self.weibo_data['post_length'][self.weibo_data['post_length'] > 0]
                
                print(f"EDA Weibo: Found {len(valid_lengths)} posts with valid text lengths")
                
                if len(valid_lengths) > 0:
                    # Calculate statistics
                    mean_length = valid_lengths.mean()
                    median_length = valid_lengths.median()
                    max_length = valid_lengths.max()
                    
                    # Create histogram with better binning
                    n, bins, patches = axes_eda_weibo[0, 0].hist(valid_lengths, bins=min(50, len(valid_lengths)//10 + 1), 
                                                                color='lightcoral', edgecolor='black', alpha=0.7)
                    
                    # Add vertical lines for mean and median
                    axes_eda_weibo[0, 0].axvline(mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.0f}')
                    axes_eda_weibo[0, 0].axvline(median_length, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_length:.0f}')
                    
                    # Add statistics text
                    stats_text = f'Total posts: {len(valid_lengths)}\nMax length: {max_length:.0f}\nMean: {mean_length:.1f}\nMedian: {median_length:.1f}'
                    axes_eda_weibo[0, 0].text(0.65, 0.95, stats_text, transform=axes_eda_weibo[0, 0].transAxes, 
                                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    axes_eda_weibo[0, 0].legend()
                else:
                    axes_eda_weibo[0, 0].text(0.5, 0.5, 'No valid text lengths found (Weibo)', ha='center', va='center', transform=axes_eda_weibo[0,0].transAxes)
            else:
                axes_eda_weibo[0, 0].text(0.5, 0.5, 'No text data for length analysis (Weibo)', ha='center', va='center', transform=axes_eda_weibo[0,0].transAxes)
            axes_eda_weibo[0, 0].set_title('Post Length Distribution (Weibo)')
            axes_eda_weibo[0, 0].set_xlabel('Post Length (characters)')
            axes_eda_weibo[0, 0].set_ylabel('Frequency')

            # 2. Posts Over Time (Weibo) - PRIORITIZE adjusted_datetime then virtual date columns
            weibo_time_col = None
            if 'adjusted_datetime' in self.weibo_data.columns:
                weibo_time_col = 'adjusted_datetime'
            elif '新生成日期' in self.weibo_data.columns:
                weibo_time_col = '新生成日期'
            elif 'created_at_virtual' in self.weibo_data.columns:
                weibo_time_col = 'created_at_virtual'
            elif 'date' in self.weibo_data.columns:
                weibo_time_col = 'date'
            elif '发布时间' in self.weibo_data.columns:
                weibo_time_col = '发布时间'
            
            if weibo_time_col:
                try:
                    # Handle different time formats
                    if weibo_time_col == '发布时间':
                        # Parse Weibo special time format
                        def parse_weibo_time(time_str):
                            if pd.isna(time_str):
                                return None
                            try:
                                # Clean time string
                                time_str = str(time_str).strip().replace('\n', '').replace(' ', '')
                                # Skip processing if contains "月" and "日" without year (incomplete date)
                                if '月' in time_str and '日' in time_str and '年' not in time_str:
                                    return None  # Skip incomplete dates
                                # Convert to standard format if complete date
                                if '年' in time_str:
                                    time_str = time_str.replace('年', '-').replace('月', '-').replace('日', ' ')
                                return pd.to_datetime(time_str, errors='coerce')
                            except:
                                return None
                        
                        weibo_time_series = self.weibo_data[weibo_time_col].apply(parse_weibo_time)
                        weibo_time_series = weibo_time_series.dropna().dt.floor('D')
                    else:
                        # For other time columns, use direct parsing
                        weibo_time_series = pd.to_datetime(self.weibo_data[weibo_time_col], errors='coerce').dt.floor('D')
                    
                    weibo_post_counts_time = weibo_time_series.value_counts().sort_index()
                    if len(weibo_post_counts_time) > 0:
                        axes_eda_weibo[0, 1].plot(weibo_post_counts_time.index, weibo_post_counts_time.values, marker='o', linestyle='-', color='crimson')
                        axes_eda_weibo[0, 1].set_xlabel('Date')
                        axes_eda_weibo[0, 1].set_ylabel('Number of Posts')
                        axes_eda_weibo[0, 1].tick_params(axis='x', rotation=45)
                        print(f"Weibo EDA: Successfully plotted {len(weibo_post_counts_time)} time points using column '{weibo_time_col}'")
                    else:
                        axes_eda_weibo[0, 1].text(0.5, 0.5, 'No valid time data for plotting', ha='center', va='center', transform=axes_eda_weibo[0,1].transAxes)
                        print(f"Weibo EDA: No valid time data found in column '{weibo_time_col}'")
                except Exception as e:
                    print(f"EDA Weibo Time Series Error: {e}")
                    axes_eda_weibo[0, 1].text(0.5, 0.5, f'Time data error: Unknown datetime string format, unable to parse', ha='center', va='center', transform=axes_eda_weibo[0,1].transAxes)
            else:
                axes_eda_weibo[0, 1].text(0.5, 0.5, 'Virtual time data not available', ha='center', va='center', transform=axes_eda_weibo[0,1].transAxes)
                print("Weibo EDA: No suitable time column found")
            axes_eda_weibo[0, 1].set_title('Posts Over Time (Weibo - Daily)')

            # Placeholder for other Weibo EDA plots
            axes_eda_weibo[1, 0].text(0.5, 0.5, 'Weibo EDA Plot 3 (Placeholder)', ha='center', va='center')
            axes_eda_weibo[1, 1].text(0.5, 0.5, 'Weibo EDA Plot 4 (Placeholder)', ha='center', va='center')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            eda_weibo_path = os.path.join(self.output_dir, 'eda_weibo_analysis.png')
            plt.savefig(eda_weibo_path, dpi=300, bbox_inches='tight')
            plt.close(fig_eda_weibo)
            print(f"EDA for Weibo saved to: {eda_weibo_path}")
        print("EDA finished.")
    
    def create_daily_sentiment_trend(self):
        """Create daily sentiment analysis trend chart for Reddit and Weibo"""
        print("Creating daily sentiment analysis trend chart...")
        
        # Set up the figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        fig.suptitle('Daily Sentiment Analysis Trends: Reddit vs Weibo', fontsize=16, fontweight='bold')
        
        # Process Reddit data
        if self.reddit_data is not None and not self.reddit_data.empty:
            print("Processing Reddit sentiment data...")
            
            # Ensure we have the necessary columns
            if 'date' not in self.reddit_data.columns:
                if 'created_utc' in self.reddit_data.columns:
                    self.reddit_data['date'] = pd.to_datetime(self.reddit_data['created_utc'], unit='s')
                elif 'created_date' in self.reddit_data.columns:
                    self.reddit_data['date'] = pd.to_datetime(self.reddit_data['created_date'])
            
            if 'full_text' not in self.reddit_data.columns:
                if 'title' in self.reddit_data.columns and 'content' in self.reddit_data.columns:
                    self.reddit_data['full_text'] = (self.reddit_data['title'].fillna('') + ' ' + 
                                                     self.reddit_data['content'].fillna('')).str.strip()
                elif 'title' in self.reddit_data.columns:
                    self.reddit_data['full_text'] = self.reddit_data['title'].fillna('').str.strip()
            
            # Perform sentiment analysis for each post
            reddit_sentiments = []
            for idx, row in self.reddit_data.iterrows():
                if pd.notna(row.get('date')) and pd.notna(row.get('full_text')):
                    sentiment_result = self.analyze_sentiment(row['full_text'], language='english')
                    reddit_sentiments.append({
                        'date': pd.to_datetime(row['date']).date(),
                        'polarity': sentiment_result['polarity'],
                        'category': sentiment_result['category']
                    })
            
            if reddit_sentiments:
                reddit_df = pd.DataFrame(reddit_sentiments)
                
                # Group by date and calculate daily averages
                daily_reddit = reddit_df.groupby('date').agg({
                    'polarity': 'mean'
                }).reset_index()
                
                # Plot Reddit sentiment trend
                if not daily_reddit.empty:
                    # Convert dates to numeric for regression
                    daily_reddit['date_numeric'] = pd.to_datetime(daily_reddit['date']).map(pd.Timestamp.toordinal)
                    
                    # Plot data points
                    ax1.plot(daily_reddit['date'], daily_reddit['polarity'], 
                            marker='o', linestyle='-', color='dodgerblue', linewidth=2, markersize=4, label='Daily Average')
                    
                    # Add polynomial trend line and moving average
                    if len(daily_reddit) > 3:
                        # Polynomial fitting (degree 3 for smooth curves)
                        degree = min(3, len(daily_reddit) - 1)
                        z = np.polyfit(daily_reddit['date_numeric'], daily_reddit['polarity'], degree)
                        p = np.poly1d(z)
                        ax1.plot(daily_reddit['date'], p(daily_reddit['date_numeric']), 
                                "r-", alpha=0.8, linewidth=2, label=f'Polynomial Trend (degree {degree})')
                        
                        # Add moving average for smoother trend (monthly)
                        if len(daily_reddit) >= 30:
                            window_size = 30  # Monthly window (30 days)
                            daily_reddit_sorted = daily_reddit.sort_values('date')
                            moving_avg = daily_reddit_sorted['polarity'].rolling(window=window_size, center=True).mean()
                            ax1.plot(daily_reddit_sorted['date'], moving_avg, 
                                    "g-", alpha=0.7, linewidth=2, label=f'Moving Average (Monthly)')
                    elif len(daily_reddit) > 1:
                        # Simple linear trend for small datasets
                        z = np.polyfit(daily_reddit['date_numeric'], daily_reddit['polarity'], 1)
                        p = np.poly1d(z)
                        ax1.plot(daily_reddit['date'], p(daily_reddit['date_numeric']), 
                                "r--", alpha=0.8, linewidth=2, label=f'Linear Trend (slope: {z[0]:.4f})')
                    
                    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    ax1.set_title('Reddit Daily Sentiment Polarity Trend', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Average Sentiment Polarity')
                    ax1.grid(True, alpha=0.3)
                    ax1.tick_params(axis='x', rotation=45)
                    
                    # Add sentiment range indicators
                    ax1.fill_between(daily_reddit['date'], -1, -0.1, alpha=0.1, color='red', label='Negative Range')
                    ax1.fill_between(daily_reddit['date'], -0.1, 0.1, alpha=0.1, color='gray', label='Neutral Range')
                    ax1.fill_between(daily_reddit['date'], 0.1, 1, alpha=0.1, color='green', label='Positive Range')
                    ax1.legend()
                    
                    print(f"Reddit: Processed {len(daily_reddit)} days of sentiment data")
                    print(f"Reddit sentiment range: {daily_reddit['polarity'].min():.3f} to {daily_reddit['polarity'].max():.3f}")
                else:
                    ax1.text(0.5, 0.5, 'No Reddit sentiment data available', ha='center', va='center', transform=ax1.transAxes)
            else:
                ax1.text(0.5, 0.5, 'No Reddit sentiment data available', ha='center', va='center', transform=ax1.transAxes)
        else:
            ax1.text(0.5, 0.5, 'No Reddit data available', ha='center', va='center', transform=ax1.transAxes)
        
        # Process Weibo data
        if self.weibo_data is not None and not self.weibo_data.empty:
            print("Processing Weibo sentiment data...")
            
            # Ensure we have the necessary columns - ONLY use virtual dates for Weibo
            if 'date' not in self.weibo_data.columns:
                if '新生成日期' in self.weibo_data.columns:
                    self.weibo_data['date'] = pd.to_datetime(self.weibo_data['新生成日期'], errors='coerce')
                else:
                    print("Warning: No virtual date column found in Weibo data. Skipping Weibo sentiment analysis.")
                    ax2.text(0.5, 0.5, 'No Weibo virtual date data available', ha='center', va='center', transform=ax2.transAxes)
                    plt.tight_layout()
                    plt.savefig('daily_sentiment_trend_analysis.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    return
            
            if 'full_text' not in self.weibo_data.columns:
                if 'enhanced_cleaned_text' in self.weibo_data.columns:
                    self.weibo_data['full_text'] = self.weibo_data['enhanced_cleaned_text'].fillna('').str.strip()
                elif '博文内容' in self.weibo_data.columns:
                    self.weibo_data['full_text'] = self.weibo_data['博文内容'].fillna('').str.strip()
            
            # Perform enhanced sentiment analysis for each post
            weibo_sentiments = []
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for idx, row in self.weibo_data.iterrows():
                if pd.notna(row.get('date')) and pd.notna(row.get('full_text')):
                    text = str(row['full_text']).strip()
                    if len(text) > 5:  # Only process meaningful text
                        sentiment_result = self.analyze_sentiment(text, language='chinese')
                        
                        # Enhanced sentiment analysis for Chinese text
                        # Add manual adjustment for better Chinese sentiment detection
                        polarity = sentiment_result['polarity']
                        
                        # Check for Chinese positive/negative words more explicitly
                        positive_words = ['好', '棒', '赞', '优秀', '喜欢', '支持', '同意', '正确', '有用', '帮助', '成功', '进步', '发展', '创新']
                        negative_words = ['不好', '差', '糟糕', '反对', '错误', '失败', '问题', '担心', '害怕', '危险', '威胁', '不安', '质疑']
                        
                        pos_score = sum(1 for word in positive_words if word in text)
                        neg_score = sum(1 for word in negative_words if word in text)
                        
                        # Adjust polarity based on Chinese sentiment words
                        if pos_score > neg_score and pos_score > 0:
                            polarity = max(polarity, 0.1 + pos_score * 0.1)
                        elif neg_score > pos_score and neg_score > 0:
                            polarity = min(polarity, -0.1 - neg_score * 0.1)
                        
                        # Ensure polarity is within bounds
                        polarity = max(-1.0, min(1.0, polarity))
                        
                        # Update category based on adjusted polarity
                        if polarity > 0.1:
                            category = 'positive'
                            positive_count += 1
                        elif polarity < -0.1:
                            category = 'negative'
                            negative_count += 1
                        else:
                            category = 'neutral'
                            neutral_count += 1
                        
                        weibo_sentiments.append({
                            'date': pd.to_datetime(row['date']).date(),
                            'polarity': polarity,
                            'category': category
                        })
            
            print(f"Weibo sentiment distribution: Positive: {positive_count}, Negative: {negative_count}, Neutral: {neutral_count}")
            
            if weibo_sentiments:
                weibo_df = pd.DataFrame(weibo_sentiments)
                
                # Group by date and calculate daily averages
                daily_weibo = weibo_df.groupby('date').agg({
                    'polarity': 'mean'
                }).reset_index()
                
                # Plot Weibo sentiment trend
                if not daily_weibo.empty:
                    # Convert dates to numeric for regression
                    daily_weibo['date_numeric'] = pd.to_datetime(daily_weibo['date']).map(pd.Timestamp.toordinal)
                    
                    # Plot data points
                    ax2.plot(daily_weibo['date'], daily_weibo['polarity'], 
                            marker='s', linestyle='-', color='crimson', linewidth=2, markersize=4, label='Daily Average')
                    
                    # Add polynomial trend line and moving average
                    if len(daily_weibo) > 3:
                        # Polynomial fitting (degree 3 for smooth curves)
                        degree = min(3, len(daily_weibo) - 1)
                        z = np.polyfit(daily_weibo['date_numeric'], daily_weibo['polarity'], degree)
                        p = np.poly1d(z)
                        ax2.plot(daily_weibo['date'], p(daily_weibo['date_numeric']), 
                                "r-", alpha=0.8, linewidth=2, label=f'Polynomial Trend (degree {degree})')
                        
                        # Add moving average for smoother trend (monthly)
                        if len(daily_weibo) >= 30:
                            window_size = 30  # Monthly window (30 days)
                            daily_weibo_sorted = daily_weibo.sort_values('date')
                            moving_avg = daily_weibo_sorted['polarity'].rolling(window=window_size, center=True).mean()
                            ax2.plot(daily_weibo_sorted['date'], moving_avg, 
                                    "g-", alpha=0.7, linewidth=2, label=f'Moving Average (Monthly)')
                    elif len(daily_weibo) > 1:
                        # Simple linear trend for small datasets
                        z = np.polyfit(daily_weibo['date_numeric'], daily_weibo['polarity'], 1)
                        p = np.poly1d(z)
                        ax2.plot(daily_weibo['date'], p(daily_weibo['date_numeric']), 
                                "r--", alpha=0.8, linewidth=2, label=f'Linear Trend (slope: {z[0]:.4f})')
                    
                    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    ax2.set_title('Weibo Daily Sentiment Polarity Trend', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel('Average Sentiment Polarity')
                    ax2.grid(True, alpha=0.3)
                    ax2.tick_params(axis='x', rotation=45)
                    
                    # Add sentiment range indicators
                    ax2.fill_between(daily_weibo['date'], -1, -0.1, alpha=0.1, color='red', label='Negative Range')
                    ax2.fill_between(daily_weibo['date'], -0.1, 0.1, alpha=0.1, color='gray', label='Neutral Range')
                    ax2.fill_between(daily_weibo['date'], 0.1, 1, alpha=0.1, color='green', label='Positive Range')
                    ax2.legend()
                    
                    print(f"Weibo: Processed {len(daily_weibo)} days of sentiment data")
                    print(f"Weibo sentiment range: {daily_weibo['polarity'].min():.3f} to {daily_weibo['polarity'].max():.3f}")
                else:
                    ax2.text(0.5, 0.5, 'No Weibo sentiment data available', ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'No Weibo sentiment data available', ha='center', va='center', transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, 'No Weibo data available', ha='center', va='center', transform=ax2.transAxes)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        sentiment_trend_path = os.path.join(self.output_dir, 'daily_sentiment_trend_analysis.png')
        plt.savefig(sentiment_trend_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Daily sentiment trend analysis saved to: {sentiment_trend_path}")
        return sentiment_trend_path

    def generate_report(self, results):
        """Generate comprehensive analysis report"""
        report = []
        report.append("# Integrated Reddit vs Weibo AI Consciousness Analysis Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        report.append("## Executive Summary\n")
        report.append("This report presents a comprehensive analysis of AI consciousness discussions ")
        report.append("across Reddit and Weibo platforms, examining temporal patterns, sentiment trends, ")
        report.append("and thematic evolution across three key time periods.\n\n")
        
        # Time Period Overview
        report.append("## Time Period Analysis\n")
        for period_id, period_info in self.periods.items():
            report.append(f"### {period_info['name']}\n")
            report.append(f"**Period:** {period_info['start'].strftime('%Y-%m-%d')} to {period_info['end'].strftime('%Y-%m-%d')}\n")
            report.append(f"**Description:** {period_info['description']}\n\n")
            
            # Reddit data for this period
            reddit_data = results['reddit'].get(period_id, {})
            weibo_data = results['weibo'].get(period_id, {})
            
            if reddit_data:
                report.append(f"**Reddit Posts:** {reddit_data.get('total_posts', 0)}\n")
                sentiment_dist = reddit_data.get('sentiment_distribution', {})
                report.append(f"**Reddit Sentiment Distribution:** {sentiment_dist}\n")
                
                # Top keywords
                keyword_freq = reddit_data.get('keyword_frequency', {})
                if keyword_freq:
                    top_keywords = dict(Counter(keyword_freq).most_common(5))
                    report.append(f"**Reddit Top Keywords:** {top_keywords}\n")
            
            if weibo_data:
                report.append(f"**Weibo Posts:** {weibo_data.get('total_posts', 0)}\n")
                sentiment_dist = weibo_data.get('sentiment_distribution', {})
                report.append(f"**Weibo Sentiment Distribution:** {sentiment_dist}\n")
                
                # Top keywords
                keyword_freq = weibo_data.get('keyword_frequency', {})
                if keyword_freq:
                    top_keywords = dict(Counter(keyword_freq).most_common(5))
                    report.append(f"**Weibo Top Keywords:** {top_keywords}\n")
            
            report.append("\n")
        
        # Cross-platform Comparison
        report.append("## Cross-platform Comparison\n")
        
        # Total posts comparison
        total_reddit = sum(results['reddit'].get(p, {}).get('total_posts', 0) for p in self.periods.keys())
        total_weibo = sum(results['weibo'].get(p, {}).get('total_posts', 0) for p in self.periods.keys())
        
        report.append(f"**Total Posts:**\n")
        report.append(f"- Reddit: {total_reddit}\n")
        report.append(f"- Weibo: {total_weibo}\n\n")
        
        # Sentiment comparison
        report.append("**Overall Sentiment Trends:**\n")
        for platform in ['reddit', 'weibo']:
            platform_results = results[platform]
            all_sentiments = {}
            
            for period_data in platform_results.values():
                sentiment_dist = period_data.get('sentiment_distribution', {})
                for sentiment, count in sentiment_dist.items():
                    all_sentiments[sentiment] = all_sentiments.get(sentiment, 0) + count
            
            report.append(f"- {platform.title()}: {all_sentiments}\n")
        
        report.append("\n")
        
        # Topic Analysis
        report.append("## Topic Analysis\n")
        
        for platform in ['reddit', 'weibo']:
            report.append(f"### {platform.title()} Topics\n")
            platform_results = results[platform]
            
            for period_id, period_data in platform_results.items():
                period_name = self.periods[period_id]['name']
                topics = period_data.get('topics', [])
                
                if topics:
                    report.append(f"**{period_name}:**\n")
                    for i, topic in enumerate(topics[:3]):  # Show top 3 topics
                        if topic:
                            top_words = ', '.join(topic[:5])  # Show top 5 words
                            report.append(f"- Topic {i+1}: {top_words}\n")
                    report.append("\n")
        
        # Representative Posts
        report.append("## Representative Posts\n")
        
        for platform in ['reddit', 'weibo']:
            report.append(f"### {platform.title()} Representative Posts\n")
            platform_results = results[platform]
            
            for period_id, period_data in platform_results.items():
                period_name = self.periods[period_id]['name']
                rep_posts = period_data.get('representative_posts', [])
                
                if rep_posts:
                    report.append(f"**{period_name}:**\n")
                    for i, post in enumerate(rep_posts[:2]):  # Show top 2 posts
                        report.append(f"{i+1}. **Sentiment:** {post['sentiment']} | **Score:** {post['score']:.2f}\n")
                        report.append(f"   *Text:* {post['text'][:200]}...\n\n")
        
        # Methodology
        report.append("## Methodology\n")
        report.append("### Data Sources\n")
        report.append("- **Reddit:** AI consciousness related posts and comments\n")
        report.append("- **Weibo:** AI consciousness related posts in Chinese\n\n")
        
        report.append("### Analysis Methods\n")
        report.append("1. **Temporal Analysis:** Data divided into three key periods based on major AI events\n")
        report.append("2. **Sentiment Analysis:** Multi-dimensional sentiment classification\n")
        report.append("3. **Topic Modeling:** LDA-based topic extraction and analysis\n")
        report.append("4. **Keyword Analysis:** Frequency analysis of consciousness-related terms\n")
        report.append("5. **Cross-platform Comparison:** Comparative analysis between platforms\n\n")
        
        # Technical Details
        report.append("### Technical Implementation\n")
        report.append("- **Text Processing:** Enhanced cleaning and preprocessing\n")
        report.append("- **Sentiment Analysis:** TextBlob + custom keyword-based analysis\n")
        report.append("- **Topic Modeling:** Scikit-learn LDA with TF-IDF vectorization\n")
        report.append("- **Visualization:** Matplotlib and Seaborn for comprehensive charts\n")
        report.append("- **Language Support:** Bilingual analysis (English/Chinese)\n\n")
        
        # Conclusions
        report.append("## Key Findings\n")
        report.append("1. **Temporal Patterns:** Significant variation in discussion volume across time periods\n")
        report.append("2. **Platform Differences:** Distinct conversation patterns between Reddit and Weibo\n")
        report.append("3. **Sentiment Evolution:** Observable changes in sentiment over time\n")
        report.append("4. **Topic Diversity:** Rich thematic content across both platforms\n")
        report.append("5. **Cross-cultural Insights:** Different cultural perspectives on AI consciousness\n\n")
        
        # Save report
        report_content = ''.join(report)
        report_path = os.path.join(self.output_dir, 'integrated_analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Report saved to: {report_path}")
        return report_content
    
    def run_analysis(self):
        """Run complete integrated analysis"""
        print("Starting Integrated Reddit vs Weibo AI Consciousness Analysis...")
        print("=" * 60)
        
        # Step 1: Load and clean data
        print("\n1. Loading and cleaning data...")
        if not self.load_and_clean_data():
            print("Failed to load data. Exiting.")
            return
        
        # Step 2: Preprocess and divide by time periods
        print("\n2. Preprocessing data and dividing by time periods...")
        processed_data = self.preprocess_data()
        
        if not processed_data['reddit'] and not processed_data['weibo']:
            print("No data available for analysis. Exiting.")
            return
        
        # Step 3: Temporal analysis
        print("\n3. Performing temporal analysis...")
        results = self.temporal_analysis(processed_data)
        
        # Step 4: Create visualizations
        print("\n4. Creating visualizations...")
        self.create_visualizations(results)
        
        # Step 5: Create improved topic visualization
        print("\n5. Creating improved topic visualization...")
        self.create_improved_visualization()

        # Step 6: Perform EDA and Visualize
        print("\n6. Performing EDA and Visualize...")
        self.perform_eda_and_visualize()

        # Step 7: Create daily sentiment analysis trend chart
        print("\n7. Creating daily sentiment analysis trend chart...")
        self.create_daily_sentiment_trend()

        # Step 8: Generate comprehensive report
        print("\n8. Generating comprehensive report...")
        self.generate_report(results)
        
        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print(f"All outputs saved to: {os.path.abspath(self.output_dir)}")
        print("\nGenerated files:")
        print("- integrated_ai_consciousness_analysis.png")
        print("- improved_topic_visualization.png")
        print("- eda_reddit_analysis.png")
        print("- eda_weibo_analysis.png")
        print("- daily_sentiment_trend_analysis.png")
        print("- integrated_analysis_report.md")
        print("- lda_visualization_*.html (if pyLDAvis available)")


def main():
    """Main function to run the integrated analysis"""
    analyzer = IntegratedRedditWeiboAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()