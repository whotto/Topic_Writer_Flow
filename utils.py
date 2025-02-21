#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""工具函数模块

包含API集成和辅助功能：
- EXA AI API集成
- DeepKSeek API集成
- Unsplash API集成
- 飞书API集成
- 文件处理工具
"""

import json
import os
import re  # 添加re模块导入
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import cached_property
from hashlib import md5
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import openai  # 添加 openai 导入
import requests
from diskcache import Cache
from loguru import logger
from markdown import Markdown
from PIL import Image
from psutil import virtual_memory
from ratelimit import limits, sleep_and_retry
from requests.adapters import HTTPAdapter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from urllib3.util.retry import Retry

from config import load_config


# 配置请求会话
def create_session():
    session = requests.Session()
    retry = Retry(
        total=3,  # 减少重试次数
        backoff_factor=0.5,  # 减少重试等待时间
        status_forcelist=[500, 502, 503, 504, 429],
        allowed_methods=["HEAD", "GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.verify = False  # 禁用SSL验证
    requests.packages.urllib3.disable_warnings()
    return session

class APIClientBase:
    """API客户端基类"""
    def __init__(self, api_key: str):
        self.session = create_session()
        self.api_key = api_key
        self.base_url = ""
        
    @retry(stop=stop_after_attempt(3), 
           wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError))
           )
    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """统一请求处理"""
        url = f"{self.base_url}{endpoint}"
        headers = kwargs.get('headers', {})
        headers.update({"Authorization": f"Bearer {self.api_key}"})
        
        try:
            response = self.session.request(
                method,
                url,
                headers=headers,
                timeout=(2, 20),  # 减少超时时间
                **kwargs
            )
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            logger.error(f"API请求失败: {e.response.status_code} {e.response.text}")
            raise

class ExaAIClient(APIClientBase):
    """继承自基类的EXA客户端"""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.exa.ai"
        self.session = create_session()
        self.headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.cache = Cache('/tmp/exa_cache', 
                          size_limit=5000000,  # 5MB缓存
                          ttl=3600)  # 1小时过期

    def _call_api(self, method: str, endpoint: str, **kwargs):
        """统一的API调用方法"""
        try:
            response = super()._request(method, endpoint, **kwargs)
            
            if response.status_code != 200:
                raise ValueError(f"API错误 {response.status_code}")
            
            return response.json()
                
        except Exception as e:
            logger.error(f"API调用失败: {str(e)}")
            raise

    def search_content(self, topic: str, num_results: int=3, highlights_per_url: int=2):
        """搜索内容"""
        try:
            # 移除书名号和特殊字符
            clean_topic = topic.strip('《》').replace('：', ' ').replace('，', ' ')
        
            endpoint = "/search"
            data = {
                "query": clean_topic,
                "numResults": num_results,
                "startPublishedDate": "2022-01-01T00:00:00.000Z",
                "type": "neural",  # 使用神经搜索
                "useAutoprompt": True,
                "contents": {
                    "text": True,
                    "highlights": {
                        "highlightsPerUrl": highlights_per_url,
                        "numSentences": 3,
                        "query": clean_topic
                    },
                    "summary": {
                        "query": clean_topic
                    }
                },
                "includeDomains": ["zhihu.com", "csdn.net", "juejin.cn", "51cto.com"],
                "category": "technology"
            }
            
            logger.info(f"正在搜索: {clean_topic}")
            response = self._call_api("POST", endpoint, json=data)
            
            results = response.get('results', [])
            
            if not results:
                logger.info(f"未找到相关内容")
                return {
                    'summaries': ["暂无相关内容"],
                    'keywords': [clean_topic],
                    'topic': topic  # 保留原始主题
                }
            
            # 提取内容
            summaries = []
            keywords = set([clean_topic])
            
            for result in results:
                try:
                    contents = result.get('contents', {})
                    if not contents:
                        continue
                        
                    # 尝试提取摘要
                    summary = contents.get('summary', '')
                    if summary and len(summary) > 50:
                        summaries.append(summary)
                        continue
                            
                    # 尝试提取高亮内容
                    highlights = contents.get('highlights', [])
                    if highlights:
                        highlight_text = ' '.join([h.get('text', '') for h in highlights if h.get('text')])
                        if len(highlight_text) > 50:
                            summaries.append(highlight_text)
                            continue
                                
                    # 尝试提取正文
                    text = contents.get('text', '')
                    if text and len(text) > 100:
                        summaries.append(text[:500])
                            
                except Exception as e:
                    logger.error(f"处理结果失败: {str(e)}")
                    continue
            
            if not summaries:
                summaries = ["未找到相关内容摘要"]
            
            result_data = {
                'summaries': summaries[:5],  # 限制摘要数量
                'keywords': list(keywords)[:10],  # 限制关键词数量
                'topic': topic  # 保留原始主题
            }
            
            logger.info(f"搜索完成")
            return result_data
                
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return {
                'summaries': ["搜索失败，请稍后重试"],
                'keywords': [clean_topic],
                'topic': topic  # 保留原始主题
            }

    def _process_result(self, result: Dict[str, Any]) -> Tuple[str, set]:
        """增强版结果处理"""
        try:
            # 检查result是否包含必要字段
            if not isinstance(result, dict):
                logger.error(f"无效的结果格式: {type(result)}")
                return "", set()
            
            contents = result.get('contents', {})
            # 按照API文档的结构提取内容
            highlights = []
            if isinstance(contents.get('highlights'), list):
                highlights = [h.get('text', '') for h in contents['highlights'] if isinstance(h, dict)]
            
            # 提取摘要
            summary = contents.get('summary', '')
            if summary and isinstance(summary, str):
                if len(summary) >= 100:
                    return summary, set()
            
            # 处理高亮内容
            if highlights:
                highlight_text = ' '.join(highlights)
                if len(highlight_text) >= 100:
                    return highlight_text, set()
            
            # 处理全文内容
            text = contents.get('text', '')
            if text and isinstance(text, str):
                if len(text) >= 200:
                    return self._generate_summary(text, 500), set()
            
            return "", set()
        except Exception as e:
            logger.error(f"处理结果失败: {str(e)}")
            return "", set()

    def _calculate_readability(self, text: str) -> float:
        """简易可读性评分算法"""
        try:
            # 计算平均句子长度
            sentences = text.split('.')
            avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
            
            # 计算生僻词比例（使用内置词库）
            common_words = {
                '的', '是', '在', '和', '有', '为', '等', '与', '中', '大', '要',
                '我们', '可以', '通过', '进行', '有效', '提升', '案例', '应用',
                '行业', '实战', '效果', '工具', '方法', '分析', '数据', '客户',
                '服务', '产品', '市场', '技术', '管理', '系统', '平台', '解决方案'
            }
            words = text.lower().split()
            rare_ratio = 1 - len([w for w in words if w in common_words])/len(words)
            
            return min(0.8 + (avg_sentence_len < 25)*0.1 - rare_ratio*0.2, 1.0)
        except Exception as e:
            logger.error(f"计算可读性失败: {str(e)}")
            return 0.5

    def _extract_entities(self, text: str) -> List[str]:
        """简化版实体提取"""
        try:
            import jieba
            words = jieba.lcut(text)
            return [w for w in words if len(w) > 1]  # 返回长度大于1的词语作为实体
        except Exception as e:
            logger.error(f"实体提取失败: {str(e)}")
            return []

    def _expand_content(self, text: str) -> str:
        """内容扩展增强"""
        expansion_prompt = f"""
        请对以下内容进行专业扩展，要求：
        1. 补充具体案例，注明企业名称和应用效果
        2. 添加行业最新数据（2023-2024年）
        3. 增加实施步骤说明
        4. 最终字数需达到2000字以上
        
        原始内容：
        {text}
        """
        return self._call_api("POST", "/expand", json={"text": text}, headers=self.headers)

    def _summarize_highlights(self, highlights: List[Dict]) -> str:
        """智能摘要生成"""
        return "\n".join([h['text'] for h in highlights if 'text' in h])

    def _generate_summary(self, text: str, max_length: int=500) -> str:
        """生成内容摘要"""
        return text[:max_length] + "..." if len(text) > max_length else text

class OpenRouterDeepSeekClient:
    """使用 OpenRouter API 调用的 DeepSeek 客户端"""
    
    def __init__(self, openrouter_api_key: str):
        """初始化客户端"""
        self.client = openai.OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com",
                "X-Title": "AI Content Generator"
            }
        )
        self.model_name = "deepseek/deepseek-r1-distill-llama-70b"
        
    @retry(stop=stop_after_attempt(3), 
           wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_api(self, messages: List[Dict], **kwargs) -> Optional[str]:
        """统一的API调用方法"""
        try:
            # 添加日志记录请求信息
            logger.info(f"调用 OpenRouter API，模型: {self.model_name}")
            logger.info(f"请求参数: {kwargs}")
            
            # 添加进度提示
            logger.info("正在等待API响应...")
            start_time = time.time()
            
            # 使用新版 OpenAI 库的方式调用 API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                timeout=180,  # 设置3分钟超时
                **kwargs
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"API响应时间: {elapsed_time:.2f}秒")
            
            if not response or not response.choices:
                raise ValueError("API返回空响应")
                
            content = response.choices[0].message.content
            logger.info("API调用成功")
            return content
            
        except Exception as e:
            logger.error(f"API调用失败: {str(e)}")
            logger.error(f"请求信息 - 模型: {self.model_name}, 消息数: {len(messages)}")
            raise

    def generate_article(self, topic: str, **kwargs) -> Dict[str, Any]:
        """使用 OpenRouter API 调用 DeepSeek 模型生成文章
        
        Args:
            topic: 文章主题
            
        Returns:
            Dict 包含文章内容的字典
        """
        try:
            if not topic or not isinstance(topic, str):
                logger.error("无效的文章主题")
                return {
                    "标题": "",
                    "正文": "生成失败：无效的主题",
                    "正文字数": 0
                }
                
            # 从 kwargs 中获取关键词和素材
            keywords = kwargs.get('keywords', '')
            article_materials = kwargs.get('article_materials', '')
                
            # 构建增强版提示词
            prompt = f"""你是一位精通多种写作流派与表达技巧的文体转换大师。请基于以下素材，根据主题《{topic}》撰写一篇富有感染力和教育价值的微信公众号文章。

写作要求：
1. 内容结构（2000字以上）：
   - 【导语】用具象类比和生动画面引发读者兴趣
   - 【现状分析】结合最新行业数据，以对话方式与读者交流
   - 【案例解读】至少3个具体应用案例，用生动形象阐释抽象概念
   - 【实操建议】以问答形式详细指导，预答读者疑惑
   - 【结语】引导读者思考，展望未来

2. 写作特征：
   - 多用"你"与读者直接对话
   - 用生动具体的类比解释专业概念
   - 适时设问，引导读者思考
   - 巧妙植入诙谐元素增添趣味
   - 多用主动句式，语言简洁有力
   - 使用「」作为中文引号
   
3. 行文规范：
   - 段落自然过渡，逻辑清晰
   - 保持语言流畅，避免生硬
   - 专业术语必须配合通俗解释
   - 观点鲜明，论述有力
   - 确保改写后核心含义不变

关键词：{keywords}
参考素材：{article_materials}

请开始创作，直接输出成品文本，不要做任何编辑说明。"""
            
            logger.info(f"开始生成文章: {topic}")
            logger.info(f"关键词: {keywords}")
            logger.info(f"素材长度: {len(article_materials)} 字符")
            
            # 调用 OpenRouter API
            messages = [
                {"role": "system", "content": "你是一位专业的内容创作者，擅长撰写微信公众号文章。你的文章总是详尽且超过2000字。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self._call_api(
                messages=messages,
                max_tokens=4000,  # 增加 max_tokens
                temperature=kwargs.get('temperature', 0.7),
                top_p=0.9,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            # 处理响应
            if not response:
                logger.error("API返回空响应")
                return {
                    "标题": topic,
                    "正文": "生成失败：API返回为空",
                    "正文字数": 0
                }
                
            content = response.strip()
            if not content:
                logger.error("生成的内容为空")
                return {
                    "标题": topic,
                    "正文": "生成失败：内容为空",
                    "正文字数": 0
                }
                
            # 将生成的内容格式化为字典
            result = {
                "标题": topic,
                "正文": content,
                "正文字数": len(content)
            }
            
            logger.info(f"生成文章成功，字数: {len(content)}")
            
            # 字数验证（至少2000字）
            if len(content) < 2000:
                logger.warning(f"文章字数不足: {len(content)}字，尝试补充内容")
                # 尝试补充内容而不是完全重新生成
                supplement_prompt = f"""请在保持原文结构的基础上，为以下文章补充更多细节和案例，使总字数超过2000字：

{content}"""
                
                try:
                    supplement = self._call_api(
                        messages=[
                            {"role": "system", "content": "你是一位专业的内容编辑，擅长扩充和完善文章内容。"},
                            {"role": "user", "content": supplement_prompt}
                        ],
                        max_tokens=4000,
                        temperature=0.7
                    )
                    
                    if supplement and len(supplement) > len(content):
                        result["正文"] = supplement
                        result["正文字数"] = len(supplement)
                        logger.info(f"成功补充内容，最终字数: {len(supplement)}")
                except Exception as e:
                    logger.error(f"补充内容失败: {str(e)}")
            
            return result
                
        except Exception as e:
            logger.error(f"生成文章失败: {str(e)}")
            return {
                "标题": topic if isinstance(topic, str) else "",
                "正文": "生成失败：发生错误",
                "正文字数": 0
            }

    def generate_content(self, prompt: str, system_prompt: str = None, **kwargs) -> Dict[str, Any]:
        """生成内容的通用方法
        
        Args:
            prompt: 提示词
            system_prompt: 系统提示词（可选）
            **kwargs: 其他参数
            
        Returns:
            Dict 包含生成内容的字典
        """
        try:
            # 构建消息列表
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # 添加进度提示
            logger.info("开始生成内容...")
            start_time = time.time()
            
            response = self._call_api(
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 2000),
                temperature=kwargs.get('temperature', 0.8),
                presence_penalty=kwargs.get('presence_penalty', 0.2),
                frequency_penalty=kwargs.get('frequency_penalty', 0.3)
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"内容生成完成，用时: {elapsed_time:.2f}秒")
            
            if response:
                content = response.strip()
                
                # 提取标题（如果有）
                title = ""
                if content.startswith('#'):
                    title_end = content.find('\n')
                    if title_end > 0:
                        title = content[1:title_end].strip()
                        content = content[title_end:].strip()
                
                # 处理正文格式
                content = self._format_content(content, kwargs.get('platform', 'wechat'))
                
                # 处理标签
                tags = []
                if '#' in content:
                    # 提取文末的标签
                    main_content = []
                    for line in content.split('\n'):
                        if line.strip().startswith('#'):
                            tags.extend([tag.strip() for tag in line.split('#') if tag.strip()])
                        else:
                            main_content.append(line)
                    content = '\n'.join(main_content).strip()
                
                # 如果没有提取到标签，生成默认标签
                if not tags:
                    tags = ["人工智能", "技术创新", "经验分享"]
                
                return {
                    "title": title,
                    "content": content,
                    "word_count": len(content),
                    "tags": ' '.join([f"#{tag}" for tag in tags])
                }
            else:
                logger.error("API返回内容为空")
                return None
                
        except Exception as e:
            logger.error(f"生成内容失败: {str(e)}")
            return None

    def _format_content(self, content: str, platform: str) -> str:
        """格式化内容（总入口）"""
        if not content:
            return ""

        # 1. 移除提示性文字
        prompt_words = [
            r'【开篇】', r'【背景】', r'【分析】', r'【方案】', r'【总结】',
            r'【导语】', r'【现状分析】', r'【案例解读】', r'【实操建议】', r'【结语】',
            r'【互动环节】', r'【互动引导】', r'【行动号召】',
            r'开篇：', r'背景：', r'分析：', r'方案：', r'总结：',
            r'导语：', r'现状：', r'案例：', r'建议：', r'结语：',
            r'互动环节：', r'互动引导：', r'行动号召：',
            r'\*\*互动环节\*\*', r'\*\*思考问题\*\*', r'\*\*延伸阅读\*\*',
            r'让我们开始吧', r'让我们一起来看看', r'接下来', r'首先', r'最后',
            r'一起来探讨', r'写在最后', r'本文小结', r'总结一下'
        ]
        
        # 使用正则表达式移除提示性文字（包括后面的换行符）
        for word in prompt_words:
            # 使用 r 前缀确保正确处理转义字符
            content = re.sub(fr"{word}\\s*\\n*", "", content)
            # 处理可能的标题形式
            content = re.sub(fr"^#{{1,6}}\\s*{word}\\s*\\n*", "", content, flags=re.MULTILINE)

        # 2. 清理多余的空行
        content = re.sub(r'\n{3,}', '\n\n', content)

        # 3. 清理段落开头的"首先"、"其次"等过渡词
        transition_words = [
            r'^首先[，,]?', r'^其次[，,]?', r'^然后[，,]?', r'^接下来[，,]?',
            r'^最后[，,]?', r'^总的来说[，,]?', r'^总而言之[，,]?'
        ]
        for word in transition_words:
            content = re.sub(word, '', content, flags=re.MULTILINE)

        # 4. 处理标题格式
        content = re.sub(r'^#{{1,6}}\\s*(.+?)(?:：|:)', r'## \1', content, flags=re.MULTILINE)

        # 5. 优化分隔符使用
        content = re.sub(r'\n*---\n*', '\n\n---\n\n', content)

        # 6. 处理引用格式
        content = re.sub(r'(?<!>)\n>', '\n\n>', content)
        content = re.sub(r'>\s*\n(?!>)', '>\n\n', content)

        # 7. 处理列表格式
        content = re.sub(r'(?<!-)\n-', '\n\n-', content)
        content = re.sub(r'(?<!\d\.)\n\d\.', '\n\n1.', content)

        return content.strip()

    def _remove_prompt_words(self, content: str) -> str:
        """移除提示性文字"""
        # 移除到 _format_content 方法中
        pass

    def generate_image_keywords(self, topic: str) -> List[str]:
        """生成图片关键词
        
        Args:
            topic: 文章主题
            
        Returns:
            关键词列表
        """
        try:
            prompt = f"""请为主题《{topic}》生成3个用于Unsplash图库搜索的英文关键词。要求：
1. 每行一个关键词
2. 使用简单的英文词组
3. 关键词要具体且视觉化
4. 避免抽象概念
5. 适合搜索高质量配图

直接输出关键词，不要其他任何内容。

示例输出：
business meeting
data analysis
modern office"""
            
            # 构建消息列表
            messages = [
                {"role": "system", "content": "你是一位专业的图片搜索专家。请直接输出3个英文关键词，每行一个，不要其他任何内容。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self._call_api(
                messages=messages,
                max_tokens=100,
                temperature=0.5,
                top_p=0.9
            )
            
            if response:
                # 清理和验证关键词
                keywords = []
                for line in response.strip().split('\n'):
                    keyword = line.strip()
                    if (
                        keyword and 
                        keyword.replace(' ', '').isalnum() and  # 只包含字母数字和空格
                        len(keyword.split()) <= 3  # 最多3个单词
                    ):
                        keywords.append(keyword)
                
                if keywords:
                    logger.info(f"成功生成关键词: {keywords[:3]}")
                    return keywords[:3]
            
            logger.warning(f"生成的关键词无效，使用备用策略")
            return self._get_backup_keywords(topic)
                
        except Exception as e:
            logger.error(f"生成图片关键词失败: {str(e)}")
            return self._get_backup_keywords(topic)
            
    def _get_backup_keywords(self, topic: str) -> List[str]:
        """获取备用关键词"""
        logger.info("使用备用关键词策略")
        # 移除书名号和特殊字符
        clean_topic = topic.strip('《》').replace('：', ' ').split()[0]
        
        # 根据主题类型返回不同的关键词组合
        if any(kw in topic.lower() for kw in ['python', '编程', '开发']):
            return [
                "python programming",
                "code development",
                "software engineering"
            ]
        elif any(kw in topic.lower() for kw in ['ai', '人工智能', '深度学习']):
            return [
                "artificial intelligence",
                "deep learning",
                "neural network"
            ]
        elif any(kw in topic.lower() for kw in ['金融', '理财', 'roi']):
            return [
                "financial technology",
                "business analytics",
                "investment management"
            ]
        else:
            return [
                clean_topic,
                "modern technology",
                "business innovation"
            ]

    def _retry_generate(self, topic: str, **kwargs) -> Dict[str, Any]:
        """重试生成文章的辅助方法"""
        try:
            # 使用新的kwargs，避免参数冲突
            new_kwargs = kwargs.copy()
            new_kwargs['temperature'] = 0.8
            return self.generate_article(topic, **new_kwargs)
        except Exception as e:
            logger.error(f"重试生成文章失败: {str(e)}")
            return None

class UnsplashClient:
    """Unsplash API客户端"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.unsplash.com"
        self.headers = {
            "Authorization": f"Client-ID {api_key}",
            "Accept-Version": "v1"
        }
        self.session = create_session()
        logger.info("初始化 Unsplash 客户端")

    def search_photos(self, query: str, per_page: int = 3) -> List[Dict[str, Any]]:
        """搜索图片
        
        Args:
            query: 搜索关键词
            per_page: 返回结果数量
            
        Returns:
            图片信息列表
        """
        try:
            logger.info(f"开始搜索图片，关键词: {query}")
            endpoint = "/search/photos"
            params = {
                "query": query,
                "per_page": per_page,
                "orientation": "landscape"
            }
            
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            result = response.json()
            
            if not result.get('results'):
                logger.warning(f"未找到相关图片: {query}")
                return []
                
            logger.info(f"成功获取 {len(result['results'])} 张图片")
            return result['results']
            
        except Exception as e:
            logger.error(f"搜索图片失败: {str(e)}")
            logger.exception(e)
            return []

    def download_image(self, url: str) -> Optional[bytes]:
        """下载并验证图片"""
        try:
            response = self.session.get(url, timeout=5)
            image_data = response.content
            
            # 验证图片格式
            try:
                with Image.open(BytesIO(image_data)) as img:
                    if img.size[0] > 1200 or img.size[1] > 1200:
                        img.thumbnail((1200, 1200))
                        buffer = BytesIO()
                        img.save(buffer, format=img.format)
                        image_data = buffer.getvalue()
                    logger.info("图片格式验证成功")
                    return image_data
            except Exception as e:
                logger.error(f"图片处理失败: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"图片下载失败: {str(e)}")
            return None

    def get_random_photo(self, query: str = None) -> Dict[str, Any]:
        """获取随机图片

        Args:
            query: 可选的搜索关键词

        Returns:
            图片信息字典
        """
        try:
            logger.info(f"开始获取随机图片，关键词: {query if query else '无'}")
            endpoint = f"{self.base_url}/photos/random"
            params = {}
            if query:
                params["query"] = query
            
            response = self.session.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            result = response.json()
            
            logger.info("成功获取随机图片")
            return result
            
        except Exception as e:
            logger.error(f"获取随机图片失败: {str(e)}")
            logger.exception(e)
            return {}

class FileHandler:
    """文件处理工具"""
    @staticmethod
    def read_markdown_topics(file_path: str) -> List[str]:
        """读取Markdown格式的选题文件

        Args:
            file_path: 文件路径

        Returns:
            选题列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 修改解析逻辑，适应当前的选题文件格式
            topics = []
            for line in content.split('\n'):
                line = line.strip()
                if line and '|' in line:
                    # 分割表格行
                    columns = [col.strip() for col in line.split('|')]
                    # 检查是否包含《》格式的标题
                    for col in columns:
                        if col.startswith('《') and col.endswith('》'):
                            topics.append(col)
                            break
            return topics
        except Exception as e:
            logger.error(f"读取选题文件失败: {str(e)}")
            return []

    @staticmethod
    def save_markdown(content: Dict[str, Any], output_dir: str, platform: str = 'wechat', image_urls: List[str] = None) -> str:
        """保存Markdown文件"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成规范的文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 确保title存在且为字符串
            title = content.get('title', '')
            if not isinstance(title, str):
                title = str(title)
            
            # 清理标题
            title = re.sub(r'[<>:"/\\|?*]', '_', title)
            title = title.strip('《》')  # 移除书名号
            
            # 生成文件名
            if platform == 'xiaohongshu':
                filename = f"{timestamp}_【干货分享】{title}.md"
            else:
                filename = f"《{title}》_{timestamp}.md"
            
            filepath = os.path.join(output_dir, filename)
            
            # 获取文章内容
            article_content = content.get('content', '')
            
            # 移除文章开头的重复标题
            article_content = re.sub(r'^#\s*《?' + re.escape(title) + r'》?\s*\n+', '', article_content)
            article_content = re.sub(r'^#.*?\n+', '', article_content)  # 移除其他可能的标题
            
            # 构建文章内容
            md_parts = []
            
            if platform == 'wechat':
                md_parts.extend([
                    f"# {title}",
                    "",
                    f"*{datetime.now().strftime('%Y-%m-%d %H:%M')}*",
                    "",
                    "---",
                    "",
                    article_content.strip(),
                    "",
                    "---",
                    "",
                    f"*字数：{len(article_content)}*"
                ])
            else:  # xiaohongshu
                md_parts.extend([
                    f"# 【干货分享】{title}",
                    "",
                    article_content.strip(),
                    "",
                    content.get('tags', '')
                ])
            
            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(md_parts))
            
            logger.info(f"文件已保存: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"保存文件失败: {str(e)}")
            return ""

    @staticmethod
    def save_article(content: Dict[str, Any], output_dir: str, platform: str = 'wechat') -> str:
        """保存文章到Markdown文件

        Args:
            content: 文章内容字典
            output_dir: 输出目录
            platform: 平台(wechat/xiaohongshu)

        Returns:
            保存的文件路径
        """
        try:
            # 基础检查
            if not isinstance(content, dict) or 'title' not in content or 'content' not in content:
                logger.error("无效的文章内容格式")
                return ""

            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            title = content['title'].replace('/', '_').replace('\\', '_')
            filename = f"{timestamp}_{title}.md"
            file_path = os.path.join(output_dir, filename)

            # 构建Markdown内容
            md_parts = []
            
            if platform == 'wechat':
                # 微信公众号文章格式
                md_parts.extend([
                    f"# {content['title']}",
                    "",
                    "## 文章信息",
                    f"- 创建时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"- 字数：{len(content['content'])}",
                    "",
                    "## 正文",
                    ""
                ])

                # 处理正文内容（包含引用和图片）
                body_lines = content['content'].split('\n')
                current_section = []
                
                for line in body_lines:
                    if '[此处需要配图：' in line:
                        # 如果有累积的段落，先添加
                        if current_section:
                            md_parts.append('\n'.join(current_section))
                            md_parts.append("")
                            current_section = []
                            
                        # 获取图片索引
                        img_index = len([p for p in md_parts if '![' in p])
                        if img_index < len(content.get('images', [])):
                            md_parts.extend([
                                "---",
                                f"![配图{img_index + 1}]({content['images'][img_index]})",
                                "---",
                                ""
                            ])
                    elif '[来源：' in line:
                        # 处理引用
                        current_section.append(line)
                    else:
                        current_section.append(line)

                # 添加最后的段落
                if current_section:
                    md_parts.append('\n'.join(current_section))

                # 提取并添加参考来源
                references = re.findall(r'\[来源：(.*?)\]', content['content'])
                if references:
                    md_parts.extend([
                        "",
                        "## 参考来源",
                        ""
                    ])
                    for i, ref in enumerate(references, 1):
                        md_parts.append(f"{i}. {ref}")

            else:  # xiaohongshu
                # 小红书笔记格式
                md_parts.extend([
                    f"# {content['title']}",
                    "",
                    content['content'],
                    ""
                ])

                # 添加图片（最多3张）
                if 'images' in content:
                    for img_url in content['images'][:3]:
                        md_parts.extend([
                            "---",
                            f"![图片]({img_url})",
                            "---",
                            ""
                        ])

                # 添加标签
                if 'tags' in content:
                    md_parts.append(content['tags'])

            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(md_parts))

            logger.info(f"文章已保存到: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"保存文章失败: {str(e)}")
            return ""

    def edit_file(self, target_file: str, instructions: str, code_edit: str):
        """编辑文件内容"""
        try:
            # 读取文件内容
            with open(target_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换内容
            new_content = content.replace(
                """class OpenRouterDeepSeekClient:
    \"\"\"使用 OpenRouter API 调用的 DeepSeek 客户端\"\"\"

    def __init__(self, openrouter_api_key: str):
        \"\"\"初始化客户端\"\"\"
        self.openai_client = OpenAI(
            api_key=openrouter_api_key,
            api_base="https://openrouter.ai/api/v1",
        )
        self.model_name = "deepseek/deepseek-chat"
        self.headers = {
            "HTTP-Referer": "<YOUR_SITE_URL>",
            "X-Title": "<YOUR_SITE_NAME>",
        }""",
                """class OpenRouterDeepSeekClient:
    \"\"\"使用 OpenRouter API 调用的 DeepSeek 客户端\"\"\"
    
    def __init__(self, openrouter_api_key: str):
        \"\"\"初始化客户端\"\"\"
        # 设置 OpenRouter API 配置
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = openrouter_api_key
        
        self.model_name = "deepseek/deepseek-chat"
        self.headers = {
            "HTTP-Referer": "https://github.com",
            "X-Title": "AI Content Generator",
        }"""
            )
            
            # 写入文件
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            return True
            
        except Exception as e:
            logger.error(f"编辑文件失败: {str(e)}")
            return False

def _insert_images(content: str, images: List[str], platform: str) -> str:
    """智能插入图片"""
    if not images:
        return content
    
    # 分段处理
    paragraphs = content.split('\n\n')
    result = []
    image_count = 0
    max_images = 3 if platform == 'xiaohongshu' else 5
    
    # 计算关键位置
    total_paragraphs = len(paragraphs)
    intro_end = total_paragraphs // 4  # 文章前1/4
    main_body_middle = total_paragraphs // 2  # 文章中间
    conclusion_start = total_paragraphs * 3 // 4  # 文章后3/4
    
    # 标记重要段落
    important_positions = {
        intro_end,
        main_body_middle,
        conclusion_start
    }
    
    for i, para in enumerate(paragraphs):
        if image_count >= max_images:
            result.append(para)
            continue
        
        # 添加段落
        result.append(para)
        
        # 判断是否应该在此处添加图片
        should_add_image = (
            # 基本条件
            para.strip() and
            not para.startswith('#') and
            not para.startswith('>') and
            image_count < len(images) and
            # 满足以下条件之一：
            (
                # 1. 在重要位置
                i in important_positions or
                # 2. 段落较长且包含完整句子
                (len(para) > 150 and ('。' in para or '！' in para or '？' in para))
            )
        )
        
        if should_add_image:
            # 确保段落结束且有适当的空白
            result.append("")
            
            if platform == 'wechat':
                result.extend([
                    "---",
                    "",
                    f"![配图{image_count + 1}]({images[image_count]})",
                    "",
                    "---"
                ])
            else:  # xiaohongshu
                result.extend([
                    f"![图片{image_count + 1}]({images[image_count]})",
                    ""
                ])
            
            result.append("")  # 确保图片后有空行
            image_count += 1
    
    # 如果还有未使用的图片，添加到文章末尾
    if image_count < min(len(images), max_images):
        result.extend(["", "---", ""])
        
        while image_count < min(len(images), max_images):
            if platform == 'wechat':
                result.extend([
                    f"![配图{image_count + 1}]({images[image_count]})",
                    "",
                ])
            else:
                result.extend([
                    f"![图片{image_count + 1}]({images[image_count]})",
                    ""
                ])
            image_count += 1
    
    return '\n'.join(result)