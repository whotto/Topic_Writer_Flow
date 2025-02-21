#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自动化微信公众号及小红书图文生成系统

此脚本用于自动化生成微信公众号文章和小红书笔记，包括：
- 从Markdown文件中读取选题
- 使用EXA AI进行内容检索
- 通过Unsplash获取配图
- 生成并存储文章内容
- 同步到飞书文档
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from config import load_config
from utils import FileHandler, ExaAIClient, OpenRouterDeepSeekClient, UnsplashClient, _insert_images
import uuid
from apscheduler.schedulers.blocking import BlockingScheduler
from concurrent.futures import ThreadPoolExecutor, as_completed

# 获取logger实例
logger = logging.getLogger(__name__)

class ContentGenerator:
    def __init__(self):
        """初始化内容生成器
        """
        self.config = load_config()
        self.topics = []
        self.output_dir = 'output'
        self.deepseek = OpenRouterDeepSeekClient(self.config.OPENROUTER_API_KEY)
        self.exa_client = ExaAIClient(self.config.EXA_API_KEY)
        self.unsplash_client = UnsplashClient(self.config.UNSPLASH_API_KEY)
        
        # 统一管理系统提示词
        self.prompts = {
            'wechat_system': """你是一位精通多种写作流派与表达技巧的文体转换大师。你擅长将专业内容转化为通俗易懂的文章，并确保内容既专业又有趣。
你的写作特点：
1. 文章结构清晰，层次分明
2. 善于使用类比和举例来解释复杂概念
3. 能将专业术语转化为通俗易懂的语言
4. 擅长设置悬念和互动，增强文章吸引力
5. 注重数据支撑，增强文章说服力
6. 每篇文章都确保2000字以上，内容充实

你的目标读者是：
1. 对该领域感兴趣但缺乏专业背景的读者
2. 寻求实用建议和解决方案的从业者
3. 期待了解行业趋势和最新发展的决策者""",
            
            'xiaohongshu_system': """你是一位深受欢迎的小红书达人，擅长将专业内容转化为有趣、实用的笔记。
你的写作特点：
1. 文风活泼亲切，擅长使用表情符号
2. 善于讲故事，让内容更有代入感
3. 专注于实用性建议和干货分享
4. 擅长设置互动话题，引导读者评论
5. 善于使用标题吸引眼球
6. 注重分点论述，让内容更易消化

你的目标读者是：
1. 喜欢轻松愉快氛围的年轻群体
2. 寻求干货和实用建议的职场人
3. 关注个人成长和技能提升的学习者""",
            
            'image_system': """你是一位专业的图片搜索专家，擅长为内容选择最合适的配图。
你的工作职责：
1. 分析文章主题和内容重点
2. 提取适合视觉呈现的关键概念
3. 生成准确的英文搜索关键词
4. 确保关键词具有可搜索性
5. 避免过于抽象的概念
6. 优先选择场景类和实物类的关键词

输出要求：
1. 每行一个英文关键词
2. 关键词要简洁且具体
3. 确保关键词与主题高度相关
4. 避免使用特殊字符
5. 每个关键词不超过3个单词
6. 只输出关键词，不要其他任何内容"""
        }

    def read_topics(self) -> List[str]:
        """从Markdown文件中读取选题列表

        Returns:
            选题标题列表
        """
        topics_file = os.path.join(os.getcwd(), 'topic.md')
        if not os.path.exists(topics_file):
            logger.error(f"选题文件不存在: {topics_file}")
            return []
        return FileHandler.read_markdown_topics(topics_file)

    def process_images(self, topic: str, keywords: List[str]) -> List[str]:
        """处理图片
        
        Args:
            topic: 文章主题
            keywords: 图片关键词列表
            
        Returns:
            图片URL列表
        """
        try:
            # 添加图片搜索容错机制
            if not keywords:
                logger.warning("使用备用关键词策略")
                keywords = [topic.split('：')[0], "technology"]
            
            # 添加图片质量过滤
            image_urls = []
            for keyword in keywords:
                logger.info(f"开始搜索关键词 '{keyword}' 的图片")
                photos = self.unsplash_client.search_photos(keyword)
                if photos:
                    # 获取图片URL
                    for photo in photos:
                        if 'urls' in photo and 'regular' in photo['urls']:
                            if photo['width'] >= 1200 and photo['height'] >= 800:
                                image_urls.append(photo['urls']['regular'])
            
            return image_urls
            
        except Exception as e:
            logger.error(f"处理图片失败: {str(e)}")
            logger.exception(e)
            return []

    def generate_content(self, topic: str) -> Dict[str, Any]:
        """生成内容
        
        Args:
            topic: 文章主题
            
        Returns:
            Dict 包含生成内容的字典
        """
        try:
            # 1. 并行执行EXA搜索和图片关键词生成
            with ThreadPoolExecutor(max_workers=2) as executor:
                search_future = executor.submit(self.exa_client.search_content, topic)
                image_keywords_future = executor.submit(self.deepseek.generate_image_keywords, topic)
                
                search_result = search_future.result()
                image_keywords = image_keywords_future.result()
            
            if not search_result or not search_result.get('summaries'):
                logger.error(f"EXA返回空结果，响应结构: {search_result}")
                return self._generate_default_content(topic)
            
            materials = search_result.get('summaries', [])
            keywords = search_result.get('keywords', [])
            
            # 2. 并行处理图片搜索
            image_urls = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for keyword in image_keywords:
                    futures.append(executor.submit(self.unsplash_client.search_photos, keyword))
                
                for future in as_completed(futures):
                    photos = future.result()
                    if photos and len(photos) > 0:
                        url = photos[0]['urls']['regular']
                        if url not in image_urls:
                            image_urls.append(url)
            
            # 3. 构建增强版提示词
            article_prompt = f"""你是一位精通多种写作流派与表达技巧的文体转换大师。请基于以下素材，撰写一篇主题为《{topic}》的微信公众号文章。

素材内容：
{chr(10).join(materials[:3])}

关键概念：
{', '.join(keywords[:5])}

写作要求：
1. 文章结构（2000字以上）：
   - 【开篇】用生动的场景或故事引入，点明主题价值
   - 【背景】结合行业现状和痛点，引出问题
   - 【分析】用数据和案例深入剖析，突出核心观点
   - 【方案】提供清晰可行的解决方案
   - 【总结】呼应开篇，展望未来

2. 内容特点：
   - 每个部分都要有小标题
   - 重要观点用加粗标记
   - 数据要标注来源
   - 适当使用类比和比喻
   - 语言要生动有趣
   - 使用「」作为中文引号

3. 互动设计：
   - 设置3-5个思考问题
   - 每部分结束设置互动引导
   - 结尾设置行动号召

4. 排版要求：
   - 使用Markdown格式
   - 合理分段，段落间要空行
   - 重要内容使用引用格式
   - 在关键位置预留配图位置
   - 使用"---"作为分隔符

请开始创作，直接输出成品文本。"""

            # 4. 生成文章
            article = self.deepseek.generate_article(
                topic=topic,
                keywords=', '.join(keywords[:5]),
                article_materials=article_prompt,
                system_prompt=self.prompts['wechat_system']
            )
            
            if not article or not isinstance(article, dict):
                logger.error("文章生成失败")
                return self._generate_default_content(topic)
            
            # 5. 处理文章格式和图片插入
            content = article['正文']
            content = self.deepseek._format_content(content, 'wechat')
            content = _insert_images(content, image_urls, 'wechat')
            
            # 6. 生成小红书笔记
            xiaohongshu_prompt = f"""请将以下文章改写成一篇吸引人的小红书笔记，要求：

1. 标题要求：
   - 简洁有力，突出核心价值
   - 使用数字、问号等提高点击率
   - 长度控制在15-20字
   - 必须使用表情符号点缀

2. 正文要求：
   - 字数控制在600-800字
   - 开头用表情符号吸引眼球
   - 分段描述，每段2-3句话
   - 多用"你"、"我"等第一人称
   - 语言要口语化、有趣
   - 重点内容加粗处理
   - 适当加入互动性问句

3. 结构要求：
   - 开篇点题，抛出痛点
   - 分点阐述，每点都用表情符号开头
   - 总结核心观点
   - 结尾设置互动引导

4. 标签要求：
   - 3-5个相关标签
   - 包含话题标签和领域标签
   - 标签要热门且相关

原文主题：{topic}
参考内容：
{content[:500]}

请按照小红书的风格重新创作，确保内容专业性的同时增加趣味性和互动性。"""
            
            xiaohongshu_note = self.deepseek.generate_content(
                prompt=xiaohongshu_prompt,
                system_prompt=self.prompts['xiaohongshu_system'],
                max_tokens=1000,  # 减少token数量
                temperature=0.8
            )
            
            if not xiaohongshu_note:
                xiaohongshu_note = {
                    'title': f"【干货分享】{topic}",
                    'content': "内容生成失败，请稍后重试...",
                    'tags': "#技术干货 #经验分享 #实战技巧",
                }
            
            # 7. 处理小红书笔记格式和图片
            xiaohongshu_note['content'] = self.deepseek._format_content(
                xiaohongshu_note['content'], 
                'xiaohongshu'
            )
            xiaohongshu_note['content'] = _insert_images(
                xiaohongshu_note['content'], 
                image_urls[:3],
                'xiaohongshu'
            )
            
            return {
                'wechat': {
                    'title': topic,
                    'content': content,
                    'images': image_urls
                },
                'xiaohongshu': xiaohongshu_note
            }
            
        except Exception as e:
            logger.error(f"生成内容失败: {str(e)}")
            return self._generate_default_content(topic)

    def _generate_default_content(self, topic: str) -> Dict:
        """生成默认内容防止崩溃"""
        return {
            'wechat': {
                'title': topic,
                'content': f"很抱歉，关于《{topic}》的内容生成失败，可能原因：\n1. 暂无相关行业案例\n2. 技术资料尚未公开\n3. 系统临时故障",
                'word_count': 0,
                'images': []
            },
            'xiaohongshu': {
                'title': f"【行业洞察】{topic}",
                'content': f"当前暂无{topic}相关案例分享，推荐关注我们的技术专栏获取最新动态",
                'tags': "#系统错误 #技术前沿"
            }
        }

    def save_content(self, content: Dict[str, Any]) -> bool:
        """增强版保存逻辑"""
        try:
            # 获取图片URL
            image_urls = content['wechat'].get('images', [])
            
            # 公众号保存
            wechat_path = FileHandler.save_markdown(
                content['wechat'], 
                os.path.join(self.output_dir, 'wechat'),
                'wechat',
                image_urls=image_urls[:3]  # 最多插入3张图
            )
            
            # 小红书保存
            xiaohongshu_path = FileHandler.save_markdown(
                content['xiaohongshu'],
                os.path.join(self.output_dir, 'xiaohongshu'), 
                'xiaohongshu',
                image_urls=image_urls[:4]  # 小红书最多4张
            )
            
            return True
            
        except Exception as e:
            logger.error(f"保存流程异常：{str(e)}")
            return False

    def sync_to_feishu(self, content: Dict[str, Any]) -> bool:
        pass

def main():
    """主函数"""
    try:
        # 初始化内容生成器
        generator = ContentGenerator()
        
        # 读取选题
        topics = generator.read_topics()
        if not topics:
            logger.error("未找到有效选题")
            return
            
        logger.info(f"成功读取 {len(topics)} 个选题")
        
        # 处理每个选题
        for topic in topics:
            try:
                logger.info(f"开始处理选题: {topic}")
                
                # 生成内容
                content = generator.generate_content(topic)
                if not content:
                    logger.error(f"生成内容失败: {topic}")
                    continue
                    
                # 保存内容
                if generator.save_content(content):
                    logger.info(f"内容已保存: {topic}")
                else:
                    logger.error(f"保存内容失败: {topic}")
                    
            except Exception as e:
                logger.error(f"处理选题出错: {topic}, 错误: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")

def auto_publish():
    scheduler = BlockingScheduler()
    # 每天9点执行生成任务
    scheduler.add_job(main, 'cron', hour=9)
    scheduler.start()

if __name__ == '__main__':
    main()