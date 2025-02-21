#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""配置管理模块

用于加载和验证环境变量配置
"""

import os
from typing import Dict
from dotenv import load_dotenv
from loguru import logger
import logging
from datetime import datetime
from pydantic_settings import BaseSettings

# 定义日志文件路径
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log')

class Settings(BaseSettings):
    EXA_API_KEY: str
    OPENROUTER_API_KEY: str
    UNSPLASH_API_KEY: str
    OUTPUT_DIR: str = "output"
    TOPICS_FILE: str = "topics.md"
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

def load_config() -> Settings:
    """加载并验证配置"""
    config = Settings()
    # 替换原有验证逻辑
    logger.info("配置加载成功")
    return config