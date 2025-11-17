import os
import re
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

import jieba
import jieba.analyse
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()


class TextAnalysisAgent:
    """
    文本分析Agent，负责从转录文本中提取关键词、生成会议总结
    """

    def __init__(self,
                 keyword_extraction_method: str = "jieba",  # jieba or llm
                 max_keywords: int = 10):
        """
        初始化文本分析Agent
        
        Args:
            llm_model: 用于生成总结的LLM模型（默认使用环境变量中的配置）
            keyword_extraction_method: 关键词提取方法 (jieba 或 llm)
            max_keywords: 最大关键词数量
        """
        # 从环境变量获取配置
        self.llm_model = os.getenv("DASHSCOPE_LLM_MODEL_NAME", "deepseek-r1")
        self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        # 注意：base_url不应该包含"chat/completions"路径，因为ChatOpenAI会自动添加
        self.dashscope_llm_base_url = os.getenv("DASHSCOPE_LLM_BASE_URL",
                                                "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.keyword_extraction_method = keyword_extraction_method
        self.max_keywords = max_keywords

        # 初始化LLM
        if self.dashscope_api_key:
            self.llm = ChatOpenAI(
                model=self.llm_model,
                api_key=self.dashscope_api_key,
                base_url=self.dashscope_llm_base_url
            )
        else:
            self.llm = None
            print("警告: 未找到DASHSCOPE_API_KEY环境变量，将使用纯本地方法")

    def extract_keywords(self, text: str) -> List[Tuple[str, float]]:
        """
        从文本中提取关键词
        
        Args:
            text: 输入文本
            
        Returns:
            关键词和权重的列表
        """
        if self.keyword_extraction_method == "jieba" or not self.llm:
            # 使用jieba进行关键词提取
            return self._extract_keywords_jieba(text)
        else:
            # 使用LLM进行关键词提取
            return self._extract_keywords_llm(text)

    def _extract_keywords_jieba(self, text: str) -> List[Tuple[str, float]]:
        """
        使用jieba提取关键词

        Args:
            text: 输入文本

        Returns:
            关键词和权重的列表
        """
        # 移除多余空格和特殊字符
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\\n\\r]+', ' ', text)

        # 使用TF-IDF提取关键词
        keywords = jieba.analyse.extract_tags(
            text,
            topK=self.max_keywords,
            withWeight=True,
        )

        # 如果TF-IDF提取效果不佳，使用TextRank
        if len(keywords) < 3:
            keywords = jieba.analyse.textrank(
                text,
                topK=self.max_keywords,
                withWeight=True,
            )

        return keywords

    def _extract_keywords_llm(self, text: str) -> List[Tuple[str, float]]:
        """
        使用LLM提取关键词
        
        Args:
            text: 输入文本
            
        Returns:
            关键词和权重的列表
        """
        if not self.llm:
            raise ValueError("LLM未初始化，请确保DASHSCOPE_API_KEY环境变量已设置")

        # 构建提示
        prompt = ChatPromptTemplate.from_template("""
        请从以下会议文本中提取最重要的{max_keywords}个关键词，并为每个关键词分配一个0-1之间的权重，表示其重要性。
        权重总和应接近1。
        
        会议文本:
        {text}
        
        请以JSON格式返回，例如：
        {{"keywords": [["关键词1", 0.2], ["关键词2", 0.15], ...]}}
        """)

        # 设置输出解析器
        parser = JsonOutputParser()

        # 构建链
        chain = prompt | self.llm | parser

        # 执行链
        try:
            result = chain.invoke({
                "text": text,
                "max_keywords": self.max_keywords
            })

            # 确保结果格式正确
            if "keywords" in result:
                return result["keywords"]
            else:
                # 如果格式不正确，使用备用方法
                return self._extract_keywords_jieba(text)
        except Exception as e:
            print(f"使用LLM提取关键词时出错: {e}")
            # 失败时回退到jieba方法
            return self._extract_keywords_jieba(text)

    def generate_summary(self, text: str, summary_type: str = "comprehensive") -> str:
        """
        生成会议总结
        
        Args:
            text: 输入文本
            summary_type: 总结类型 (comprehensive: 详细总结, concise: 简洁总结)
            
        Returns:
            生成的总结文本
        """
        if self.llm:
            # 使用LLM生成总结
            return self._generate_summary_llm(text, summary_type)
        else:
            # 使用简单的本地方法生成总结
            return self._generate_summary_local(text)

    def _generate_summary_llm(self, text: str, summary_type: str = "comprehensive") -> str:
        """
        使用LLM生成总结
        
        Args:
            text: 输入文本
            summary_type: 总结类型
            
        Returns:
            生成的总结文本
        """
        if not self.llm:
            raise ValueError("LLM未初始化，请确保DASHSCOPE_API_KEY环境变量已设置")

        # 根据总结类型设置提示内容
        if summary_type == "comprehensive":
            summary_prompt = "请提供详细的会议总结，包括讨论的主要议题、达成的共识、提出的行动项和决策。"
        else:
            summary_prompt = "请提供简洁的会议摘要，突出重点内容，控制在200字以内。"

        # 构建提示
        prompt = ChatPromptTemplate.from_template("""
        你是一个专业的会议记录助手，请根据以下会议文本生成会议总结。
        
        {summary_prompt}
        
        会议文本:
        {text}
        
        请直接输出总结文本，不要添加额外的说明。
        """)

        # 设置输出解析器
        parser = StrOutputParser()

        # 构建链
        chain = prompt | self.llm | parser

        # 执行链
        try:
            return chain.invoke({
                "text": text,
                "summary_prompt": summary_prompt
            })
        except Exception as e:
            print(f"使用LLM生成总结时出错: {e}")
            # 失败时回退到本地方法
            return self._generate_summary_local(text)

    def _generate_summary_local(self, text: str) -> str:
        """
        使用本地方法生成简单总结
        
        Args:
            text: 输入文本
            
        Returns:
            生成的简单总结文本
        """
        # 提取关键词
        keywords = self._extract_keywords_jieba(text)
        keyword_str = ", ".join([kw[0] for kw in keywords[:5]])

        # 简单统计
        sentences = re.split(r'[。！？；\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 构建简单总结
        summary = f"会议主要讨论了关于{keyword_str}等内容。会议包含{len(sentences)}个主要观点。"

        # 添加第一个和最后一个句子作为上下文
        if len(sentences) > 0:
            summary += f" 开始讨论：{sentences[0][:50]}..."
        if len(sentences) > 1:
            summary += f" 结束讨论：{sentences[-1][:50]}..."

        return summary

    def analyze_speaker_turns(self, text: str) -> Dict[str, Any]:
        """
        分析说话人轮流发言情况
        
        Args:
            text: 带说话人标记的输入文本，格式如"A: 发言内容\nB: 发言内容"
            
        Returns:
            说话人统计信息
        """
        # 简单的说话人识别模式
        speaker_pattern = r'^([A-Za-z0-9]+):'
        lines = text.strip().split('\n')

        speaker_counts = Counter()
        speaker_texts = {}

        current_speaker = None
        current_text = []

        for line in lines:
            match = re.match(speaker_pattern, line.strip())
            if match:
                # 处理前一个说话人的内容
                if current_speaker:
                    speaker_counts[current_speaker] += 1
                    if current_speaker not in speaker_texts:
                        speaker_texts[current_speaker] = []
                    speaker_texts[current_speaker].extend(current_text)

                # 开始新的说话人
                current_speaker = match.group(1)
                current_text = [line[match.end():].strip()]
            else:
                # 延续当前说话人的内容
                current_text.append(line.strip())

        # 处理最后一个说话人的内容
        if current_speaker:
            speaker_counts[current_speaker] += 1
            if current_speaker not in speaker_texts:
                speaker_texts[current_speaker] = []
            speaker_texts[current_speaker].extend(current_text)

        # 计算每个说话人的发言长度
        speaker_lengths = {}
        for speaker, texts in speaker_texts.items():
            full_text = ' '.join(texts)
            speaker_lengths[speaker] = len(full_text)

        return {
            "speaker_counts": dict(speaker_counts),
            "speaker_text_lengths": speaker_lengths,
            "total_turns": sum(speaker_counts.values())
        }

    def process_meeting_text(self, text: str) -> Dict[str, Any]:
        """
        处理完整的会议文本，返回综合分析结果
        
        Args:
            text: 会议文本
            
        Returns:
            包含关键词、总结、说话人分析等的综合结果
        """
        # 提取关键词
        keywords = self.extract_keywords(text)

        # 生成总结
        comprehensive_summary = self.generate_summary(text, "comprehensive")
        concise_summary = self.generate_summary(text, "concise")

        # 分析说话人
        speaker_analysis = self.analyze_speaker_turns(text)

        return {
            "keywords": keywords,
            "comprehensive_summary": comprehensive_summary,
            "concise_summary": concise_summary,
            "speaker_analysis": speaker_analysis,
            "raw_text": text
        }


# 示例用法
if __name__ == "__main__":
    # 示例会议文本
    sample_text = """
    A: 今天我们讨论一下项目的进度和下一步计划。
    B: 目前前端开发已经完成了80%，主要问题是用户认证模块还有一些bug。
    A: 那后端的API接口开发情况如何？
    C: 后端接口已经全部完成，正在进行单元测试。
    B: 我们需要在下周前完成所有bug修复，然后进行集成测试。
    A: 好的，那我们下周三进行一次项目评审会议。
    """

    # 创建文本分析Agent (自动使用.env中的deepseek模型配置)
    agent = TextAnalysisAgent()

    # 处理会议文本
    result = agent.process_meeting_text(sample_text)

    # 打印结果
    print("=== 关键词 ===")
    for keyword, weight in result["keywords"]:
        print(f"{keyword}: {weight:.4f}")

    print("\n=== 详细总结 ===")
    print(result["comprehensive_summary"])

    print("\n=== 简洁总结 ===")
    print(result["concise_summary"])

    print("\n=== 说话人分析 ===")
    print(f"发言次数: {result['speaker_analysis']['speaker_counts']}")
    print(f"发言长度: {result['speaker_analysis']['speaker_text_lengths']}")
