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

# Load environment variables
load_dotenv()


class TextAnalysisAgent:
    """
    Text Analysis Agent, responsible for extracting keywords and generating meeting summaries from transcribed text
    """

    def __init__(self,
                 keyword_extraction_method: str = "jieba",  # jieba or llm
                 max_keywords: int = 10):
        """
        Initialize text analysis Agent
        
        Args:
            keyword_extraction_method: Keyword extraction method (jieba or llm)
            max_keywords: Maximum number of keywords
        """
        # Get configuration from environment variables
        self.llm_model = os.getenv("DASHSCOPE_LLM_MODEL_NAME", "deepseek-r1")
        self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        # Note: base_url should not include "chat/completions" path as ChatOpenAI adds it automatically
        self.dashscope_llm_base_url = os.getenv("DASHSCOPE_LLM_BASE_URL",
                                                "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.keyword_extraction_method = keyword_extraction_method
        self.max_keywords = max_keywords

        # Initialize LLM
        if self.dashscope_api_key:
            self.llm = ChatOpenAI(
                model=self.llm_model,
                api_key=self.dashscope_api_key,
                base_url=self.dashscope_llm_base_url
            )
        else:
            self.llm = None
            print("Warning: DASHSCOPE_API_KEY environment variable not found, will use pure local methods")

    def extract_keywords(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract keywords from text
        
        Args:
            text: Input text
            
        Returns:
            List of keywords and weights
        """
        if self.keyword_extraction_method == "jieba" or not self.llm:
            # Use jieba for keyword extraction
            return self._extract_keywords_jieba(text)
        else:
            # Use LLM for keyword extraction
            return self._extract_keywords_llm(text)

    def _extract_keywords_jieba(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract keywords using jieba

        Args:
            text: Input text

        Returns:
            List of keywords and weights
        """
        # Remove extra spaces and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\n\r]+', ' ', text)

        # Use TF-IDF to extract keywords
        keywords = jieba.analyse.extract_tags(
            text,
            topK=self.max_keywords,
            withWeight=True,
        )

        # If TF-IDF extraction is not effective, use TextRank
        if len(keywords) < 3:
            keywords = jieba.analyse.textrank(
                text,
                topK=self.max_keywords,
                withWeight=True,
            )

        return keywords

    def _extract_keywords_llm(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract keywords using LLM
        
        Args:
            text: Input text
            
        Returns:
            List of keywords and weights
        """
        if not self.llm:
            raise ValueError("LLM not initialized, please ensure DASHSCOPE_API_KEY environment variable is set")

        # Build prompt
        prompt = ChatPromptTemplate.from_template("""
        Please extract the most important {max_keywords} keywords from the following meeting text, and assign a weight between 0-1 to each keyword indicating its importance.
        The sum of weights should be close to 1.
        
        Meeting text:
        {text}
        
        Please return in JSON format, for example:
        {{{{"keywords": [["keyword1", 0.2], ["keyword2", 0.15], ...]}}}}
        """)

        # Set output parser
        parser = JsonOutputParser()

        # Build chain
        chain = prompt | self.llm | parser

        # Execute chain
        try:
            result = chain.invoke({
                "text": text,
                "max_keywords": self.max_keywords
            })

            # Ensure result format is correct
            if "keywords" in result:
                return result["keywords"]
            else:
                # If format is incorrect, use fallback method
                return self._extract_keywords_jieba(text)
        except Exception as e:
            print(f"Error when extracting keywords using LLM: {e}")
            # Fall back to jieba method if failed
            return self._extract_keywords_jieba(text)

    def generate_summary(self, text: str, summary_type: str = "comprehensive") -> str:
        """
        Generate meeting summary
        
        Args:
            text: Input text
            summary_type: Summary type (comprehensive: detailed summary, concise: brief summary)
            
        Returns:
            Generated summary text
        """
        if self.llm:
            # Use LLM to generate summary
            return self._generate_summary_llm(text, summary_type)
        else:
            # Use simple local method to generate summary
            return self._generate_summary_local(text)

    def _generate_summary_llm(self, text: str, summary_type: str = "comprehensive") -> str:
        """
        Generate summary using LLM
        
        Args:
            text: Input text
            summary_type: Summary type
            
        Returns:
            Generated summary text
        """
        if not self.llm:
            raise ValueError("LLM not initialized, please ensure DASHSCOPE_API_KEY environment variable is set")

        # Set prompt content based on summary type
        if summary_type == "comprehensive":
            summary_prompt = "Please provide a detailed meeting summary, including main topics discussed, consensus reached, action items proposed, and decisions made."
        else:
            summary_prompt = "Please provide a concise meeting summary, highlighting key points, limited to 200 characters."

        # Build prompt
        prompt = ChatPromptTemplate.from_template("""
        You are a professional meeting recording assistant. Please generate a meeting summary based on the following meeting text.
        
        {summary_prompt}
        
        Meeting text:
        {text}
        
        Please output the summary text directly without additional explanation.
        """)

        # Set output parser
        parser = StrOutputParser()

        # Build chain
        chain = prompt | self.llm | parser

        # Execute chain
        try:
            return chain.invoke({
                "text": text,
                "summary_prompt": summary_prompt
            })
        except Exception as e:
            print(f"Error when generating summary using LLM: {e}")
            # Fall back to local method if failed
            return self._generate_summary_local(text)

    def _generate_summary_local(self, text: str) -> str:
        """
        Generate simple summary using local method
        
        Args:
            text: Input text
            
        Returns:
            Generated simple summary text
        """
        # Extract keywords
        keywords = self._extract_keywords_jieba(text)
        keyword_str = ", ".join([kw[0] for kw in keywords[:5]])

        # Simple statistics
        sentences = re.split(r'[。！？；\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Build simple summary
        summary = f"The meeting mainly discussed content related to {keyword_str}. The meeting contains {len(sentences)} main points."

        # Add first and last sentences as context
        if len(sentences) > 0:
            summary += f" Start of discussion: {sentences[0][:50]}..."
        if len(sentences) > 1:
            summary += f" End of discussion: {sentences[-1][:50]}..."

        return summary

    def analyze_speaker_turns(self, text: str) -> Dict[str, Any]:
        """
        Analyze speaker turns
        
        Args:
            text: Input text with speaker markers, format like "A: Speech content\nB: Speech content"
            
        Returns:
            Speaker statistics
        """
        # Simple speaker recognition pattern
        speaker_pattern = r'^([A-Za-z0-9]+):'
        lines = text.strip().split('\n')

        speaker_counts = Counter()
        speaker_texts = {}

        current_speaker = None
        current_text = []

        for line in lines:
            match = re.match(speaker_pattern, line.strip())
            if match:
                # Process previous speaker's content
                if current_speaker:
                    speaker_counts[current_speaker] += 1
                    if current_speaker not in speaker_texts:
                        speaker_texts[current_speaker] = []
                    speaker_texts[current_speaker].extend(current_text)

                # Start new speaker
                current_speaker = match.group(1)
                current_text = [line[match.end():].strip()]
            else:
                # Continue current speaker's content
                current_text.append(line.strip())

        # Process last speaker's content
        if current_speaker:
            speaker_counts[current_speaker] += 1
            if current_speaker not in speaker_texts:
                speaker_texts[current_speaker] = []
            speaker_texts[current_speaker].extend(current_text)

        # Calculate each speaker's speech length
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
        Process complete meeting text and return comprehensive analysis results
        
        Args:
            text: Meeting text
            
        Returns:
            Comprehensive results including keywords, summary, speaker analysis, etc.
        """
        # Extract keywords
        keywords = self.extract_keywords(text)

        # Generate summaries
        comprehensive_summary = self.generate_summary(text, "comprehensive")
        concise_summary = self.generate_summary(text, "concise")

        # Analyze speakers
        speaker_analysis = self.analyze_speaker_turns(text)

        return {
            "keywords": keywords,
            "comprehensive_summary": comprehensive_summary,
            "concise_summary": concise_summary,
            "speaker_analysis": speaker_analysis,
            "raw_text": text
        }


# Example usage
if __name__ == "__main__":
    # Example meeting text
    sample_text = """
    A: Let's discuss the project progress and next steps today.
    B: Currently, front-end development is 80% complete. The main issue is that there are still some bugs in the user authentication module.
    A: How is the backend API interface development going?
    C: Backend interfaces are all completed and undergoing unit testing.
    B: We need to fix all bugs before next week, then conduct integration testing.
    A: Okay, let's have a project review meeting next Wednesday.
    """

    # Create Text Analysis Agent (automatically uses deepseek model configuration from .env)
    agent = TextAnalysisAgent()

    # Process meeting text
    result = agent.process_meeting_text(sample_text)

    # Print results
    print("=== Keywords ===")
    for keyword, weight in result["keywords"]:
        print(f"{keyword}: {weight:.4f}")

    print("\n=== Comprehensive Summary ===")
    print(result["comprehensive_summary"])

    print("\n=== Concise Summary ===")
    print(result["concise_summary"])

    print("\n=== Speaker Analysis ===")
    print(f"Speaker turns: {result['speaker_analysis']['speaker_counts']}")
    print(f"Speech lengths: {result['speaker_analysis']['speaker_text_lengths']}")