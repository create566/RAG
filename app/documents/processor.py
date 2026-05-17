"""
文档处理服务 - 对标Java的文档处理流水线
"""
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import re
import math

from app.core.logging import get_logger

logger = get_logger(__name__)


class ChunkStrategy:
    """切块策略"""

    @staticmethod
    def structural_chunk(text: str, headers: List[str] = None) -> List[str]:
        """基于文档结构的切块 - 对标Java的结构切块"""
        if not headers:
            headers = ["#", "##", "###", "第", "章", "节", "一、", "二、", "三、"]
        paragraphs = text.split("\n\n")
        # 如果没有明显段落结构，按行处理
        if len(paragraphs) <= 1:
            paragraphs = text.split("\n")

        chunks = []
        current_chunk = []

        for line in paragraphs:
            line_stripped = line.strip()
            if not line_stripped:
                if current_chunk:
                    chunk_text = "\n".join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                    current_chunk = []
                continue

            is_header = any(line_stripped.startswith(h) or h in line_stripped[:10] for h in headers)

            if is_header and current_chunk:
                chunk_text = "\n".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk = []

            current_chunk.append(line_stripped)

        if current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks if chunks else [text]

    @staticmethod
    def recursive_chunk(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """递归分块 - 对标Java的RecursiveTextSplit"""
        if len(text) <= max_chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + max_chunk_size

            if end < len(text):
                for sep in ["。", "！", "？", "；", "\n", "，"]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap if end < len(text) else end
            # 防止死循环
            if start <= 0 or start >= len(text):
                break

        return chunks

    @staticmethod
    def semantic_chunk(
        text: str,
        max_chunk_size: int = 800,
        min_chunk_size: int = 100,
        similarity_threshold: float = 0.15,
        config: Dict = None
    ) -> List[str]:
        """
        语义分块 - 基于句子相似度自动检测主题边界

        原理：
        1. 将文本拆为句子
        2. 计算相邻句子组的词重叠度（Jaccard 相似度 + IDF 加权）
        3. 在低相似度处切开，表示主题切换
        4. 在 chunk 大小约束下合并高相似度句子
        """
        config = config or {}

        # 1. 拆句（中英文句号、换行等）
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return [text] if text.strip() else []

        # 2. 计算每句的词集合
        sent_tokens = []
        for s in sentences:
            tokens = _tokenize(s)
            # 去停用词（简单版）
            tokens = [t for t in tokens if len(t) > 1 and t not in _STOP_WORDS]
            sent_tokens.append(set(tokens))

        # 3. 计算 IDF 权重（用于加权相似度）
        doc_count = len(sentences)
        idf = {}
        for tokens in sent_tokens:
            for t in tokens:
                idf[t] = idf.get(t, 0) + 1
        for t in idf:
            idf[t] = math.log((doc_count + 1) / (idf[t] + 1)) + 1

        # 4. 计算相邻句子间的加权相似度
        similarities = []
        for i in range(len(sent_tokens) - 1):
            sim = _weighted_jaccard(sent_tokens[i], sent_tokens[i+1], idf)
            similarities.append(sim)

        # 5. 动态阈值：均值 - 0.3 * 标准差，低于此阈值说明是主题边界
        if similarities:
            mean_sim = sum(similarities) / len(similarities)
            std_sim = (sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)) ** 0.5
            threshold = max(similarity_threshold, mean_sim - 0.3 * std_sim)
        else:
            threshold = similarity_threshold

        # 6. 按边界合并句子为 chunks
        chunks = []
        current = sentences[0]
        current_len = len(sentences[0])

        for i in range(1, len(sentences)):
            sent = sentences[i]
            sent_len = len(sent)
            sim = similarities[i-1] if i-1 < len(similarities) else 1.0

            # 如果超尺寸或遇到低相似度边界，且当前块足够大 → 切
            should_split = (
                (current_len + sent_len > max_chunk_size) or
                (sim < threshold and current_len >= min_chunk_size)
            )

            if should_split and current.strip():
                chunks.append(current.strip())
                current = sent
                current_len = sent_len
            else:
                current += sent
                current_len += sent_len

        if current.strip():
            # 最后一块如果太大，再用递归分块兜底
            if len(current) > max_chunk_size * 1.5:
                chunks.extend(ChunkStrategy.recursive_chunk(current, max_chunk_size))
            else:
                chunks.append(current.strip())

        return chunks if chunks else [text]

    @staticmethod
    async def llm_chunk(
        text: str,
        llm_service=None,
        max_chunk_size: int = 1000,
        config: Dict = None
    ) -> List[str]:
        """
        LLM智能切块 - 让模型判断最优分块边界

        处理低质量/复杂文档时使用，模型理解文档内容后确定切分点。
        默认关闭，仅在 structural + recursive + semantic 效果不佳时启用。
        """
        if not llm_service:
            return ChunkStrategy.semantic_chunk(text, max_chunk_size=max_chunk_size)

        # 如果文本太长，先粗略分块再让 LLM 处理每段
        if len(text) > 4000:
            coarse_chunks = ChunkStrategy.structural_chunk(text)
            if len(coarse_chunks) <= 1:
                coarse_chunks = ChunkStrategy.recursive_chunk(text, 4000, 0)
        else:
            coarse_chunks = [text]

        all_chunks = []
        for coarse in coarse_chunks:
            if len(coarse) < 200:
                all_chunks.append(coarse)
                continue

            prompt = f"""请分析以下文档内容，在合适的位置用 `---CHUNK---` 标记分块边界。

分块规则：
1. 保持语义完整性：一个主题/知识点尽量放在同一个块中
2. 控制块大小：每块约 300-800 字，不要切太碎也不要整段不切
3. 在自然边界切分：段落切换、话题转换处切开
4. 保留上下文：分块后每块应该能独立理解

文档内容：
{coarse}

请直接返回带 `---CHUNK---` 标记的完整文本，不要添加额外解释。"""

            try:
                response = await llm_service.chat(prompt, temperature=0.3, max_tokens=4096)
                # 去除可能的 markdown 代码块标记
                response = response.strip()
                if response.startswith("```"):
                    lines = response.split("\n")
                    response = "\n".join(lines[1:]) if len(lines) > 1 else response
                if response.endswith("```"):
                    response = response[:-3].strip()

                # 按标记切分
                parts = response.split("---CHUNK---")
                for part in parts:
                    part = part.strip()
                    if part:
                        # 如果 LLM 给的块仍然太大，递归分块兜底
                        if len(part) > max_chunk_size * 1.5:
                            all_chunks.extend(ChunkStrategy.recursive_chunk(part, max_chunk_size))
                        else:
                            all_chunks.append(part)
            except Exception as e:
                logger.warning(f"[LLM CHUNK] LLM切块失败: {e}，降级为语义分块")
                all_chunks.extend(ChunkStrategy.semantic_chunk(coarse, max_chunk_size=max_chunk_size))

        return all_chunks if all_chunks else [text]


# ---- 语义分块辅助函数 ----

_STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
}


def _split_sentences(text: str) -> List[str]:
    """将文本拆分为句子列表"""
    # 中英文句子分隔符
    pattern = r'(?<=[。！？\.\!\?\n])(?=[^\s])'
    sentences = re.split(pattern, text)
    # 过滤空句
    result = []
    for s in sentences:
        s = s.strip()
        if s:
            result.append(s)
    return result if result else [text]


def _tokenize(text: str) -> List[str]:
    """简单分词：中文字符级 + 英文词级 混合"""
    tokens = []
    # 提取中文单字（作为 unigram）+ 双字（bigram）
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    # 中文字符加入
    tokens.extend(chinese_chars)
    # 中文双字 bigram
    for i in range(len(chinese_chars) - 1):
        tokens.append(chinese_chars[i] + chinese_chars[i+1])
    # 英文单词
    english_words = re.findall(r'[a-zA-Z]{2,}', text.lower())
    tokens.extend(english_words)
    return tokens


def _weighted_jaccard(set1: set, set2: set, idf: Dict) -> float:
    """加权 Jaccard 相似度"""
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    union = set1 | set2
    if not union:
        return 0.0
    # 加权：交集词用 IDF 加权
    weighted_intersection = sum(idf.get(t, 1) for t in intersection)
    weighted_union = sum(idf.get(t, 1) for t in union)
    return weighted_intersection / weighted_union if weighted_union > 0 else 0.0


class DocumentProcessor:
    """文档处理器 - 对标Java的DocumentPreprocessService"""

    def __init__(self, tika_client=None, config: Dict = None):
        self.tika_client = tika_client
        self.config = config or {}

    async def parse_document(self, file_path: str) -> str:
        """解析文档 - 对标Java的TikaReaderHandler"""
        suffix = Path(file_path).suffix.lower()

        if suffix == ".pdf":
            return await self._parse_pdf(file_path)
        elif suffix == ".docx":
            return await self._parse_docx(file_path)
        elif suffix == ".txt":
            return await self._parse_txt(file_path)
        elif suffix == ".md":
            return await self._parse_md(file_path)
        else:
            return await self._parse_with_tika(file_path)

    async def _parse_pdf(self, file_path: str) -> str:
        """解析PDF"""
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"PDF解析失败: {str(e)}"

    async def _parse_docx(self, file_path: str) -> str:
        """解析DOCX"""
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            return f"DOCX解析失败: {str(e)}"

    async def _parse_txt(self, file_path: str) -> str:
        """解析TXT"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    async def _parse_md(self, file_path: str) -> str:
        """解析Markdown"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    async def _parse_with_tika(self, file_path: str) -> str:
        """使用Tika解析"""
        return "使用Tika解析"


class ChunkingPipeline:
    """
    组合式切块流水线 - 对标Java的组合切块引擎

    策略组合逻辑（按文档类型自动编排）：
    1. structural   — 主干策略，保留文档天然边界
    2. recursive    — 兜底策略，控制块大小不超标
    3. semantic     — 优化策略，在结构基础上精修语义边界
    4. llm          — 增强策略，处理低质量/复杂文档（需LLM服务）

    流水线可配置为 list 形式，例如：
      ["structural", "recursive", "semantic"]
      ["structural", "recursive", "semantic", "llm"]
    """

    def __init__(self, strategies: List[str] = None, config: Dict = None, llm_service=None):
        self.strategies = strategies or ["structural", "recursive"]
        self.config = config or {}
        self.llm_service = llm_service

    async def execute(self, text: str) -> List[str]:
        """
        按流水线顺序执行切块。
        每步输出的 chunks 作为下一步的输入：
        - structural:  整文 → 结构块
        - recursive:   对每个超尺寸的结构块再次切分
        - semantic:    对每个块做边界精修（合并/拆分）
        - llm:         对疑难块用 LLM 重新分块
        """
        result = [text]

        for strategy in self.strategies:
            if not result:
                break
            result = await self._apply_strategy(strategy, result)

        return result

    async def _apply_strategy(self, strategy: str, chunks: List[str]) -> List[str]:
        """将某个策略应用到所有 chunks，并对超限块递归处理"""
        output = []
        max_size = self.config.get("parent_chunk_size", 2000)

        for chunk in chunks:
            if strategy == "structural":
                # 结构切块只对第一步的原文有意义，对已切块跳过
                sub = ChunkStrategy.structural_chunk(
                    chunk,
                    headers=self.config.get("headers", ["#", "##", "###", "第", "章", "节", "一、"])
                )
            elif strategy == "recursive":
                # 递归分块：拆分超过 max_size 的块
                sub = await self._recursive_limit(chunk, max_size)
            elif strategy == "semantic":
                # 语义分块：精修边界，仅在块足够大时才做
                if len(chunk) > 300:
                    sub = ChunkStrategy.semantic_chunk(
                        chunk,
                        max_chunk_size=self.config.get("child_chunk_size", 500),
                        min_chunk_size=self.config.get("min_chunk_size", 100),
                        similarity_threshold=self.config.get("semantic_similarity_threshold", 0.15),
                        config=self.config
                    )
                else:
                    sub = [chunk]
            elif strategy == "llm":
                # LLM 切块：仅对内容杂乱、前面的策略可能切不好的块启用
                if len(chunk) > 500:
                    sub = await ChunkStrategy.llm_chunk(
                        chunk,
                        llm_service=self.llm_service,
                        max_chunk_size=self.config.get("child_chunk_size", 500),
                        config=self.config
                    )
                else:
                    sub = [chunk]
            else:
                sub = [chunk]

            output.extend(sub)

        return output

    async def _recursive_limit(self, chunk: str, max_size: int) -> List[str]:
        """递归限制块大小，在自然边界切断"""
        if len(chunk) <= max_size:
            return [chunk]
        return ChunkStrategy.recursive_chunk(
            chunk,
            max_chunk_size=max_size,
            overlap=self.config.get("overlap", 50)
        )


class TextSplitter:
    """文本分割器 - 对标Java的组合切块引擎"""

    def __init__(self, strategy: str = "structural", config: Dict = None, llm_service=None):
        self.strategy = strategy
        self.config = config or {}
        self.llm_service = llm_service

    async def split(self, text: str) -> List[str]:
        """执行切分 - 支持组合策略（逗号分隔）"""
        # 解析策略列表（支持 "structural,recursive,semantic" 组合写法）
        if "," in self.strategy:
            strategies = [s.strip() for s in self.strategy.split(",")]
        else:
            strategies = [self.strategy]

        pipeline = ChunkingPipeline(
            strategies=strategies,
            config=self.config,
            llm_service=self.llm_service
        )
        return await pipeline.execute(text)