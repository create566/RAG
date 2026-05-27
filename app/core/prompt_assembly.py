"""
Prompt组装服务 - 对标Java的RagPromptAssemblyService
"""
from typing import Dict, Any, List

from app.core.logging import get_logger

logger = get_logger(__name__)


class RagPromptAssemblyService:
    """RAG Prompt组装服务"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.per_sub_question_budget = self.config.get("per_sub_question_budget", 1500)
        self.total_budget = self.config.get("total_budget", 4000)

    def assemble(self, plan, retrieval_context) -> Dict[str, str]:
        """组装Prompt - 对标Java的assemble方法"""
        evidence_parts = []
        reference_id_map = []

        sub_questions = retrieval_context.sub_question_evidence_list
        references = retrieval_context.flatten_references()

        logger.info(f"[PROMPT] sub_questions count: {len(sub_questions)}, references count: {len(references)}")

        # 构建引用
        ref_idx = 1
        for sq_evidence in sub_questions:
            if not sq_evidence.references:
                continue

            sq_text = f"子问题{ref_idx}: {sq_evidence.sub_question}\n"
            sq_text += "相关证据:\n"

            for ref in sq_evidence.references[:3]:
                content = ref.get("content_preview", "")
                if len(content) > self.per_sub_question_budget:
                    content = self._truncate_at_paragraph_boundary(content, self.per_sub_question_budget)

                # 获取来源信息
                doc_name = ref.get("document_name", "未知文档")
                section_path = ref.get("section_path", "")
                chunk_index = ref.get("chunk_index", 0)

                ref_id = f"[{ref_idx}]"
                source_info = f"{ref_id} {content}"
                if section_path:
                    source_info += f"\n   来源：{doc_name} / {section_path}"
                else:
                    source_info += f"\n   来源：{doc_name}"
                sq_text += source_info + "\n"
                reference_id_map.append(ref)
                ref_idx += 1

            evidence_parts.append(sq_text)

        # 合并所有证据
        all_evidence = "\n\n".join(evidence_parts)

        # 构建系统Prompt - 严格忠实约束
        system_prompt = """你是严格基于上下文的问答助手，必须遵守以下规则：
1. 只允许使用上下文提供的信息回答，严禁脑补、拓展、篡改、新增任何内容
2. 上下文没有提到的信息，一律回答"无相关信息"，禁止自行发挥
3. 严格忠于原文条款，不修改数值、不增减规则、不合并条件
4. 精简作答，只输出核心答案，不冗余、不堆砌无关内容

重要格式要求：
- 每条证据的来源格式：【文件名】章节名
- 回答时必须显示完整来源，例如："根据【差旅与费用报销管理办法】3.2 提前申请时限..."
- 不要省略来源信息，不要只写 [1]，要写完整
"""

        # 构建用户Prompt
        user_prompt = f"""用户问题: {plan.original_question}

证据:
{all_evidence}

请严格基于以上证据回答，遵守以下规则：
1. 只使用提供的证据，禁止脑补、发挥、篡改
2. 上下文没有的信息，一律回答"无相关信息"
3. 引用时使用 [编号] 格式，如 [1][2]，并附完整来源：【文件名】第X章
4. 精简输出，不堆砌无关内容
"""

        logger.info(f"[PROMPT] Assembled - system_prompt: {len(system_prompt)} chars, user_prompt: {len(user_prompt)} chars, evidence: {len(all_evidence)} chars")

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "total_budget": len(all_evidence),
            "per_sub_question_budget": self.per_sub_question_budget,
            "rendered_reference_count": len(reference_id_map)
        }

    def _truncate_at_paragraph_boundary(self, content: str, max_len: int) -> str:
        """在段落边界截断内容"""
        if len(content) <= max_len or max_len <= 0:
            return content

        truncated = content[:max_len]
        # 查找换行符（段落边界）
        last_newline = truncated.rfind('\n')
        if last_newline > max_len * 0.3:
            return truncated[:last_newline].strip() + "..."
        return truncated.strip() + "..."