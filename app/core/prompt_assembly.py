"""
Prompt组装服务 - 对标Java的RagPromptAssemblyService
"""
from typing import Dict, Any, List


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

        print(f"[PROMPT] sub_questions count: {len(sub_questions)}, references count: {len(references)}")

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

                ref_id = f"[{ref_idx}]"
                sq_text += f"{ref_id} {content}\n"
                reference_id_map.append(ref)
                ref_idx += 1

            evidence_parts.append(sq_text)

        # 合并所有证据
        all_evidence = "\n\n".join(evidence_parts)

        # 构建系统Prompt
        system_prompt = f"""你是一个基于文档的知识助手。你的职责是：
1. 仅基于提供的证据回答用户问题
2. 如果证据不足，明确告知用户
3. 在回答时标注来源编号，如 [1][2]
4. 保持回答准确、简洁

证据来源格式：[编号] 内容
"""

        # 构建用户Prompt
        user_prompt = f"""用户问题: {plan.original_question}

证据:
{all_evidence}

请基于以上证据回答用户问题。要求：
1. 仅使用提供的证据
2. 在引用时使用 [编号] 格式
3. 如果无法回答，明确说明
"""

        print(f"[PROMPT] Assembled - system_prompt: {len(system_prompt)} chars, user_prompt: {len(user_prompt)} chars, evidence: {len(all_evidence)} chars")

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