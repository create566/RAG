"""
RAGas 评估脚本
用于评估 RAG 系统的检索和回答质量

使用方式:
    python eval_rag.py

评估指标:
    - faithfulness: 回答是否忠实于检索到的上下文
    - answer_relevancy: 回答与问题的相关性
    - context_precision: 上下文块的相关性排序质量
    - context_recall: 检索到的上下文是否覆盖了正确答案
"""
import asyncio
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall
)
from datasets import Dataset
from app.config import get_settings
from app.services.chat_service import get_chat_service
from app.services.document_service import get_document_service
from app.models.chat import ChatRequest, ChatQueryMode
from ragas.llms.base import LangchainLLMWrapper, RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_dashscope import ChatDashScope, DashScopeEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings


# ============================================================
# 测试数据集 - 根据你的实际文档内容设计
# ============================================================
TEST_DATASET = [
    {
        "user_input": "差旅报销的餐补标准是多少？",
        "reference": "国内出差餐补标准为100元/天，不需要提供发票。当天往返且出差时长不足8小时的，不发放餐补。",
    },
    {
        "user_input": "紧急出差有什么特殊规则？",
        "reference": "如因客户现场重大故障、监管检查，生产事故等原因导致无法提前申请，员工可先执行出差，但必须在出发后24小时内完成补提流程，并在备注中标注'紧急支援出差'。",
    },
    {
        "user_input": "酒店住宿的标准是什么？",
        "reference": "一线城市(北京、上海、深圳、广州)住宿上限500元/晚；新一线城市(杭州、成都、苏州、武汉、南京)450元/晚；其他城市350元/晚。",
    },
    {
        "user_input": "报销时限有什么要求？",
        "reference": "员工应在出差结束后10个工作日内提交报销；发票最晚应在费用发生后的30个自然日内完成上传；每月26日至月末为财务关账期。",
    },
    {
        "user_input": "客服平台的灰度发布比例是多少？",
        "reference": "灰度租户默认不超过生产总量的10%。",
    },
    {
        "user_input": "知识运营专员的职责是什么？",
        "reference": "知识运营专员负责知识整理、问法扩写、召回评估、版本发布。",
    },
]


async def run_single_evaluation(question: str, reference: str) -> dict:
    """
    对单个问题运行完整的 RAG 流程并返回评估数据
    """
    print(f"\n{'='*60}")
    print(f"测试问题: {question}")
    print(f"参考答案: {reference}")
    print('='*60)

    try:
        # 1. 创建 ChatService
        chat_service = get_chat_service()

        # 2. 构造请求
        request = ChatRequest(
            user_id=2,  # 假设用户ID为2
            question=question,
            chat_mode=ChatQueryMode.AUTO_DOCUMENT,
        )

        # 3. 获取回答（非流式）
        response = await chat_service.chat(request)

        # 4. 获取检索上下文
        retrieved_contexts = []
        if hasattr(response, 'sources') and response.sources:
            for source in response.sources:
                context_text = getattr(source, 'content', None) or f"{source.document_name}: {source.section_path}"
                retrieved_contexts.append(context_text)
        elif hasattr(response, 'trace') and response.trace:
            # 从trace中获取检索上下文
            notes = response.trace.get('retrieval_notes', [])
            retrieved_contexts = notes if notes else ["无检索上下文"]

        result = {
            "user_input": question,
            "response": response.answer if hasattr(response, 'answer') else str(response),
            "retrieved_contexts": retrieved_contexts if retrieved_contexts else ["无检索上下文"],
            "reference": reference,
        }

        print(f"系统回答: {result['response'][:200]}...")
        print(f"检索上下文数: {len(retrieved_contexts)}")

        return result

    except Exception as e:
        print(f"评估出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            "user_input": question,
            "response": f"错误: {e}",
            "retrieved_contexts": [["错误"]],
            "reference": reference,
        }


async def run_evaluation():
    """
    运行完整的 RAGas 评估流程
    """
    print("=" * 60)
    print("RAG 系统评估开始")
    print("=" * 60)

    # 1. 初始化服务
    print("\n[1/3] 初始化服务...")
    settings = get_settings()
    print(f"  - LLM Provider: {settings.llm.provider}")
    print(f"  - Embedding Model: {settings.embedding.model}")

    # 2. 运行测试
    print("\n[2/3] 运行 RAG 流程...")
    results = []
    for i, test_case in enumerate(TEST_DATASET):
        print(f"\n  [{i+1}/{len(TEST_DATASET)}] ", end="")
        result = await run_single_evaluation(
            test_case["user_input"],
            test_case["reference"]
        )
        results.append(result)

    # 3. 转换为 Dataset 格式
    print("\n\n[3/3] 执行 RAGas 评估...")

    # 构建 datasets
    dataset_dict = {
        "user_input": [r["user_input"] for r in results],
        "response": [r["response"] for r in results],
        "retrieved_contexts": [r["retrieved_contexts"] for r in results],
        "reference": [r["reference"] for r in results],
    }

    print("\n数据集统计:")
    print(f"  - 问题数量: {len(dataset_dict['user_input'])}")
    print(f"  - 回答长度: {sum(len(r) for r in dataset_dict['response']) / len(dataset_dict['response']):.0f} 字符(平均)")

    # 创建 Dataset
    dataset = Dataset.from_dict(dataset_dict)

    # 4. 定义评估指标
    metrics = [
        faithfulness,           # 回答忠实度
        answer_relevancy,      # 回答相关性
        context_precision,     # 上下文精确度
        context_recall,        # 上下文召回率
    ]

    # 5. 初始化 LLM 和 Embeddings（支持 DashScope / Ollama）
    eval_provider = os.getenv("EVAL_LLM_PROVIDER", "dashscope")  # 可设为 "ollama"
    print(f"\n初始化评估用 LLM ({eval_provider})...")
    run_cfg = RunConfig(timeout=120, max_retries=3, max_workers=4)

    if eval_provider == "ollama":
        llm = LangchainLLMWrapper(ChatOllama(
            model="gemma4:e4b",
            base_url="http://localhost:11434",
        ), run_config=run_cfg)
        embeddings = LangchainEmbeddingsWrapper(DashScopeEmbeddings(
            model="text-embedding-v1",
            api_key=os.getenv("DASHSCOPE_API_KEY") or settings.dashscope_api_key,
        ), run_config=run_cfg)
        print("  - LLM模型: gemma4:e4b (Ollama)")
        print("  - Embedding模型: gemma4:e4b (Ollama)")
    else:
        llm = LangchainLLMWrapper(ChatDashScope(
            model="qwen-turbo",
            api_key=os.getenv("DASHSCOPE_API_KEY") or settings.dashscope_api_key,
        ), run_config=run_cfg)
        embeddings = LangchainEmbeddingsWrapper(DashScopeEmbeddings(
            model="text-embedding-v1",
            api_key=os.getenv("DASHSCOPE_API_KEY") or settings.dashscope_api_key,
        ), run_config=run_cfg)
        print("  - LLM模型: qwen-turbo (DashScope)")
        print("  - Embedding模型: text-embedding-v1 (DashScope)")

    # 将 embeddings 设置到需要它的指标
    answer_relevancy.embeddings = embeddings

    # 6. 执行评估
    print("\n正在评估(这可能需要几分钟)...\n")

    try:
        result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings, raise_exceptions=True)

        # 6. 输出结果
        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)

        # 打印整体分数
        print("\n【整体评分】")
        scores = result.scores
        metric_names = list(scores[0].keys())
        for metric_name in metric_names:
            metric_display = {
                "faithfulness": "回答忠实度",
                "answer_relevancy": "回答相关性",
                "context_precision": "上下文精确度",
                "context_recall": "上下文召回率",
            }.get(metric_name, metric_name)
            print(f"  {metric_display}: {scores[0][metric_name]:.4f}")

        # 打印详细结果表格
        print("\n【详细结果】")
        df = result.to_pandas()
        print(df.to_string(index=False))

        # 7. 保存结果
        output_file = "eval_results01.csv"
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n结果已保存到: {output_file}")

        return result

    except Exception as e:
        print(f"\n评估失败: {e}")
        print("\n可能的原因:")
        print("  1. API Key 未设置或额度不足")
        print("  2. 检索服务(Chroma/ES)未启动")
        print("  3. 网络连接问题")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主入口"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                  RAG 系统评估脚本 (RAGas)                       ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  评估指标:                                                     ║
    ║    - faithfulness      回答是否忠实于检索到的上下文                ║
    ║    - answer_relevancy  回答与问题的相关性                        ║
    ║    - context_precision 上下文块的相关性排序质量                   ║
    ║    - context_recall    检索到的上下文是否覆盖正确答案              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    result = asyncio.run(run_evaluation())

    if result:
        print("\n[OK] 评估完成!")
    else:
        print("\n[FAIL] 评估失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
