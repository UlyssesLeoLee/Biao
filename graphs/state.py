from typing import Annotated, List, Optional
from typing_extensions import TypedDict
import operator


class SectionResult(TypedDict):
    name: str
    content: str
    status: str  # pending | enhanced | generated | skipped


class BiddingState(TypedDict):
    # 输入
    input_document: str          # 原始标书内容
    user_requirements: str       # 用户需求描述
    document_type: str           # 标书类型（投标书/招标书/其他）
    operation_mode: str          # 模式: enhance | generate | analyze

    # 分析结果
    document_structure: str      # 文档结构分析
    identified_sections: List[str]   # 识别到的章节
    missing_sections: List[str]  # 缺失章节
    improvement_points: List[str]    # 待改进点

    # RAG 检索结果
    retrieved_context: str       # 检索到的参考内容

    # 处理结果
    section_results: Annotated[List[SectionResult], operator.add]
    current_section_index: int   # 当前处理的章节索引

    # 最终输出
    final_document: str          # 最终输出的标书
    summary: str                 # 处理总结

    # 流程控制
    error_message: Optional[str]
    processing_log: Annotated[List[str], operator.add]
