"""組裝 RAG Pipeline for 虎尾地標辨識

這裡是四層的「接線盒」：要換 data/retriever/generator，只改這一檔。
"""

import os

from src.rag import RAGPipeline
from src.rag.data import CSVDataSource, GoogleSheetDataSource
from src.rag.generator import GeminiGenerator
from src.rag.retriever import AllInPromptRetriever

from . import schema

# 虎尾地標 Google Sheet 的公開 CSV export URL
DEFAULT_SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1Xcvnj35efvDa1anm7ojOAwRUv6n7hKVhN8UTpb4Fmf8/export?format=csv"
)


# -------- Prompt 組裝（app 層職責，不進 src/rag/） --------

def build_context(rows: list[dict]) -> str:
    """把地標候選列組成人類可讀的 context 區塊。"""
    context = f"虎尾地標資料庫（共 {len(rows)} 個地點）：\n\n"
    for i, row in enumerate(rows, 1):
        name = (row.get(schema.KEY_COLUMN) or "").strip()
        context += f"{i}. {name}\n"
        for _field, col, label in schema.FEATURE_COLUMNS:
            val = (row.get(col) or "").strip()
            if not val:
                continue
            if col == schema.COL_SUMMARY:
                val = val[: schema.SUMMARY_MAX_LEN]
            context += f"   {label}：{val}\n"
        context += "\n"
    return context


def build_prompt(payload: dict, query) -> str:
    """Gemini prompt template — 視覺辨識地標。"""
    context = build_context(payload.get("rows", []))
    return f"""{context}

請根據照片的視覺特徵（建築風格、材質、文字、雕塑、場景），從上面的地點清單中選出最匹配的一個。

規則：
- 如果照片明顯不是 18 個地標中任何一個（例：動物、自拍、植物、非建築物、外地照片），回答 name 為 'unknown'，confidence 為 'low'。寧可說『不確定』也不要瞎掰。
- 如果照片上有文字（碑文、門牌、布條），優先用文字判斷
- 如果沒有文字，用建築風格、材質、結構來比對

請用以下 JSON 格式回答：
{{"name": "地點名稱", "reason": "判斷依據", "confidence": "high/medium/low"}}

只輸出 JSON。"""


# -------- 三層組裝 --------

def build_data_source(csv_path: str | None = None):
    """預設用 GoogleSheet；若傳 csv_path 則改成本地 CSV。"""
    if csv_path:
        return CSVDataSource(csv_path, key_column=schema.KEY_COLUMN)
    return GoogleSheetDataSource(DEFAULT_SHEET_CSV_URL, key_column=schema.KEY_COLUMN)


def build_pipeline(api_key: str | None = None, csv_path: str | None = None) -> RAGPipeline:
    """組出虎尾地標 RAG pipeline。

    Args:
        api_key:  Gemini API key（預設讀 GOOGLE_API_KEY env）
        csv_path: 若提供，DataSource 改用本地 CSV；否則用 Google Sheet
    """
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 GOOGLE_API_KEY（請設 env 或傳入 api_key）")

    data_source = build_data_source(csv_path)
    retriever = AllInPromptRetriever(
        data_source=data_source,
        key_field=schema.KEY_COLUMN,
        filter_fn=schema.row_is_valid,
    )
    generator = GeminiGenerator(
        api_key=api_key,
        prompt_builder=build_prompt,
        model="gemini-2.5-flash",
    )
    return RAGPipeline(
        data_source=data_source,
        retriever=retriever,
        generator=generator,
    )
