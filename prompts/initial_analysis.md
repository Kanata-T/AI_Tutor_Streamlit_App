ユーザーから提供された画像 (存在する場合) とテキストクエリを分析してください。

ユーザーのテキストクエリ: 「{query_text}」
{image_instructions}

以下の情報を抽出し、指定されたJSON形式で出力してください。
1.  **ocr_text**: 画像が存在する場合、画像から抽出した主要なテキスト。存在しない場合は null。
2.  **request_category**: ユーザーの要求が以下のどのカテゴリに最も当てはまるか1つ選択（文法, 語彙, 長文読解, 英作文, 英文和訳, 構文解釈, 和文英訳, 添削, その他）。
3.  **topic**: 分類された主題領域や具体的な内容。
4.  **summary**: 特定された中核的な質問やタスクの簡潔な要約。
5.  **ambiguity**: このまま直接詳細に解説できるほど要求が明確か ("clear")、それとも診断的な質問を通じた明確化が必要か ("ambiguous")。
6.  **reason_for_ambiguity**: "ambiguous" と判断した場合、その理由を簡潔に。明確な場合は null。

出力は必ず以下のJSON形式のみで返してください。他のテキストは含めないでください。
```json
{{
    "ocr_text": "...",
    "request_category": "...",
    "topic": "...",
    "summary": "...",
    "ambiguity": "clear" | "ambiguous",
    "reason_for_ambiguity": "..."
}}