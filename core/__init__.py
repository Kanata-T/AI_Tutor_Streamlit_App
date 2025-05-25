# core/__init__.py
from .state_manager import (
    initialize_session_state,
    get_current_step,
    set_current_step,
    add_message,
    store_user_input,
    store_initial_analysis_result,
    store_processed_image_details, # タスク1で追加
    store_problem_context,       # タスク1で追加
    store_guidance_plan,         # ★タスク2で追加★
    set_processing_status,
    set_explanation_style,
    reset_for_new_session,
    # store_clarification_analysis,       # analyze_user_clarification_logicが削除されたため、これも不要になる可能性が高い
    add_clarification_history_message,
    store_generated_explanation,
    # ステップ定数
    STEP_INPUT_SUBMISSION,
    STEP_INITIAL_ANALYSIS,
    STEP_CLARIFICATION_NEEDED,
    # STEP_CLARIFICATION_LOOP, # 未使用なら削除またはコメントアウト
    STEP_PLAN_GUIDANCE,          # ★タスク2で追加★
    STEP_SELECT_STYLE,
    STEP_GENERATE_EXPLANATION,
    STEP_FOLLOW_UP_LOOP,
    STEP_CONFIRM_UNDERSTANDING,
    STEP_SUMMARIZE,
    STEP_SESSION_END
)
from .tutor_logic import (
    perform_initial_analysis_logic,
    generate_clarification_question_logic,
    # analyze_user_clarification_logic, # ← これを削除
    perform_guidance_planning_logic,   # ★タスク2で追加★
    _create_problem_context_summary, # ★タスク1で追加、必要ならエクスポート★
    generate_explanation_logic,
    generate_followup_response_logic,
    generate_summary_logic,
    analyze_student_performance_logic
)

# (オプション) __all__ を定義している場合は、そこからも analyze_user_clarification_logic を削除し、
# 新しい関数や定数を追加してください。
# 例:
# __all__ = [
#     # state_manager からエクスポートする名前
#     "initialize_session_state", "get_current_step", "set_current_step", "add_message",
#     "store_user_input", "store_initial_analysis_result", "store_processed_image_details",
#     "store_problem_context", "store_guidance_plan", "set_processing_status",
#     "set_explanation_style", "reset_for_new_session", # "store_clarification_analysis",
#     "add_clarification_history_message", "store_generated_explanation",
#     "STEP_INPUT_SUBMISSION", "STEP_INITIAL_ANALYSIS", "STEP_CLARIFICATION_NEEDED",
#     "STEP_PLAN_GUIDANCE", "STEP_SELECT_STYLE", "STEP_GENERATE_EXPLANATION",
#     "STEP_FOLLOW_UP_LOOP", "STEP_CONFIRM_UNDERSTANDING", "STEP_SUMMARIZE", "STEP_SESSION_END",

#     # tutor_logic からエクスポートする名前
#     "perform_initial_analysis_logic",
#     "generate_clarification_question_logic",
#     "perform_guidance_planning_logic",
#     "_create_problem_context_summary",
#     "generate_explanation_logic",
#     "generate_followup_response_logic",
#     "generate_summary_logic",
#     "analyze_student_performance_logic",
# ]