# core/__init__.py
from .state_manager import (
    initialize_session_state,
    get_current_step,
    set_current_step,
    add_message,
    store_user_input,
    store_initial_analysis_result,
    set_processing_status,
    reset_for_new_session,
    # ステップ定数もエクスポートしておくと便利
    STEP_INPUT_SUBMISSION,
    STEP_INITIAL_ANALYSIS,
    STEP_CLARIFICATION_NEEDED,
    STEP_CLARIFICATION_LOOP,
    STEP_SELECT_STYLE,
    STEP_GENERATE_EXPLANATION,
    STEP_FOLLOW_UP_LOOP,
    STEP_CONFIRM_UNDERSTANDING,
    STEP_SUMMARIZE,
    STEP_SESSION_END
)
from .tutor_logic import perform_initial_analysis_logic