"""Pydantic 只驗證輸出格式，不替模型修正答案或判斷使用者意圖。"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Route = Literal["product_inquiry", "returns", "customer_support", "out_of_scope",
                "place_order", "order_tracking", "clarification"]
Slot = Literal["product_name", "quantity", "order_id", "delivery_address", "customer_name"]


class StrictOutput(BaseModel):
    """禁止額外欄位與自動將字串數字轉為數值，避免掩蓋格式錯誤。"""
    model_config = ConfigDict(extra="forbid", strict=True)


class ExtractedSlots(StrictOutput):
    """欄位必須全部出現；不知道時填 null，不能任意編造。"""
    product_name: str | None
    quantity: int | None = Field(ge=1)
    order_id: str | None
    delivery_address: str | None
    customer_name: str | None


class RouterOutput(StrictOutput):
    """對應 scheme.json 的 router_output_spec；此測試集每筆只有單一意圖。"""
    route: Route
    intent: str
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    status: Literal["ready", "missing_information", "ambiguous_intent", "out_of_scope",
                    "blocked", "awaiting_confirmation"]
    extracted_slots: ExtractedSlots
    missing_slots: list[Slot]
    next_action: Literal["query_database", "ask_for_information", "generate_response",
                         "handoff_to_human", "reject_request", "request_order_confirmation",
                         "simulate_place_order", "request_cancellation_confirmation",
                         "simulate_cancel_order"]
