"""分開檢查 JSON、Schema、路由語意；保留未修復的原始模型回答。"""
import json

from pydantic import ValidationError

from .schemas import RouterOutput


def score_output(raw, expected_route):
    """Pydantic 合法不代表分對類；路由必須另外對照預期標籤。"""
    result = {"json_valid": False, "schema_valid": False, "route_correct": False,
              "end_to_end_success": False, "predicted_route": None, "validation_error": ""}
    try:
        # Python 預設允許 NaN / Infinity，這裡拒絕非標準 JSON 數值。
        def bad_constant(value):
            raise ValueError(f"Non-standard JSON constant: {value}")
        def unique_keys(pairs):
            """拒絕重複欄位，避免兩個 route 被默默覆寫。"""
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError(f"Duplicate JSON key: {key}")
                obj[key] = value
            return obj
        obj = json.loads(raw, parse_constant=bad_constant, object_pairs_hook=unique_keys)
        result["json_valid"] = True
        # 語意分類與 Schema 分開記錄：即使其他欄位格式錯，仍可看到 route 猜測。
        if isinstance(obj, dict):
            route = obj.get("route")
            result["predicted_route"] = route if isinstance(route, str) else None
            result["route_correct"] = route == expected_route
        RouterOutput.model_validate(obj)
        result["schema_valid"] = True
        result["end_to_end_success"] = result["route_correct"]
    except (ValueError, TypeError, ValidationError) as error:
        result["validation_error"] = str(error)
    return result
