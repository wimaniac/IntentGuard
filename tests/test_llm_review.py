"""Kiểm thử gợi ý DeepSeek độc lập với mạng và dữ liệu review thật."""

import json

import pytest

from intentguard.llm_review import suggest_review


class FakeResponse:
    """Mô phỏng phản hồi HTTP JSON của DeepSeek cho test."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        """Trả về phản hồi giả cho context manager."""
        return self

    def __exit__(self, *_args):
        """Kết thúc context manager mà không chặn lỗi."""
        return False

    def read(self):
        """Trả payload ở dạng byte như HTTP response."""
        return json.dumps(self.payload).encode("utf-8")


def test_suggestion_requires_human_save_and_sends_one_case(monkeypatch) -> None:
    """Gợi ý chỉ gọi API cho câu chọn và trả cấu trúc đã kiểm tra."""
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": "reject",
                                    "reason": "Đã thuộc intent cũ",
                                    "closest_intent": "transaction_charged_twice",
                                    "rewritten_text": None,
                                    "in_scope": False,
                                    "support_request": True,
                                    "faithful_rewrite": False,
                                }
                            )
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr("intentguard.llm_review.urlopen", fake_urlopen)
    result = suggest_review("Tôi bị tính phí hai lần", "LOAN", ["transaction_charged_twice"], "secret")
    assert result.decision == "reject"
    assert captured["body"]["messages"][1]["content"].count("Tôi bị tính phí hai lần") == 1
    assert captured["timeout"] == 30


def test_suggestion_rejects_unknown_intent(monkeypatch) -> None:
    """Intent model tự bịa không được hiển thị như nhãn hợp lệ."""
    monkeypatch.setattr(
        "intentguard.llm_review.urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": "reject",
                                    "reason": "trùng",
                                    "closest_intent": "invented",
                                    "rewritten_text": None,
                                    "in_scope": False,
                                    "support_request": True,
                                    "faithful_rewrite": False,
                                }
                            )
                        },
                    }
                ]
            }
        ),
    )
    with pytest.raises(ValueError, match="intent không nằm"):
        suggest_review("Câu hỏi", "LOAN", ["known"], "secret")
