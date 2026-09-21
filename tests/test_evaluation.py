"""Kiểm thử ghép đúng in-domain và OOD theo vai trò validation/test."""

from scripts.calibrate_and_evaluate import _in_domain_reference


def test_ood_validation_uses_validation_reference() -> None:
    """Validation OOD không được dùng in-domain test làm đối chứng."""
    validation = object()
    test = object()
    assert _in_domain_reference("massive_validation", validation, test) is validation
    assert _in_domain_reference("bank_test", validation, test) is test
