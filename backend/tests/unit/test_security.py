"""Password hashing and cookie-signing primitives (app/core/security.py).
No DB/Redis needed here — these are pure functions."""

from app.core.security import hash_password, sign_token, unsign_token, verify_password


def test_hash_password_does_not_return_the_plaintext():
    hashed = hash_password("correct-horse-battery-staple")
    assert hashed != "correct-horse-battery-staple"
    assert "correct-horse-battery-staple" not in hashed


def test_hash_password_uses_argon2():
    # Argon2 hashes are self-describing strings like $argon2id$v=19$...
    assert hash_password("some-password").startswith("$argon2")


def test_hash_password_is_salted_so_two_hashes_of_the_same_password_differ():
    a = hash_password("same-password")
    b = hash_password("same-password")
    assert a != b


def test_verify_password_accepts_the_correct_password():
    hashed = hash_password("correct-horse-battery-staple")
    assert verify_password("correct-horse-battery-staple", hashed) is True


def test_verify_password_rejects_the_wrong_password():
    hashed = hash_password("correct-horse-battery-staple")
    assert verify_password("wrong-password", hashed) is False


def test_verify_password_rejects_a_malformed_hash_instead_of_raising():
    assert verify_password("anything", "not-a-real-argon2-hash") is False


def test_sign_and_unsign_token_round_trips():
    token = "some-opaque-session-token"
    signed = sign_token(token)
    assert signed != token
    assert unsign_token(signed) == token


def test_unsign_token_rejects_a_tampered_signature():
    signed = sign_token("some-opaque-session-token")
    tampered = signed[:-1] + ("0" if signed[-1] != "0" else "1")
    assert unsign_token(tampered) is None


def test_unsign_token_rejects_a_tampered_token_with_a_valid_looking_suffix():
    signed = sign_token("some-opaque-session-token")
    _, _, signature = signed.rpartition(".")
    forged = f"a-different-token.{signature}"
    assert unsign_token(forged) is None


def test_unsign_token_rejects_garbage_input():
    assert unsign_token("not-a-signed-value-at-all") is None
    assert unsign_token("") is None
