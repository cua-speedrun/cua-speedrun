from __future__ import annotations

from cua_speedrun.commands import local_evaluations as local_module
from cua_speedrun.commands.local_evaluations import LocalEvaluations
from cua_speedrun.commands.paths import InstallationPaths
from cua_speedrun.config import load_dotenv
from cua_speedrun.service.db import User, make_session_factory
from cua_speedrun.service.store import LocalStore
from cua_speedrun.service.usersecrets import decrypt


def _service(tmp_path) -> LocalEvaluations:
    session_factory = make_session_factory(f"sqlite:///{tmp_path / 'platform.db'}")
    with session_factory() as session:
        user = User(handle="dev", quota_tier="dev")
        session.add(user)
        session.commit()
        user_id = user.id
    return LocalEvaluations(
        paths=InstallationPaths.resolve(tmp_path),
        session_factory=session_factory,
        store=LocalStore(tmp_path / "store"),
        user_id=user_id,
    )


def test_local_cli_submit_imports_dotenv_modal_pair_for_detached_worker(
    tmp_path, monkeypatch
) -> None:
    service = _service(tmp_path)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "MODAL_TOKEN_ID=ak-from-dotenv\n"
        "MODAL_TOKEN_SECRET=as-from-dotenv\n"
    )
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.setenv("CS_SECRET_KEY", "test-encryption-key")
    load_dotenv(dotenv)

    queued = {}
    monkeypatch.setattr(
        local_module,
        "benchmark_id_for_name",
        lambda _session_factory, _benchmark: 1,
    )
    monkeypatch.setattr(
        local_module,
        "queue_evaluation",
        lambda **kwargs: queued.update(kwargs) or {"run_id": 7},
    )

    result = service.submit(
        submission_zip=b"unused by mocked queue",
        name="agent",
        track="track",
        benchmark="benchmark",
        compute="modal",
        environment="modal",
        allocate_gpu=False,
        parallel_evaluations=1,
        saved_environment_names=(),
        evaluation_environment={},
    )

    assert result == {"run_id": 7}
    assert queued["evaluation_environment"] == {}
    with service.session_factory() as session:
        user = session.get(User, service.user_id)
        assert user.modal_token_id == "ak-from-dotenv"
        assert user.modal_token_secret_enc != "as-from-dotenv"
        assert decrypt(user.modal_token_secret_enc) == "as-from-dotenv"


def test_exported_modal_pair_replaces_old_local_credentials(
    tmp_path, monkeypatch
) -> None:
    service = _service(tmp_path)
    monkeypatch.setenv("CS_SECRET_KEY", "test-encryption-key")
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-exported")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-exported")

    service._sync_modal_credentials_from_process("modal", "modal-native")

    with service.session_factory() as session:
        user = session.get(User, service.user_id)
        assert user.modal_token_id == "ak-exported"
        assert decrypt(user.modal_token_secret_enc) == "as-exported"


def test_partial_modal_pair_is_rejected_only_for_remote_topology(
    tmp_path, monkeypatch
) -> None:
    import pytest

    service = _service(tmp_path)
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-only-id")
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    # Local-only execution does not need or consume provider credentials.
    service._sync_modal_credentials_from_process("local", "local")

    with pytest.raises(ValueError, match="must both be set"):
        service._sync_modal_credentials_from_process("modal", "modal")


def test_invalid_modal_pair_is_not_persisted(tmp_path, monkeypatch) -> None:
    import pytest

    service = _service(tmp_path)
    monkeypatch.setenv("MODAL_TOKEN_ID", "wrong-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "wrong-secret")

    with pytest.raises(ValueError, match="invalid Modal token pair"):
        service._sync_modal_credentials_from_process("modal", "modal")

    with service.session_factory() as session:
        user = session.get(User, service.user_id)
        assert user.modal_token_id is None
        assert user.modal_token_secret_enc is None
