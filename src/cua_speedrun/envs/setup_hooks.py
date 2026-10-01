"""Require successful provisioning before exposing a task to an agent."""

from contextlib import contextmanager


@contextmanager
def checked_setup(env):
    """Turn Gym-Anything hook warnings and nonzero exits into setup failures."""
    if not getattr(env.env_spec, "hooks", None) and not getattr(env.task_spec, "hooks", None):
        yield
        return
    reporter = env._reporter
    execute = env._runner.exec
    active = None

    class Reporter:
        def stage_start(self, stage):
            nonlocal active
            active = stage
            if reporter is not None:
                reporter.stage_start(stage)

        def stage_done(self, stage):
            nonlocal active
            active = None
            if reporter is not None:
                reporter.stage_done(stage)

        def stage_fail(self, stage, error):
            if reporter is not None:
                reporter.stage_fail(stage, error)
            raise RuntimeError(f"Environment setup failed at {stage}: {error}")

    def checked_exec(*args, **kwargs):
        code = execute(*args, **kwargs)
        if active is not None and code not in (None, 0):
            log_path = {
                "pre_start_hook": "/tmp/env_setup_pre_start.log",
                "post_start_hook": "/tmp/env_setup_post_start.log",
                "pre_task_hook": "/tmp/task_pre_task.log",
            }.get(active)
            detail = ""
            capture = getattr(env._runner, "exec_capture", None)
            if log_path and capture is not None:
                try:
                    detail = capture(f"tail -n 60 {log_path} 2>/dev/null")[-8000:]
                except Exception:
                    pass
            raise RuntimeError(f"{active} exited with status {code}\n{detail}".rstrip())
        return code

    env._reporter = Reporter()
    env._runner.exec = checked_exec
    try:
        yield
    finally:
        env._runner.exec = execute
        env._reporter = reporter
