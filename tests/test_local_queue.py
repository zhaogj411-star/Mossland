from scripts.tools.local_queue.api import TaskRecord, TaskResult
from scripts.tools.local_queue.lease import acquire_task
from scripts.tools.local_queue.manifest import write_manifest
from scripts.tools.local_queue.paths import done_path, output_dir
from scripts.tools.local_queue.runner import LostLeaseError, commit_result


def test_stale_worker_cannot_commit_after_lease_is_reclaimed(tmp_path):
    root = tmp_path / "queue"
    job = "job"
    task = TaskRecord(task_id="abcdef", task_type="copy", payload={})

    assert acquire_task(root, job, task, "worker-a", lease_ttl=0)
    assert acquire_task(root, job, task, "worker-b", lease_ttl=300)

    stale_output = root / "tmp" / "worker-a" / "abcdef.json"
    stale_output.parent.mkdir(parents=True)
    stale_output.write_text('{"worker":"a"}\n', encoding="utf-8")

    try:
        commit_result(root, job, task, "worker-a", TaskResult(stale_output, {}))
    except LostLeaseError:
        pass
    else:
        raise AssertionError("stale worker unexpectedly committed after losing lease")

    assert not done_path(root, job, task.task_id).exists()
    assert not stale_output.exists()
    assert not output_dir(root, job, task.task_id).exists()


def test_current_lease_owner_can_commit(tmp_path):
    root = tmp_path / "queue"
    job = "job"
    task = TaskRecord(task_id="abcdef", task_type="copy", payload={})

    assert acquire_task(root, job, task, "worker-a", lease_ttl=300)

    output = root / "tmp" / "worker-a" / "abcdef.json"
    output.parent.mkdir(parents=True)
    output.write_text('{"worker":"a"}\n', encoding="utf-8")

    commit_result(root, job, task, "worker-a", TaskResult(output, {"ok": True}))

    marker = done_path(root, job, task.task_id)
    assert marker.exists()
    assert not output.exists()
    assert (output_dir(root, job, task.task_id) / "abcdef.json").exists()


def test_write_manifest_refuses_existing_job(tmp_path):
    root = tmp_path / "queue"
    job = "job"
    task = TaskRecord(task_id="abcdef", task_type="copy", payload={})

    assert write_manifest(root, job, iter([task]), {"job": job}) == 1

    try:
        write_manifest(root, job, iter([task]), {"job": job})
    except FileExistsError:
        pass
    else:
        raise AssertionError("write_manifest unexpectedly appended to an existing job")
