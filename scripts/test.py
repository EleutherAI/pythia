from filelock import FileLock
import shutil

dir_path = "/om2/user/sunnyd/amber_cache/"
total, used, free = shutil.disk_usage(dir_path)
print(f"Free space: {free // (2**30)} GiB")

import os

lock_path = os.path.join(dir_path, 'lock.lock')
if os.path.exists(lock_path):
        print(f"Lock file exists. Owner: {os.stat(lock_path).st_uid}")

lock = FileLock(lock_path)
with lock:
    with open("/om/user/sunnyd/amber_cache/test.txt", "a") as f:
        f.write("foo")
