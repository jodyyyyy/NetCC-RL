"""One bounded FIFO writer for a run's JSONL records.

Both limits include in-flight batches. Producers block rather than drop events.
JSON serialization and file access belong to the worker; the producer makes an
owned CPU-only snapshot and estimates a conservative serialized byte bound.
Flush means Python/file-system flush, as with the old logger, not fsync.
"""
from collections import deque
import json
import math
from pathlib import Path
from threading import Condition, Thread
from time import perf_counter


def _snapshot(value):
    if value is None or isinstance(value, bool):
        return value, 6
    if isinstance(value, int):
        return value, len(str(value))+1
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('nonfinite JSON record')
        return value, 32
    if isinstance(value, str):
        return value, 6*len(value)+2
    if isinstance(value, (list, tuple)):
        items = [_snapshot(item) for item in value]
        return [item[0] for item in items], 2+sum(item[1]+1 for item in items)
    if isinstance(value, dict):
        result, size = {}, 2
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError('JSON record keys must be strings')
            copied, bound = _snapshot(item)
            result[key] = copied
            size += 6*len(key)+4+bound
        return result, size
    raise TypeError('writer accepts only CPU JSON primitives, never tensors')


class JsonlWriter:
    def __init__(self, root, *, max_items=1024, max_bytes=8*1024*1024,
                 batch_items=128, batch_bytes=256*1024, flush_interval=.25):
        if min(max_items,max_bytes,batch_items,batch_bytes) < 1 or not math.isfinite(flush_interval) or flush_interval <= 0:
            raise ValueError('writer limits must be positive')
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_items, self.max_bytes = max_items, max_bytes
        self.batch_items, self.batch_bytes = batch_items, batch_bytes
        self.flush_interval = flush_interval
        self._cv, self._queue = Condition(), deque()
        self._closing = self._closed = False
        self._error = None
        self._pending_items = self._pending_bytes = 0
        self._stats = dict(max_pending_items=0, max_pending_bytes=0,
                           enqueue_wait_seconds=0., enqueue_wait_count=0,
                           writer_seconds=0., serialization_seconds=0., disk_write_seconds=0.,
                           snapshot_seconds=0., records_written=0, bytes_written=0,
                           batches_written=0, failures=0)
        self._thread = Thread(target=self._run, name='netcc-rl-jsonl-writer', daemon=False)
        self._thread.start()

    def check(self):
        with self._cv:
            self._check()

    def _check(self):
        if self._error is not None:
            raise OSError('background JSONL writer failed') from self._error

    def submit(self, name, data):
        if Path(name).name != name or not name.endswith('.jsonl'):
            raise ValueError('writer needs a local .jsonl filename')
        snapshot_start = perf_counter()
        payload, bound = _snapshot(data)
        snapshot_seconds = perf_counter()-snapshot_start
        bound += 1  # newline
        if bound > self.max_bytes:
            raise ValueError('record exceeds writer byte limit; no record was dropped silently')
        with self._cv:
            self._check()
            if self._closing:
                raise RuntimeError('writer is closed')
            start = perf_counter()
            waited = False
            while self._pending_items >= self.max_items or self._pending_bytes+bound > self.max_bytes:
                waited = True
                self._cv.wait()
                self._check()
                if self._closing:
                    raise RuntimeError('writer is closed')
            if waited:
                self._stats['enqueue_wait_seconds'] += perf_counter()-start
                self._stats['enqueue_wait_count'] += 1
            self._stats['snapshot_seconds'] += snapshot_seconds
            self._queue.append((name,payload,bound))
            self._pending_items += 1
            self._pending_bytes += bound
            self._stats['max_pending_items'] = max(self._stats['max_pending_items'], self._pending_items)
            self._stats['max_pending_bytes'] = max(self._stats['max_pending_bytes'], self._pending_bytes)
            self._cv.notify_all()

    def _open(self, name):
        return (self.root/name).open('a', encoding='utf-8')

    def _run(self):
        handles = {}
        try:
            while True:
                with self._cv:
                    self._cv.wait_for(lambda: self._queue or self._closing)
                    if not self._queue:
                        break
                    deadline = perf_counter()+self.flush_interval
                    while (not self._closing and len(self._queue) < min(self.batch_items,self.max_items)
                           and self._pending_bytes < self.batch_bytes):
                        remaining = deadline-perf_counter()
                        if remaining <= 0:
                            break
                        self._cv.wait(remaining)
                    batch, size = [], 0
                    while self._queue and len(batch) < self.batch_items:
                        item = self._queue[0]
                        if batch and size+item[2] > self.batch_bytes:
                            break
                        batch.append(self._queue.popleft())
                        size += item[2]
                start = perf_counter()
                written, touched = 0, set()
                serialization_seconds = disk_seconds = 0.
                for name, payload, _ in batch:
                    stamp = perf_counter()
                    line = json.dumps(payload,allow_nan=False)+'\n'
                    serialization_seconds += perf_counter()-stamp
                    stamp = perf_counter()
                    if name not in handles:
                        handles[name] = self._open(name)
                    handles[name].write(line)
                    disk_seconds += perf_counter()-stamp
                    touched.add(name)
                    written += len(line.encode('utf-8'))
                stamp = perf_counter()
                for name in touched:
                    handles[name].flush()
                disk_seconds += perf_counter()-stamp
                with self._cv:
                    self._pending_items -= len(batch)
                    self._pending_bytes -= size
                    self._stats['writer_seconds'] += perf_counter()-start
                    self._stats['serialization_seconds'] += serialization_seconds
                    self._stats['disk_write_seconds'] += disk_seconds
                    self._stats['records_written'] += len(batch)
                    self._stats['bytes_written'] += written
                    self._stats['batches_written'] += 1
                    self._cv.notify_all()
        except BaseException as error:
            with self._cv:
                self._error = error
                self._stats['failures'] += 1
                self._cv.notify_all()
        finally:
            for handle in handles.values():
                try:
                    handle.close()
                except BaseException as error:
                    with self._cv:
                        self._error = self._error or error
                        self._stats['failures'] += 1
                        self._cv.notify_all()

    def flush(self):
        with self._cv:
            self._cv.wait_for(lambda: self._pending_items == 0 or self._error is not None)
            self._check()

    def close(self):
        with self._cv:
            self._closing = True
            self._cv.notify_all()
        self._thread.join()
        with self._cv:
            self._closed = True
            self._check()

    def diagnostics(self):
        with self._cv:
            return dict(self._stats, max_items=self.max_items, max_bytes=self.max_bytes,
                        pending_items=self._pending_items,pending_bytes=self._pending_bytes,
                        closed=self._closed, failure=repr(self._error) if self._error else None,
                        batch_items=self.batch_items,batch_bytes=self.batch_bytes,flush_interval=self.flush_interval,
                        byte_accounting='conservative JSON bound, includes in-flight batch')

    def __enter__(self):
        return self

    def __exit__(self, kind, error, tb):
        try:
            self.close()
        except BaseException as writer_error:
            if error is not None:
                raise error.with_traceback(tb) from writer_error
            raise
