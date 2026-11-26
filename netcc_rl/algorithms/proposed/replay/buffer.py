"""Episode replay with every retained real transition eligible as a start.

Rows store incoming actions/rewards, including an initial observation-only row.
Circular overwrite may remove that row: only consecutive retained row pairs
form transitions. Padding is finite storage, never a training label.
"""

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage


class Buffer:
    boundary_version = "single_step_real_transition_tail_v1"

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = torch.device(cfg.device)
        self._capacity = min(cfg.buffer_size, cfg.steps)
        if self._capacity < 1:
            raise ValueError("replay capacity must hold one transition")
        # Worst case: every transition is its own episode (two rows each).
        self._row_capacity = 2 * self._capacity
        self._num_eps = 0
        self._next_row_id = 0
        self._eligible_starts = torch.empty(0, dtype=torch.long)

    @property
    def capacity(self):
        return self._capacity

    @property
    def row_capacity(self):
        return self._row_capacity

    @property
    def num_transitions(self):
        return len(self._eligible_starts)

    def diagnostics(self):
        return dict(
            replay_requested_transitions=self.cfg.buffer_size,
            replay_transition_capacity=self.capacity,
            replay_transitions=self.num_transitions,
            replay_physical_row_capacity=self.row_capacity,
            replay_physical_rows=len(self._buffer) if self._num_eps else 0,
            replay_storage_device=str(self._storage_device) if self._num_eps else None,
        )

    @property
    def num_eps(self):
        return self._num_eps

    def _init(self, td):
        bytes_per_row = sum(v.numel() * v.element_size() for v in td.values()) / len(td)
        mem_free = (
            torch.cuda.mem_get_info(self._device)[0]
            if self._device.type == "cuda"
            else 0
        )
        self._storage_device = (
            self._device
            if 2.5 * bytes_per_row * self.row_capacity < mem_free
            else torch.device("cpu")
        )
        print(
            f"Buffer transition capacity: {self.capacity:,}; physical rows: {self.row_capacity:,}; storage: {self._storage_device}"
        )
        return ReplayBuffer(
            storage=LazyTensorStorage(self.row_capacity, device=self._storage_device),
            pin_memory=False,
            prefetch=0,
        )

    def add(self, td):
        """Keep whole-episode collection; accept even a one-transition episode."""
        if td.ndim != 1 or len(td) < 2:
            raise ValueError("an episode needs an initial and a next observation")
        if not torch.isfinite(td["obs"]).all() or any(
            not torch.isfinite(td[key][1:]).all()
            for key in ("action", "reward", "terminated")
        ):
            raise ValueError("episode contains incomplete real transitions")
        if td["terminated"][1:-1].bool().any():
            raise ValueError("episode contains transitions after termination")
        td = td.clone()
        td["episode"] = torch.full_like(td["reward"], self._num_eps, dtype=torch.int64)
        td["episode_step"] = torch.arange(len(td), device=td.device)
        td["replay_row_id"] = torch.arange(
            self._next_row_id, self._next_row_id + len(td), device=td.device
        )
        if not self._num_eps:
            self._buffer = self._init(td)
        # A single long episode can exceed storage. Keep its last C pairs.
        self._buffer.extend(td[-(self.capacity + 1) :])
        self._next_row_id += len(td)
        self._num_eps += 1
        rows = self._buffer[:]
        ids, eps, steps = (
            rows[key].cpu() for key in ("replay_row_id", "episode", "episode_step")
        )
        nxt = (torch.arange(len(rows)) + 1) % self.row_capacity
        nxt = nxt.clamp_max(len(rows) - 1)
        valid = (
            (ids[nxt] == ids + 1)
            & (eps[nxt] == eps)
            & (steps[nxt] == steps + 1)
            & ~rows["terminated"].cpu().nan_to_num().bool()
        )
        # Retain the newest C real transitions, regardless of reset-row count.
        candidates = valid.nonzero().flatten()
        if len(candidates) > self.capacity:
            cutoff = ids[candidates].topk(self.capacity).values.min()
            candidates = candidates[ids[candidates] >= cutoff]
        self._eligible_starts = candidates
        return self._num_eps

    def load(self, td):
        for episode in td.unbind(0):
            self.add(episode)
        return self._num_eps

    def _prepare_batch(self, td):
        td = td.to(self._device, non_blocking=True)
        reward = td["reward"][1:].unsqueeze(-1).contiguous()
        terminated = td.get("terminated", None)
        terminated = (
            torch.zeros_like(reward)
            if terminated is None
            else terminated[1:].unsqueeze(-1).contiguous()
        )
        task = td.get("task", None)
        return (
            td["obs"].contiguous(),
            td["action"][1:].contiguous(),
            reward,
            terminated,
            None if task is None else task[0].contiguous(),
        )

    def _sample_rows(self, start_indices=None):
        if not len(self._eligible_starts):
            raise RuntimeError("no complete retained transitions in replay")
        if start_indices is None:
            # Membership is guaranteed by construction. Do not rescan/sort the
            # full retained start set on every internally generated minibatch.
            starts = self._eligible_starts[
                torch.randint(len(self._eligible_starts), (self.cfg.batch_size,))
            ]
        else:
            starts = torch.as_tensor(start_indices, dtype=torch.long).flatten()
            if (
                len(starts) != self.cfg.batch_size
                or not torch.isin(starts, self._eligible_starts).all()
            ):
                raise ValueError(
                    "start_indices must select B complete retained transitions"
                )
        n = self.cfg.horizon + 1
        indices = (starts[None] + torch.arange(n)[:, None]) % self.row_capacity
        # An unfilled buffer has no wrapped data. Clamp retrieval, then check IDs.
        rows = self._buffer[
            indices.clamp_max(len(self._buffer) - 1).to(self._storage_device)
        ]
        ids, eps = rows["replay_row_id"], rows["episode"]
        offsets = torch.arange(n, device=ids.device)[:, None]
        valid = (
            (ids == ids[0] + offsets)
            & (eps == eps[0])
            & (indices.to(ids.device) < len(self._buffer))
        )
        valid = valid.cumprod(0).bool()
        lengths = valid.sum(0)
        # Repeat the actual final observation, never another episode's reset.
        row = torch.minimum(
            torch.arange(n, device=ids.device)[:, None], lengths[None] - 1
        )
        column = torch.arange(len(starts), device=ids.device)[None].expand(n, -1)
        rows = rows[row, column]
        rows["row_valid_mask"] = valid
        return rows

    def sample(self, *, start_indices=None):
        """H consecutive single-step transitions and a real-transition mask.

        Tail padding repeats the final real observation; labels are zeroed and
        masked. Starts are uniformly sampled from retained real transitions.
        """
        td = self._sample_rows(start_indices).to(self._device)
        obs, action, reward, terminated, task = self._prepare_batch(td)
        rows = td["row_valid_mask"]
        real = rows[1:]
        terminal = torch.where(real[..., None], terminated, 0).bool()[..., 0]
        alive = torch.cat(
            [torch.ones_like(terminal[:1]), (~terminal[:-1]).cumprod(0).bool()], 0
        )
        real = real & alive
        action, reward, terminated = (
            torch.where(real[..., None], x, 0) for x in (action, reward, terminated)
        )
        return dict(
            model_batch=(obs, action, reward, terminated, task), model_valid_mask=real
        )
