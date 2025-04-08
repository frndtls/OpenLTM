import abc
import torch
from functools import cached_property
from einops import einsum, rearrange, repeat
from torch import nn


class Projection(nn.Module, abc.ABC):
    def __init__(self, proj_width: int, num_heads: int, **kwargs):
        super().__init__()
        self.proj_width = proj_width
        self.num_heads = num_heads

    @abc.abstractmethod
    def forward(self, x, seq_id): ...

class RotaryProjection(Projection):
    def __init__(self, *, proj_width: int, num_heads: int, max_len: int = 512, base: int = 10000):
        super().__init__(proj_width, num_heads)
        assert (
            self.proj_width % 2 == 0
        ), f"proj_width must be even, got {self.proj_width}"
        self.register_buffer(
            "theta",
            1.0
            / torch.pow(
                base,
                torch.arange(0, self.proj_width, 2, dtype=torch.float)
                / self.proj_width,
            ),
            persistent=False,
        )
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self._init_freq(max_len=max_len)

    def _init_freq(self, max_len: int):
        if self.cos is None or self.cos.size(-2) < max_len:
            position = torch.arange(
                max_len, device=self.theta.device, dtype=self.theta.dtype
            )
            m_theta = einsum(position, self.theta,
                             "length, width -> length width")
            m_theta = repeat(m_theta, "length width -> length (width 2)")
            self.register_buffer("cos", torch.cos(m_theta), persistent=False)
            self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    @staticmethod
    def _rotate(x):
        x1, x2 = rearrange(x, "... (dim r) -> r ... dim", r=2)
        return rearrange([-x2, x1], "r ... dim -> ... (dim r)", r=2)  # noqa

    def forward(self, x, seq_id, patch_mark: torch.Tensor = None):
        self._init_freq(max_len=seq_id.max() + 1)
        rot_cos = self.cos[seq_id]
        rot_sin = self.sin[seq_id]
        return rot_cos * x + rot_sin * self._rotate(x)
    
class TimeRotaryProjection(Projection):
    def __init__(self, *, proj_width: int, num_heads: int, max_len: int = 100, base: int = 10000, duration: int = 60 * 60 * 96, abs_index: bool = False, anchor_list, clock_list):
        super().__init__(proj_width, num_heads)
        assert (
            self.proj_width % 2 == 0
        ), f"proj_width must be even, got {self.proj_width}"
        
        self.duration = duration
        self.D = 2 * torch.pi * self.duration
        
        # self.anchors = [1, 60, 60*60, 60*60*24, 60*60*24*7, 60*60*24*30, 60*60*24*365] 
        self.anchors = anchor_list
        self.base_list = [(self.anchors[i+1] // self.anchors[i]) for i in range(len(self.anchors) -1)] # 6
        self.abs_index = abs_index
        # self.num_segment = len(self.base_list) # 6
        
        # self.num_clocks = proj_width // 2 # 32
        
        # # TODO
        # self.clock_list = [self.num_clocks // self.num_segment] * self.num_segment # 6
        # reminder = self.num_clocks % self.num_segment
        # for i in range(reminder):
        #     self.clock_list[i] += 1
            
        # self.clock_list = [self.clock_list[i] * 2 for i in range(self.num_segment)]
        self.clock_list = clock_list
        
        theta_list = []
        for i in range(len(self.base_list)):
            theta_list.append(
                (self.D / self.anchors[i])
                / torch.pow(
                    self.base_list[i],
                    torch.arange(0, self.clock_list[i], 2, dtype=torch.float)
                    / self.clock_list[i],
                )
            )
        print(theta_list)
        self.register_buffer(
            "theta",
            torch.cat(theta_list, dim=0),
            persistent=False,
        )
        # self.register_buffer(
        #     "theta",
        #     1.0
        #     / torch.pow(
        #         base,
        #         torch.arange(0, self.proj_width, 2, dtype=torch.float)
        #         / self.proj_width,
        #     ),
        #     persistent=False,
        # )
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self._init_freq(max_len=max_len)

    def _init_freq(self, max_len: int):
        if self.cos is None or self.cos.size(-2) < max_len:
            position = torch.arange(
                max_len, device=self.theta.device, dtype=self.theta.dtype
            )
            m_theta = einsum(position, self.theta,
                             "length, width -> length width")
            m_theta = repeat(m_theta, "length width -> length (width 2)")
            self.register_buffer("cos", torch.cos(m_theta), persistent=False)
            self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    def _init_freq_abs(self, patch_mark: torch.Tensor):
        # [B N]
        m_theta = einsum(patch_mark, self.theta,
                    "B length, width -> B length width") # [B N dim/2]
        m_theta = repeat(m_theta, "B length width -> B length (width 2)") # [B N dim]
        self.register_buffer("cos", torch.cos(m_theta), persistent=False)
        self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    @staticmethod
    def _rotate(x):
        x1, x2 = rearrange(x, "... (dim r) -> r ... dim", r=2)
        return rearrange([-x2, x1], "r ... dim -> ... (dim r)", r=2)  # noqa

    def forward(self, x, seq_id, patch_mark=None):
        if not self.abs_index:
            # [B H L E]
            self._init_freq(max_len=seq_id.max() + 1)
            rot_cos = self.cos[seq_id]
            # print("rot_cos", self.cos.shape) # [max_len dim]
            # print("seq_id", seq_id.shape) # [B H N]
            # print("cos", rot_cos.shape) # [B H N dim]
            # print('x', x.shape) # [B H N dim]
            rot_sin = self.sin[seq_id]
            return rot_cos * x + rot_sin * self._rotate(x)
        else:
            # print('patch_mark', patch_mark.shape) # [B N]
            _, H, _, _ = x.shape
            self._init_freq_abs(patch_mark=patch_mark)
            rot_cos = self.cos.unsqueeze(1).repeat(1, H, 1, 1)
            rot_sin = self.sin.unsqueeze(1).repeat(1, H, 1, 1)
            # print("rot_cos", self.cos.shape) # [max_len dim]
            # print("seq_id", seq_id.shape) # [B H N]
            # print('x', x.shape) # [B H N dim]
            return rot_cos * x + rot_sin * self._rotate(x)


class QueryKeyProjection(nn.Module):
    def __init__(self, dim: int, num_heads: int, proj_layer, kwargs=None, partial_factor=None):
        super().__init__()
        if partial_factor is not None:
            assert (
                0.0 <= partial_factor[0] < partial_factor[1] <= 1.0
            ), f"got {partial_factor[0]}, {partial_factor[1]}"
        assert num_heads > 0 and dim % num_heads == 0

        # print(dim, num_heads)
        self.head_dim = dim // num_heads
        self.partial_factor = partial_factor
        self.query_proj = proj_layer(
            proj_width=self.proj_width,
            num_heads=num_heads,
            **(kwargs or {}),
        )
        self.key_proj = self.query_proj

    @cached_property
    def proj_width(self) -> int:
        if self.partial_factor is None:
            return self.head_dim
        return int(self.head_dim * (self.partial_factor[1] - self.partial_factor[0]))

    @cached_property
    def split_sizes(self):
        if self.partial_factor is None:
            return 0, self.head_dim, 0
        return (
            int(self.partial_factor[0] * self.head_dim),
            self.proj_width,
            int((1.0 - self.partial_factor[1]) * self.head_dim),
        )

    def forward(self, query, key, query_id, kv_id, patch_mark=None):
        if self.partial_factor is not None:
            queries = list(query.split(self.split_sizes, dim=-1))
            keys = list(key.split(self.split_sizes, dim=-1))
            queries[1] = self.query_proj(queries[1], seq_id=query_id, patch_mark=patch_mark)
            keys[1] = self.key_proj(keys[1], seq_id=kv_id, patch_mark=patch_mark)
            query = torch.cat(queries, dim=-1)
            key = torch.cat(keys, dim=-1)
        else:
            query = self.query_proj(query, seq_id=query_id, patch_mark=patch_mark)
            key = self.key_proj(key, seq_id=kv_id, patch_mark=patch_mark)
        return query, key
