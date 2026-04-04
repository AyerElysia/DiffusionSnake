"""
WAN 文本提示编码工具。

- 使用 T5 将可变长度的文本提示编码为定长张量（pad 到 max_sequence_length）；
- 将每条样本按 `num_videos_per_prompt` 扩展，方便视频生成批处理；
- 暴露 encode_prompt 供管线直接调用。
"""

import torch
from typing import Any, Callable, Dict, List, Optional, Union

def _get_t5_prompt_embeds(
    text_encoder,
    tokenizer,
    prompt: Union[str, List[str]] = None,
    max_sequence_length: int = 226,
    num_videos_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """
    基于 T5 编码器生成 token 级隐藏态，并按最大序列长度补零对齐。

    返回形状：[batch*num_videos_per_prompt, max_sequence_length, hidden_dim]
    """

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds

def encode_prompt(
    text_encoder,
    tokenizer,
    prompt: Union[str, List[str]],
    max_sequence_length: int = 226,
    num_videos_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    ):
    """
    将 prompt 编码为 T5 隐藏状态。

    参数
    - prompt：字符串或字符串列表；
    - max_sequence_length：编码后 pad 的最大长度；
    - num_videos_per_prompt：每条文本复制的次数；
    - device/dtype：放置设备与精度。

    返回
    - prompt_embeds：形状 [batch*num_videos_per_prompt, max_sequence_length, hidden_dim]
    """
    device = text_encoder[0].device
    dtype = text_encoder[0].dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    prompt_embeds = _get_t5_prompt_embeds(
        text_encoder=text_encoder[0],
        tokenizer=tokenizer[0],
        prompt=prompt,
        max_sequence_length=max_sequence_length,
        num_videos_per_prompt=num_videos_per_prompt,
        device=device,
        dtype=dtype,
    )

    return prompt_embeds
