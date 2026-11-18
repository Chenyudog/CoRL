"""Load the local openpi policy with optional CPU vision for 8 GB GPUs."""
import dataclasses
from unittest.mock import patch


def materialize_nonpersistent_buffers(model):
    """Recreate deterministic, non-checkpoint buffers with original formulas."""
    import torch
    for module_name, module in model.named_modules():
        for name, buffer in list(module._buffers.items()):
            if buffer is None or not buffer.is_meta:
                continue
            if name == 'inv_freq' and hasattr(module, 'rope_init_fn'):
                inv_freq, scaling = module.rope_init_fn(module.config, torch.device('cpu'))
                module.original_inv_freq = inv_freq
                module.attention_scaling = scaling
                value = inv_freq.to(dtype=buffer.dtype)
            elif name == 'position_ids':
                value = torch.arange(buffer.shape[-1], dtype=buffer.dtype, device='cpu').expand(buffer.shape)
            else:
                raise RuntimeError(f'Unknown meta buffer needs its original initializer: {module_name}.{name}')
            assert value.shape == buffer.shape and value.dtype == buffer.dtype
            module._buffers[name] = value
    assert all(not b.is_meta for b in model.buffers())


def load_into_meta(model, filename):
    """Assign mapped checkpoint tensors with the original model's exact dtypes."""
    from safetensors import safe_open
    expected = model.state_dict()
    assert expected and all(t.is_meta for t in expected.values())
    state = {}
    with safe_open(filename, framework='pt', device='cpu') as reader:
        for name in reader.keys():
            tensor = reader.get_tensor(name)
            assert name in expected and tensor.shape == expected[name].shape, name
            if tensor.dtype != expected[name].dtype:
                tensor = tensor.to(dtype=expected[name].dtype)
            state[name] = tensor
        aliases = reader.metadata() or {}
        for dropped, kept in aliases.items():
            assert dropped in expected and kept in state, (dropped, kept)
            assert expected[dropped].dtype == state[kept].dtype
            state[dropped] = state[kept]
    assert set(state) == set(expected)
    model.load_state_dict(state, strict=True, assign=True)
    model.paligemma_with_expert.paligemma.tie_weights()
    materialize_nonpersistent_buffers(model)
    assert all(not p.is_meta for p in model.parameters())
    assert all(t.dtype == expected[name].dtype for name, t in model.state_dict().items())


def load_policy(checkpoint, vision_on_cpu, embeddings_on_cpu=False):
    import torch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.policies import policy_config
    from openpi.training.config import get_config

    config = get_config("pi05_libero")
    config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, pytorch_compile_mode=None)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings_on_cpu = embeddings_on_cpu or vision_on_cpu

    def load_model(model_config, train_config, weight_path):
        # Assign the checkpoint into an empty model to avoid a full random copy.
        with torch.device("meta"):
            model = PI0Pytorch(model_config)
        if device == "cuda" and not vision_on_cpu:
            model.paligemma_with_expert.paligemma.tie_weights()
            expert_head = model.paligemma_with_expert.gemma_expert.lm_head
            required = sum(param.numel() * param.element_size() for param in model.parameters())
            required -= sum(param.numel() * param.element_size() for param in expert_head.parameters())
            if embeddings_on_cpu:
                embeddings = model.paligemma_with_expert.paligemma.language_model.embed_tokens
                required -= sum(param.numel() * param.element_size() for param in embeddings.parameters())
            free, _ = torch.cuda.mem_get_info()
            if free < required:
                raise SystemExit(
                    f"GPU inference requires {required / 2**20:.0f} MiB for GPU-resident weights alone, "
                    f"but CUDA reports only {free / 2**20:.0f} MiB free. "
                    "Try again when the GPU is free, or use --vision-on-cpu."
                )
        load_into_meta(model, weight_path)
        if device == "cuda":
            prepare_gpu_transfer(model, vision_on_cpu, embeddings_on_cpu)
        return model

    with patch.object(type(config.model), "load_pytorch", load_model):
        return policy_config.create_trained_policy(config, checkpoint, pytorch_device=device)


def prepare_gpu_transfer(model, vision_on_cpu, embeddings_on_cpu):
    backbone = model.paligemma_with_expert
    paligemma = backbone.paligemma
    expert = backbone.gemma_expert
    original_to = model.to

    def guard_unused_head(module, args):
        raise RuntimeError("The vocabulary output head was unexpectedly called during action inference.")

    expert.lm_head.register_forward_pre_hook(guard_unused_head)
    paligemma.lm_head.register_forward_pre_hook(guard_unused_head)
    skipped = [(paligemma, "lm_head"), (expert, "lm_head")]

    if vision_on_cpu:
        skipped.extend((
            (paligemma.model, "vision_tower"),
            (paligemma.model, "multi_modal_projector"),
            (paligemma.language_model, "embed_tokens"),
        ))
        original_embed_image = backbone.embed_image

        def embed_image(image):
            return original_embed_image(image.to("cpu")).to(image.device)

        backbone.embed_image = embed_image

    if embeddings_on_cpu:
        skipped.append((paligemma.language_model, "embed_tokens"))
        original_embed_text = backbone.embed_language_tokens

        def embed_tokens(tokens):
            return original_embed_text(tokens.to("cpu")).to(tokens.device)

        backbone.embed_language_tokens = embed_tokens

    def move_policy_to_device(*args, **kwargs):
        # Skip unused heads, then restore the shared vocabulary weight after transfer.
        saved = [(parent, name, getattr(parent, name)) for parent, name in skipped]
        for parent, name, _ in saved:
            setattr(parent, name, None)
        try:
            result = original_to(*args, **kwargs)
        finally:
            for parent, name, module in saved:
                setattr(parent, name, module)
            del model.to
        paligemma.tie_weights()
        return result

    model.to = move_policy_to_device
