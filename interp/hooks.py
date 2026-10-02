"""Forward hooks on TransformerActor's residual stream: read (ActivationCapture)
and write (SteeringHook).

Sites (all (batch, n_tokens, embed_dim) activations):

    embed       obs_encoder output -- per-token obs embedding, before position/
                profile fusion
    tok_in      forward_pre_hook on actor.transformer -- the residual-stream
                entry (after input_proj + profile embedding)
    resid_L{i}  output of transformer.layers[i] -- NOTE the layer returns a
                TUPLE (x, attn_weights); capture reads [0], steering rebuilds
                the tuple so need_weights=True paths stay intact
    final_norm  transformer.norm output -- feeds fc_mean/fc_logstd directly,
                the most surgical steering point (no re-normalization after)

SteeringHook adds ``alpha * vector`` to the site activation. alpha is in RAW
activation units -- callers scale by the direction artifact's act_std to talk
in sigmas. alpha == 0.0 returns None from the hook (PyTorch keeps the original
output), so the baseline is bitwise-identical BY CONSTRUCTION, not by luck.
"""

import numpy as np
import torch


def resolve_site(actor, site: str):
    """Map a site name to (module, kind) where kind selects the hook flavor:
    'tensor' (plain output), 'tuple' ((x, attn) output), 'pre' (input arg 0)."""
    if site == "embed":
        return actor.obs_encoder, "tensor"
    if site == "tok_in":
        return actor.transformer, "pre"
    if site == "final_norm":
        return actor.transformer.norm, "tensor"
    if site.startswith("resid_L"):
        idx = int(site[len("resid_L"):])
        layers = actor.transformer.layers
        if not 0 <= idx < len(layers):
            raise ValueError(f"{site}: actor has {len(layers)} layers")
        return layers[idx], "tuple"
    raise ValueError(f"Unknown site {site!r}")


def default_sites(actor) -> list[str]:
    """All capture sites for this actor, residual layers included."""
    n_layers = len(actor.transformer.layers)
    return (["embed", "tok_in"]
            + [f"resid_L{i}" for i in range(n_layers)]
            + ["final_norm"])


class ActivationCapture:
    """Read-only forward hooks buffering the LATEST forward pass per site.

    Usage:
        with ActivationCapture(actor, sites) as cap:
            actor.get_action(...)
            acts = cap.snapshot()   # {site: (n_tokens, embed_dim) float32}

    Hooks only .detach() references -- they never modify the computation, so
    capture-on and capture-off rollouts produce identical actions.
    """

    def __init__(self, actor, sites):
        self.actor = actor
        self.sites = list(sites)
        self._buf: dict[str, torch.Tensor] = {}
        self._handles = []

    def _make_hook(self, site, kind):
        if kind == "pre":
            def hook(module, args):
                self._buf[site] = args[0].detach()
                return None
            return hook, True
        if kind == "tuple":
            def hook(module, args, output):
                self._buf[site] = output[0].detach()
                return None
            return hook, False

        def hook(module, args, output):
            self._buf[site] = output.detach()
            return None
        return hook, False

    def __enter__(self):
        for site in self.sites:
            module, kind = resolve_site(self.actor, site)
            fn, is_pre = self._make_hook(site, kind)
            handle = (module.register_forward_pre_hook(fn) if is_pre
                      else module.register_forward_hook(fn))
            self._handles.append(handle)
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def snapshot(self) -> dict[str, np.ndarray]:
        """Latest pass per site as {site: (n_tokens, embed_dim) float32}."""
        out = {}
        for site, t in self._buf.items():
            if t.ndim != 3 or t.shape[0] != 1:
                raise RuntimeError(
                    f"Site {site}: expected (1, n_tokens, d), got {tuple(t.shape)}"
                )
            out[site] = t[0].to("cpu", torch.float32).numpy().copy()
        return out


class SteeringHook:
    """Add alpha * vector to ONE site's activation during forward.

    Args:
        actor: the TransformerActor to steer.
        site: one of resolve_site's names (default steering site for the
            pipeline is "final_norm").
        vector: (embed_dim,) direction; steered as given (unit-normalize and
            scale alpha by act_std upstream to talk in sigmas).
        alpha: initial coefficient, RAW activation units. 0.0 = exact no-op.
        turbine_mask: optional (n_tokens,) bool -- steer only these tokens
            (per-turbine steering); None steers every token.

    ``set_alpha`` is the live knob (the Phase-4 slider calls it between env
    steps). Use as a context manager, or attach()/remove() explicitly.
    """

    def __init__(self, actor, site: str, vector, alpha: float = 0.0,
                 turbine_mask=None):
        self.actor = actor
        self.site = site
        self.module, self.kind = resolve_site(actor, site)
        self.vector = torch.as_tensor(
            np.asarray(vector), dtype=torch.float32).reshape(-1)
        self.alpha = float(alpha)
        self.turbine_mask = (
            None if turbine_mask is None
            else torch.as_tensor(np.asarray(turbine_mask), dtype=torch.float32).reshape(-1)
        )
        self._handle = None

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def _delta(self, ref: torch.Tensor) -> torch.Tensor:
        """(d,) or (n_tokens, d) additive offset matching ref's device/dtype."""
        v = self.vector.to(device=ref.device, dtype=ref.dtype)
        delta = self.alpha * v
        if self.turbine_mask is not None:
            mask = self.turbine_mask.to(device=ref.device, dtype=ref.dtype)
            if mask.shape[0] != ref.shape[1]:
                raise RuntimeError(
                    f"turbine_mask has {mask.shape[0]} entries but the pass "
                    f"has {ref.shape[1]} tokens"
                )
            delta = mask[:, None] * delta  # (n_tokens, d), broadcasts over batch
        return delta

    def _hook_pre(self, module, args):
        if self.alpha == 0.0:
            return None
        x = args[0]
        return (x + self._delta(x),) + tuple(args[1:])

    def _hook_tuple(self, module, args, output):
        if self.alpha == 0.0:
            return None
        x = output[0]
        return (x + self._delta(x),) + tuple(output[1:])

    def _hook_tensor(self, module, args, output):
        if self.alpha == 0.0:
            return None
        return output + self._delta(output)

    def attach(self):
        if self._handle is not None:
            raise RuntimeError("SteeringHook already attached")
        if self.kind == "pre":
            self._handle = self.module.register_forward_pre_hook(self._hook_pre)
        elif self.kind == "tuple":
            self._handle = self.module.register_forward_hook(self._hook_tuple)
        else:
            self._handle = self.module.register_forward_hook(self._hook_tensor)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.remove()
        return False
