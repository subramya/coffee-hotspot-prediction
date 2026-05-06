"""
gat_model.py
------------
ST-GAT-GRU: Spatio-Temporal Graph Attention Network + GRU for
morning subway hotspot classification.

Architecture
------------
  Ridership branch : (N, T, 1)  → GRU         → (N, T, hidden_dim)
  Café branch      : (N, 1)     → Linear+ReLU → (N, 1, hidden_dim) broadcast T
  Weather branch   : (T, 2)     → Linear+ReLU → (1, T, hidden_dim) broadcast N

  Fusion           : concat 3×hidden_dim → Linear  → (N, T, embed_dim)

  GAT (per timestep):
    Layer 1: GATConv(embed_dim,    embed_dim, heads=4, concat=True)  → (N, 4*embed_dim)
    Layer 2: GATConv(4*embed_dim,  embed_dim, heads=4, concat=False) → (N, embed_dim)
    Result over T steps: (N, T, embed_dim)

  Outer GRU        : (N, T, embed_dim) → last hidden → (N, gru_hidden)
  Head             : Linear(gru_hidden → 1) → sigmoid → (N,)

forward() signature
-------------------
  x_ride    : (N, T, 1)
  x_cafe    : (N, 1)
  x_weather : (T, 2)   — one row per day, shared across stations
  edge_index: (2, E)
  edge_weight: (E,)

branch_forward() signature
--------------------------
  Same inputs as forward() plus:
  zero_branches : set of str from {'ride', 'cafe', 'weather'}
  Zeroes out the named branches before fusion — used for CCI computation.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class STGATGRUModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        embed_dim: int  = 64,
        gru_hidden: int = 64,
        gat_heads: int  = 4,
        dropout: float  = 0.3,
    ):
        """
        Parameters
        ----------
        hidden_dim : output size of each branch encoder
        embed_dim  : station embedding size after fusion (and GAT in/out)
        gru_hidden : hidden size of the outer temporal GRU
        gat_heads  : number of attention heads in both GAT layers
        dropout    : dropout rate applied after GAT layers and in GRU
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim  = embed_dim
        self.gru_hidden = gru_hidden
        self.dropout_p  = dropout

        # ── Branch encoders ───────────────────────────────────────────────────

        # Ridership: GRU over the T-day sequence
        self.ride_gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,   # input: (N, T, 1)
        )

        # Café: static scalar per station
        self.cafe_enc = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
        )

        # Weather: 2-dim daily vector (temp, precip)
        self.weather_enc = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
        )

        # ── Fusion ────────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, embed_dim),
            nn.ReLU(),
        )

        # ── GAT layers ───────────────────────────────────────────────────────
        # Layer 1: concat=True → output is heads * embed_dim
        self.gat1 = GATConv(
            in_channels=embed_dim,
            out_channels=embed_dim,
            heads=gat_heads,
            concat=True,
            dropout=dropout,
        )
        # Layer 2: concat=False → output is embed_dim (average of heads)
        self.gat2 = GATConv(
            in_channels=gat_heads * embed_dim,
            out_channels=embed_dim,
            heads=gat_heads,
            concat=False,
            dropout=dropout,
        )
        self.gat_drop = nn.Dropout(dropout)

        # ── Outer temporal GRU ────────────────────────────────────────────────
        # Integrates the sequence of per-timestep spatial embeddings
        self.temporal_gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,   # input: (N, T, embed_dim)
        )

        # ── Classification head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, 1),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode_branches(self, x_ride, x_cafe, x_weather):
        """
        Encode each branch and return (N, T, hidden_dim) tensors for all three.

        x_ride    : (N, T, 1)
        x_cafe    : (N, 1)
        x_weather : (T, 2)
        """
        N, T, _ = x_ride.shape

        # Ridership: GRU over T steps → hidden state at every step
        ride_enc, _ = self.ride_gru(x_ride)          # (N, T, hidden_dim)

        # Café: encode once, then tile across T timesteps
        cafe_enc = self.cafe_enc(x_cafe)              # (N, hidden_dim)
        cafe_enc = cafe_enc.unsqueeze(1).expand(N, T, -1)   # (N, T, hidden_dim)

        # Weather: encode each day, then tile across N stations
        weather_enc = self.weather_enc(x_weather)     # (T, hidden_dim)
        weather_enc = weather_enc.unsqueeze(0).expand(N, T, -1)  # (N, T, hidden_dim)

        return ride_enc, cafe_enc, weather_enc

    def _gat_pass(self, x, edge_index, edge_weight, _debug=False):
        """Single GAT forward: (N, embed_dim) → (N, embed_dim)."""
        if _debug:
            print(f"    [gat_debug] x.requires_grad={x.requires_grad}  "
                  f"x.shape={tuple(x.shape)}  "
                  f"edge_index.device={edge_index.device}  "
                  f"x.device={x.device}")
        assert x.device == edge_index.device, (
            f"Device mismatch: x on {x.device}, edge_index on {edge_index.device}")
        x = self.gat1(x, edge_index, edge_attr=edge_weight)
        if _debug:
            print(f"    [gat_debug] after gat1: requires_grad={x.requires_grad}")
        x = torch.relu(x)
        x = self.gat_drop(x)
        x = self.gat2(x, edge_index, edge_attr=edge_weight)
        x = torch.relu(x)
        return x                                      # (N, embed_dim)

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(self, x_ride, x_cafe, x_weather, edge_index, edge_weight, _debug=False):
        """
        Parameters
        ----------
        x_ride      : (N, T, 1)
        x_cafe      : (N, 1)
        x_weather   : (T, 2)
        edge_index  : (2, E)  — precomputed from adjacency matrix
        edge_weight : (E,)

        Returns
        -------
        logits : (N,)  — raw logits (caller applies sigmoid)
        """
        N, T, _ = x_ride.shape

        ride_enc, cafe_enc, weather_enc = self._encode_branches(
            x_ride, x_cafe, x_weather
        )

        # Fuse all 3 branches at each timestep → (N, T, embed_dim)
        fused = self.fusion(
            torch.cat([ride_enc, cafe_enc, weather_enc], dim=-1)
        )

        # Apply GAT independently at each timestep
        spatial_embeds = []
        for t in range(T):
            node_feat = fused[:, t, :]               # (N, embed_dim)
            spatial_embeds.append(
                self._gat_pass(node_feat, edge_index, edge_weight,
                               _debug=(_debug and t == 0)))
        spatial_seq = torch.stack(spatial_embeds, dim=1)  # (N, T, embed_dim)

        # Outer GRU over T spatial embeddings per station
        _, h_n = self.temporal_gru(spatial_seq)      # h_n: (1, N, gru_hidden)
        h_n = h_n.squeeze(0)                         # (N, gru_hidden)

        # Classification head — returns raw logits (caller applies sigmoid)
        logits = self.head(h_n).squeeze(-1)           # (N,)
        return logits

    def branch_forward(
        self,
        x_ride,
        x_cafe,
        x_weather,
        edge_index,
        edge_weight,
        zero_branches: set = None,
    ):
        """
        Same as forward() but with specified branches zeroed out.
        Used for the Coffee Contribution Index and ablation studies.

        Parameters
        ----------
        zero_branches : set of str
            Any combination of {'ride', 'cafe', 'weather'}.
            Named branches will be replaced with zeros before fusion.

        Returns
        -------
        probs : (N,)
        """
        if zero_branches is None:
            zero_branches = set()

        N, T, _ = x_ride.shape

        ride_enc, cafe_enc, weather_enc = self._encode_branches(
            x_ride, x_cafe, x_weather
        )

        if 'ride' in zero_branches:
            ride_enc = torch.zeros_like(ride_enc)
        if 'cafe' in zero_branches:
            cafe_enc = torch.zeros_like(cafe_enc)
        if 'weather' in zero_branches:
            weather_enc = torch.zeros_like(weather_enc)

        fused = self.fusion(
            torch.cat([ride_enc, cafe_enc, weather_enc], dim=-1)
        )

        spatial_embeds = []
        for t in range(T):
            node_feat = fused[:, t, :]
            spatial_embeds.append(self._gat_pass(node_feat, edge_index, edge_weight))
        spatial_seq = torch.stack(spatial_embeds, dim=1)

        _, h_n = self.temporal_gru(spatial_seq)
        h_n = h_n.squeeze(0)

        logits = self.head(h_n).squeeze(-1)
        return torch.sigmoid(logits)   # branch_forward always returns probs for CCI


# ════════════════════════════════════════════════════════════════════════════════
# Simplified model — no outer temporal GRU
# branches → fusion → GAT (once) → linear head
# Used while diagnosing convergence; avoids vanishing gradients through
# 7 unrolled GAT timesteps.
# ════════════════════════════════════════════════════════════════════════════════

class SimplifiedGATModel(nn.Module):
    """
    Shallow ST-GAT without the outer temporal GRU.

    Input shapes
    ------------
    x_ride    : (N, seq_len)   — flattened ridership window per station
    x_cafe    : (N, 1)         — static café density
    x_weather : (2,)           — today's (tmax, prcp), shared across stations
    edge_index: (2, E)
    edge_weight:(E,)

    Output
    ------
    logits : (N,)  — raw logits (caller applies sigmoid)
    """

    def __init__(
        self,
        seq_len:    int   = 7,
        hidden_dim: int   = 64,
        embed_dim:  int   = 64,
        gat_heads:  int   = 4,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ── Branch encoders ───────────────────────────────────────────────────
        self.ride_enc    = nn.Sequential(nn.Linear(seq_len, hidden_dim), nn.ReLU())
        self.cafe_enc    = nn.Sequential(nn.Linear(1,       hidden_dim), nn.ReLU())
        self.weather_enc = nn.Sequential(nn.Linear(2,       hidden_dim), nn.ReLU())

        # ── Fusion ────────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, embed_dim),
            nn.ReLU(),
        )

        # ── GAT (2 layers) ───────────────────────────────────────────────────
        self.gat1 = GATConv(embed_dim,          embed_dim,
                            heads=gat_heads, concat=True,  dropout=dropout)
        self.gat2 = GATConv(gat_heads*embed_dim, embed_dim,
                            heads=gat_heads, concat=False, dropout=dropout)
        self.gat_drop = nn.Dropout(dropout)

        # ── Head ─────────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x_ride, x_cafe, x_weather,
                edge_index, edge_weight, _debug=False):
        N = x_ride.shape[0]

        ride_enc    = self.ride_enc(x_ride)                          # (N, H)
        cafe_enc    = self.cafe_enc(x_cafe)                          # (N, H)
        weather_enc = self.weather_enc(x_weather).unsqueeze(0).expand(N, -1)  # (N, H)

        fused = self.fusion(
            torch.cat([ride_enc, cafe_enc, weather_enc], dim=-1))    # (N, embed)

        if _debug:
            print(f"    [gat_debug] fused.requires_grad={fused.requires_grad}  "
                  f"edge_index.device={edge_index.device}  fused.device={fused.device}")

        x = self.gat1(fused, edge_index, edge_attr=edge_weight)
        x = torch.relu(x)
        x = self.gat_drop(x)
        x = self.gat2(x, edge_index, edge_attr=edge_weight)
        x = torch.relu(x + fused)   # residual: prevents node-feature over-smoothing

        if _debug:
            print(f"    [gat_debug] gat_out.requires_grad={x.requires_grad}")

        logits = self.head(x).squeeze(-1)                            # (N,)
        return logits

    def branch_forward(self, x_ride, x_cafe, x_weather,
                       edge_index, edge_weight, zero_branches=None):
        """Zero out named branches for CCI computation. Returns sigmoid probs."""
        if zero_branches is None:
            zero_branches = set()
        N = x_ride.shape[0]

        ride_enc    = self.ride_enc(x_ride)
        cafe_enc    = self.cafe_enc(x_cafe)
        weather_enc = self.weather_enc(x_weather).unsqueeze(0).expand(N, -1)

        if 'ride'    in zero_branches: ride_enc    = torch.zeros_like(ride_enc)
        if 'cafe'    in zero_branches: cafe_enc    = torch.zeros_like(cafe_enc)
        if 'weather' in zero_branches: weather_enc = torch.zeros_like(weather_enc)

        fused = self.fusion(
            torch.cat([ride_enc, cafe_enc, weather_enc], dim=-1))

        x = self.gat1(fused, edge_index, edge_attr=edge_weight)
        x = torch.relu(x)
        x = self.gat_drop(x)
        x = self.gat2(x, edge_index, edge_attr=edge_weight)
        x = torch.relu(x + fused)   # residual
        return torch.sigmoid(self.head(x).squeeze(-1))


# ── Standalone validation ─────────────────────────────────────────────────────

if __name__ == "__main__":
    N = 123   # stations
    T = 7     # sequence length
    E = 3450  # approx edges (2 * 1725 undirected)

    model = STGATGRUModel(hidden_dim=64, embed_dim=64, gru_hidden=64,
                          gat_heads=4, dropout=0.3)
    model.eval()

    x_ride    = torch.randn(N, T, 1)
    x_cafe    = torch.randn(N, 1)
    x_weather = torch.randn(T, 2)
    src = torch.randint(0, N, (E,))
    dst = torch.randint(0, N, (E,))
    edge_index  = torch.stack([src, dst], dim=0)
    edge_weight = torch.rand(E)

    with torch.no_grad():
        logits = model.forward(x_ride, x_cafe, x_weather, edge_index, edge_weight)
        probs  = torch.sigmoid(logits)

    print("── STGATGRUModel architecture ──────────────────────────────────")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total params : {total_params:,}")
    print(f"  Output logits: {tuple(logits.shape)}  (expect ({N},))")
    print(f"  Sigmoid range: [{probs.min():.4f}, {probs.max():.4f}]")

    with torch.no_grad():
        probs_no_cafe = model.branch_forward(
            x_ride, x_cafe, x_weather, edge_index, edge_weight,
            zero_branches={'cafe'})
    mean_delta = (probs - probs_no_cafe).abs().mean().item()
    print(f"  Mean |Δprob| (zero café): {mean_delta:.4f}")
    print("────────────────────────────────────────────────────────────────")
