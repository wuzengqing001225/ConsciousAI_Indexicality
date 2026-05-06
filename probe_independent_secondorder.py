"""
Probe: Second-Order Meta-Representation (Independent Parameters)
================================================================
Extends probe_independent.py with two experiment batteries designed to
strengthen evidence that P3 (behavioral self-monitoring) is genuine meta-
representation rather than simple closed-loop error correction.

Batteries:
  B1: Second-order decodability with orthogonality & causal mediation
      - First-order probe: h_t → actual_token_{t-1}
      - Second-order probe: h_t → was_corrupted_{t-1}
      - Indirect baseline: corruption estimate from first-order predictions
      - Orthogonality: are B1 and B2 weight directions independent?
      - Causal mediation: does P3 survive ablation of first-order subspace?
      - Thermometer baseline: no-echo agents should fail to detect corruption

  B2: Counterfactual intention perturbation
      - FORCE: override intended message to random non-intended token
              (echo=actual but actual≠intended)
      - CORRUPT: normal intended, but echo corrupted (echo≠actual==intended)
      - CLEAN: no intervention
      Measure: speak rate, retransmission rate at t_c+1
      Predictions disambiguate which comparison triggers P3

Companion to `probe_independent.py`.
"""

# %% ── 1. IMPORTS & CONFIG ──────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json, os, warnings
from scipy import stats as scipy_stats
warnings.filterwarnings('ignore')

# ── Save directory ──
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

if os.path.exists('/content/drive/MyDrive'):
    SAVE_DIR = "/content/drive/MyDrive/EL"
elif os.path.exists('/workspace'):
    SAVE_DIR = "/workspace/EL"
else:
    SAVE_DIR = "./EL"

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Model prefixes ──
MODEL_PREFIX = 'independent'
MODEL_PREFIX_NOECHO = 'independent_noecho'
PROBE_PREFIX = 'independent_secondorder_probe'

# ── Environment (must match training) ──
N_S = 3; N_C_TRAIN = 6; N_C_TEST = 3; N_C = N_C_TRAIN + N_C_TEST
N_A = 3; T_EP = 10; VOCAB = 6; SILENCE = VOCAB; V = VOCAB + 1
HIDDEN = 128
INPUT_DIM = N_S + N_C + V + V + T_EP

N_SEEDS = 10
SEEDS_A = list(range(42, 42 + N_SEEDS))
SEEDS_B = list(range(142, 142 + N_SEEDS))


# %% ── 2. AGENT CLASS ──────────────────────────────────────────────────
class GRUAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(INPUT_DIM, HIDDEN)
        self.msg_h = nn.Linear(HIDDEN, V)
        self.act_h = nn.Linear(HIDDEN, N_A)
        self.act_h2 = nn.Linear(HIDDEN, N_A)
        self.val_h = nn.Linear(HIDDEN, 1)

    def forward(self, x, h):
        h_new = self.gru(x, h)
        return (self.msg_h(h_new), self.act_h(h_new), self.act_h2(h_new),
                self.val_h(h_new).squeeze(-1), h_new)

    def sample(self, x, h):
        ml, al_o, al_s, v, h_new = self.forward(x, h)
        md = torch.distributions.Categorical(logits=ml)
        ad_o = torch.distributions.Categorical(logits=al_o)
        ad_s = torch.distributions.Categorical(logits=al_s)
        msg = md.sample(); act_o = ad_o.sample(); act_s = ad_s.sample()
        lp = md.log_prob(msg) + ad_o.log_prob(act_o) + ad_s.log_prob(act_s)
        ent = md.entropy() + ad_o.entropy() + ad_s.entropy()
        return msg, (act_o, act_s), lp, v, ent, h_new


# %% ── 3. HELPERS ────────────────────────────────────────────────────────
_T_EYE = None

def make_input(s, c, msg, echo, t):
    global _T_EYE
    s_oh = F.one_hot(s, N_S).float()
    if t >= 1:
        s_oh = torch.zeros_like(s_oh)
    if _T_EYE is None or _T_EYE.device != s.device:
        _T_EYE = torch.eye(T_EP, device=s.device)
    t_oh = _T_EYE[t].expand(*s.shape, T_EP)
    return torch.cat([s_oh, F.one_hot(c, N_C).float(),
                      F.one_hot(msg, V).float(),
                      F.one_hot(echo, V).float(), t_oh], dim=-1)


def get_target_other(s_other, c_self, t, dynamic=True):
    if dynamic: return (s_other + c_self + t) % N_A
    return (s_other + c_self) % N_A


def load_agent(prefix, tag, seed):
    model_name = f"{prefix}_{tag}_seed{seed}_model.pth"
    model_path = os.path.join(SAVE_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"  WARNING: {model_path} not found")
        return None
    agent = GRUAgent().to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    return agent


def load_all_agents(prefix=MODEL_PREFIX):
    agents_A, agents_B = [], []
    for i in range(N_SEEDS):
        a = load_agent(prefix, 'A', SEEDS_A[i])
        b = load_agent(prefix, 'B', SEEDS_B[i])
        if a is not None and b is not None:
            agents_A.append(a)
            agents_B.append(b)
    return agents_A, agents_B


def train_linear_probe(X, y, n_classes, n_epochs=200, lr=1e-2):
    """Train linear probe with 80/20 train/test split."""
    X = X.detach()
    y = y.detach()
    n = X.shape[0]
    perm = torch.randperm(n, device=X.device)
    n_train = int(0.8 * n)
    X_train, X_test = X[perm[:n_train]], X[perm[n_train:]]
    y_train, y_test = y[perm[:n_train]], y[perm[n_train:]]

    with torch.enable_grad():
        probe = nn.Linear(X.shape[1], n_classes).to(X.device)
        opt = torch.optim.Adam(probe.parameters(), lr=lr)
        for _ in range(n_epochs):
            logits = probe(X_train)
            loss = F.cross_entropy(logits, y_train)
            opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        preds = probe(X_test).argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
    return acc, probe


@torch.no_grad()
def collect_probe_data(agent_left, agent_right, n=20000, epsilon=0.25,
                       corruption_type='uniform'):
    """Collect h_t, intended, actual, corrupt per timestep.

    corruption_type:
      'uniform'       — replace with uniform random token (training distribution)
      'fixed0'        — always replace with token 0
      'adjacent'      — replace with (intended + 1) % V
      'silence'       — always replace with SILENCE token
      'policy_matched' — resample from agent's own softmax distribution (≠ intended)
    """
    train_c = list(range(N_C_TRAIN))

    s_L = torch.randint(0, N_S, (n,), device=device)
    s_R = torch.randint(0, N_S, (n,), device=device)
    c_L = torch.tensor(np.random.choice(train_c, n), device=device)
    c_R = torch.tensor(np.random.choice(train_c, n), device=device)

    mL = torch.full((n,), SILENCE, device=device, dtype=torch.long)
    mR = torch.full((n,), SILENCE, device=device, dtype=torch.long)
    echoL = torch.full((n,), SILENCE, device=device, dtype=torch.long)
    echoR = torch.full((n,), SILENCE, device=device, dtype=torch.long)

    hL = torch.zeros(n, HIDDEN, device=device)
    hR = torch.zeros(n, HIDDEN, device=device)

    h_list, intended_list, actual_list = [], [], []
    match_list, s_self_list, s_other_list = [], [], []
    corrupt_list = []

    for t in range(T_EP):
        xL = make_input(s_L, c_L, mR, echoL, t)
        xR = make_input(s_R, c_R, mL, echoR, t)

        h_new_L = agent_left.gru(xL, hL)
        ml_L = agent_left.msg_h(h_new_L)
        mL_intended = torch.distributions.Categorical(logits=ml_L).sample()

        mask = torch.rand(n, device=device) < epsilon
        if corruption_type == 'uniform':
            replacement = torch.randint(0, V, (n,), device=device)
        elif corruption_type == 'fixed0':
            replacement = torch.zeros(n, device=device, dtype=torch.long)
        elif corruption_type == 'adjacent':
            replacement = (mL_intended + 1) % V
        elif corruption_type == 'silence':
            replacement = torch.full((n,), SILENCE, device=device, dtype=torch.long)
        elif corruption_type == 'policy_matched':
            # Resample from agent's own softmax distribution, excluding intended
            probs = F.softmax(ml_L, dim=-1)  # [n, V]
            # Zero out the intended token to force ≠ intended
            probs_masked = probs.scatter(1, mL_intended.unsqueeze(1), 0.0)
            probs_masked = probs_masked / (probs_masked.sum(dim=1, keepdim=True) + 1e-8)
            replacement = torch.multinomial(probs_masked, 1).squeeze(1)
        else:
            raise ValueError(f"Unknown corruption_type: {corruption_type}")
        mL_transmitted = torch.where(mask, replacement, mL_intended)

        h_new_R = agent_right.gru(xR, hR)
        ml_R = agent_right.msg_h(h_new_R)
        mR_transmitted = torch.distributions.Categorical(logits=ml_R).sample()

        h_list.append(h_new_L.clone())
        intended_list.append(mL_intended.clone())
        actual_list.append(mL_transmitted.clone())
        match_list.append((mL_intended == mL_transmitted).long())
        s_self_list.append(s_L.clone())
        s_other_list.append(s_R.clone())
        corrupt_list.append(mask.long())

        echoL = mL_transmitted; echoR = mR_transmitted
        mL = mL_transmitted; mR = mR_transmitted
        hL = h_new_L; hR = h_new_R

    return {
        'h': torch.stack(h_list),
        'intended': torch.stack(intended_list),
        'actual': torch.stack(actual_list),
        'match': torch.stack(match_list),
        's_self': torch.stack(s_self_list),
        's_other': torch.stack(s_other_list),
        'corrupt': torch.stack(corrupt_list),
    }


# %% ── 4. BATTERY 1: SECOND-ORDER DECODABILITY ─────────────────────────
@torch.no_grad()
def run_secondorder_battery(agent_left, agent_right, agent_left_noecho,
                            agent_right_noecho, seed_idx):
    """
    Battery 1: Second-order decodability with orthogonality & causal mediation.
    Tests whether the hidden state encodes corruption detection as independent
    meta-representation, not just derived from first-order token predictions.
    """
    results = {}

    print(f"    B1.1: Collecting probe data (echo-trained agents)...")
    data = collect_probe_data(agent_left, agent_right, n=12000, epsilon=0.25)

    print(f"    B1.2: Training first-order and second-order probes...")
    b1_results = {}

    for t in range(2, T_EP):  # t >= 2 so t-1 has echo to process
        h_t = data['h'][t]
        actual_tm1 = data['actual'][t-1]
        corrupt_tm1 = data['corrupt'][t-1]
        intended_tm1 = data['intended'][t-1]

        # Probe A: h_t → actual_token_{t-1} (first-order)
        acc_A, probe_A = train_linear_probe(h_t, actual_tm1, V, n_epochs=150)
        b1_results[f't{t}_probeA_actual_acc'] = acc_A

        # Probe B: h_t → was_corrupted_{t-1} (second-order)
        acc_B, probe_B = train_linear_probe(h_t, corrupt_tm1, 2, n_epochs=150)
        b1_results[f't{t}_probeB_corrupt_acc'] = acc_B

        # Indirect baseline: predict corruption from Probe A predictions
        with torch.no_grad():
            preds_A = probe_A(h_t).argmax(dim=1)
            indirect_corrupt = (preds_A != intended_tm1).long()
            # Recompute accuracy on this derived signal
            perm = torch.randperm(h_t.shape[0], device=device)
            n_test = h_t.shape[0] // 5
            indir_acc = (indirect_corrupt[perm[:n_test]] == corrupt_tm1[perm[:n_test]]).float().mean().item()
        b1_results[f't{t}_indirect_corrupt_acc'] = indir_acc

        # Orthogonality: cosine similarity between probe directions
        # W_A: [V, H] — its row space is the subspace A uses in H-dim space
        # W_B: [2, H] — B's decision direction in H-dim space
        W_A = probe_A.weight.detach()  # [V, H]
        W_B = probe_B.weight.detach()  # [2, H]
        B_dir = W_B[1] - W_B[0]  # [H], corrupted vs clean direction

        # SVD of W_A to get its row space basis in H-dim space
        # W_A = U @ diag(S) @ Vh, where Vh: [V, H] — rows are right singular vectors
        _, S_A, Vh_A = torch.linalg.svd(W_A, full_matrices=False)  # Vh_A: [V, H]
        rank_A = (S_A > 1e-6).sum().item()
        A_basis = Vh_A[:rank_A].T  # [H, rank_A], columns span A's row space

        # Project B_dir onto A's row space
        proj_coef = A_basis.T @ B_dir  # [rank_A]
        proj_vec = A_basis @ proj_coef  # [H]
        cos_sim = (torch.norm(proj_vec) / (torch.norm(B_dir) + 1e-8)).item()
        orth = 1.0 - cos_sim
        b1_results[f't{t}_orthogonality'] = orth
        b1_results[f't{t}_cos_sim_AB'] = cos_sim

        # Causal mediation: project h_t to nullspace of A's row space, re-run B
        # Nullspace basis: Vh_A[rank_A:] gives directions orthogonal to A's row space
        _, _, Vh_A_full = torch.linalg.svd(W_A, full_matrices=True)  # Vh_A_full: [H, H]
        V_null = Vh_A_full[rank_A:].T  # [H, H-rank_A], nullspace of A's row space
        # Project h onto nullspace (remove A-decodable information)
        h_proj_null = h_t @ V_null @ V_null.T  # stays in H-dim space
        # Train B on nullspace-projected h
        acc_B_ablated, _ = train_linear_probe(h_proj_null, corrupt_tm1, 2, n_epochs=150)
        b1_results[f't{t}_probeB_corrupt_acc_ablated'] = acc_B_ablated
        # Chance-corrected mediation retention
        chance_acc = max(corrupt_tm1.float().mean().item(), 1 - corrupt_tm1.float().mean().item())
        above_chance_orig = acc_B - chance_acc
        above_chance_ablated = acc_B_ablated - chance_acc
        mediation_retention_raw = acc_B_ablated / (acc_B + 1e-8)
        mediation_retention_corrected = above_chance_ablated / (above_chance_orig + 1e-8)
        b1_results[f't{t}_chance_acc'] = chance_acc
        b1_results[f't{t}_mediation_retention_raw'] = mediation_retention_raw
        b1_results[f't{t}_mediation_retention'] = mediation_retention_corrected

    # Thermometer baseline: no-echo agents
    print(f"    B1.3: Thermometer baseline (no-echo agents)...")
    if agent_left_noecho is not None and agent_right_noecho is not None:
        data_noecho = collect_probe_data(agent_left_noecho, agent_right_noecho, n=8000, epsilon=0.25)
        for t in range(2, T_EP):  # all timesteps
            h_t = data_noecho['h'][t]
            corrupt_tm1 = data_noecho['corrupt'][t-1]
            acc_B_noecho, _ = train_linear_probe(h_t, corrupt_tm1, 2, n_epochs=150)
            b1_results[f't{t}_probeB_corrupt_acc_noecho'] = acc_B_noecho

    # ── Cross-condition probe transfer: echo ↔ no-echo ──
    print(f"    B1.3b: Cross-condition probe transfer (echo ↔ no-echo)...")
    cross_results = {}
    if agent_left_noecho is not None and agent_right_noecho is not None:
        for t in range(2, T_EP):
            h_echo = data['h'][t]
            corrupt_echo = data['corrupt'][t - 1]
            h_noecho = data_noecho['h'][t]
            corrupt_noecho = data_noecho['corrupt'][t - 1]

            # Train on echo, test on no-echo
            _, probe_echo = train_linear_probe(h_echo, corrupt_echo, 2, n_epochs=150)
            with torch.no_grad():
                preds_on_noecho = probe_echo(h_noecho).argmax(dim=1)
                acc_echo2noecho = (preds_on_noecho == corrupt_noecho).float().mean().item()
            cross_results[f't{t}_echo_to_noecho'] = acc_echo2noecho

            # Train on no-echo, test on echo
            _, probe_noecho = train_linear_probe(h_noecho, corrupt_noecho, 2, n_epochs=150)
            with torch.no_grad():
                preds_on_echo = probe_noecho(h_echo).argmax(dim=1)
                acc_noecho2echo = (preds_on_echo == corrupt_echo).float().mean().item()
            cross_results[f't{t}_noecho_to_echo'] = acc_noecho2echo

        # s_self cross-transfer control: if coordinate systems differ generically,
        # s_self probe should also fail to transfer; if only corruption probe fails,
        # that's corruption-specific representational uniqueness
        for t in range(2, T_EP):
            h_echo = data['h'][t]
            s_self_echo = data['s_self'][t]
            h_noecho = data_noecho['h'][t]
            s_self_noecho = data_noecho['s_self'][t]

            # s_self: self-test baselines (don't assume 1.0)
            acc_ss_self_echo, probe_ss_echo = train_linear_probe(h_echo, s_self_echo, N_S, n_epochs=150)
            acc_ss_self_noecho, probe_ss_noecho = train_linear_probe(h_noecho, s_self_noecho, N_S, n_epochs=150)
            cross_results[f't{t}_sself_self_echo'] = acc_ss_self_echo
            cross_results[f't{t}_sself_self_noecho'] = acc_ss_self_noecho

            # s_self: train on echo, test on no-echo
            with torch.no_grad():
                preds_ss = probe_ss_echo(h_noecho).argmax(dim=1)
                acc_ss_e2n = (preds_ss == s_self_noecho).float().mean().item()
            cross_results[f't{t}_sself_echo_to_noecho'] = acc_ss_e2n

            # s_self: train on no-echo, test on echo
            with torch.no_grad():
                preds_ss2 = probe_ss_noecho(h_echo).argmax(dim=1)
                acc_ss_n2e = (preds_ss2 == s_self_echo).float().mean().item()
            cross_results[f't{t}_sself_noecho_to_echo'] = acc_ss_n2e

        # Print early timesteps
        for t in range(2, min(T_EP, 5)):
            e2n = cross_results.get(f't{t}_echo_to_noecho', 0)
            n2e = cross_results.get(f't{t}_noecho_to_echo', 0)
            ss_e2n = cross_results.get(f't{t}_sself_echo_to_noecho', 0)
            ss_n2e = cross_results.get(f't{t}_sself_noecho_to_echo', 0)
            print(f"      t={t}: corrupt e→n={e2n:.3f} n→e={n2e:.3f} | s_self e→n={ss_e2n:.3f} n→e={ss_n2e:.3f}")

    results['B1_cross_transfer'] = cross_results
    results['B1'] = b1_results

    # ── Generalization test: train ProbeB on uniform, test on novel corruption types ──
    # Tests across ALL timesteps t=2..9, with shuffle baseline and no-echo control
    print(f"    B1.4: Generalization test (corruption-type transfer, all timesteps)...")
    gen_results = {}
    novel_types = ['fixed0', 'adjacent', 'silence', 'policy_matched']

    # Pre-collect data for all novel corruption types (echo-trained agents)
    novel_data = {}
    for ctype in novel_types:
        novel_data[ctype] = collect_probe_data(agent_left, agent_right, n=8000,
                                               epsilon=0.25, corruption_type=ctype)

    # Also collect novel corruption data for no-echo agents (if available)
    novel_data_noecho = {}
    if agent_left_noecho is not None and agent_right_noecho is not None:
        data_noecho_uniform = collect_probe_data(agent_left_noecho, agent_right_noecho,
                                                 n=8000, epsilon=0.25, corruption_type='uniform')
        for ctype in novel_types:
            novel_data_noecho[ctype] = collect_probe_data(
                agent_left_noecho, agent_right_noecho, n=8000,
                epsilon=0.25, corruption_type=ctype)

    for t in range(2, T_EP):
        h_t = data['h'][t]
        corrupt_tm1 = data['corrupt'][t - 1]

        # Train ProbeB on uniform corruption data at this timestep
        acc_uniform, probe_B_uniform = train_linear_probe(h_t, corrupt_tm1, 2, n_epochs=200)
        gen_results[f't{t}_uniform'] = acc_uniform

        # Shuffle baseline: same probe evaluated on shuffled labels → should be ~chance
        perm_shuffle = torch.randperm(corrupt_tm1.shape[0], device=device)
        corrupt_shuffled = corrupt_tm1[perm_shuffle]
        with torch.no_grad():
            preds_shuf = probe_B_uniform(h_t).argmax(dim=1)
            # Compare predictions to SHUFFLED labels (breaks true correlation)
            acc_shuffle = (preds_shuf == corrupt_shuffled).float().mean().item()
        gen_results[f't{t}_shuffle_baseline'] = acc_shuffle

        # Test on each novel corruption type
        for ctype in novel_types:
            h_novel = novel_data[ctype]['h'][t]
            corrupt_novel = novel_data[ctype]['corrupt'][t - 1]
            with torch.no_grad():
                preds_novel = probe_B_uniform(h_novel).argmax(dim=1)
                acc_novel = (preds_novel == corrupt_novel).float().mean().item()
            gen_results[f't{t}_{ctype}'] = acc_novel

        # No-echo generalization control: train ProbeB on no-echo uniform, test on novel
        if agent_left_noecho is not None and agent_right_noecho is not None:
            h_noecho = data_noecho_uniform['h'][t]
            corrupt_noecho = data_noecho_uniform['corrupt'][t - 1]
            acc_noecho_uniform, probe_B_noecho = train_linear_probe(
                h_noecho, corrupt_noecho, 2, n_epochs=200)
            gen_results[f't{t}_noecho_uniform'] = acc_noecho_uniform
            for ctype in novel_types:
                h_ne_novel = novel_data_noecho[ctype]['h'][t]
                corrupt_ne_novel = novel_data_noecho[ctype]['corrupt'][t - 1]
                with torch.no_grad():
                    preds_ne = probe_B_noecho(h_ne_novel).argmax(dim=1)
                    acc_ne = (preds_ne == corrupt_ne_novel).float().mean().item()
                gen_results[f't{t}_noecho_{ctype}'] = acc_ne

        if t <= 4:  # print detail for early timesteps only
            print(f"      t={t}: uniform={acc_uniform:.3f}, shuffle={acc_shuffle:.3f}, "
                  + ", ".join(f"{c}={gen_results[f't{t}_{c}']:.3f}" for c in novel_types))

    results['B1_generalization'] = gen_results

    return results


# %% ── 5. BATTERY 2: COUNTERFACTUAL INTENTION PERTURBATION ──────────────
@torch.no_grad()
def run_counterfactual_intention(agent_left, agent_right, n=20000, t_c=2):
    """
    Battery 2: Counterfactual intention perturbation.
    Distinguishes between:
      (A) P3 compares intention vs echo (detect echo mismatch)
      (B) P3 compares intention vs actual (detect own output mismatch)

    Three conditions:
      CLEAN: normal episode, no intervention
      FORCE: override intended to random ≠ intended; echo returns forced token
             → echo==actual but actual≠intended
      CORRUPT: normal intended; corrupt echo
               → echo≠actual==intended
    """
    assert t_c <= T_EP - 2
    train_c = list(range(N_C_TRAIN))

    conditions = {}

    for cond in ['clean', 'force', 'corrupt', 'force_noecho']:
        s_L = torch.randint(0, N_S, (n,), device=device)
        s_R = torch.randint(0, N_S, (n,), device=device)
        c_L = torch.tensor(np.random.choice(train_c, n), device=device)
        c_R = torch.tensor(np.random.choice(train_c, n), device=device)

        mL = torch.full((n,), SILENCE, device=device, dtype=torch.long)
        mR = torch.full((n,), SILENCE, device=device, dtype=torch.long)
        echoL = torch.full((n,), SILENCE, device=device, dtype=torch.long)
        echoR = torch.full((n,), SILENCE, device=device, dtype=torch.long)

        hL = torch.zeros(n, HIDDEN, device=device)
        hR = torch.zeros(n, HIDDEN, device=device)

        speak_rate = 0.0
        retx_intended_rate = 0.0
        retx_actual_rate = 0.0
        mL_intended_at_tc = None
        mL_actual_at_tc = None

        # force_noecho behaves like force at t_c, but silences echo at t_c
        is_force_variant = cond in ('force', 'force_noecho')

        for t in range(min(T_EP, t_c + 3)):
            xL = make_input(s_L, c_L, mR, echoL, t)
            xR = make_input(s_R, c_R, mL, echoR, t)

            mL_intended, _, _, _, _, hL = agent_left.sample(xL, hL)
            mR_intended, _, _, _, _, hR = agent_right.sample(xR, hR)

            mL_actual = mL_intended.clone()
            mR_actual = mR_intended.clone()

            # Apply condition at t_c
            if t == t_c:
                mL_intended_at_tc = mL_intended.clone()

                if is_force_variant:
                    # Override to random token != intended (per sample)
                    mL_actual = torch.randint(0, V, (n,), device=device)
                    collision = (mL_actual == mL_intended)
                    mL_actual[collision] = (mL_actual[collision] + 1) % V
                elif cond == 'corrupt':
                    pass  # actual stays as intended; echo corrupted below

                mL_actual_at_tc = mL_actual.clone()

            # Set echo channels
            if t == t_c and cond == 'corrupt':
                # Corrupt echo: random token != actual, so mismatch is guaranteed
                rand_echo = torch.randint(0, V, (n,), device=device)
                collision = (rand_echo == mL_actual)
                rand_echo[collision] = (rand_echo[collision] + 1) % V
                echoL = rand_echo
            elif t == t_c and cond == 'force_noecho':
                # Force at t_c but silence the echo — agent doesn't hear back
                echoL = torch.full((n,), SILENCE, device=device, dtype=torch.long)
            else:
                echoL = mL_actual

            echoR = mR_actual
            mL = mL_actual
            mR = mR_actual

            if t == t_c + 1:
                spoke_mask = (mL_intended != SILENCE)
                speak_rate = spoke_mask.float().mean().item()
                # Retx conditioned on agent actually speaking at t_c+1
                if spoke_mask.sum() > 0:
                    retx_intended_rate = (mL_intended[spoke_mask] == mL_intended_at_tc[spoke_mask]).float().mean().item()
                    retx_actual_rate = (mL_intended[spoke_mask] == mL_actual_at_tc[spoke_mask]).float().mean().item()
                else:
                    retx_intended_rate = 0.0
                    retx_actual_rate = 0.0

        conditions[cond] = {
            'speak_rate': speak_rate,
            'retx_intended_rate': retx_intended_rate,
            'retx_actual_rate': retx_actual_rate,
        }

    # Compute contrasts
    results = {
        'B2_clean_speak': conditions['clean']['speak_rate'],
        'B2_force_speak': conditions['force']['speak_rate'],
        'B2_corrupt_speak': conditions['corrupt']['speak_rate'],
        'B2_force_noecho_speak': conditions['force_noecho']['speak_rate'],
        'B2_force_vs_clean_speak': conditions['force']['speak_rate'] - conditions['clean']['speak_rate'],
        'B2_corrupt_vs_clean_speak': conditions['corrupt']['speak_rate'] - conditions['clean']['speak_rate'],
        'B2_force_noecho_vs_clean_speak': conditions['force_noecho']['speak_rate'] - conditions['clean']['speak_rate'],
        # Retx of intended token (what agent wanted to say)
        'B2_clean_retx_intended': conditions['clean']['retx_intended_rate'],
        'B2_force_retx_intended': conditions['force']['retx_intended_rate'],
        'B2_corrupt_retx_intended': conditions['corrupt']['retx_intended_rate'],
        'B2_force_noecho_retx_intended': conditions['force_noecho']['retx_intended_rate'],
        # Retx of actual token (what was transmitted, differs from intended in FORCE)
        'B2_clean_retx_actual': conditions['clean']['retx_actual_rate'],
        'B2_force_retx_actual': conditions['force']['retx_actual_rate'],
        'B2_corrupt_retx_actual': conditions['corrupt']['retx_actual_rate'],
        'B2_force_noecho_retx_actual': conditions['force_noecho']['retx_actual_rate'],
    }

    return results



# %% ── 6. MAIN RUNNER ──────────────────────────────────────────────────
print("\n" + "="*80)
print(f"  {PROBE_PREFIX.upper()} — Second-Order Meta-Representation Diagnostics")
print("="*80)

print(f"\nLoading agents...")
agents_A, agents_B = load_all_agents(MODEL_PREFIX)
agents_A_noecho, agents_B_noecho = load_all_agents(MODEL_PREFIX_NOECHO)

if not agents_A:
    print("No agents found. Exiting.")
    exit()

print(f"Loaded {len(agents_A)} agent pairs (echo-trained)")
print(f"Loaded {len(agents_A_noecho)} agent pairs (no-echo)")

all_results = []

print(f"\n{'='*80}")
print(f"  BATTERY 1: Second-Order Decodability & Causal Mediation")
print(f"{'='*80}")

b1_results = []
for i in range(len(agents_A)):
    print(f"\n  Seed {i} (seeds A={SEEDS_A[i]}, B={SEEDS_B[i]}):")
    noecho_left = agents_A_noecho[i] if i < len(agents_A_noecho) else None
    noecho_right = agents_B_noecho[i] if i < len(agents_B_noecho) else None
    res = run_secondorder_battery(agents_A[i], agents_B[i], noecho_left, noecho_right, i)
    b1_results.append(res)

    if 'B1' in res:
        b1 = res['B1']
        # Print summary stats
        t3_accs = [b1.get(f't{t}_probeB_corrupt_acc', 0) for t in range(2, min(T_EP, 5))]
        t3_orth = [b1.get(f't{t}_orthogonality', 0) for t in range(2, min(T_EP, 5))]
        print(f"    ProbeB (corrupt detection) avg: {np.mean(t3_accs):.3f}")
        print(f"    Orthogonality avg: {np.mean(t3_orth):.3f}")
    if 'B1_generalization' in res:
        gen = res['B1_generalization']
        # Print compact summary: policy_matched at t=3 (the hardest, most informative test)
        pm3 = gen.get('t3_policy_matched', None)
        shuf3 = gen.get('t3_shuffle_baseline', None)
        uni3 = gen.get('t3_uniform', None)
        print(f"    Generalization (t=3): uniform={uni3:.3f}, policy_matched={pm3:.3f}, shuffle={shuf3:.3f}" if pm3 is not None else "    Generalization: N/A")

print(f"\n{'='*80}")
print(f"  BATTERY 2: Counterfactual Intention Perturbation")
print(f"{'='*80}")

b2_results = []
for i in range(len(agents_A)):
    print(f"\n  Seed {i} (seeds A={SEEDS_A[i]}, B={SEEDS_B[i]}):")
    res = run_counterfactual_intention(agents_A[i], agents_B[i], n=20000, t_c=2)
    b2_results.append(res)
    print(f"    CLEAN speak: {res['B2_clean_speak']:.3f}")
    print(f"    FORCE speak: {res['B2_force_speak']:.3f} (Δ={res['B2_force_vs_clean_speak']:+.3f})")
    print(f"    CORRUPT speak: {res['B2_corrupt_speak']:.3f} (Δ={res['B2_corrupt_vs_clean_speak']:+.3f})")
    print(f"    FORCE_NOECHO speak: {res['B2_force_noecho_speak']:.3f} (Δ={res['B2_force_noecho_vs_clean_speak']:+.3f})")

all_results = {
    'B1': b1_results,
    'B2': b2_results,
    '_meta': {
        'probe_prefix': PROBE_PREFIX,
        'n_seeds': len(agents_A),
    }
}

# %% ── 7. PLOT & SAVE ───────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  PLOTTING & SAVING RESULTS")
print(f"{'='*80}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"{PROBE_PREFIX} — Second-Order Meta-Representation Analysis",
             fontsize=14, fontweight='bold')

# [0,0]: Probe B accuracy (echo vs no-echo agents) per timestep
ax = axes[0, 0]
ts = list(range(2, T_EP))
b1_acc = []
b1_acc_noecho = []
for t in ts:
    acc_vals = [b['B1'].get(f't{t}_probeB_corrupt_acc', 0.5) for b in b1_results]
    acc_vals_noecho = [b['B1'].get(f't{t}_probeB_corrupt_acc_noecho', 0.5) for b in b1_results if f't{t}_probeB_corrupt_acc_noecho' in b['B1']]
    b1_acc.append(np.mean(acc_vals))
    b1_acc_noecho.append(np.mean(acc_vals_noecho) if acc_vals_noecho else 0.5)
ax.plot(ts, b1_acc, 'o-', color='#27ae60', label='Echo-trained', linewidth=2, markersize=8)
if any(b1_acc_noecho):
    ax.plot(ts, b1_acc_noecho, 's--', color='#e74c3c', label='No-echo', linewidth=2, markersize=8)
ax.axhline(0.5, c='gray', ls=':', alpha=0.5, label='chance')
ax.set_xlabel("Timestep"); ax.set_ylabel("Accuracy")
ax.set_title("B1: ProbeB Corruption Detection (lag-1)")
ax.legend(); ax.grid(alpha=0.3)

# [0,1]: Probe B direct vs indirect accuracy comparison
ax = axes[0, 1]
direct_accs = []
indirect_accs = []
for t in range(2, T_EP):
    d_vals = [b['B1'].get(f't{t}_probeB_corrupt_acc', 0) for b in b1_results]
    i_vals = [b['B1'].get(f't{t}_indirect_corrupt_acc', 0) for b in b1_results]
    direct_accs.append(np.mean(d_vals))
    indirect_accs.append(np.mean(i_vals))
ax.plot(range(2, T_EP), direct_accs, 'o-', color='#8e44ad', label='Direct (ProbeB)', linewidth=2)
ax.plot(range(2, T_EP), indirect_accs, 's--', color='#f39c12', label='Indirect (from ProbeA)', linewidth=2)
ax.axhline(0.5, c='gray', ls=':', alpha=0.5)
ax.set_xlabel("Timestep"); ax.set_ylabel("Accuracy")
ax.set_title("B1: Direct vs Indirect Corruption Detection")
ax.legend(); ax.grid(alpha=0.3)

# [0,2]: Causal mediation: Probe B accuracy before/after ablating Probe A
ax = axes[0, 2]
original_accs = []
ablated_accs = []
for t in range(2, T_EP):
    o_vals = [b['B1'].get(f't{t}_probeB_corrupt_acc', 0) for b in b1_results]
    a_vals = [b['B1'].get(f't{t}_probeB_corrupt_acc_ablated', 0) for b in b1_results]
    original_accs.append(np.mean(o_vals))
    ablated_accs.append(np.mean(a_vals))
x = np.arange(len(range(2, T_EP)))
w = 0.35
ax.bar(x - w/2, original_accs, w, label='Original', color='#27ae60', alpha=0.8)
ax.bar(x + w/2, ablated_accs, w, label='Nullspace-ablated', color='#e74c3c', alpha=0.8)
ax.axhline(0.5, c='gray', ls=':', alpha=0.5)
ax.set_ylabel("Accuracy"); ax.set_title("B1: Causal Mediation Ablation")
ax.set_xticks(x); ax.set_xticklabels(range(2, T_EP))
ax.set_xlabel("Timestep")
ax.legend(); ax.grid(alpha=0.3)

# [1,0]: B2 — Counterfactual intention: speak rates for CLEAN/FORCE/CORRUPT/FORCE_NOECHO
ax = axes[1, 0]
clean_sp = np.mean([b.get('B2_clean_speak', 0) for b in b2_results])
force_sp = np.mean([b.get('B2_force_speak', 0) for b in b2_results])
corrupt_sp = np.mean([b.get('B2_corrupt_speak', 0) for b in b2_results])
fne_sp = np.mean([b.get('B2_force_noecho_speak', 0) for b in b2_results])
clean_err = np.std([b.get('B2_clean_speak', 0) for b in b2_results])
force_err = np.std([b.get('B2_force_speak', 0) for b in b2_results])
corrupt_err = np.std([b.get('B2_corrupt_speak', 0) for b in b2_results])
fne_err = np.std([b.get('B2_force_noecho_speak', 0) for b in b2_results])
x_pos = [0, 1, 2, 3]
ax.bar(x_pos, [clean_sp, force_sp, corrupt_sp, fne_sp],
       yerr=[clean_err, force_err, corrupt_err, fne_err],
       color=['#95a5a6', '#3498db', '#e74c3c', '#9b59b6'], alpha=0.8, capsize=5)
ax.set_xticks(x_pos); ax.set_xticklabels(['CLEAN', 'FORCE', 'CORRUPT', 'FORCE\n_NOECHO'], fontsize=8)
ax.set_ylabel("Speak Rate")
ax.set_title("B2: Speak Rate at t_c+1")
ax.grid(alpha=0.3, axis='y')

# [1,1]: B2 — Per-seed scatter: FORCE vs CORRUPT speak rate (equivalence test)
ax = axes[1, 1]
force_per_seed = [b.get('B2_force_speak', 0) for b in b2_results]
corrupt_per_seed = [b.get('B2_corrupt_speak', 0) for b in b2_results]
seed_labels = [f"s{i}" for i in range(len(b2_results))]
ax.scatter(force_per_seed, corrupt_per_seed, s=100, alpha=0.7,
           color='#2c3e50', edgecolor='k', zorder=3)
for i, lab in enumerate(seed_labels):
    ax.annotate(lab, (force_per_seed[i], corrupt_per_seed[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
# Identity line
max_v = max(max(force_per_seed), max(corrupt_per_seed)) * 1.1 + 0.05
ax.plot([0, max_v], [0, max_v], 'k--', alpha=0.4, label='y = x (equivalence)')
ax.set_xlabel("FORCE speak rate"); ax.set_ylabel("CORRUPT speak rate")
ax.set_title("B2: FORCE vs CORRUPT Equivalence (per seed)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_xlim(0, max_v); ax.set_ylim(0, max_v)

# [1,2]: B2 — Retx breakdown: does agent retransmit INTENDED or ACTUAL?
ax = axes[1, 2]
# In FORCE condition, intended != actual, so retx_intended vs retx_actual
# diverges. FORCE_NOECHO tests whether echo drives the retx content.
force_retx_int = np.mean([b.get('B2_force_retx_intended', 0) for b in b2_results])
force_retx_act = np.mean([b.get('B2_force_retx_actual', 0) for b in b2_results])
corrupt_retx_int = np.mean([b.get('B2_corrupt_retx_intended', 0) for b in b2_results])
corrupt_retx_act = np.mean([b.get('B2_corrupt_retx_actual', 0) for b in b2_results])
fne_retx_int = np.mean([b.get('B2_force_noecho_retx_intended', 0) for b in b2_results])
fne_retx_act = np.mean([b.get('B2_force_noecho_retx_actual', 0) for b in b2_results])
x_pos = np.arange(3)
w = 0.35
ax.bar(x_pos - w/2, [force_retx_int, corrupt_retx_int, fne_retx_int], w,
       label='retx == INTENDED', color='#3498db', alpha=0.8)
ax.bar(x_pos + w/2, [force_retx_act, corrupt_retx_act, fne_retx_act], w,
       label='retx == ACTUAL', color='#e67e22', alpha=0.8)
ax.set_xticks(x_pos); ax.set_xticklabels(['FORCE', 'CORRUPT', 'FORCE\n_NOECHO'], fontsize=8)
ax.set_ylabel("Retx Rate at t_c+1")
ax.set_title("B2: Does agent retransmit INTENDED or ACTUAL?")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plot_path = os.path.join(SAVE_DIR, f'{PROBE_PREFIX}_results.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved plot: {plot_path}")

# Summary statistics
print(f"\n{'='*80}")
print(f"  SUMMARY")
print(f"{'='*80}")

def _mean_safe(lst):
    vals = [v for v in lst if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0

print(f"\nB1: Second-Order Decodability")
print(f"  ProbeB corruption detection (lag-1, t=2): {_mean_safe([b['B1'].get('t2_probeB_corrupt_acc', 0.5) for b in b1_results]):.3f}")
print(f"  Orthogonality (avg): {_mean_safe([np.mean([b['B1'].get(f't{t}_orthogonality', 0) for t in range(2, min(T_EP, 5))]) for b in b1_results]):.3f}")
print(f"  Mediation retention corrected (avg): {_mean_safe([np.mean([b['B1'].get(f't{t}_mediation_retention', 0) for t in range(2, min(T_EP, 5))]) for b in b1_results]):.3f}")
print(f"  Mediation retention raw (avg): {_mean_safe([np.mean([b['B1'].get(f't{t}_mediation_retention_raw', 0) for t in range(2, min(T_EP, 5))]) for b in b1_results]):.3f}")
print(f"  Chance baseline (avg): {_mean_safe([np.mean([b['B1'].get(f't{t}_chance_acc', 0.75) for t in range(2, min(T_EP, 5))]) for b in b1_results]):.3f}")

print(f"\nB1 Generalization (corruption-type transfer, all timesteps):")
print(f"  {'type':>16s}  " + "  ".join(f"t={t}" for t in range(2, T_EP)))
for ctype in ['uniform', 'fixed0', 'adjacent', 'silence', 'policy_matched', 'shuffle_baseline']:
    row = []
    for t in range(2, T_EP):
        vals = [b.get('B1_generalization', {}).get(f't{t}_{ctype}', None) for b in b1_results]
        vals = [v for v in vals if v is not None]
        row.append(f"{np.mean(vals):.3f}" if vals else "  N/A")
    print(f"  {ctype:>16s}  " + "  ".join(row))
# No-echo control
print(f"\n  No-echo generalization control:")
print(f"  {'type':>16s}  " + "  ".join(f"t={t}" for t in range(2, T_EP)))
for ctype in ['noecho_uniform'] + [f'noecho_{c}' for c in ['fixed0', 'adjacent', 'silence', 'policy_matched']]:
    row = []
    for t in range(2, T_EP):
        vals = [b.get('B1_generalization', {}).get(f't{t}_{ctype}', None) for b in b1_results]
        vals = [v for v in vals if v is not None]
        row.append(f"{np.mean(vals):.3f}" if vals else "  N/A")
    print(f"  {ctype:>16s}  " + "  ".join(row))

# Cross-transfer with self-test baseline comparison
print(f"\nB1 Cross-transfer: corruption probe (vs self-test baseline):")
print(f"  {'t':>4s}  {'e→e':>6s}  {'e→n':>6s}  {'gap':>6s}  |  {'n→n':>6s}  {'n→e':>6s}  {'gap':>6s}")
for t in range(2, T_EP):
    e2e = np.mean([b['B1'].get(f't{t}_probeB_corrupt_acc', 0) for b in b1_results])
    e2n = np.mean([b.get('B1_cross_transfer', {}).get(f't{t}_echo_to_noecho', 0) for b in b1_results])
    n2n = np.mean([b['B1'].get(f't{t}_probeB_corrupt_acc_noecho', 0) for b in b1_results])
    n2e = np.mean([b.get('B1_cross_transfer', {}).get(f't{t}_noecho_to_echo', 0) for b in b1_results])
    print(f"  t={t:>2d}  {e2e:>6.3f}  {e2n:>6.3f}  {e2e-e2n:>+6.3f}  |  {n2n:>6.3f}  {n2e:>6.3f}  {n2n-n2e:>+6.3f}")

# s_self cross-transfer control (with real self-test baselines)
print(f"\nB1 Cross-transfer: s_self probe (coordinate-system control):")
print(f"  {'t':>4s}  {'ss_e→e':>7s}  {'ss_e→n':>7s}  {'ss_gap':>7s}  |  {'cor_gap':>7s}  {'diff':>7s}")
for t in range(2, T_EP):
    ss_self = np.mean([b.get('B1_cross_transfer', {}).get(f't{t}_sself_self_echo', 0) for b in b1_results])
    ss_e2n = np.mean([b.get('B1_cross_transfer', {}).get(f't{t}_sself_echo_to_noecho', 0) for b in b1_results])
    sself_gap = ss_self - ss_e2n
    e2e = np.mean([b['B1'].get(f't{t}_probeB_corrupt_acc', 0) for b in b1_results])
    e2n = np.mean([b.get('B1_cross_transfer', {}).get(f't{t}_echo_to_noecho', 0) for b in b1_results])
    corrupt_gap = e2e - e2n
    print(f"  t={t:>2d}  {ss_self:>7.3f}  {ss_e2n:>7.3f}  {sself_gap:>+7.3f}  |  {corrupt_gap:>+7.3f}  {corrupt_gap - sself_gap:>+7.3f}")

# Paired specificity test: is corruption cross-transfer gap > s_self cross-transfer gap?
print(f"\nB1 Cross-transfer specificity test (corrupt_gap vs sself_gap, paired):")
print(f"  {'t':>4s}  {'cor_gap':>8s}  {'ss_gap':>8s}  {'diff':>8s}  {'t_p':>7s}  {'all>0':>5s}")
for t in range(2, T_EP):
    corrupt_gaps = np.array([
        b['B1'].get(f't{t}_probeB_corrupt_acc', np.nan) -
        b.get('B1_cross_transfer', {}).get(f't{t}_echo_to_noecho', np.nan)
        for b in b1_results
    ])
    sself_gaps = np.array([
        b.get('B1_cross_transfer', {}).get(f't{t}_sself_self_echo', np.nan) -
        b.get('B1_cross_transfer', {}).get(f't{t}_sself_echo_to_noecho', np.nan)
        for b in b1_results
    ])
    valid = ~(np.isnan(corrupt_gaps) | np.isnan(sself_gaps))
    if valid.sum() >= 2:
        cg = corrupt_gaps[valid]
        sg = sself_gaps[valid]
        diff = cg - sg
        t_stat, t_p = scipy_stats.ttest_rel(cg, sg)
        all_pos = all(d > 0 for d in diff)
        print(f"  t={t:>2d}  {cg.mean():>+8.3f}  {sg.mean():>+8.3f}  {diff.mean():>+8.3f}  {t_p:>7.4f}  {'yes' if all_pos else 'no':>5s}")
    else:
        print(f"  t={t:>2d}  insufficient data")

# Same specificity test, noecho→echo direction
print(f"\nB1 Cross-transfer specificity test (noecho→echo direction):")
print(f"  {'t':>4s}  {'cor_gap':>8s}  {'ss_gap':>8s}  {'diff':>8s}  {'t_p':>7s}  {'all>0':>5s}")
for t in range(2, T_EP):
    corrupt_gaps_n = np.array([
        b['B1'].get(f't{t}_probeB_corrupt_acc_noecho', np.nan) -
        b.get('B1_cross_transfer', {}).get(f't{t}_noecho_to_echo', np.nan)
        for b in b1_results
    ])
    sself_gaps_n = np.array([
        b.get('B1_cross_transfer', {}).get(f't{t}_sself_self_noecho', np.nan) -
        b.get('B1_cross_transfer', {}).get(f't{t}_sself_noecho_to_echo', np.nan)
        for b in b1_results
    ])
    valid_n = ~(np.isnan(corrupt_gaps_n) | np.isnan(sself_gaps_n))
    if valid_n.sum() >= 2:
        cg_n = corrupt_gaps_n[valid_n]
        sg_n = sself_gaps_n[valid_n]
        diff_n = cg_n - sg_n
        t_stat_n, t_p_n = scipy_stats.ttest_rel(cg_n, sg_n)
        all_pos_n = all(d > 0 for d in diff_n)
        print(f"  t={t:>2d}  {cg_n.mean():>+8.3f}  {sg_n.mean():>+8.3f}  {diff_n.mean():>+8.3f}  {t_p_n:>7.4f}  {'yes' if all_pos_n else 'no':>5s}")
    else:
        print(f"  t={t:>2d}  insufficient data")

print(f"\nB1 Cross-transfer significance test (self vs cross):")
print(f"  {'t':>4s}  {'e→e vs e→n diff':>16s}  {'Cohen_d':>8s}  {'t_p':>7s}  {'W_p':>7s}  {'all>0':>5s}")
for t in range(2, T_EP):
    e2e_vals = np.array([b['B1'].get(f't{t}_probeB_corrupt_acc', np.nan) for b in b1_results])
    e2n_vals = np.array([b.get('B1_cross_transfer', {}).get(f't{t}_echo_to_noecho', np.nan) for b in b1_results])
    mask = ~(np.isnan(e2e_vals) | np.isnan(e2n_vals))
    if mask.sum() >= 2:
        diff = e2e_vals[mask] - e2n_vals[mask]
        d_mean = diff.mean()
        d_std = diff.std(ddof=1) if diff.std(ddof=1) > 0 else 1e-8
        cohens_d = d_mean / d_std
        t_stat, t_p = scipy_stats.ttest_rel(e2e_vals[mask], e2n_vals[mask])
        try:
            w_stat, w_p = scipy_stats.wilcoxon(e2e_vals[mask], e2n_vals[mask])
        except ValueError:
            w_p = 1.0
        all_positive = all(d > 0 for d in diff)
        print(f"  t={t:>2d}  {d_mean:>+16.3f}  {cohens_d:>+8.2f}  {t_p:.4f}  {w_p:.4f}  {'yes' if all_positive else 'no':>5s}")

# Paired t-test + Wilcoxon + Cohen's d: echo vs no-echo corruption detection
ttest_results = {}
print(f"\nB1 Statistical tests (echo vs no-echo corruption detection, n={len(b1_results)} seeds):")
print(f"  Note: with n={len(b1_results)}, the 'all>0' direction-consistency column is the most reliable indicator;")
print(f"        t-test and Wilcoxon p-values have low power at this sample size.")
print(f"  {'t':>4s}  {'echo':>6s}  {'noecho':>6s}  {'diff':>6s}  {'Cohen_d':>8s}  {'t_p':>7s}  {'W_p':>7s}  {'all>0':>5s}  sig")
for t in range(2, T_EP):
    echo_vals = [b['B1'].get(f't{t}_probeB_corrupt_acc', None) for b in b1_results]
    noecho_vals = [b['B1'].get(f't{t}_probeB_corrupt_acc_noecho', None) for b in b1_results]
    pairs = [(e, n) for e, n in zip(echo_vals, noecho_vals) if e is not None and n is not None]
    if len(pairs) >= 2:
        e_arr = np.array([p[0] for p in pairs])
        n_arr = np.array([p[1] for p in pairs])
        diff = e_arr - n_arr
        t_stat, t_p = scipy_stats.ttest_rel(e_arr, n_arr)
        # Cohen's d for paired samples
        d_mean = diff.mean()
        d_std = diff.std(ddof=1) if diff.std(ddof=1) > 0 else 1e-8
        cohens_d = d_mean / d_std
        # Wilcoxon signed-rank (non-parametric)
        try:
            w_stat, w_p = scipy_stats.wilcoxon(e_arr, n_arr)
        except ValueError:
            w_p = 1.0  # all differences zero
        # Direction consistency: all seeds positive?
        all_positive = all(d > 0 for d in diff)
        sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"
        print(f"  t={t:>2d}  {np.mean(e_arr):.3f}  {np.mean(n_arr):.3f}  {d_mean:+.3f}  {cohens_d:>+8.2f}  {t_p:.4f}  {w_p:.4f}  {'yes' if all_positive else 'no':>5s}  {sig}")
        ttest_results[f't{t}'] = {
            'echo_mean': float(np.mean(e_arr)),
            'noecho_mean': float(np.mean(n_arr)),
            'diff': float(d_mean),
            'cohens_d': float(cohens_d),
            't_stat': float(t_stat), 't_p': float(t_p),
            'w_p': float(w_p),
            'n_pairs': len(pairs),
            'all_positive': bool(all_positive),
        }
    else:
        print(f"  t={t:>2d}  insufficient paired data")

all_results['B1_paired_ttest'] = ttest_results

print(f"\nB2: Counterfactual Intention")
print(f"  CLEAN speak rate: {np.mean([b.get('B2_clean_speak', 0) for b in b2_results]):.3f}")
print(f"  FORCE speak rate: {np.mean([b.get('B2_force_speak', 0) for b in b2_results]):.3f}")
print(f"  CORRUPT speak rate: {np.mean([b.get('B2_corrupt_speak', 0) for b in b2_results]):.3f}")
print(f"  FORCE_NOECHO speak rate: {np.mean([b.get('B2_force_noecho_speak', 0) for b in b2_results]):.3f}")
print(f"  FORCE effect: {np.mean([b.get('B2_force_vs_clean_speak', 0) for b in b2_results]):+.3f}")
print(f"  CORRUPT effect: {np.mean([b.get('B2_corrupt_vs_clean_speak', 0) for b in b2_results]):+.3f}")
print(f"  FORCE_NOECHO effect: {np.mean([b.get('B2_force_noecho_vs_clean_speak', 0) for b in b2_results]):+.3f}")
# Retx breakdown: did agent retransmit INTENDED or ACTUAL?
print(f"  FORCE retx==INTENDED: {np.mean([b.get('B2_force_retx_intended', 0) for b in b2_results]):.3f}")
print(f"  FORCE retx==ACTUAL:   {np.mean([b.get('B2_force_retx_actual', 0) for b in b2_results]):.3f}")
print(f"  CORRUPT retx==INTENDED: {np.mean([b.get('B2_corrupt_retx_intended', 0) for b in b2_results]):.3f}")
print(f"  CORRUPT retx==ACTUAL:   {np.mean([b.get('B2_corrupt_retx_actual', 0) for b in b2_results]):.3f}")
print(f"  FORCE_NOECHO retx==INTENDED: {np.mean([b.get('B2_force_noecho_retx_intended', 0) for b in b2_results]):.3f}")
print(f"  FORCE_NOECHO retx==ACTUAL:   {np.mean([b.get('B2_force_noecho_retx_actual', 0) for b in b2_results]):.3f}")

# Save JSON (after all results are computed)
json_path = os.path.join(SAVE_DIR, f'{PROBE_PREFIX}_results.json')
with open(json_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved JSON: {json_path}")
print(f"\n{'='*80}\n")
