"""
Probe: Communication Repair Diagnostics (Independent Parameters, No-Echo Ablation)
===================================================================================
Loads agents trained WITHOUT echo channel and runs the same diagnostic battery
as probe_independent.py. This is the train-time echo ablation control:
communication should be preserved while self-monitoring (P3) should be abolished.

Companion to `train_independent_noecho.py`.
"""

# %% ── 1. IMPORTS & CONFIG ──────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json, os, warnings
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

# ── Model prefix ──
MODEL_PREFIX = 'independent_noecho'
PROBE_PREFIX = 'independent_noecho_probe'

# On-manifold evaluation: keep echo at SILENCE throughout all rollouts,
# matching training conditions where agents never received echo feedback.
NO_ECHO_EVAL = True

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


def load_agent(tag, seed):
    model_name = f"{MODEL_PREFIX}_{tag}_seed{seed}_model.pth"
    model_path = os.path.join(SAVE_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"  WARNING: {model_path} not found")
        return None
    agent = GRUAgent().to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    return agent


def load_all_agents():
    agents_A, agents_B = [], []
    for i in range(N_SEEDS):
        a = load_agent('A', SEEDS_A[i])
        b = load_agent('B', SEEDS_B[i])
        if a is not None and b is not None:
            agents_A.append(a)
            agents_B.append(b)
    return agents_A, agents_B


# %% ── 4. MUTUAL INFORMATION ───────────────────────────────────────────
def mutual_information_vec(tok, var, n_tok, n_var):
    """Compute mutual information I(tok; var) in nats."""
    N = tok.shape[0]
    idx = tok * n_var + var
    joint = torch.bincount(idx, minlength=n_tok * n_var).float().reshape(n_tok, n_var)
    joint = joint / N
    px = joint.sum(dim=1, keepdim=True).clamp(min=1e-12)
    py = joint.sum(dim=0, keepdim=True).clamp(min=1e-12)
    mask = joint > 1e-10
    mi = (joint[mask] * (joint[mask].log() - (px * py)[mask].log())).sum()
    return mi.item()


@torch.no_grad()
def compute_mi_analysis(agent_left, agent_right, n=50000):
    """Compute mutual information between messages and states for both agents.
    ON-MANIFOLD: echo channels are kept at SILENCE throughout, matching
    training conditions for no-echo agents. Reports MI for both directions."""
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
    mi_results = {}
    for t in range(T_EP):
        xL = make_input(s_L, c_L, mR, echoL, t)
        xR = make_input(s_R, c_R, mL, echoR, t)
        mL_new, _, _, _, _, hL = agent_left.sample(xL, hL)
        mR_new, _, _, _, _, hR = agent_right.sample(xR, hR)
        # L's messages (A→B)
        mi_L_self = mutual_information_vec(mL_new, s_L, V, N_S)
        mi_L_other = mutual_information_vec(mL_new, s_R, V, N_S)
        mi_results[f't{t}_L_mi_self'] = mi_L_self
        mi_results[f't{t}_L_mi_other'] = mi_L_other
        # R's messages (B→A)
        mi_R_self = mutual_information_vec(mR_new, s_R, V, N_S)
        mi_R_other = mutual_information_vec(mR_new, s_L, V, N_S)
        mi_results[f't{t}_R_mi_self'] = mi_R_self
        mi_results[f't{t}_R_mi_other'] = mi_R_other
        # On-manifold: echo stays SILENCE (no-echo training condition)
        # echoL and echoR remain SILENCE — do NOT update them
        mL = mL_new; mR = mR_new
    ts = list(range(1, T_EP))
    L_self_avg = np.mean([mi_results[f't{t}_L_mi_self'] for t in ts])
    L_other_avg = np.mean([mi_results[f't{t}_L_mi_other'] for t in ts])
    R_self_avg = np.mean([mi_results[f't{t}_R_mi_self'] for t in ts])
    R_other_avg = np.mean([mi_results[f't{t}_R_mi_other'] for t in ts])
    mi_results['L_avg_mi_self'] = L_self_avg
    mi_results['L_avg_mi_other'] = L_other_avg
    mi_results['R_avg_mi_self'] = R_self_avg
    mi_results['R_avg_mi_other'] = R_other_avg
    mi_results['avg_mi_self'] = (L_self_avg + R_self_avg) / 2
    mi_results['avg_mi_other'] = (L_other_avg + R_other_avg) / 2
    mi_results['avg_mi_gap'] = mi_results['avg_mi_self'] - mi_results['avg_mi_other']
    if mi_results['avg_mi_other'] > 1e-10:
        mi_results['avg_mi_ratio'] = mi_results['avg_mi_self'] / mi_results['avg_mi_other']
    return mi_results


# %% ── 5. T1: CONDITIONAL TRIGGER RATE ─────────────────────────────────
@torch.no_grad()
def test_conditional_trigger(agent_left, agent_right, n=20000):
    """Corrupt LEFT's token at t_c, measure LEFT's speak rate at t_c+1."""
    train_c = list(range(N_C_TRAIN))
    results = {}

    T_CS = [1, 2, 3, 4, 5]
    for t_c in T_CS:
        assert t_c <= T_EP - 2, f"t_c={t_c} too close to T_EP={T_EP}"
    for t_c in T_CS:
        for corrupt in [False, True]:
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

            speak_after = None
            retx_after = None
            retx_cond_after = None
            mL_at_tc = None
            for t in range(min(T_EP, t_c + 3)):
                xL = make_input(s_L, c_L, mR, echoL, t)
                xR = make_input(s_R, c_R, mL, echoR, t)

                mL_intended, _, _, _, _, hL = agent_left.sample(xL, hL)
                mR_intended, _, _, _, _, hR = agent_right.sample(xR, hR)

                mL_transmitted = mL_intended.clone()
                mR_transmitted = mR_intended.clone()

                if t == t_c and corrupt:
                    mL_transmitted = torch.randint(0, V, (n,), device=device)

                if t == t_c:
                    mL_at_tc = mL_intended.clone()

                if not NO_ECHO_EVAL:
                    echoL = mL_transmitted
                    echoR = mR_transmitted
                mL = mL_transmitted
                mR = mR_transmitted

                if t == t_c + 1:
                    speak_after = (mL_intended != SILENCE).float().mean().item()
                    retx_after = (mL_intended == mL_at_tc).float().mean().item()
                    active_mask = (mL_at_tc != SILENCE)
                    if active_mask.any():
                        retx_cond_after = (mL_intended[active_mask] == mL_at_tc[active_mask]).float().mean().item()
                    else:
                        retx_cond_after = float('nan')

            tag = 'corrupt' if corrupt else 'clean'
            results[f'speak_t{t_c}+1_{tag}'] = speak_after
            results[f'retx_t{t_c}+1_{tag}'] = retx_after
            results[f'retx_cond_t{t_c}+1_{tag}'] = retx_cond_after

    for t_c in T_CS:
        c_key = f'speak_t{t_c}+1_corrupt'
        n_key = f'speak_t{t_c}+1_clean'
        if c_key in results and n_key in results:
            results[f'contrast_t{t_c}'] = results[c_key] - results[n_key]
        rc_key = f'retx_t{t_c}+1_corrupt'
        rn_key = f'retx_t{t_c}+1_clean'
        if rc_key in results and rn_key in results:
            results[f'retx_contrast_t{t_c}'] = results[rc_key] - results[rn_key]
        rcc_key = f'retx_cond_t{t_c}+1_corrupt'
        rcn_key = f'retx_cond_t{t_c}+1_clean'
        if rcc_key in results and rcn_key in results:
            import math
            vc = results[rcc_key]; vn = results[rcn_key]
            if not (math.isnan(vc) or math.isnan(vn)):
                results[f'retx_cond_contrast_t{t_c}'] = vc - vn
            else:
                results[f'retx_cond_contrast_t{t_c}'] = float('nan')

    return results


# %% ── 6. T2: SENDER VS RECEIVER ASYMMETRY (full rollout) ─────────────
@torch.no_grad()
def test_sender_receiver_asymmetry(agent_left, agent_right, n=20000, t_c=2):
    assert t_c <= T_EP - 2, f"t_c={t_c} too close to T_EP={T_EP}"
    train_c = list(range(N_C_TRAIN))
    results_by_mode = {}

    for corrupt in [False, True]:
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

        mL_at_tc = None
        for t in range(min(T_EP, t_c + 3)):
            xL = make_input(s_L, c_L, mR, echoL, t)
            xR = make_input(s_R, c_R, mL, echoR, t)

            mL_intended, _, _, _, _, hL = agent_left.sample(xL, hL)
            mR_intended, _, _, _, _, hR = agent_right.sample(xR, hR)

            mL_transmitted = mL_intended.clone()
            mR_transmitted = mR_intended.clone()

            if t == t_c and corrupt:
                mL_transmitted = torch.randint(0, V, (n,), device=device)

            if t == t_c:
                mL_at_tc = mL_intended.clone()

            if not NO_ECHO_EVAL:
                echoL = mL_transmitted
                echoR = mR_transmitted
            mL = mL_transmitted
            mR = mR_transmitted

            if t == t_c + 1:
                tag = 'corrupt' if corrupt else 'clean'
                results_by_mode[f'L_speak_{tag}'] = (mL_intended != SILENCE).float().mean().item()
                results_by_mode[f'R_speak_{tag}'] = (mR_intended != SILENCE).float().mean().item()
                results_by_mode[f'L_retx_{tag}'] = (mL_intended == mL_at_tc).float().mean().item()
                active_mask = (mL_at_tc != SILENCE)
                if active_mask.any():
                    results_by_mode[f'L_retx_cond_{tag}'] = (mL_intended[active_mask] == mL_at_tc[active_mask]).float().mean().item()
                    results_by_mode[f'L_retx_cond_n_{tag}'] = int(active_mask.sum().item())
                else:
                    results_by_mode[f'L_retx_cond_{tag}'] = float('nan')
                    results_by_mode[f'L_retx_cond_n_{tag}'] = 0

    results = {}
    results['L_contrast'] = results_by_mode.get('L_speak_corrupt', 0) - results_by_mode.get('L_speak_clean', 0)
    results['R_contrast'] = results_by_mode.get('R_speak_corrupt', 0) - results_by_mode.get('R_speak_clean', 0)
    results['asymmetry'] = results['L_contrast'] - results['R_contrast']
    results['L_retx_contrast'] = results_by_mode.get('L_retx_corrupt', 0) - results_by_mode.get('L_retx_clean', 0)
    results['L_retx_corrupt'] = results_by_mode.get('L_retx_corrupt', 0)
    results['L_retx_clean'] = results_by_mode.get('L_retx_clean', 0)
    import math
    _cond_c = results_by_mode.get('L_retx_cond_corrupt', float('nan'))
    _cond_n = results_by_mode.get('L_retx_cond_clean', float('nan'))
    if not (math.isnan(_cond_c) or math.isnan(_cond_n)):
        results['L_retx_cond_contrast'] = _cond_c - _cond_n
    else:
        results['L_retx_cond_contrast'] = float('nan')
    results['L_retx_cond_corrupt'] = _cond_c
    results['L_retx_cond_clean'] = _cond_n
    results['L_retx_cond_n_corrupt'] = results_by_mode.get('L_retx_cond_n_corrupt', 0)
    results['L_retx_cond_n_clean'] = results_by_mode.get('L_retx_cond_n_clean', 0)
    return results


# %% ── 6b. CHANNEL-SEPARATED CORRUPTION ───────────────────────────────
@torch.no_grad()
def test_echo_only_corrupt(agent_left, agent_right, n=20000, t_c=2):
    """Corrupt only L's OWN echo (echoL), leave mL_transmitted clean.
    If L re-speaks at t_c+1, the trigger is own-echo self-monitoring."""
    assert t_c <= T_EP - 2, f"t_c={t_c} too close to T_EP={T_EP}"
    train_c = list(range(N_C_TRAIN))
    results_by_mode = {}

    for corrupt in [False, True]:
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

        for t in range(min(T_EP, t_c + 3)):
            xL = make_input(s_L, c_L, mR, echoL, t)
            xR = make_input(s_R, c_R, mL, echoR, t)

            mL_intended, _, _, _, _, hL = agent_left.sample(xL, hL)
            mR_intended, _, _, _, _, hR = agent_right.sample(xR, hR)

            mL_transmitted = mL_intended.clone()
            mR_transmitted = mR_intended.clone()

            if not NO_ECHO_EVAL:
                if t == t_c and corrupt:
                    echoL_corrupt = torch.randint(0, V, (n,), device=device)
                else:
                    echoL_corrupt = mL_transmitted
                echoL = echoL_corrupt
                echoR = mR_transmitted
            mL = mL_transmitted
            mR = mR_transmitted

            if t == t_c + 1:
                tag = 'corrupt' if corrupt else 'clean'
                results_by_mode[f'L_speak_{tag}'] = (mL_intended != SILENCE).float().mean().item()
                results_by_mode[f'R_speak_{tag}'] = (mR_intended != SILENCE).float().mean().item()

    results = {}
    results['echo_only_L_contrast'] = results_by_mode.get('L_speak_corrupt', 0) - results_by_mode.get('L_speak_clean', 0)
    results['echo_only_R_contrast'] = results_by_mode.get('R_speak_corrupt', 0) - results_by_mode.get('R_speak_clean', 0)
    return results


@torch.no_grad()
def test_receiver_only_corrupt(agent_left, agent_right, n=20000, t_c=2):
    """Corrupt only mL_transmitted (to R), leave L's own echo clean.
    If L still re-speaks, the trigger can't be own-echo self-monitoring."""
    assert t_c <= T_EP - 2, f"t_c={t_c} too close to T_EP={T_EP}"
    train_c = list(range(N_C_TRAIN))
    results_by_mode = {}

    for corrupt in [False, True]:
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

        for t in range(min(T_EP, t_c + 3)):
            xL = make_input(s_L, c_L, mR, echoL, t)
            xR = make_input(s_R, c_R, mL, echoR, t)

            mL_intended, _, _, _, _, hL = agent_left.sample(xL, hL)
            mR_intended, _, _, _, _, hR = agent_right.sample(xR, hR)

            mL_transmitted = mL_intended.clone()
            mR_transmitted = mR_intended.clone()

            if t == t_c and corrupt:
                mL_to_R = torch.randint(0, V, (n,), device=device)
            else:
                mL_to_R = mL_transmitted

            if not NO_ECHO_EVAL:
                echoL = mL_transmitted
                echoR = mR_transmitted
            mL = mL_to_R
            mR = mR_transmitted

            if t == t_c + 1:
                tag = 'corrupt' if corrupt else 'clean'
                results_by_mode[f'L_speak_{tag}'] = (mL_intended != SILENCE).float().mean().item()
                results_by_mode[f'R_speak_{tag}'] = (mR_intended != SILENCE).float().mean().item()

    results = {}
    results['recv_only_L_contrast'] = results_by_mode.get('L_speak_corrupt', 0) - results_by_mode.get('L_speak_clean', 0)
    results['recv_only_R_contrast'] = results_by_mode.get('R_speak_corrupt', 0) - results_by_mode.get('R_speak_clean', 0)
    return results


# %% ── 7. DOWNSTREAM BENEFIT ──────────────────────────────────────────
@torch.no_grad()
def test_downstream_benefit(agent_left, agent_right, n=20000, t_c=2,
                            end_t=6):
    assert t_c <= T_EP - 2, f"t_c={t_c} too close to T_EP={T_EP}"
    assert end_t < T_EP, f"end_t={end_t} must be < T_EP={T_EP}"
    """Strict counterfactual — same episodes, same corruption, fork at t_c+1.
    Compare two conditions on IDENTICAL episodes:
      (A) Allow-repair: sender communicates freely after corruption
      (B) Gag: sender forced to SILENCE from t_c+1 onward
    Both branches share the same s_L, s_R, c_L, c_R, AND the same corruption
    draw at t_c. They diverge only at the sender's post-corruption output.
    """
    train_c = list(range(N_C_TRAIN))

    s_L = torch.randint(0, N_S, (n,), device=device)
    s_R = torch.randint(0, N_S, (n,), device=device)
    c_L = torch.tensor(np.random.choice(train_c, n), device=device)
    c_R = torch.tensor(np.random.choice(train_c, n), device=device)

    corrupt_token = torch.randint(0, V, (n,), device=device)

    def _roll(gag_sender):
        mL = torch.full((n,), SILENCE, device=device, dtype=torch.long)
        mR = torch.full((n,), SILENCE, device=device, dtype=torch.long)
        echoL = torch.full((n,), SILENCE, device=device, dtype=torch.long)
        echoR = torch.full((n,), SILENCE, device=device, dtype=torch.long)

        hL = torch.zeros(n, HIDDEN, device=device)
        hR = torch.zeros(n, HIDDEN, device=device)

        r_other_R = torch.zeros(n, device=device)
        count = 0

        for t in range(min(T_EP, end_t + 1)):
            xL = make_input(s_L, c_L, mR, echoL, t)
            xR = make_input(s_R, c_R, mL, echoR, t)

            mL_intended, (aL_o, aL_s), _, _, _, hL = agent_left.sample(xL, hL)
            mR_intended, (aR_o, aR_s), _, _, _, hR = agent_right.sample(xR, hR)

            mL_transmitted = mL_intended.clone()
            mR_transmitted = mR_intended.clone()

            if t == t_c:
                mL_transmitted = corrupt_token.clone()

            if gag_sender and t >= t_c + 1:
                mL_transmitted = torch.full_like(mL_transmitted, SILENCE)

            if not NO_ECHO_EVAL:
                echoL = mL_transmitted
                echoR = mR_transmitted
            mL = mL_transmitted
            mR = mR_transmitted

            if t_c + 1 <= t <= end_t:
                tR_o = get_target_other(s_L, c_R, t, dynamic=True)
                r_other_R += (aR_o == tR_o).float()
                count += 1

        h_R_last = hR.clone()
        r_other_R_mean = (r_other_R / max(count, 1)).mean().item()
        return r_other_R_mean, h_R_last

    rA, hRA = _roll(gag_sender=False)
    rB, hRB = _roll(gag_sender=True)

    def _decode(h, s):
        h_flat = h.float(); s_flat = s.long()
        return train_linear_probe(h_flat, s_flat, N_S, n_epochs=100)

    acc_A = _decode(hRA, s_L)
    acc_B = _decode(hRB, s_L)

    return {
        'r_other_R_with_repair': rA,
        'r_other_R_no_repair':   rB,
        'benefit_r_other':       rA - rB,
        'decode_sL_from_hR_with_repair': acc_A,
        'decode_sL_from_hR_no_repair':   acc_B,
        'benefit_decode':        acc_A - acc_B,
    }


# %% ── 8. T3: PROBE h → intended / actual / s_self / s_other ──────────
@torch.no_grad()
def collect_probe_data(agent_left, agent_right, n=20000, epsilon=0.3):
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
        mL_transmitted = torch.where(mask, torch.randint(0, V, (n,), device=device), mL_intended)

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

        if not NO_ECHO_EVAL:
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


def train_linear_probe(X, y, n_classes, n_epochs=200, lr=1e-2):
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
    return acc


def run_probes(data):
    results = {}
    for t in range(T_EP):
        h_t = data['h'][t]
        results[f't{t}_intended'] = train_linear_probe(h_t, data['intended'][t], V)
        results[f't{t}_actual'] = train_linear_probe(h_t, data['actual'][t], V)
        results[f't{t}_match'] = train_linear_probe(h_t, data['match'][t], 2)
        results[f't{t}_s_self'] = train_linear_probe(h_t, data['s_self'][t], N_S)
        results[f't{t}_s_other'] = train_linear_probe(h_t, data['s_other'][t], N_S)
    return results


# %% ── 8b. REPRESENTATION-LEVEL CORRUPTION DETECTION ─────────────────
def probe_corruption_detection_from_h(data):
    """Linear classifier from sender's h_t to predict whether corruption
    happened at t (same-step) and at t-1 (lagged).  Above-chance accuracy
    indicates latent detection in the representation, independent of
    behavioral re-speak."""
    results = {}
    for t in range(T_EP):
        h_t = data['h'][t]
        y_t = data['corrupt'][t]
        results[f'corr_detect_t{t}_sameStep'] = train_linear_probe(h_t, y_t, 2)
        if t >= 1:
            y_tm1 = data['corrupt'][t-1]
            results[f'corr_detect_t{t}_lag1'] = train_linear_probe(h_t, y_tm1, 2)
    return results


# %% ── 9. T5: NO-ECHO ABLATION ────────────────────────────────────────
@torch.no_grad()
def test_no_echo_ablation(agent_left, agent_right, n=20000, t_c=2):
    """Ablate L's own echo channel ONLY from t_c onwards.
    For t < t_c the rollout is fully on-manifold (echo flows normally);
    at t = t_c we start replacing echoL with SILENCE.  This isolates the
    effect of self-monitoring AT the critical step, rather than forcing the
    agent off-distribution for the whole episode (which would just measure
    policy collapse)."""
    assert t_c <= T_EP - 2, f"t_c={t_c} too close to T_EP={T_EP}"
    train_c = list(range(N_C_TRAIN))
    results = {}

    for corrupt in [False, True]:
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

        speak_after = None
        for t in range(min(T_EP, t_c + 3)):
            xL = make_input(s_L, c_L, mR, echoL, t)
            xR = make_input(s_R, c_R, mL, echoR, t)

            mL_intended, _, _, _, _, hL = agent_left.sample(xL, hL)
            mR_intended, _, _, _, _, hR = agent_right.sample(xR, hR)

            mL_transmitted = mL_intended.clone()
            if t == t_c and corrupt:
                mL_transmitted = torch.randint(0, V, (n,), device=device)

            if not NO_ECHO_EVAL:
                if t >= t_c:
                    echoL = torch.full_like(echoL, SILENCE)
                else:
                    echoL = mL_transmitted
                echoR = mR_intended.clone()
            mL = mL_transmitted
            mR = mR_intended.clone()

            if t == t_c + 1:
                speak_after = (mL_intended != SILENCE).float().mean().item()

        tag = 'corrupt' if corrupt else 'clean'
        results[f'no_echo_speak_{tag}'] = speak_after

    results['no_echo_contrast'] = (results.get('no_echo_speak_corrupt', 0) -
                                    results.get('no_echo_speak_clean', 0))
    return results


# %% ── 10. RUN ALL TESTS ──────────────────────────────────────────────
def run_tests_for_pair(agent_left, agent_right, label, seed_idx,
                        run_full=True):
    """Full battery for A-vs-B; reduced for controls when run_full=False."""
    print(f"\n  --- {label} seed {seed_idx} ---")
    res = {}

    print(f"  T1: Conditional Trigger Rate...")
    t1 = test_conditional_trigger(agent_left, agent_right)
    res.update(t1)
    for t_c in [1, 2, 3]:
        c = t1.get(f'contrast_t{t_c}', 0)
        print(f"    t_c={t_c}: corrupt={t1.get(f'speak_t{t_c}+1_corrupt', 0):.3f} "
              f"clean={t1.get(f'speak_t{t_c}+1_clean', 0):.3f} contrast={c:+.3f}")

    print(f"  T2: Sender vs Receiver Asymmetry (full rollout)...")
    t2 = test_sender_receiver_asymmetry(agent_left, agent_right)
    res.update(t2)
    print(f"    L(sender) contrast={t2['L_contrast']:+.3f}  "
          f"R(receiver) contrast={t2['R_contrast']:+.3f}  "
          f"asymmetry={t2['asymmetry']:+.3f}")

    print(f"  T2b: Echo-only corruption...")
    t2b = test_echo_only_corrupt(agent_left, agent_right)
    res.update(t2b)
    print(f"    echo_only L_contrast={t2b['echo_only_L_contrast']:+.3f}  "
          f"R_contrast={t2b['echo_only_R_contrast']:+.3f}")

    print(f"  T2c: Receiver-only corruption...")
    t2c = test_receiver_only_corrupt(agent_left, agent_right)
    res.update(t2c)
    print(f"    recv_only L_contrast={t2c['recv_only_L_contrast']:+.3f}  "
          f"R_contrast={t2c['recv_only_R_contrast']:+.3f}")

    if not run_full:
        return res

    print(f"  T_downstream: Downstream benefit of repair...")
    td = test_downstream_benefit(agent_left, agent_right)
    res.update(td)
    print(f"    r_other_R with/without repair: {td['r_other_R_with_repair']:.3f} / "
          f"{td['r_other_R_no_repair']:.3f}  Δ={td['benefit_r_other']:+.3f}")
    print(f"    decode s_L from h_R with/without: {td['decode_sL_from_hR_with_repair']:.3f} / "
          f"{td['decode_sL_from_hR_no_repair']:.3f}  Δ={td['benefit_decode']:+.3f}")

    print(f"  T3: Probes (h → intended/actual/match/s_self/s_other)...")
    data = collect_probe_data(agent_left, agent_right, n=15000, epsilon=0.3)
    t3 = run_probes(data)
    res.update(t3)
    print(f"    @ t=2: intended={t3.get('t2_intended', 0):.3f} "
          f"actual={t3.get('t2_actual', 0):.3f} match={t3.get('t2_match', 0):.3f}")
    print(f"    @ t=2: s_self={t3.get('t2_s_self', 0):.3f} "
          f"s_other={t3.get('t2_s_other', 0):.3f}")

    print(f"  T3b: Corruption detection from h...")
    t3b = probe_corruption_detection_from_h(data)
    res.update(t3b)
    same_t2 = t3b.get('corr_detect_t2_sameStep', 0)
    lag_t3  = t3b.get('corr_detect_t3_lag1', 0)
    print(f"    @ t=2 sameStep={same_t2:.3f}  @ t=3 lag1={lag_t3:.3f} (chance=0.5)")

    print(f"  T5: No-echo Ablation...")
    t5 = test_no_echo_ablation(agent_left, agent_right)
    res.update(t5)
    print(f"    No-echo contrast={t5['no_echo_contrast']:+.3f}")

    print(f"  MI: Mutual Information Analysis...")
    mi = compute_mi_analysis(agent_left, agent_right)
    res.update(mi)
    print(f"    avg_mi_self={mi.get('avg_mi_self', 0):.4f}  "
          f"avg_mi_other={mi.get('avg_mi_other', 0):.4f}  "
          f"avg_mi_ratio={mi.get('avg_mi_ratio', 0):.4f}")

    return res


# %% ── 11. MAIN ────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"  {PROBE_PREFIX.upper()} PROBE — Communication Repair Diagnostics")
print("="*70)

agents_A, agents_B = load_all_agents()
if not agents_A:
    print("No agents found. Exiting.")
    exit()

print(f"\nLoaded {len(agents_A)} agent pairs")

# ── Main test: A-vs-B (trained partners) — FULL BATTERY ──
print(f"\n{'='*70}")
print(f"  A-vs-B (TRAINED PARTNERS) — FULL BATTERY  (PRIMARY)")
print(f"{'='*70}")
ab_results = []
for i in range(len(agents_A)):
    res = run_tests_for_pair(agents_A[i], agents_B[i], "A-vs-B", i, run_full=True)
    ab_results.append(res)

# ── Control: A-vs-A — cross-seed self-pair ──
print(f"\n{'='*70}")
print(f"  A-vs-A (CONTROL, cross-seed) — T1/T2/T2b/T2c")
print(f"{'='*70}")
aa_results = []
for i in range(len(agents_A)):
    j = (i + 1) % len(agents_A)
    res = run_tests_for_pair(agents_A[i], agents_A[j], "A-vs-A", i, run_full=False)
    aa_results.append(res)

# ── Control: B-vs-B — cross-seed self-pair ──
print(f"\n{'='*70}")
print(f"  B-vs-B (CONTROL, cross-seed) — T1/T2/T2b/T2c")
print(f"{'='*70}")
bb_results = []
for i in range(len(agents_B)):
    j = (i + 1) % len(agents_B)
    res = run_tests_for_pair(agents_B[i], agents_B[j], "B-vs-B", i, run_full=False)
    bb_results.append(res)


# %% ── 12. PLOT ────────────────────────────────────────────────────────
def plot_all():
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(f"{PROBE_PREFIX} — Probe Results with Hole-Fixes",
                 fontsize=14, fontweight='bold')

    color_main = '#27ae60'
    color_ctrl_aa = '#2980b9'
    color_ctrl_bb = '#e74c3c'

    # (0,0): T1 trigger contrast by timestep
    ax = axes[0, 0]
    t_cs = [1, 2, 3, 4, 5]
    for label, results, color, ls in [
        ('A-vs-B', ab_results, color_main, '-'),
        ('A-vs-A', aa_results, color_ctrl_aa, '--'),
        ('B-vs-B', bb_results, color_ctrl_bb, '--'),
    ]:
        contrasts = [np.mean([r.get(f'contrast_t{t_c}', 0) for r in results]) for t_c in t_cs]
        ax.plot(t_cs, contrasts, f'o{ls}', color=color, label=label, linewidth=2)
    ax.axhline(0, c='gray', ls='--', alpha=0.5)
    ax.set_xlabel("Corruption timestep t_c"); ax.set_ylabel("Trigger Contrast")
    ax.set_title("T1: Conditional Trigger Rate"); ax.legend()

    # (0,1): T2 L/R contrast across pair types
    ax = axes[0, 1]
    x = np.arange(3)
    w = 0.35
    for cfg_i, (label, results, color) in enumerate([
        ('A-vs-B', ab_results, color_main),
        ('A-vs-A', aa_results, color_ctrl_aa),
        ('B-vs-B', bb_results, color_ctrl_bb),
    ]):
        l_c = [r.get('L_contrast', 0) for r in results]
        r_c = [r.get('R_contrast', 0) for r in results]
        ax.bar(cfg_i - w/2, np.mean(l_c), w, yerr=np.std(l_c), color=color, alpha=0.9, capsize=4)
        ax.bar(cfg_i + w/2, np.mean(r_c), w, yerr=np.std(r_c), color=color, alpha=0.4, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(['A-vs-B', 'A-vs-A', 'B-vs-B'])
    ax.set_ylabel("Speak Contrast (t_c+1)")
    ax.set_title("T2: L (dark) vs R (light) contrast")
    ax.axhline(0, c='gray', ls='--', alpha=0.5)

    # (0,2): T2b/T2c channel-separated — A-vs-B only
    ax = axes[0, 2]
    ab_echo = [r.get('echo_only_L_contrast', 0) for r in ab_results]
    ab_recv = [r.get('recv_only_L_contrast', 0) for r in ab_results]
    ab_full = [r.get('L_contrast', 0) for r in ab_results]
    xs = np.arange(3); w = 0.6
    ax.bar(0, np.mean(ab_full), w, yerr=np.std(ab_full), color='#8e44ad', alpha=0.8, capsize=4, label='full')
    ax.bar(1, np.mean(ab_echo), w, yerr=np.std(ab_echo), color='#27ae60', alpha=0.8, capsize=4, label='echo-only')
    ax.bar(2, np.mean(ab_recv), w, yerr=np.std(ab_recv), color='#e67e22', alpha=0.8, capsize=4, label='recv-only')
    ax.set_xticks(xs); ax.set_xticklabels(['full', 'echoL only', 'recvL only'])
    ax.set_ylabel("L_contrast (t_c+1)")
    ax.set_title("T2b/c: Channel-Separated (A-vs-B)")
    ax.axhline(0, c='gray', ls='--', alpha=0.5)

    # (1,0): Downstream benefit
    ax = axes[1, 0]
    r_with = [r.get('r_other_R_with_repair', 0) for r in ab_results]
    r_no   = [r.get('r_other_R_no_repair', 0)   for r in ab_results]
    w = 0.35
    ax.bar(0 - w/2, np.mean(r_with), w, yerr=np.std(r_with), color='#27ae60', alpha=0.8, capsize=4, label='with repair')
    ax.bar(0 + w/2, np.mean(r_no),   w, yerr=np.std(r_no),   color='#95a5a6', alpha=0.8, capsize=4, label='sender gagged')
    ax.set_xticks([0]); ax.set_xticklabels(['A-vs-B'])
    ax.set_ylabel("Receiver r_other (post-corruption window)")
    ax.set_title("Downstream Benefit (T3-func)")
    ax.legend(fontsize=8)

    # (1,1): Downstream decoding benefit
    ax = axes[1, 1]
    d_with = [r.get('decode_sL_from_hR_with_repair', 0) for r in ab_results]
    d_no   = [r.get('decode_sL_from_hR_no_repair', 0)   for r in ab_results]
    ax.bar(0 - w/2, np.mean(d_with), w, yerr=np.std(d_with), color='#27ae60', alpha=0.8, capsize=4, label='with repair')
    ax.bar(0 + w/2, np.mean(d_no),   w, yerr=np.std(d_no),   color='#95a5a6', alpha=0.8, capsize=4, label='sender gagged')
    ax.axhline(1/N_S, c='k', ls=':', alpha=0.5, label='chance')
    ax.set_xticks([0]); ax.set_xticklabels(['A-vs-B'])
    ax.set_ylabel("decode(s_L | h_R)")
    ax.set_title("Decode Sender State from Receiver h")
    ax.legend(fontsize=8)

    # (1,2): Corruption detection from h
    ax = axes[1, 2]
    ts = list(range(1, T_EP))
    same = [np.mean([r.get(f'corr_detect_t{t}_sameStep', 0.5) for r in ab_results]) for t in ts]
    lag  = [np.mean([r.get(f'corr_detect_t{t}_lag1', 0.5)     for r in ab_results]) for t in ts]
    ax.plot(ts, same, 'o-', color='#8e44ad', label='sameStep', linewidth=2)
    ax.plot(ts, lag,  's--', color='#e67e22', label='lag1', linewidth=2)
    ax.axhline(0.5, c='gray', ls=':', alpha=0.5, label='chance')
    ax.set_xlabel("Timestep"); ax.set_ylabel("Probe Accuracy")
    ax.set_title("Corruption Detection from h (T3b)")
    ax.legend(fontsize=8)

    # (2,0): T3 probe intended vs actual
    ax = axes[2, 0]
    intended = [np.mean([r.get(f't{t}_intended', 0) for r in ab_results]) for t in range(T_EP)]
    actual = [np.mean([r.get(f't{t}_actual', 0) for r in ab_results]) for t in range(T_EP)]
    ax.plot(range(T_EP), intended, 'o-', color=color_main, label='intended', linewidth=2)
    ax.plot(range(T_EP), actual, 's--', color=color_main, label='actual', alpha=0.5)
    ax.axhline(1/V, c='gray', ls=':', alpha=0.5, label='chance')
    ax.set_xlabel("Timestep"); ax.set_ylabel("Probe Accuracy")
    ax.set_title("T3: Probe h → Intended vs Actual Token"); ax.legend(fontsize=8)

    # (2,1): T3 probe s_self / s_other
    ax = axes[2, 1]
    s_self = [np.mean([r.get(f't{t}_s_self', 0) for r in ab_results]) for t in range(T_EP)]
    s_other = [np.mean([r.get(f't{t}_s_other', 0) for r in ab_results]) for t in range(T_EP)]
    ax.plot(range(T_EP), s_self, 'o-', color=color_main, label='s_self', linewidth=2)
    ax.plot(range(T_EP), s_other, 's--', color=color_main, label='s_other', alpha=0.5)
    ax.axhline(1/N_S, c='gray', ls=':', alpha=0.5, label='chance')
    ax.set_xlabel("Timestep"); ax.set_ylabel("Probe Accuracy")
    ax.set_title("T3: h → s_self / s_other"); ax.legend(fontsize=8)

    # (2,2): No-echo ablation vs full
    ax = axes[2, 2]
    full = [r.get('contrast_t2', 0) for r in ab_results]
    noecho = [r.get('no_echo_contrast', 0) for r in ab_results]
    w = 0.3
    ax.bar(0-w/2, np.mean(full),   w, yerr=np.std(full),   color=color_main, alpha=0.8, capsize=3, edgecolor='k', label='Full')
    ax.bar(0+w/2, np.mean(noecho), w, yerr=np.std(noecho), color=color_main, alpha=0.3, capsize=3, edgecolor='k', label='No-echo')
    ax.set_xticks([0]); ax.set_xticklabels(['A-vs-B'])
    ax.set_ylabel("Trigger Contrast (t_c=2)")
    ax.set_title("T5: No-echo Ablation")
    ax.legend(fontsize=8)
    ax.axhline(0, c='gray', ls='--', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, f'{PROBE_PREFIX}_probe_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved: {plot_path}")

plot_all()

# Save JSON
raw = {
    'A-vs-B': ab_results,
    'A-vs-A': aa_results,
    'B-vs-B': bb_results,
    '_meta': {
        'model_prefix': MODEL_PREFIX,
        'probe_prefix': PROBE_PREFIX,
        'n_seeds': len(ab_results),
        'tests': ['controls', 'channel_separated', 'downstream_benefit',
                  'representation_corruption_detection', 'mi_analysis'],
    }
}
json_path = os.path.join(SAVE_DIR, f'{PROBE_PREFIX}_probe_results.json')
with open(json_path, 'w') as f:
    json.dump(raw, f, indent=2, default=str)
print(f"\nSaved JSON: {json_path}")

# Summary
print(f"\n{'='*70}")
print(f"  SUMMARY — {PROBE_PREFIX}")
print(f"{'='*70}")

def _mean(results, key, default=0):
    vals = [r.get(key, default) for r in results]
    return float(np.nanmean(vals))

# A-vs-B
print(f"\n  A-vs-B (trained pair, PRIMARY):")
print(f"    T1 contrast_t2:          {_mean(ab_results,'contrast_t2'):+.3f}")
print(f"    T2 L_contrast:           {_mean(ab_results,'L_contrast'):+.3f}")
print(f"    T2 R_contrast:           {_mean(ab_results,'R_contrast'):+.3f}")
print(f"    T2 asymmetry:            {_mean(ab_results,'asymmetry'):+.3f}")
print(f"    T2b echo-only L_contrast:{_mean(ab_results,'echo_only_L_contrast'):+.3f}")
print(f"    T2c recv-only L_contrast:{_mean(ab_results,'recv_only_L_contrast'):+.3f}")
print(f"    T2 L_retx_contrast:      {_mean(ab_results,'L_retx_contrast'):+.3f}")
print(f"    T2 L_retx_corrupt:       {_mean(ab_results,'L_retx_corrupt'):.3f}")
print(f"    T2 L_retx_clean:         {_mean(ab_results,'L_retx_clean'):.3f}")
print(f"    T1 retx_cond at t_c=1:   Δ={_mean(ab_results,'retx_cond_contrast_t1'):+.3f}")
print(f"    T1 retx_cond at t_c=2:   Δ={_mean(ab_results,'retx_cond_contrast_t2'):+.3f}")
print(f"    T3 s_other (t=2):        {_mean(ab_results,'t2_s_other'):.3f}")
print(f"    T3b corr_detect t=2 same:{_mean(ab_results,'corr_detect_t2_sameStep'):.3f}")
print(f"    T3b corr_detect t=3 lag1:{_mean(ab_results,'corr_detect_t3_lag1'):.3f}")
print(f"    Downstream r_other Δ:    {_mean(ab_results,'benefit_r_other'):+.3f}")
print(f"    Downstream decode Δ:     {_mean(ab_results,'benefit_decode'):+.3f}")
print(f"    T5 no_echo_contrast:     {_mean(ab_results,'no_echo_contrast'):+.3f}")
print(f"    MI avg_mi_self:          {_mean(ab_results,'avg_mi_self'):.4f}")
print(f"    MI avg_mi_other:         {_mean(ab_results,'avg_mi_other'):.4f}")
print(f"    MI avg_mi_ratio:         {_mean(ab_results,'avg_mi_ratio'):.4f}")

# Controls)
print(f"\n  A-vs-A (control, cross-seed):")
print(f"    T1 contrast_t2:          {_mean(aa_results,'contrast_t2'):+.3f}")
print(f"    T2 L_contrast:           {_mean(aa_results,'L_contrast'):+.3f}")
print(f"    T2b echo-only L_contrast:{_mean(aa_results,'echo_only_L_contrast'):+.3f}")

print(f"\n  B-vs-B (control, cross-seed):")
print(f"    T1 contrast_t2:          {_mean(bb_results,'contrast_t2'):+.3f}")
print(f"    T2 L_contrast:           {_mean(bb_results,'L_contrast'):+.3f}")
print(f"    T2b echo-only L_contrast:{_mean(bb_results,'echo_only_L_contrast'):+.3f}")

# Verdict logic
print(f"\n  ── Verdict ──")
ab_L = _mean(ab_results, 'L_contrast')
ab_echo = _mean(ab_results, 'echo_only_L_contrast')
ab_recv = _mean(ab_results, 'recv_only_L_contrast')
ab_ne = _mean(ab_results, 'no_echo_contrast')
ab_det_lag1 = _mean(ab_results, 'corr_detect_t3_lag1')
ab_det_same = _mean(ab_results, 'corr_detect_t2_sameStep')
ab_ben = _mean(ab_results, 'benefit_r_other')
aa_L = _mean(aa_results, 'L_contrast')
bb_L = _mean(bb_results, 'L_contrast')

echo_driven = ab_echo > 0.03 and ab_recv < 0.02
controls_note = abs(aa_L) < 0.03 and abs(bb_L) < 0.03
ablation_kills = ab_ne < ab_L * 0.5
repr_detects = ab_det_lag1 > 0.80
functional = ab_ben > 0.02
ab_retx = _mean(ab_results, 'L_retx_contrast')
ab_retx_c = _mean(ab_results, 'L_retx_corrupt')
ab_retx_n = _mean(ab_results, 'L_retx_clean')
ab_retx_cond_t2 = _mean(ab_results, 'L_retx_cond_contrast')
import math
print(f"    echo-driven trigger (T2b>>T2c):     {echo_driven}")
print(f"    controls (AA,BB) L_contrast:        AA={aa_L:+.3f} BB={bb_L:+.3f} (expected ≈ AB for sender-internal P3)")
print(f"    no-echo ablation kills signal:      {ablation_kills}")
print(f"    repr detects (lag1>{0.80:.2f}):       {repr_detects}  (lag1={ab_det_lag1:.3f}, sameStep={ab_det_same:.3f}≈baseline)")
print(f"    functional downstream benefit:      {functional}  (Δ r_other={ab_ben:+.3f})")
print(f"    retx raw T2 m(t+1)==m(t):           {ab_retx:+.3f}  (corrupt={ab_retx_c:.3f}, clean={ab_retx_n:.3f})")
print(f"    retx cond T1 (SILENCE excl):")
retx_cond_any_valid = False
for _tc in [1, 2, 3]:
    _rc = _mean(ab_results, f'retx_cond_t{_tc}+1_corrupt')
    _rn = _mean(ab_results, f'retx_cond_t{_tc}+1_clean')
    _sp = _mean(ab_results, f'speak_t{_tc}+1_clean')
    if not (math.isnan(_rc) or math.isnan(_rn)):
        _delta = _rc - _rn
        print(f"      t_c={_tc}: cond_retx Δ={_delta:+.3f}  (corrupt={_rc:.3f} clean={_rn:.3f})  speak_clean={_sp:.3f}")
        retx_cond_any_valid = True
    else:
        print(f"      t_c={_tc}: NaN (agents silent at t_c, speak_clean={_sp:.3f})")
print(f"    retx cond per-seed at t_c=1:")
for i, r in enumerate(ab_results):
    _rc1 = r.get('retx_cond_t1+1_corrupt', float('nan'))
    _rn1 = r.get('retx_cond_t1+1_clean', float('nan'))
    _sp1 = r.get('speak_t1+1_clean', 0)
    if not (math.isnan(_rc1) or math.isnan(_rn1)):
        print(f"      seed {i}: cond_retx Δ={_rc1-_rn1:+.3f}  (corrupt={_rc1:.3f} clean={_rn1:.3f})  speak_clean={_sp1:.3f}")
    else:
        print(f"      seed {i}: NaN (silent at t=1, speak_clean={_sp1:.3f})")
n_retx_pos = 0; n_retx_neg = 0; n_retx_zero = 0; n_retx_nan = 0
for r in ab_results:
    _rc = r.get('retx_cond_t1+1_corrupt', float('nan'))
    _rn = r.get('retx_cond_t1+1_clean', float('nan'))
    if math.isnan(_rc) or math.isnan(_rn):
        n_retx_nan += 1
    else:
        d = _rc - _rn
        if d > 0.02: n_retx_pos += 1
        elif d < -0.02: n_retx_neg += 1
        else: n_retx_zero += 1
_rc1_all = _mean(ab_results, 'retx_cond_t1+1_corrupt')
_rn1_all = _mean(ab_results, 'retx_cond_t1+1_clean')
if not (math.isnan(_rc1_all) or math.isnan(_rn1_all)):
    ab_retx_cond = _rc1_all - _rn1_all
else:
    ab_retx_cond = float('nan')
print(f"      per-seed: {n_retx_pos} retransmit, {n_retx_neg} re-signal, {n_retx_zero} neutral, {n_retx_nan} NaN")
if n_retx_nan == len(ab_results):
    retx_verdict = "N/A (no seeds have active tokens at t_c=1)"
elif n_retx_pos > 0 and n_retx_neg == 0 and n_retx_zero <= 1:
    retx_verdict = "consistent retransmission across active seeds"
elif n_retx_neg > 0 and n_retx_pos == 0:
    retx_verdict = "consistent adaptive re-signaling"
elif n_retx_pos > 0 and (n_retx_neg > 0 or n_retx_zero > 1):
    retx_verdict = f"heterogeneous: {n_retx_pos} retransmit, {n_retx_zero} neutral, {n_retx_neg} re-signal (output form varies by seed)"
else:
    retx_verdict = f"no retransmission (0/{len(ab_results)} seeds), response is token-switching or silence"
print(f"      → {retx_verdict}")
n_ok = sum([echo_driven, ablation_kills, repr_detects, functional])
print(f"    Score: {n_ok}/4 core criteria satisfied")
if n_ok >= 3:
    print(f"    → ECHO-BASED SELF-MONITORING WITH FUNCTIONAL BENEFIT (independent params)")
elif n_ok >= 2:
    print(f"    → Partial signal; interpret cautiously")
else:
    print(f"    → No coherent repair signature")

print(f"\n{'='*70}")
