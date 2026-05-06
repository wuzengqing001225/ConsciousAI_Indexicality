"""
Independent-Parameter Training: Self-Referential Communication in Multi-Agent RL
=================================================================================
Two agents (A, B) with fully independent parameters learn to communicate via
discrete tokens under task pressure. The environment requires each agent to
take actions depending on the other's private state, making communication necessary.

Architecture: GRU (d=128) with three output heads (message, act-other, act-self).
Echo channel: agent observes a (possibly corrupted) copy of its own previous message.
POMDP: private state s_i visible only at t=0, masked for t >= 1.

Four-phase curriculum (performance-gated, 60k total updates):
  Phase 1 (12-18k): Additive reward, dynamic targets, build coordination
  Phase 2 (4-8k):   AND reward, enforce coupling
  Phase 3 (12k):    AND + corruption ramp (ε: 0→0.25) + speak cost ramp
  Phase 4 (remainder): AND + ε held + speak cost ramp (sparsification)

NO_ECHO: When False, echo channel contains the transmitted message.
"""

# %% ── 1. IMPORTS & CONFIG ───────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math, warnings, time, json, os
warnings.filterwarnings('ignore')

# ── Drive & Path Setup ──
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

# ── Tee stdout to log ──
import sys
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, d):
        for s in self.streams:
            try: s.write(d); s.flush()
            except Exception: pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass
NO_ECHO = False    # echo channel contains transmitted message
MODEL_PREFIX = 'independent' if not NO_ECHO else 'independent_noecho'
_log_path = os.path.join(SAVE_DIR, f'{MODEL_PREFIX}_train_log.txt')
_log_fh = open(_log_path, 'a', buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_fh)
print(f"[log] tee -> {_log_path}")

# ── GPU Setup ──
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
GPU_MEM_GB = 0.0
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name()}")
    GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM: {GPU_MEM_GB:.1f} GB")

# ── Environment ──
N_S        = 3
N_C_TRAIN  = 6
N_C_TEST   = 3
N_C        = N_C_TRAIN + N_C_TEST
N_A        = 3
T_EP       = 10
VOCAB      = 6
SILENCE    = VOCAB
V          = VOCAB + 1

P_CHANGE   = 0.0

# ── Training ──
def _auto_batch():
    if GPU_MEM_GB >= 70:   return 131072   # A100 80GB
    if GPU_MEM_GB >= 35:   return  65536
    if GPU_MEM_GB >= 20:   return  32768
    return 16384
BATCH_SIZE = _auto_batch()
HIDDEN     = 128
LR         = 3e-4
N_UPDATES  = 60000

# ── Resume ──
RESUME_FROM = 0

# ── 4-phase curriculum with additive+dynamic from step 0 ──
# Phase 1: r=additive, dynamic target, ALL heads
PHASE1_MIN            = 12000
PHASE1_MAX            = 18000
PHASE1_R_THRESH       = 0.85
PHASE1_MSG_ENT_THRESH = 1.5
# Phase 2: r=AND, dynamic; all heads
PHASE2_DUR_MIN        = 4000
PHASE2_DUR_MAX        = 8000
PHASE2_R_THRESH       = 0.70
# Phase 3: AND + ε ramp 0→EPSILON_MAX + cost ramp 0→PHASE3_COST_TARGET (fixed dur)
PHASE3_DUR            = 12000
PHASE3_COST_TARGET    = 0.005
EPSILON_MAX           = 0.25
# Phase 4: AND + ε held + cost ramp to PHASE4_COST_TARGET (absorbs remainder)
PHASE4_COST_TARGET    = 0.025

REWARD_GATE_WINDOW    = 500
DIAG_ABORT_ON_FAIL    = False

USE_BF16   = True
GAMMA      = 0.99

# Per-head entropy coefficients with msg linear ramp
ENT_MSG_HI         = 0.03
ENT_MSG_LO         = 0.0
ENT_MSG_RAMP_STEPS = 5000
ENT_ACT            = 0.01
ENT_HOLD_FRAC      = 0.65
ENT_DECAY_FLOOR    = 0.1

SILENCE_PEN = 0.05

# ── Seeds: independent ranges ──
N_SEEDS = 10
SEEDS_A = list(range(42, 42 + N_SEEDS))
SEEDS_B = list(range(142, 142 + N_SEEDS))
TOK = [str(i) for i in range(VOCAB)] + ['SIL']

B = BATCH_SIZE

# A+ INPUT: s_oh + c_oh + msg_oh + echo_oh + t_oh
INPUT_DIM = N_S + N_C + V + V + T_EP

print(f"\nIndependent-Parameter Training:")
print(f"  NO_ECHO={NO_ECHO}  {'(no-echo ablation)' if NO_ECHO else '(normal echo)'}")
print(f"  Env: N_S={N_S} N_C={N_C} N_A={N_A} V={V} T_EP={T_EP}")
print(f"  INPUT_DIM={INPUT_DIM}")
print(f"  4-phase curriculum (per-seed gated, additive+dynamic from step 0):")
print(f"    P1: r=(ro+rs)/2  dyn=T  heads=all  "
      f"[{PHASE1_MIN}..{PHASE1_MAX}] gate r_o∧r_s≥{PHASE1_R_THRESH} msg_ent≤{PHASE1_MSG_ENT_THRESH}")
print(f"    P2: r=AND        dyn=T  heads=all  "
      f"[+{PHASE2_DUR_MIN}..+{PHASE2_DUR_MAX}] gate r_AND≥{PHASE2_R_THRESH}")
print(f"    P3: AND ε 0→{EPSILON_MAX} cost 0→{PHASE3_COST_TARGET}  [+{PHASE3_DUR} fixed]")
print(f"    P4: AND ε={EPSILON_MAX} cost {PHASE3_COST_TARGET}→{PHASE4_COST_TARGET}  [remainder]")
print(f"  BATCH_SIZE={BATCH_SIZE}  HIDDEN={HIDDEN}  N_UPDATES={N_UPDATES}")
print(f"  Per-head ENT: msg {ENT_MSG_HI}→{ENT_MSG_LO} over {ENT_MSG_RAMP_STEPS} steps; "
      f"act={ENT_ACT} (const until {ENT_HOLD_FRAC*100:.0f}% anneal)")
print(f"  Agent A seeds: {SEEDS_A}")
print(f"  Agent B seeds: {SEEDS_B}")
print(f"  Diagnostic abort on fail: {DIAG_ABORT_ON_FAIL}")
print(f"  Random baseline: {1/N_A:.3f}")


# %% ── 2. GRU AGENT (individual, not parallel) ──────────────────────────
class GRUAgent(nn.Module):
    def __init__(self, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.gru = nn.GRUCell(INPUT_DIM, HIDDEN)
        self.msg_h = nn.Linear(HIDDEN, V)
        self.act_h = nn.Linear(HIDDEN, N_A)
        self.act_h2 = nn.Linear(HIDDEN, N_A)
        self.val_h = nn.Linear(HIDDEN, 1)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, HIDDEN, device=next(self.parameters()).device)

    def forward_step(self, x, h):
        h_new = self.gru(x, h)
        ml = self.msg_h(h_new)
        al_o = self.act_h(h_new)
        al_s = self.act_h2(h_new)
        vl = self.val_h(h_new).squeeze(-1)
        return (ml, al_o, al_s, vl), h_new

    def sample_step(self, x, h):
        """Returns per-head lp/ent so callers can select heads by phase.

        Returns:
          msg, (act_o, act_s),
          (lp_msg, lp_o, lp_s), vl, (ent_msg, ent_o, ent_s),
          h_new
        """
        (ml, al_o, al_s, vl), h_new = self.forward_step(x, h)
        md = torch.distributions.Categorical(logits=ml)
        ad_o = torch.distributions.Categorical(logits=al_o)
        ad_s = torch.distributions.Categorical(logits=al_s)
        msg = md.sample(); act_o = ad_o.sample(); act_s = ad_s.sample()
        lp_msg = md.log_prob(msg)
        lp_o = ad_o.log_prob(act_o)
        lp_s = ad_s.log_prob(act_s)
        ent_msg = md.entropy()
        ent_o = ad_o.entropy()
        ent_s = ad_s.entropy()
        return msg, (act_o, act_s), (lp_msg, lp_o, lp_s), vl, (ent_msg, ent_o, ent_s), h_new


# %% ── 3. HELPERS ─────────────────────────────────────────────────────
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

def perturb_token(token, epsilon):
    if epsilon <= 0:
        return token, torch.zeros_like(token, dtype=torch.bool)
    mask = torch.rand_like(token.float()) < epsilon
    random_tok = torch.randint(0, V, token.shape, device=token.device)
    transmitted = torch.where(mask, random_tok, token)
    return transmitted, mask

def get_target_other(s_other, c_self, t, dynamic=True):
    if dynamic: return (s_other + c_self + t) % N_A
    return (s_other + c_self) % N_A

def get_target_self(s_self, c_self, t, dynamic=True):
    if dynamic: return (s_self + c_self + t) % N_A
    return (s_self + c_self) % N_A


# %% ── 4. PHASE LOGIC (4-phase, additive+dynamic from step 0) ────────
class PhaseController:
    """Per-seed phase tracker for 4-phase curriculum.

    States:
      phase = 1..4
      phase_start[p] = update index where phase p began
    Gates:
      P1→P2: since>=PHASE1_MIN AND mean(r_other)>=THRESH AND mean(r_self)>=THRESH
             AND msg_entropy <= PHASE1_MSG_ENT_THRESH; forced @ PHASE1_MAX
      P2→P3: since>=PHASE2_DUR_MIN AND mean(r_AND) >= PHASE2_R_THRESH;
             forced @ PHASE2_DUR_MAX
      P3→P4: fixed PHASE3_DUR
      P4: absorbs remainder of budget
    """
    def __init__(self):
        self.phase = 1
        self.phase_start = {1: 0}
        # Gate diagnostics
        self.p1_gate_passed = False
        self.p2_gate_passed = False
        self.p1_final_ro = None
        self.p1_final_rs = None
        self.p1_final_ent = None
        self.p2_final_r = None
        self.p4_collapse_warned = False

    def step(self, update, r_other_window, r_self_window,
             r_AND_window, msg_ent_window):
        """Advance phase if gate conditions met. Returns current phase."""
        if self.phase == 1:
            since = update - self.phase_start[1]
            ro_avg = float(np.mean(r_other_window)) if r_other_window else 0.0
            rs_avg = float(np.mean(r_self_window))  if r_self_window  else 0.0
            me_avg = float(np.mean(msg_ent_window)) if msg_ent_window else 2.0
            gate_ok = (ro_avg >= PHASE1_R_THRESH
                       and rs_avg >= PHASE1_R_THRESH
                       and me_avg <= PHASE1_MSG_ENT_THRESH)
            if since >= PHASE1_MIN and gate_ok:
                self.p1_gate_passed = True
                self.p1_final_ro = ro_avg
                self.p1_final_rs = rs_avg
                self.p1_final_ent = me_avg
                self.phase = 2
                self.phase_start[2] = update
                print(f"    [gate] P1→P2 PASSED @ {update}  "
                      f"r_o={ro_avg:.3f} r_s={rs_avg:.3f} "
                      f"msg_ent={me_avg:.2f} (all clear)")
            elif since >= PHASE1_MAX:
                self.p1_gate_passed = False
                self.p1_final_ro = ro_avg
                self.p1_final_rs = rs_avg
                self.p1_final_ent = me_avg
                self.phase = 2
                self.phase_start[2] = update
                print(f"    [gate] P1→P2 FORCED @ {update}  "
                      f"r_o={ro_avg:.3f} r_s={rs_avg:.3f} "
                      f"msg_ent={me_avg:.2f} (gate not met) ⚠")

        elif self.phase == 2:
            since = update - self.phase_start[2]
            rA_avg = float(np.mean(r_AND_window)) if r_AND_window else 0.0
            if since >= PHASE2_DUR_MIN and rA_avg >= PHASE2_R_THRESH:
                self.p2_gate_passed = True
                self.p2_final_r = rA_avg
                self.phase = 3
                self.phase_start[3] = update
                print(f"    [gate] P2→P3 PASSED @ {update}  "
                      f"r_AND={rA_avg:.3f} ≥ {PHASE2_R_THRESH}")
            elif since >= PHASE2_DUR_MAX:
                self.p2_gate_passed = False
                self.p2_final_r = rA_avg
                self.phase = 3
                self.phase_start[3] = update
                print(f"    [gate] P2→P3 FORCED @ {update}  "
                      f"r_AND={rA_avg:.3f} "
                      f"(failed to reach {PHASE2_R_THRESH}) ⚠")

        elif self.phase == 3:
            since = update - self.phase_start[3]
            if since >= PHASE3_DUR:
                self.phase = 4
                self.phase_start[4] = update
                print(f"    [phase] P3→P4 @ {update}  (final cost ramp)")

        return self.phase

    def curriculum_params(self, update):
        """Return (reward_mode, dynamic_target, epsilon, speak_cost).

        reward_mode ∈ {'additive', 'and'}. No 'other_only'.
        dynamic_target is ALWAYS True from step 0.
        """
        p = self.phase
        if p == 1:
            return 'additive', True, 0.0, 0.0
        if p == 2:
            return 'and', True, 0.0, 0.0
        if p == 3:
            since = update - self.phase_start[3]
            frac = min(since / PHASE3_DUR, 1.0)
            eps = EPSILON_MAX * frac
            cost = PHASE3_COST_TARGET * frac
            return 'and', True, eps, cost
        # Phase 4: absorbs remainder, ε held, cost ramp to max
        since = update - self.phase_start[4]
        remaining = max(N_UPDATES - self.phase_start[4], 1)
        frac = min(since / remaining, 1.0)
        cost = PHASE3_COST_TARGET + (PHASE4_COST_TARGET - PHASE3_COST_TARGET) * frac
        return 'and', True, EPSILON_MAX, cost


# %% ── 5. TRAINING (individual agents, per-seed) ────────────────────────
def train_seed(seed_idx, seed_A, seed_B):
    """Train one A-B pair. Returns (agent_A, agent_B, history)."""
    print(f"\n  ── Seed {seed_idx}: A={seed_A}, B={seed_B} ──")

    agent_A = GRUAgent(seed=seed_A).to(device)
    agent_B = GRUAgent(seed=seed_B).to(device)

    print(f"    [compile] disabled")

    opt_A = torch.optim.Adam(agent_A.parameters(), lr=LR)
    opt_B = torch.optim.Adam(agent_B.parameters(), lr=LR)

    train_pool = torch.tensor(list(range(N_C_TRAIN)), device=device)

    history = dict(
        reward=[], speak_A=[], speak_B=[], epsilon=[],
        r_other=[], r_self=[], silence_A=[], silence_B=[],
        msg_entropy=[], trigger_A=[], trigger_B=[],
        policy_loss=[], value_loss=[], entropy_bonus=[], speak_cost=[],
        phase=[],
    )

    pc = PhaseController()
    r_other_window, r_self_window, r_AND_window, msg_ent_window = [], [], [], []

    start_update = 0
    if RESUME_FROM > 0:
        path_A = os.path.join(SAVE_DIR, f"{MODEL_PREFIX}_A_seed{seed_A}_model.pth")
        path_B = os.path.join(SAVE_DIR, f"{MODEL_PREFIX}_B_seed{seed_B}_model.pth")
        if os.path.exists(path_A) and os.path.exists(path_B):
            agent_A.load_state_dict(torch.load(path_A, map_location=device))
            agent_B.load_state_dict(torch.load(path_B, map_location=device))
            opt_A = torch.optim.Adam(agent_A.parameters(), lr=LR)
            opt_B = torch.optim.Adam(agent_B.parameters(), lr=LR)
            start_update = RESUME_FROM
            print(f"    [resume] Loaded from step {RESUME_FROM}")

    use_amp = USE_BF16 and device.type == 'cuda'
    amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if use_amp \
              else torch.amp.autocast('cuda', enabled=False)

    t0 = time.time()
    routing_check_done = False

    for update in range(start_update, N_UPDATES):
        # Phase advancement BEFORE this step (uses last observed rewards)
        pc.step(update, r_other_window, r_self_window, r_AND_window, msg_ent_window)
        reward_mode, dynamic_target, epsilon, speak_cost = pc.curriculum_params(update)
        phase = pc.phase

        # Per-head entropy schedule.
        #   ent_msg_coef: linear ramp HI→LO over ENT_MSG_RAMP_STEPS (startup only).
        #   ent_act_coef: constant ENT_ACT.
        # Then after ENT_HOLD_FRAC*N_UPDATES, apply a linear anneal down to
        # ENT_DECAY_FLOOR × each head's coefficient (shared anneal tail).
        if update < ENT_MSG_RAMP_STEPS:
            ent_msg_base = ENT_MSG_HI + (ENT_MSG_LO - ENT_MSG_HI) * (update / ENT_MSG_RAMP_STEPS)
        else:
            ent_msg_base = ENT_MSG_LO
        ent_act_base = ENT_ACT
        frac = update / N_UPDATES
        if frac < ENT_HOLD_FRAC:
            anneal = 1.0
        else:
            anneal_progress = (frac - ENT_HOLD_FRAC) / max(1 - ENT_HOLD_FRAC, 1e-9)
            anneal = 1.0 + (ENT_DECAY_FLOOR - 1.0) * anneal_progress
        ent_msg_coef = ent_msg_base * anneal
        ent_act_coef = ent_act_base * anneal

        # Sample environment — shape [B]
        s_1 = torch.randint(0, N_S, (B,), device=device)
        s_2 = torch.randint(0, N_S, (B,), device=device)
        c_1 = train_pool[torch.randint(len(train_pool), (B,), device=device)]
        c_2 = train_pool[torch.randint(len(train_pool), (B,), device=device)]

        # Role swap: 50% of episodes
        swap_mask = torch.rand(B, device=device) < 0.5

        # Initialize
        m1 = torch.full((B,), SILENCE, device=device, dtype=torch.long)
        m2 = torch.full((B,), SILENCE, device=device, dtype=torch.long)
        echo1 = torch.full((B,), SILENCE, device=device, dtype=torch.long)
        echo2 = torch.full((B,), SILENCE, device=device, dtype=torch.long)

        hA = agent_A.init_hidden(B)
        hB = agent_B.init_hidden(B)

        # Fully independent per-agent streams
        lpA_buf, lpB_buf = [], []
        valA_buf, valB_buf = [], []
        entA_msg_buf, entA_act_buf = [], []
        entB_msg_buf, entB_act_buf = [], []
        rewA_buf, rewB_buf = [], []
        spoke_A = torch.zeros(B, device=device)
        spoke_B = torch.zeros(B, device=device)
        any_perturbed = torch.zeros(B, dtype=torch.bool, device=device)
        r_other_acc = torch.zeros(B, device=device)
        r_self_acc  = torch.zeros(B, device=device)
        r_AND_acc   = torch.zeros(B, device=device)
        msg_ent_acc = torch.zeros(1, device=device)
        costA_acc_sanity = torch.zeros(B, device=device)
        costB_acc_sanity = torch.zeros(B, device=device)

        with amp_ctx:
            for t in range(T_EP):
                x1 = make_input(s_1, c_1, m2, echo1, t)
                x2 = make_input(s_2, c_2, m1, echo2, t)

                xA = torch.where(swap_mask.unsqueeze(-1), x2, x1)
                xB = torch.where(swap_mask.unsqueeze(-1), x1, x2)

                mA_intended, (aA_o, aA_s), (lpA_msg, lpA_o, lpA_s), vA, (eA_msg, eA_o, eA_s), hA = agent_A.sample_step(xA, hA)
                mB_intended, (aB_o, aB_s), (lpB_msg, lpB_o, lpB_s), vB, (eB_msg, eB_o, eB_s), hB = agent_B.sample_step(xB, hB)

                # Unified heads from step 0 (no phase-conditional masking).
                # All three heads always in lp/ent. Per-head entropy
                # coefficients (see loss computation) give msg its own ramp.
                lpA = lpA_msg + lpA_o + lpA_s
                lpB = lpB_msg + lpB_o + lpB_s
                eA_msg_t = eA_msg
                eA_act_t = eA_o + eA_s
                eB_msg_t = eB_msg
                eB_act_t = eB_o + eB_s

                m1_intended = torch.where(swap_mask, mB_intended, mA_intended)
                m2_intended = torch.where(swap_mask, mA_intended, mB_intended)
                a1_o = torch.where(swap_mask, aB_o, aA_o)
                a1_s = torch.where(swap_mask, aB_s, aA_s)
                a2_o = torch.where(swap_mask, aA_o, aB_o)
                a2_s = torch.where(swap_mask, aA_s, aB_s)

                m1_transmitted, pmask1 = perturb_token(m1_intended, epsilon)
                m2_transmitted, pmask2 = perturb_token(m2_intended, epsilon)
                any_perturbed |= pmask1 | pmask2

                # Individual speak cost (position-level → agent-level via swap)
                cost1 = speak_cost * (m1_intended != SILENCE).float()
                cost2 = speak_cost * (m2_intended != SILENCE).float()
                costA = torch.where(swap_mask, cost2, cost1)
                costB = torch.where(swap_mask, cost1, cost2)
                costA_acc_sanity += costA
                costB_acc_sanity += costB

                tgt_1_o = get_target_other(s_2, c_1, t, dynamic=dynamic_target)
                tgt_1_s = get_target_self(s_1, c_1, t, dynamic=dynamic_target)
                tgt_2_o = get_target_other(s_1, c_2, t, dynamic=dynamic_target)
                tgt_2_s = get_target_self(s_2, c_2, t, dynamic=dynamic_target)
                r1_o = (a1_o == tgt_1_o).float()
                r1_s = (a1_s == tgt_1_s).float()
                r2_o = (a2_o == tgt_2_o).float()
                r2_s = (a2_s == tgt_2_s).float()
                if reward_mode == 'additive':
                    # Phase 1 uses additive team reward (r_o+r_s)/2 per agent.
                    r1 = (r1_o + r1_s) / 2.0
                    r2 = (r2_o + r2_s) / 2.0
                else:  # 'and'
                    r1 = r1_o * r1_s; r2 = r2_o * r2_s
                r = (r1 + r2) / 2.0

                r_other_acc += (r1_o + r2_o) / 2
                r_self_acc  += (r1_s + r2_s) / 2
                r_AND_acc   += (r1_o * r1_s + r2_o * r2_s) / 2

                with torch.no_grad():
                    for _m in [mA_intended, mB_intended]:
                        counts = torch.zeros(V, device=device)
                        counts.scatter_add_(0, _m, torch.ones(B, device=device))
                        p = counts / counts.sum().clamp(min=1)
                        msg_ent_acc += -(p * (p + 1e-10).log()).sum() / 2

                lpA_buf.append(lpA)
                lpB_buf.append(lpB)
                valA_buf.append(vA)
                valB_buf.append(vB)
                entA_msg_buf.append(eA_msg_t)
                entA_act_buf.append(eA_act_t)
                entB_msg_buf.append(eB_msg_t)
                entB_act_buf.append(eB_act_t)
                rewA_buf.append(r - costA)
                rewB_buf.append(r - costB)

                spoke_A += (mA_intended != SILENCE).float()
                spoke_B += (mB_intended != SILENCE).float()

                # Echo channel: suppress if NO_ECHO ablation
                if NO_ECHO:
                    echo1 = torch.full_like(echo1, SILENCE)
                    echo2 = torch.full_like(echo2, SILENCE)
                else:
                    echo1 = m1_transmitted
                    echo2 = m2_transmitted
                m1 = m1_transmitted
                m2 = m2_transmitted

            # ── ROUTING SANITY CHECK (iter 0 only) ──
            if not routing_check_done:
                swap_rate = swap_mask.float().mean().item()
                cost_diff = (costA_acc_sanity - costB_acc_sanity).abs().mean().item()
                print(f"    [routing-check] swap_rate={swap_rate:.3f}  "
                      f"cost|A-B|_mean={cost_diff:.4f} (nonzero iff cost routed per-agent)")
                assert 0.4 < swap_rate < 0.6, f"swap_rate {swap_rate:.3f} outside [0.4,0.6]"
                routing_check_done = True

            # Individual silence penalty
            rewA_buf[-1] = rewA_buf[-1] - SILENCE_PEN * (spoke_A == 0).float()
            rewB_buf[-1] = rewB_buf[-1] - SILENCE_PEN * (spoke_B == 0).float()

            # Normalize ADVANTAGE not RETURNS (standard A2C/PPO practice).
            def _compute_returns(rew_list):
                G = torch.zeros(B, device=device)
                out = []
                for r_ in reversed(rew_list):
                    G = r_ + GAMMA * G
                    out.insert(0, G.clone())
                return torch.stack(out)

            returns_A = _compute_returns(rewA_buf)
            returns_B = _compute_returns(rewB_buf)

            lpA_t = torch.stack(lpA_buf)
            lpB_t = torch.stack(lpB_buf)
            valA_t = torch.stack(valA_buf)
            valB_t = torch.stack(valB_buf)
            entA_msg_t = torch.stack(entA_msg_buf)
            entA_act_t = torch.stack(entA_act_buf)
            entB_msg_t = torch.stack(entB_msg_buf)
            entB_act_t = torch.stack(entB_act_buf)

            adv_A = returns_A - valA_t.detach()
            adv_B = returns_B - valB_t.detach()
            # Batch-standardize advantages
            adv_A = (adv_A - adv_A.mean()) / adv_A.std().clamp(min=1e-8)
            adv_B = (adv_B - adv_B.mean()) / adv_B.std().clamp(min=1e-8)
            policy_loss_A = -(lpA_t * adv_A).mean(dim=(0, 1))
            policy_loss_B = -(lpB_t * adv_B).mean(dim=(0, 1))
            value_loss_A = 0.5 * ((valA_t - returns_A) ** 2).mean(dim=(0, 1))
            value_loss_B = 0.5 * ((valB_t - returns_B) ** 2).mean(dim=(0, 1))
            ent_msg_A = entA_msg_t.mean(dim=(0, 1))
            ent_act_A = entA_act_t.mean(dim=(0, 1))
            ent_msg_B = entB_msg_t.mean(dim=(0, 1))
            ent_act_B = entB_act_t.mean(dim=(0, 1))

            # Per-head entropy coefficients
            loss_A = (policy_loss_A + value_loss_A
                      - ent_msg_coef * ent_msg_A - ent_act_coef * ent_act_A)
            loss_B = (policy_loss_B + value_loss_B
                      - ent_msg_coef * ent_msg_B - ent_act_coef * ent_act_B)

            # For logging only
            policy_loss = policy_loss_A + policy_loss_B
            value_loss = value_loss_A + value_loss_B
            entropy_bonus = ent_msg_A + ent_act_A + ent_msg_B + ent_act_B

        # ── SEPARATE BACKWARD + SEPARATE GRAD CLIP (no cross-agent coupling) ──
        opt_A.zero_grad()
        opt_B.zero_grad()
        loss_A.backward(retain_graph=True)
        loss_B.backward()
        nn.utils.clip_grad_norm_(agent_A.parameters(), 2.0)
        nn.utils.clip_grad_norm_(agent_B.parameters(), 2.0)
        opt_A.step()
        opt_B.step()

        # ── Logging ──
        with torch.no_grad():
            rew_val = 0.5 * (torch.stack(rewA_buf).mean().item() +
                             torch.stack(rewB_buf).mean().item())
            spkA = (spoke_A > 0).float().mean().item()
            spkB = (spoke_B > 0).float().mean().item()
            ro = r_other_acc.mean().item() / T_EP
            rs = r_self_acc.mean().item() / T_EP
            r_AND = r_AND_acc.mean().item() / T_EP
            sil_A = (spoke_A == 0).float().mean().item()
            sil_B = (spoke_B == 0).float().mean().item()
            me = msg_ent_acc.item() / T_EP

            n_pert = any_perturbed.float().sum().clamp(min=1)
            n_clean = (~any_perturbed).float().sum().clamp(min=1)
            spkA_pert  = (spoke_A * any_perturbed.float()).sum() / n_pert
            spkA_clean = (spoke_A * (~any_perturbed).float()).sum() / n_clean
            trig_A = (spkA_pert - spkA_clean).item()
            spkB_pert  = (spoke_B * any_perturbed.float()).sum() / n_pert
            spkB_clean = (spoke_B * (~any_perturbed).float()).sum() / n_clean
            trig_B = (spkB_pert - spkB_clean).item()

        # Update rolling windows for phase gates
        r_other_window.append(ro)
        r_self_window.append(rs)
        r_AND_window.append(r_AND)
        msg_ent_window.append(me)
        if len(r_other_window) > REWARD_GATE_WINDOW:
            r_other_window.pop(0)
        if len(r_self_window) > REWARD_GATE_WINDOW:
            r_self_window.pop(0)
        if len(r_AND_window) > REWARD_GATE_WINDOW:
            r_AND_window.pop(0)
        if len(msg_ent_window) > REWARD_GATE_WINDOW:
            msg_ent_window.pop(0)

        history['reward'].append(rew_val)
        history['speak_A'].append(spkA)
        history['speak_B'].append(spkB)
        history['epsilon'].append(epsilon)
        history['r_other'].append(ro)
        history['r_self'].append(rs)
        history['silence_A'].append(sil_A)
        history['silence_B'].append(sil_B)
        history['msg_entropy'].append(me)
        history['trigger_A'].append(trig_A)
        history['trigger_B'].append(trig_B)
        history['policy_loss'].append(policy_loss.item())
        history['value_loss'].append(value_loss.item())
        history['entropy_bonus'].append(entropy_bonus.item())
        history['speak_cost'].append(speak_cost)
        history['phase'].append(phase)

        # ── DIAGNOSTIC: mid-Phase-3 noise-collapse warning (once) ──
        if phase == 3 and not pc.p4_collapse_warned:
            since_p3 = update - pc.phase_start.get(3, update)
            if since_p3 >= PHASE3_DUR * 0.5 and r_AND < 0.60:
                print(f"    [DIAG WARN @ {update}] Phase 3 noise ramp caused r_AND collapse: "
                      f"{r_AND:.3f} < 0.60 → consider slowing ε ramp in next run")
                pc.p4_collapse_warned = True

        trig_avg = (trig_A + trig_B) / 2

        near_boundary = any(
            abs(update - pc.phase_start.get(p, -1e9)) < 500
            for p in [2, 3, 4]
        )
        log_interval = 200 if near_boundary else 2000

        if update % log_interval == 0:
            elapsed = time.time() - t0
            vram = f" VRAM:{torch.cuda.memory_allocated()/1e9:.1f}G" if device.type == 'cuda' else ""
            print(f"    [seed{seed_idx}] {update:5d}/{N_UPDATES} "
                  f"r:{rew_val:.3f} ro:{ro:.3f} rs:{rs:.3f} rAND:{r_AND:.3f} "
                  f"spk_A:{spkA:.2f} spk_B:{spkB:.2f} "
                  f"trig:{trig_avg:.3f} ent:{me:.2f} "
                  f"em:{ent_msg_coef:.3f} ea:{ent_act_coef:.3f} "
                  f"ε:{epsilon:.3f} sc:{speak_cost:.4f} ph:{phase}"
                  f" | {max(1,update-start_update+1)/elapsed:.1f} upd/s{vram}")

    elapsed = time.time() - t0
    n_done = N_UPDATES - start_update
    print(f"    [seed{seed_idx}] done in {elapsed:.0f}s ({n_done/max(1,elapsed):.1f} upd/s)")
    print(f"    [seed{seed_idx}] diagnostic summary:")
    print(f"      P1 gate passed: {pc.p1_gate_passed}  "
          f"(final r_o={pc.p1_final_ro} r_s={pc.p1_final_rs} ent={pc.p1_final_ent})")
    print(f"      P2 gate passed: {pc.p2_gate_passed}  (final r_AND={pc.p2_final_r})")
    print(f"      P3 ε-ramp collapse warned: {pc.p4_collapse_warned}")

    diagnostic = {
        'p1_gate_passed': bool(pc.p1_gate_passed),
        'p1_final_ro': pc.p1_final_ro,
        'p1_final_rs': pc.p1_final_rs,
        'p1_final_ent': pc.p1_final_ent,
        'p2_gate_passed': bool(pc.p2_gate_passed),
        'p2_final_r': pc.p2_final_r,
        'p4_collapse_warned': bool(pc.p4_collapse_warned),
        'phase_start': dict(pc.phase_start),
    }

    return agent_A, agent_B, history, diagnostic


def train_all_seeds():
    """Train all seed pairs sequentially."""
    all_agents_A = []
    all_agents_B = []
    all_histories = []
    all_diagnostics = []

    for si in range(N_SEEDS):
        agent_A, agent_B, hist, diag = train_seed(si, SEEDS_A[si], SEEDS_B[si])
        all_agents_A.append(agent_A)
        all_agents_B.append(agent_B)
        all_histories.append(hist)
        all_diagnostics.append(diag)

        # Save checkpoint after each seed
        for tag, agent, seed in [('A', agent_A, SEEDS_A[si]), ('B', agent_B, SEEDS_B[si])]:
            path = os.path.join(SAVE_DIR, f"{MODEL_PREFIX}_{tag}_seed{seed}_model.pth")
            torch.save(agent.state_dict(), path)
        print(f"    [seed{si}] Saved models")

        if DIAG_ABORT_ON_FAIL and (not diag['p1_gate_passed']):
            print(f"\n  [ABORT] seed {si} failed P1 gate and DIAG_ABORT_ON_FAIL=True")
            print(f"  Stopping all seeds. Review log to diagnose base-dictionary formation.")
            break

    return all_agents_A, all_agents_B, all_histories, all_diagnostics


# %% ── 6. ANALYSIS ────────────────────────────────────────────────────
@torch.no_grad()
def analyze_pair(agent_left, agent_right, label="A-B"):
    """Run repair analysis on a specific agent pair."""
    train_c = list(range(N_C_TRAIN))
    results = {}

    for mode_name, use_comm in [('comm', True), ('nocomm', False)]:
        n_ep = 4000
        s_L = torch.randint(0, N_S, (n_ep,), device=device)
        s_R = torch.randint(0, N_S, (n_ep,), device=device)
        c_L = torch.tensor(np.random.choice(train_c, n_ep), device=device)
        c_R = torch.tensor(np.random.choice(train_c, n_ep), device=device)

        mL = torch.full((n_ep,), SILENCE, device=device, dtype=torch.long)
        mR = torch.full((n_ep,), SILENCE, device=device, dtype=torch.long)
        echoL = torch.full((n_ep,), SILENCE, device=device, dtype=torch.long)
        echoR = torch.full((n_ep,), SILENCE, device=device, dtype=torch.long)

        hL = torch.zeros(n_ep, HIDDEN, device=device)
        hR = torch.zeros(n_ep, HIDDEN, device=device)

        rews = []
        for t in range(T_EP):
            xL = make_input(s_L, c_L, mR, echoL, t)
            xR = make_input(s_R, c_R, mL, echoR, t)
            mL_n, (aL_o, aL_s), _, _, _, hL = agent_left.sample_step(xL, hL)
            mR_n, (aR_o, aR_s), _, _, _, hR = agent_right.sample_step(xR, hR)

            tL_o = get_target_other(s_R, c_L, t); tL_s = get_target_self(s_L, c_L, t)
            tR_o = get_target_other(s_L, c_R, t); tR_s = get_target_self(s_R, c_R, t)
            rL = ((aL_o == tL_o).float() + (aL_s == tL_s).float()) / 2
            rR = ((aR_o == tR_o).float() + (aR_s == tR_s).float()) / 2
            rews.append(((rL + rR) / 2).mean().item())

            if use_comm:
                echoL = mL_n; echoR = mR_n
                mL = mL_n; mR = mR_n
            else:
                echoL = torch.full_like(echoL, SILENCE)
                echoR = torch.full_like(echoR, SILENCE)
                mL = torch.full_like(mL, SILENCE)
                mR = torch.full_like(mR, SILENCE)

        results[f'r_{mode_name}'] = float(np.mean(rews))
    results['delta'] = results['r_comm'] - results['r_nocomm']

    # Repair trigger check
    n_ep = 10000
    for corrupt_mode in ['no_corrupt', 'corrupt_t2']:
        s_L = torch.randint(0, N_S, (n_ep,), device=device)
        s_R = torch.randint(0, N_S, (n_ep,), device=device)
        c_L = torch.tensor(np.random.choice(train_c, n_ep), device=device)
        c_R = torch.tensor(np.random.choice(train_c, n_ep), device=device)

        mL = torch.full((n_ep,), SILENCE, device=device, dtype=torch.long)
        mR = torch.full((n_ep,), SILENCE, device=device, dtype=torch.long)
        echoL = torch.full((n_ep,), SILENCE, device=device, dtype=torch.long)
        echoR = torch.full((n_ep,), SILENCE, device=device, dtype=torch.long)
        hL = torch.zeros(n_ep, HIDDEN, device=device)
        hR = torch.zeros(n_ep, HIDDEN, device=device)

        speak_at_3 = None
        for t in range(min(T_EP, 5)):
            xL = make_input(s_L, c_L, mR, echoL, t)
            xR = make_input(s_R, c_R, mL, echoR, t)
            mL_n, _, _, _, _, hL = agent_left.sample_step(xL, hL)
            mR_n, _, _, _, _, hR = agent_right.sample_step(xR, hR)

            if t == 2 and corrupt_mode == 'corrupt_t2':
                mL_transmitted = torch.randint(0, V, (n_ep,), device=device)
            else:
                mL_transmitted = mL_n

            echoL = mL_transmitted
            echoR = mR_n
            mL = mL_transmitted
            mR = mR_n

            if t == 3:
                speak_at_3 = (mL_n != SILENCE).float().mean().item()

        if speak_at_3 is not None:
            results[f'speak_t3_{corrupt_mode}'] = speak_at_3

    if 'speak_t3_corrupt_t2' in results and 'speak_t3_no_corrupt' in results:
        results['trigger_rate_corrupt'] = results['speak_t3_corrupt_t2']
        results['trigger_rate_clean'] = results['speak_t3_no_corrupt']
        results['trigger_contrast'] = results['trigger_rate_corrupt'] - results['trigger_rate_clean']

    return results


# %% ── 7. RUN ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  TRAINING PHASE")
print(f"{'='*70}")

agents_A, agents_B, histories, diagnostics = train_all_seeds()

print(f"\n{'='*70}")
print(f"  ANALYSIS PHASE  (PRIMARY: A↔B cross; CONTROL: self-pair cross-seed)")
print(f"{'='*70}")

configs = {}

# ── PRIMARY: A-vs-B (trained partners) ──
print("\n  [A-vs-B]  (PRIMARY — trained partners)")
ab_results = []
for i in range(len(agents_A)):
    res = analyze_pair(agents_A[i], agents_B[i], f"A{SEEDS_A[i]}-B{SEEDS_B[i]}")
    tc = res.get('trigger_contrast', 0)
    print(f"    Seed {i}: delta:{res['delta']:+.3f}  "
          f"trigger: corrupt={res.get('trigger_rate_corrupt',0):.3f} "
          f"clean={res.get('trigger_rate_clean',0):.3f} contrast={tc:+.3f}")
    ab_results.append(res)
configs['A-vs-B'] = ab_results

# ── PRIMARY: B-vs-A (same trained partners, reversed role — symmetry check) ──
print("\n  [B-vs-A]  (PRIMARY — reversed role)")
ba_results = []
for i in range(len(agents_A)):
    res = analyze_pair(agents_B[i], agents_A[i], f"B{SEEDS_B[i]}-A{SEEDS_A[i]}")
    tc = res.get('trigger_contrast', 0)
    print(f"    Seed {i}: delta:{res['delta']:+.3f}  contrast={tc:+.3f}")
    ba_results.append(res)
configs['B-vs-A'] = ba_results

# ── CONTROL: A-vs-A cross-seed ──
print("\n  [A-vs-A]  (CONTROL — cross-seed self-pair)")
aa_results = []
for i in range(len(agents_A)):
    j = (i + 1) % len(agents_A)
    res = analyze_pair(agents_A[i], agents_A[j], f"A{SEEDS_A[i]}-A{SEEDS_A[j]}")
    tc = res.get('trigger_contrast', 0)
    print(f"    Seed {i}: delta:{res['delta']:+.3f}  contrast={tc:+.3f}")
    aa_results.append(res)
configs['A-vs-A'] = aa_results

# ── CONTROL: B-vs-B cross-seed ──
print("\n  [B-vs-B]  (CONTROL — cross-seed self-pair)")
bb_results = []
for i in range(len(agents_B)):
    j = (i + 1) % len(agents_B)
    res = analyze_pair(agents_B[i], agents_B[j], f"B{SEEDS_B[i]}-B{SEEDS_B[j]}")
    tc = res.get('trigger_contrast', 0)
    print(f"    Seed {i}: delta:{res['delta']:+.3f}  contrast={tc:+.3f}")
    bb_results.append(res)
configs['B-vs-B'] = bb_results


# ── Summary ──
print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")

PRIMARY = ['A-vs-B', 'B-vs-A']
CONTROL = ['A-vs-A', 'B-vs-B']

for cfg_name, results in configs.items():
    tag = 'PRIMARY' if cfg_name in PRIMARY else 'CONTROL'
    deltas = [r['delta'] for r in results]
    contrasts = [r.get('trigger_contrast', 0) for r in results]
    d_m, d_s = np.mean(deltas), np.std(deltas)
    c_m, c_s = np.mean(contrasts), np.std(contrasts)
    print(f"\n  [{cfg_name}] ({tag})")
    print(f"    Delta:            {d_m:+.3f} ± {d_s:.3f}")
    print(f"    Trigger contrast: {c_m:+.3f} ± {c_s:.3f}")
    if c_m > 0.05:
        print(f"    → REPAIR SIGNATURE DETECTED")
    elif c_m > 0.02:
        print(f"    → Weak repair signal")
    else:
        print(f"    → No repair detected")

ab_c = [r.get('trigger_contrast', 0) for r in ab_results]
print(f"\n  A-vs-B trigger contrast: {np.mean(ab_c):+.3f} ± {np.std(ab_c):.3f}")

print(f"\n  ── Per-seed diagnostic summary ──")
for i, d in enumerate(diagnostics):
    p1 = "✓" if d['p1_gate_passed'] else "✗"
    p2 = "✓" if d['p2_gate_passed'] else "✗"
    p4w = "⚠" if d.get('p4_collapse_warned') else "·"
    print(f"    seed{i}: P1{p1} (r_o={d.get('p1_final_ro')} r_s={d.get('p1_final_rs')} "
          f"ent={d.get('p1_final_ent')}) "
          f"P2{p2} (r_AND={d.get('p2_final_r')}) P3ε{p4w}  "
          f"starts={d['phase_start']}")

print(f"\n{'='*70}")


# %% ── 8. TRAINING DYNAMICS PLOT ──────────────────────────────────────
def plot_training_dynamics():
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle("Independent-Parameter Training — 4-Phase Curriculum Dynamics",
                 fontsize=14, fontweight='bold')

    def _smooth(arr, w=None):
        if w is None:
            w = min(500, max(1, len(arr) // 20))
        if w <= 1 or len(arr) < w:
            return np.array(arr)
        return np.convolve(arr, np.ones(w)/w, mode='valid')

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    def _mark_phase_starts(ax, si):
        """Mark each seed's phase transitions (per-seed, since gates differ)."""
        starts = diagnostics[si]['phase_start']
        for p, s_ in starts.items():
            if p == 1: continue
            ax.axvline(s_, c=colors[si], ls=':', alpha=0.3, linewidth=0.5)

    # 1. Reward
    ax = axes[0, 0]
    for si in range(len(histories)):
        ax.plot(_smooth(histories[si]['reward']), alpha=0.6, color=colors[si],
                label=f'seed{si}')
        _mark_phase_starts(ax, si)
    ax.set_title("Reward"); ax.set_ylabel("reward"); ax.legend(fontsize=6)

    # 2. r_other vs r_self
    ax = axes[0, 1]
    for si in range(len(histories)):
        ax.plot(_smooth(histories[si]['r_other']), color=colors[si], linewidth=1.5,
                alpha=0.7)
        ax.plot(_smooth(histories[si]['r_self']), color=colors[si], linewidth=1,
                alpha=0.4, linestyle='--')
    ax.axhline(PHASE1_R_THRESH, c='red', ls=':', alpha=0.4, label=f'P1 gate {PHASE1_R_THRESH}')
    ax.set_title("r_other (solid) / r_self (dashed)")
    ax.legend(fontsize=7)

    # 3. Speak rate
    ax = axes[0, 2]
    for si in range(len(histories)):
        spk = [(a+b)/2 for a, b in zip(histories[si]['speak_A'], histories[si]['speak_B'])]
        ax.plot(_smooth(spk), color=colors[si], linewidth=1.5, label=f'seed{si}')
    ax.set_title("Speak Rate (avg A+B)"); ax.legend(fontsize=6)

    # 4. Trigger signal
    ax = axes[1, 0]
    for si in range(len(histories)):
        trig = [(a+b)/2 for a, b in zip(histories[si]['trigger_A'], histories[si]['trigger_B'])]
        ax.plot(_smooth(trig), color=colors[si], linewidth=1.5, label=f'seed{si}')
    ax.axhline(0, c='gray', ls='-', alpha=0.3)
    ax.axhline(0.05, c='red', ls='--', alpha=0.3, label='threshold')
    ax.set_title("Repair Signal (trigger contrast)")
    ax.legend(fontsize=7)

    # 5. Message entropy
    ax = axes[1, 1]
    for si in range(len(histories)):
        ax.plot(_smooth(histories[si]['msg_entropy']), color=colors[si], linewidth=1.5)
    ax.set_title("Message Entropy"); ax.set_ylabel("nats")

    # 6. Silence rate
    ax = axes[1, 2]
    for si in range(len(histories)):
        sil = [(a+b)/2 for a, b in zip(histories[si]['silence_A'], histories[si]['silence_B'])]
        ax.plot(_smooth(sil), color=colors[si], linewidth=1.5)
    ax.set_title("Silence Rate")

    # 7. Phase trace (per seed)
    ax = axes[2, 0]
    for si in range(len(histories)):
        ax.plot(histories[si]['phase'], color=colors[si], linewidth=1.2, alpha=0.7,
                label=f'seed{si}')
    ax.set_title("Phase Trajectory"); ax.set_ylabel("phase")
    ax.set_yticks([1,2,3,4])
    ax.legend(fontsize=6)

    # 8. Curriculum: ε and cost overlaid
    ax = axes[2, 1]
    ax2 = ax.twinx()
    longest = max(range(len(histories)), key=lambda i: len(histories[i]['epsilon']))
    eps = histories[longest]['epsilon']
    sc = histories[longest]['speak_cost']
    ax.plot(eps, color='#c0392b', linewidth=2, label='ε (noise)')
    ax2.plot(sc, color='#2c3e50', linewidth=2, label='speak cost')
    ax.set_ylabel('ε', color='#c0392b')
    ax2.set_ylabel('speak cost', color='#2c3e50')
    ax.set_title(f"Curriculum schedule (seed{longest})")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    # 9. Policy loss + Value loss
    ax = axes[2, 2]
    for si in range(len(histories)):
        ax.plot(_smooth(histories[si]['policy_loss']), color=colors[si], alpha=0.7)
    ax.set_title("Policy Loss")

    for row in axes:
        for a in row:
            a.set_xlabel("Update")

    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, f'{MODEL_PREFIX}_dynamics.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved training dynamics: {plot_path}")


# %% ── 9. RESULTS PLOT ──────────────────────────────────────────────
def plot_results():
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Independent-Parameter Training — PRIMARY/CONTROL Pair Analysis",
                 fontsize=14, fontweight='bold')

    cfg_order = ['A-vs-B', 'B-vs-A', 'A-vs-A', 'B-vs-B']
    cfg_tags = ['PRIMARY', 'PRIMARY', 'CONTROL', 'CONTROL']
    cfg_colors = ['#27ae60', '#16a085', '#2980b9', '#e74c3c']

    # 1. Training reward (per seed)
    ax = axes[0, 0]
    colors_s = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    for si in range(len(histories)):
        r = histories[si]['reward']
        window = min(500, len(r) // 10 + 1)
        smoothed = np.convolve(r, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, color=colors_s[si], label=f'seed{si}')
    ax.set_xlabel("Update"); ax.set_ylabel("Reward")
    ax.set_title("Training Reward"); ax.legend(fontsize=7)

    # 2. Delta by config
    ax = axes[0, 1]
    for i, (name, tag, color) in enumerate(zip(cfg_order, cfg_tags, cfg_colors)):
        vals = [r['delta'] for r in configs[name]]
        ax.bar(i, np.mean(vals), yerr=np.std(vals), color=color, alpha=0.8, capsize=5,
               label=f'{name} ({tag})')
    ax.axhline(0.10, c='red', ls='--', alpha=0.5, label='claim threshold')
    ax.set_xticks(range(4)); ax.set_xticklabels(cfg_order, rotation=0)
    ax.set_ylabel("Delta (comm − no-comm)")
    ax.set_title("Communication Delta by Pair Config")
    ax.legend(fontsize=7)

    # 3. Trigger contrast by config
    ax = axes[0, 2]
    for i, (name, tag, color) in enumerate(zip(cfg_order, cfg_tags, cfg_colors)):
        vals = [r.get('trigger_contrast', 0) for r in configs[name]]
        ax.bar(i, np.mean(vals), yerr=np.std(vals), color=color, alpha=0.8, capsize=5)
    ax.axhline(0, c='k', ls='-', alpha=0.3)
    ax.axhline(0.05, c='red', ls='--', alpha=0.5, label='threshold')
    ax.set_xticks(range(4)); ax.set_xticklabels(cfg_order, rotation=0)
    ax.set_ylabel("Trigger Contrast")
    ax.set_title("Repair Signature by Config")
    ax.legend(fontsize=7)

    # 4. Per-seed delta scatter
    ax = axes[1, 0]
    for i, (name, color) in enumerate(zip(cfg_order, cfg_colors)):
        vals = [r['delta'] for r in configs[name]]
        xs = [i] * len(vals)
        ax.scatter(xs, vals, color=color, s=50, alpha=0.7, edgecolor='k')
    ax.axhline(0.10, c='red', ls='--', alpha=0.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(cfg_order, rotation=0)
    ax.set_ylabel("Delta"); ax.set_title("Per-seed delta scatter")

    # 5. Trigger breakdown A-vs-B (clean vs corrupt)
    ax = axes[1, 1]
    tc = [r.get('trigger_rate_corrupt', 0) for r in configs['A-vs-B']]
    tn = [r.get('trigger_rate_clean', 0) for r in configs['A-vs-B']]
    width = 0.35
    ax.bar(0 - width/2, np.mean(tc), width, yerr=np.std(tc),
           color='#e74c3c', alpha=0.7, capsize=3, label='corrupt')
    ax.bar(0 + width/2, np.mean(tn), width, yerr=np.std(tn),
           color='#3498db', alpha=0.7, capsize=3, label='clean')
    ax.set_xticks([0]); ax.set_xticklabels(['A-vs-B'])
    ax.set_ylabel("Speak Rate @ t=3")
    ax.set_title("Conditional Trigger (A-vs-B)")
    ax.legend(fontsize=8)

    # 6. Diagnostic grid
    ax = axes[1, 2]
    ax.axis('off')
    lines = ["Diagnostic per seed", ""]
    for i, d in enumerate(diagnostics):
        p1 = "✓" if d['p1_gate_passed'] else "✗"
        p2 = "✓" if d['p2_gate_passed'] else "✗"
        p4w = "⚠" if d.get('p4_collapse_warned') else " "
        lines.append(f"seed{i}: P1{p1}  P2{p2}  P3ε{p4w}")
    ax.text(0.05, 0.95, "\n".join(lines), family='monospace',
            transform=ax.transAxes, va='top', fontsize=11)

    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, f'{MODEL_PREFIX}_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved plot: {plot_path}")


plot_training_dynamics()
plot_results()

# Save JSON
raw = {}
for cfg_name, results in configs.items():
    raw[cfg_name] = [{k: v for k, v in r.items()
                       if not isinstance(v, np.ndarray)} for r in results]
raw['_diagnostics'] = diagnostics

json_path = os.path.join(SAVE_DIR, f'{MODEL_PREFIX}_results.json')
with open(json_path, 'w') as f:
    json.dump(raw, f, indent=2, default=str)
print(f"\nSaved JSON: {json_path}")

hist_path = os.path.join(SAVE_DIR, f'{MODEL_PREFIX}_histories.json')
with open(hist_path, 'w') as f:
    json.dump(histories, f)
print(f"Saved training histories: {hist_path}")

meta = {
    'version': 'independent',
    'n_updates': N_UPDATES,
    'hidden': HIDDEN,
    'curriculum': '4-phase additive+dynamic from step 0',
    'phases': {
        'P1': {'min': PHASE1_MIN, 'max': PHASE1_MAX,
               'gate': f'r_o≥{PHASE1_R_THRESH} ∧ r_s≥{PHASE1_R_THRESH} ∧ msg_ent≤{PHASE1_MSG_ENT_THRESH}',
               'reward_mode': 'additive', 'heads': 'all',
               'dynamic_target': True, 'epsilon': 0, 'cost': 0},
        'P2': {'dur_min': PHASE2_DUR_MIN, 'dur_max': PHASE2_DUR_MAX,
               'gate': f'r_AND≥{PHASE2_R_THRESH}',
               'reward_mode': 'AND', 'heads': 'all',
               'dynamic_target': True, 'epsilon': 0, 'cost': 0},
        'P3': {'dur': PHASE3_DUR, 'reward_mode': 'AND',
               'eps_ramp': f'0→{EPSILON_MAX}',
               'cost_ramp': f'0→{PHASE3_COST_TARGET}'},
        'P4': {'dur': 'remaining', 'reward_mode': 'AND',
               'epsilon': EPSILON_MAX,
               'cost_ramp': f'{PHASE3_COST_TARGET}→{PHASE4_COST_TARGET}'},
    },
    'batch_size': BATCH_SIZE,
    'per_head_ent': {
        'msg_hi': ENT_MSG_HI, 'msg_lo': ENT_MSG_LO,
        'msg_ramp_steps': ENT_MSG_RAMP_STEPS,
        'act': ENT_ACT,
        'hold_frac': ENT_HOLD_FRAC, 'decay_floor': ENT_DECAY_FLOOR,
    },
    'n_seeds': N_SEEDS,
    'cost_accounting': 'individual (Plan B)',
    'critic_optimizer': 'fully independent IPPO, split grad clip, separate backward',
}
meta_path = os.path.join(SAVE_DIR, f'{MODEL_PREFIX}_meta.json')
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)
print(f"Saved meta: {meta_path}")
