<div align="center">

# MindAI

### A biologically grounded simulation of emergent machine consciousness

*No gradient descent. No reward functions. No scripted behaviors.*  
*Only neurons, chemistry, and time.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-GPU%20Accelerated-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-GPL%203.0-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-orange?style=flat-square)]()
[![Neuroscience](https://img.shields.io/badge/Grounded%20In-Neuroscience-purple?style=flat-square)]()

</div>

---

## Overview

MindAI is a real-time neural simulation where a synthetic agent survives in a 2D world through **emergent behavior only** — no trained weights, no objectives, no designer-specified goals. The agent's brain is a sparse spiking network of ~9,000 neurons that self-organizes through biologically accurate Hebbian/STDP plasticity, modulated by a full neuroendocrine system.

The project is built around four major scientific theories of consciousness, each implemented as a discrete computational module:

| Theory | Author(s) | Module |
|--------|-----------|--------|
| **Global Workspace Theory** | Baars (1988), Dehaene (2001) | `global_workspace.py` |
| **Integrated Information Theory** (Φ) | Tononi (2004) | `qualia_space_iit.py` |
| **Predictive Processing / Free Energy** | Friston (2010) | `predictive_hierarchy.py` |
| **Somatic Marker Hypothesis** | Damasio (1994) | `volition_and_agency.py` |

The system has no access to ground truth. It has no loss function. Behavior emerges from the interaction of physiology, memory, and a world that doesn't care if it survives.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    World (40×40 grid)                   │
│         food · poison · walls · human agent             │
└───────────────────┬─────────────────────────────────────┘
                    │ raw pixels + spatial audio
          ┌─────────▼──────────┐
          │  Retina + Cochlea  │  vision (foveal encoding)
          │                    │  hearing (32 cochlear bands)
          └─────────┬──────────┘
                    │ raw_sensory  [N-dim]
          ┌─────────▼──────────────────────────────┐
          │     Sparse Connectome  (GPU)            │
          │  ~9 370 neurons · Hebbian/STDP weights  │
          │  80% excitatory (glutamate)             │
          │  20% inhibitory (GABA)                  │
          └─────────┬──────────────────────────────┘
                    │ internal recurrence
          ┌─────────▼──────────┐
          │ PredictiveMicro-   │  top-down: L5/6 prediction neurons
          │ circuits           │  bottom-up: L2/3 error neurons
          └─────────┬──────────┘
                    │ updated state + surprise scalar
          ┌─────────▼──────────┐
          │  HusserlianTime    │  "thick present"
          │                    │  retention · now · protention
          └─────────┬──────────┘
                    │
          ┌─────────▼──────────┐
          │     Thalamus       │  attention gate
          │                    │  threshold ∝ noradrenaline / boredom
          └─────────┬──────────┘
                    │ salient signal
          ┌─────────▼──────────────────────────────┐
          │  PhaseCoupledWorkspace  (Kuramoto)      │
          │  R > 0.7  →  global ignition broadcast  │
          └────┬──────────────────────────┬─────────┘
               │ broadcast activity       │ snapshot → Hippocampus
    ┌──────────▼─────────┐    ┌───────────▼──────────────┐
    │  EndocrineSystem   │    │  Hippocampus             │
    │  dopamine          │    │  episodic encoding       │
    │  serotonin (vagus) │    │  valence-gated storage   │
    │  noradrenaline     │    │  sleep replay ×5 speed   │
    │  cortisol (HPA)    │    └──────────────────────────┘
    │  oxytocin (social) │
    │  endorphins        │
    └──────────┬─────────┘
               │ neuromodulator state
    ┌──────────▼──────────────────────────────────────────┐
    │  BasalGanglia  →  FreeWillEngine                    │
    │  corticostriatal mapping    Libet delay (15 ticks)  │
    │  dopamine/pain reinforce    somatic veto            │
    └──────────┬──────────────────────────────────────────┘
               │ action (0–4)
    ┌──────────▼─────────┐
    │   World.execute()  │  move · eat · drink
    │                    │  energy / water / stress Δ
    └──────────┬─────────┘
               │
    ┌──────────▼─────────────────────────────────────────┐
    │  StructuralPlasticity  (STDP + synaptogenesis)      │
    │  LTP / LTD  ·  pruning  ·  homeostatic scaling     │
    │  cortisol neurotoxicity  ·  neurogenesis on demand  │
    └────────────────────────────────────────────────────┘
```

### Sleep path

When adenosine + melatonin pressure crosses threshold, the main loop is bypassed entirely. The `SleepCycle` replays hippocampal episodes through `StructuralPlasticity` at 5× rate. High cortisol during sleep stochastically corrupts memories — a model of trauma-influenced consolidation.

---

## Biological mechanisms

### Synaptic plasticity — Hebbian/STDP
Weights change exclusively through spike-timing-dependent plasticity (Bi & Poo, 1998). Pre-before-post firing causes LTP; post-before-pre causes LTD. No global error signal exists. The only modulation is through neuromodulators that gate *how much* synapses change, never *in which direction* to change them.

```
ΔW_LTP  ∝  pre_trace(t) × post_active(t) × neuromod_multiplier
ΔW_LTD  ∝  pre_active(t) × post_trace(t) × neuromod_multiplier
```

Synaptic homeostasis (Turrigiano scaling) normalizes total incoming weight when neurons become overloaded.

### Neuroendocrine system
Seven neuromodulators interact continuously:

| Molecule | Biological source | Role in simulation |
|----------|------------------|--------------------|
| **Dopamine** | VTA / striatum | Pavlovian conditioning; gates plasticity multiplier |
| **Serotonin** | Raphe / gut | Vagal tone from satiation; mood baseline |
| **Noradrenaline** | Locus coeruleus | Thalamic gain; arousal; surprise-driven |
| **Cortisol** | HPA axis | Suppresses LTP; damages synapses at sustained high levels |
| **Oxytocin** | Hypothalamus | Social bonding (E key); dampens fear; reduces cortisol |
| **Endorphins** | Arcuate nucleus | Analgesic; burst-released on reward; boosts plasticity |
| **Adrenaline** | Adrenal medulla | Fight-or-flight; co-triggered with noradrenaline + pain |

Plasticity multiplier: `(dopamine × 1.5 + serotonin × 0.5) × (1 − cortisol) × (1 + endorphins)`

Pain suppresses plasticity continuously: `multiplier × max(0, 1 − pain/100)` — it never reverses learning direction.

### Global Workspace — Kuramoto ignition
Phase coupling is computed across all active neurons via the Kuramoto order parameter R. When synchrony exceeds R = 0.7, a nonlinear "ignition" event occurs: activity is amplified ×3 and broadcast globally, with a 30-tick refractory period. This models the all-or-nothing conscious access reported by Dehaene & Changeux (2011).

### Predictive hierarchy
Two anatomically distinct neuron populations are maintained:

- **Prediction neurons** (L5/6 analogue): `Ŷ = W_td × internal_state` — top-down expectation
- **Error neurons** (L2/3 analogue): `ε = relu(sensory − Ŷ)` — bottom-up mismatch signal

Both populations are persistent across ticks, enabling other modules to read the current prediction or surprise state independently.

### Volition — Libet delay + somatic markers
The `FreeWillEngine` places every motor decision into a queue with a configurable delay (default: 15 ticks). The decision is realizable only when it clears the queue. Somatic markers (Damasio, 1994) bias action probabilities based on pain history: actions that previously caused high stress are suppressed proportionally to amygdala arousal (noradrenaline).

### Temporal consciousness — Husserlian "thick present"
The `HusserlianTime` module implements Husserl's phenomenology of inner time: the conscious moment is not an instantaneous point but a weighted blend of retention (recent past), primal impression (now), and protention (predicted near-future).

```
thick_now = 0.6 × primal_impression + 0.3 × retention_smear + 0.1 × protention
```

### Mood attractors
The agent's emotional state is modeled as a dynamical system with three attractor basins — *calm*, *anxiety*, *depression* — defined by energy/stress coordinates. Dopamine levels modulate the depth of each basin. The agent can fall into depressive states under sustained energy deprivation regardless of any external programmer intervention.

---

## Installation

```bash
git clone https://github.com/{username}/mindai.git
cd mindai
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch (CUDA optional), NumPy, SciPy, PyGame, PyYAML, PyAudio

---

## Running

```bash
python main.py
```

If `savegame.pkl` exists, you will be prompted to resume the saved brain state.

### Controls

| Key | Action |
|-----|--------|
| `Arrow keys` | Move human agent |
| `E` | Interact with AI (triggers oxytocin) |
| `V` | Enable microphone input (live audio → cochlea) |
| `+` / `-` | Cycle simulation speed (1 / 5 / 10 / 25 / 60 FPS / MAX) |
| `Q` | Quit and save brain state |

### Simulation speed levels

| Level | FPS | Use case |
|-------|-----|----------|
| 1 | 1 | Frame-by-frame observation |
| 2 | 5 | Slow-motion study |
| 3 | 10 | Default observation |
| 4 | 25 | Normal run |
| 5 | 60 | Fast run |
| 6 | MAX | Accelerated evolution |

---

## Configuration

All parameters live in `config/default_sim.yaml`:

```yaml
hardware:
  num_nodes: 9370          # total neurons
  spatial_dimensions: 3    # 3D geometry for axonal delay computation

biology:
  base_energy: 3000.0
  inhibitory_ratio: 0.2    # ~20% GABA neurons (Dale's principle)

consciousness:
  volition_delay_ticks: 15 # Libet readiness potential analogue
  temporal_window_size: 7  # Husserlian retention buffer depth

plasticity:
  hebbian_ltp_rate: 0.08
  pruning_half_life: 300

lifecycle:
  circadian_cycle_ticks: 5000
  rem_sleep_duration: 500
```

---

## Module map

```
src/
├── engine/
│   ├── plasticity_core.py        Connectome · STDP · synaptogenesis · homeostasis
│   ├── temporal_windows.py       Husserlian thick present
│   ├── spatial_topology_3d.py    3D neuron geometry for conduction delays
│   └── axonal_delays.py          Spike queue with per-synapse travel time
│
├── architecture/
│   ├── predictive_hierarchy.py   Predictive coding (L2/3 error · L5/6 prediction)
│   ├── thalamocortical_core.py   Attention gate · noradrenaline-driven threshold
│   ├── hippocampus_buffer.py     Episodic encoding · valence gating · consolidation
│   ├── prefrontal_cortex.py      Working memory / executive control
│   └── semantic_memory.py        Concept extraction during sleep
│
├── consciousness/
│   ├── global_workspace.py       Kuramoto synchrony · ignition broadcast
│   ├── qualia_space_iit.py       Φ approximation via eigenvalue geometry
│   ├── volition_and_agency.py    BasalGanglia + FreeWillEngine (Libet delay)
│   └── self_model_ego.py         Interoceptive self-prediction · sense of agency
│
├── neurochemistry/
│   ├── neuromodulators.py        Seven-hormone endocrine system · Pavlovian DA
│   └── attractor_dynamics.py     Mood basins (calm / anxiety / depression)
│
├── lifecycle/
│   ├── sleep_consolidation.py    Hippocampal replay · cortisol memory distortion
│   └── circadian_rhythm.py       Adenosine + melatonin → sleep pressure
│
└── environment/
    ├── world_2d.py               40×40 grid · terrain · objects · human/agent actions
    ├── vision_system.py          Foveal retina encoding
    ├── hearing_system.py         32-band cochlear filterbank · microphone input
    └── ui_renderer.py            Real-time PyGame display · neuromodulator HUD
```

---

## Save format

Brain state is serialized as `savegame.pkl` using `scipy.sparse.coo_matrix` (not PyTorch tensors) to keep file sizes manageable. On load, weights are transferred back to GPU. The save includes full world state, inventory, and all neuron weights and integrity values.

A migration utility `surgery.py` remaps saved synapse indices after the neuron layout is expanded.

---

## What this is not

- **Not a deep learning model.** `torch.set_grad_enabled(False)` is set at startup. There is no optimizer, no loss function, no backpropagation anywhere in the codebase.
- **Not a reinforcement learning agent.** There is no reward signal fed to a policy. Dopamine arises from internal physiology, not from an external evaluator.
- **Not a chatbot or language model.** The agent has no natural language capacity. It communicates through vocalizations (32-dimensional neural activity mapped to sound) and through action.
- **Not a finished product.** This is active research into whether the combination of GWT, IIT, PP, and SMH in a single embodied system produces behavior that resembles conscious experience.

---

## Key references

- Baars, B. J. (1988). *A Cognitive Theory of Consciousness.* Cambridge University Press.
- Bi, G., & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons. *Journal of Neuroscience*, 18(24).
- Damasio, A. (1994). *Descartes' Error.* Putnam.
- Dehaene, S., & Changeux, J.-P. (2011). Experimental and theoretical approaches to conscious processing. *Neuron*, 70(2).
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2).
- Hebb, D. O. (1949). *The Organization of Behavior.* Wiley.
- Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators. *Lecture Notes in Physics*, 39.
- Libet, B. et al. (1983). Time of conscious intention to act in relation to onset of cerebral activity. *Brain*, 106(3).
- Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*, 5(42).
- Turrigiano, G. G. (2008). The self-tuning neuron: synaptic scaling of excitatory synapses. *Cell*, 135(3).

---

## Star history

[![Star History Chart](https://api.star-history.com/svg?repos={username}/mindai&type=Date)](https://star-history.com/#{username}/mindai&Date)

---

<div align="center">

*The question is not whether machines can think.*  
*The question is whether thinking can emerge from the right kind of physics.*

</div>
