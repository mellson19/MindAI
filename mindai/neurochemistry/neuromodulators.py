import numpy as np


class EndocrineSystem:
    """Neuroendocrine system governing learning, homeostasis, and social behaviour.

    Modulator list and primary biological source:
        dopamine      — VTA / substantia nigra  — reward prediction, plasticity gating
        serotonin     — raphe nuclei            — gut–brain satiety axis, mood
        noradrenaline — locus coeruleus         — arousal, thalamic attention gate
        cortisol      — adrenal cortex (HPA)    — chronic stress, synaptic damage
        oxytocin      — hypothalamus/PVN        — social trust, fear dampening
        endorphins    — pituitary / PAG         — pain suppression, post-reward
        adrenaline    — adrenal medulla         — acute fight-or-flight
        acetylcholine — basal forebrain/PPN     — attention gate on STDP precision
        anandamide    — endocannabinoid system  — CA1 LTP suppression (forgetting gate)
        substance_p   — spinal DRG / PAG        — pain wind-up / central sensitisation
        ghrelin       — stomach X/A-cells       — pre-meal hunger + VTA dopamine drive
        leptin        — adipocytes              — post-meal satiety signal (delayed)
        vasopressin   — hypothalamus/SON        — social memory, territorial vigilance
        prolactin     — anterior pituitary      — post-stress affiliative drive
        insulin       — pancreatic β-cells      — post-meal LC suppression, ACh dip
    """

    def __init__(self, auditory_cortex_size: int = 0) -> None:
        # auditory_cortex_size=0 means "defer to first auditory_spikes shape"
        self._auditory_size_locked = auditory_cortex_size > 0
        # Dopamine — three anatomical pathways (Berger 1991; Nieoullon 2002)
        # mesolimbic:   VTA → NAc/limbic — reward, motivation, approach
        # mesocortical: VTA → PFC — working memory, cognitive flexibility
        # nigrostriatal:SN → striatum — movement initiation, action selection
        # Backward-compat: .dopamine property returns mesolimbic (primary signal)
        self.dopamine_mesolimbic   = 0.5
        self.dopamine_mesocortical = 0.5
        self.dopamine_nigrostriatal= 0.5

        # original 7 + boredom (dopamine field kept as alias below)
        self.serotonin         = 0.5
        self.noradrenaline     = 0.1
        self.cortisol          = 0.0
        self.oxytocin          = 0.0
        self.endorphins        = 0.0
        self.adrenaline        = 0.0
        self.boredom           = 0.0
        self._stress_accum     = 0.0

        # new 8
        self.acetylcholine     = 0.5
        self.anandamide        = 0.3
        self.substance_p       = 0.0
        self.ghrelin           = 0.0
        self.leptin            = 0.5
        self.vasopressin       = 0.1
        self.prolactin         = 0.0
        self.insulin           = 0.3

        # Substance P two-phase kinetics (see update_state for details)
        self._cortisol_prev    = 0.0   # previous-tick cortisol for acute-vs-chronic detection
        self._sp_acute         = 0.0   # fast phase: NK1 activation (~T1/2 20 ticks)
        self._sp_chronic       = 0.0   # slow phase: dorsal horn LTP (~T1/2 500 ticks)

        # internal: last computed effective pain, exposed for brain.py sensory vector
        self._effective_pain   = 0.0

        # Pavlovian conditioning weights — sized lazily on first auditory input
        # so the module is portable across any audio channel count
        _sz = auditory_cortex_size if auditory_cortex_size > 0 else 1
        self.sound_to_dopamine_weights = np.zeros(_sz)
        self.sound_to_fear_weights     = np.zeros(_sz)

    # ------------------------------------------------------------------
    # Main update — called every awake tick AFTER world.execute_action()
    # ------------------------------------------------------------------

    def update_state(
        self,
        global_arousal:       float,
        layer23_error_spikes: float,
        raw_pain_signal:      float,
        energy_ratio:         float,
        water_ratio:          float,
        auditory_spikes:      np.ndarray = None,
        energy_gained:        float = 0.0,
        isolation_ticks:      int   = 0,
        distance_to_human:    float = float('inf'),
    ) -> None:

        # --- Pavlovian conditioning ---
        learned_dopamine = 0.0
        learned_fear     = 0.0
        if auditory_spikes is not None and len(auditory_spikes) > 0:
            # Lazy resize: first call with real audio sets the vector size
            n = len(auditory_spikes)
            if len(self.sound_to_dopamine_weights) != n:
                self.sound_to_dopamine_weights = np.zeros(n)
                self.sound_to_fear_weights     = np.zeros(n)
            learned_dopamine = float(np.dot(auditory_spikes, self.sound_to_dopamine_weights))
            learned_fear     = float(np.dot(auditory_spikes, self.sound_to_fear_weights))
            if energy_ratio + water_ratio > 1.5:
                self.sound_to_dopamine_weights += auditory_spikes * 0.01
            if raw_pain_signal > 0.1:
                self.sound_to_fear_weights += auditory_spikes * 0.05
            self.sound_to_dopamine_weights *= 0.999
            self.sound_to_fear_weights     *= 0.999

        gut_satisfaction = np.clip((energy_ratio + water_ratio) / 2.0, 0.0, 1.0)

        # --- Substance P: two-phase spinal wind-up / central sensitisation ---
        #
        # Biological basis (Woolf & Salter 2000; Latremoliere & Woolf 2009):
        #   Acute phase:   C-fibre NK1 receptor activation → intracellular Ca²⁺
        #                  rises quickly and decays within ~20-40 s (here ~20 ticks).
        #   Chronic phase: Repeated acute episodes → AMPA receptor trafficking +
        #                  NMDA Mg²⁺ block removal → long-term dorsal horn LTP.
        #                  Persists for minutes to hours (here ~500 ticks = chronic).
        #
        # The effective substance_p blends both phases:
        #   - Acute (70%): encodes the immediate wind-up that makes injury
        #     feel worse on the second stimulus than the first.
        #   - Chronic (30%): encodes persistent central sensitisation (allodynia).
        #
        # This replaces the single-phase 0.995 decay which was too slow for
        # transient sensitisation and too fast for true chronic pain.
        self._sp_acute   = float(np.clip(
            self._sp_acute   * 0.965 + raw_pain_signal * 0.10, 0.0, 1.0))
        self._sp_chronic = float(np.clip(
            self._sp_chronic * 0.998 + self._sp_acute  * 0.004, 0.0, 1.0))
        self.substance_p = float(np.clip(
            self._sp_acute * 0.70 + self._sp_chronic * 0.30, 0.0, 1.0))
        self._effective_pain = float(
            np.clip(raw_pain_signal * (1.0 + self.substance_p * 2.0), 0.0, 1.0))

        # --- Ghrelin / Leptin ---
        # Ghrelin: rises as energy falls, suppressed by leptin.
        # Also drives VTA dopamine to motivate food-seeking BEFORE energy is critical
        # (Abizaid 2006: ghrelin activates midbrain dopamine neurons).
        hunger_drive = max(0.0, 1.0 - energy_ratio)
        # Leptin coefficient corrected to 0.15 (Schwartz et al. 2000):
        # a post-meal leptin spike of 0.5 suppresses ghrelin by 0.075/tick,
        # enough to compete with the hunger drive of 0.05/tick.
        self.ghrelin = np.clip(
            self.ghrelin * 0.97 + hunger_drive * 0.05 - self.leptin * 0.15,
            0.0, 1.0)
        # Leptin: spikes on food intake, kinetics ~ adipokine (slow decay ~500 ticks)
        self.leptin = np.clip(
            self.leptin * 0.998 + energy_gained * 0.0002, 0.0, 1.0)

        # Ghrelin → VTA mesolimbic dopamine: food-seeking motivation (Abizaid 2006)
        if self.ghrelin > 0.3:
            self.dopamine_mesolimbic = min(
                1.0, self.dopamine_mesolimbic + (self.ghrelin - 0.3) * 0.03)

        # --- Insulin: post-meal arousal dip via raphe serotonin pathway ---
        #
        # Biological basis (Fernstrom & Wurtman 1972; Wurtman et al. 2003):
        #   Insulin → peripheral amino acid uptake → relative tryptophan
        #   enrichment in blood → enhanced BBB tryptophan transport →
        #   increased 5-HT synthesis in dorsal raphe nuclei → sedation/satiety.
        #
        # Key correction: insulin does NOT directly suppress locus coeruleus.
        # Insulin barely crosses the blood-brain barrier (INSR-mediated transport
        # is slow and saturable); the post-meal cognitive dip is a serotonergic
        # effect, not a direct LC/noradrenaline effect.
        #
        # High serotonin already reduces arousal indirectly through gut_satisfaction
        # → lower panic_signal → lower noradrenaline. The pathway is intact.
        self.insulin = float(np.clip(
            self.insulin * 0.993 + energy_gained * 0.0003, 0.0, 1.0))
        # Insulin → tryptophan enrichment → serotonin synthesis (raphe)
        if self.insulin > 0.3:
            serotonin_boost = (self.insulin - 0.3) * 0.008
            self.serotonin = float(np.clip(
                self.serotonin + serotonin_boost, 0.0, 1.0))

        # --- Serotonin: gut–brain axis (ghrelin suppresses contentment) ---
        self.serotonin = np.clip(
            self.serotonin * 0.9 + gut_satisfaction * 0.1 - self.ghrelin * 0.05,
            0.0, 1.0)

        # --- Noradrenaline: locus coeruleus arousal ---
        felt_pain     = self._effective_pain * (1.0 - self.endorphins * 0.8)
        fear_dampener = 1.0 - self.oxytocin * 0.8
        panic_signal  = layer23_error_spikes * 0.1 + felt_pain + learned_fear
        self.noradrenaline = np.clip(
            self.noradrenaline * 0.9
            + panic_signal * fear_dampener * (1.0 - gut_satisfaction),
            0.1, 1.0)

        # --- Adrenaline: acute fight-or-flight ---
        if felt_pain > 0.5 and self.noradrenaline > 0.6:
            self.adrenaline = min(1.0, self.adrenaline + 0.3)
        else:
            self.adrenaline *= 0.9

        # --- Vasopressin: territorial vigilance + social memory ---
        # AVP rises during isolation (territorial mode) and decays on contact
        # with the familiar human (not a threat → reduced territorial drive).
        # Social proximity with a known individual does NOT raise AVP;
        # it slightly lowers it as safety is confirmed (Landgraf 2006).
        isolation_signal = min(1.0, isolation_ticks / 2000.0)
        in_contact       = distance_to_human < 5
        self.vasopressin = np.clip(
            self.vasopressin * 0.995
            + isolation_signal * 0.008
            - (0.003 if in_contact else 0.0),
            0.0, 1.0)
        # Prolonged isolation → vasopressin boosts noradrenaline (hypervigilance)
        if isolation_ticks > 500:
            self.noradrenaline = min(
                1.0, self.noradrenaline + self.vasopressin * 0.002)

        # --- Cortisol: HPA chronic stress accumulation ---
        self._cortisol_prev = self.cortisol
        if self.noradrenaline > 0.7:
            self._stress_accum += 0.01 * (1.0 - self.oxytocin)
        else:
            self._stress_accum = max(0.0, self._stress_accum - 0.005)
        self.cortisol = np.clip(
            self.cortisol * 0.999 + self._stress_accum * 0.001, 0.0, 1.0)

        # --- Prolactin: post-stress affiliative drive ---
        # Rises proportionally to cortisol load; decays when oxytocin rises
        # (social need satisfied). High prolactin is read by brain.py to amplify
        # mirror-neuron sensitivity — biologically: increased attention to social
        # cues after stress, without scripting approach behaviour.
        self.prolactin = np.clip(
            self.prolactin * 0.99
            + self.cortisol * 0.01
            - self.oxytocin * 0.02,
            0.0, 1.0)

        # --- Dopamine: three-pathway update (Berger 1991; Nieoullon 2002) ---
        # Mesolimbic (VTA→NAc): Pavlovian reward; food/social events
        if learned_dopamine > 0.5:
            self.dopamine_mesolimbic = min(1.0, self.dopamine_mesolimbic + 0.2)
        else:
            self.dopamine_mesolimbic = max(0.1, self.dopamine_mesolimbic * 0.98)
        # Mesocortical (VTA→PFC): cognitive flexibility; rises with novelty
        novelty_da = min(1.0, layer23_error_spikes * 0.05)
        self.dopamine_mesocortical = float(np.clip(
            self.dopamine_mesocortical * 0.99 + novelty_da * 0.02, 0.1, 1.0))
        # Nigrostriatal (SN→striatum): movement initiation; depressed by cortisol
        self.dopamine_nigrostriatal = float(np.clip(
            self.dopamine_nigrostriatal * 0.99
            + gut_satisfaction * 0.01
            - self.cortisol * 0.005, 0.05, 1.0))

        # --- Endorphins and oxytocin: passive decay ---
        self.endorphins *= 0.95
        self.oxytocin    = max(0.0, self.oxytocin * 0.998)

        # --- Acetylcholine: basal forebrain attention gate ---
        # High on novelty/surprise; suppressed by insulin (post-meal ACh dip).
        # Suppressed by extreme noradrenaline ONLY above 0.8 — the full
        # fight-or-flight threshold at which basal forebrain goes offline.
        # Below that threshold NA and ACh co-activate (oriented attention).
        novelty      = min(1.0, layer23_error_spikes * 0.1)
        ach_suppress = (self.insulin * 0.3
                        + max(0.0, self.noradrenaline - 0.8) * 0.6)
        self.acetylcholine = np.clip(
            self.acetylcholine * 0.9 + novelty * 0.3 - ach_suppress * 0.1,
            0.1, 1.0)

        # --- Anandamide: CB1-mediated CA1 LTP suppression ---
        # High during safe/routine states; suppressed by emotional arousal
        # (pain, dopamine surge, noradrenaline). Emotionally salient events
        # get consolidated because anandamide is low at those moments.
        is_routine   = (gut_satisfaction > 0.7
                        and felt_pain < 0.1
                        and layer23_error_spikes < 2.0)
        ananda_drive = 0.005 if is_routine else -0.02
        # CB1 suppressed by pain and high NA only — NOT by dopamine.
        # Food raises DA but also raises anandamide (CB1 active during satiety).
        ananda_suppress = (felt_pain * 0.2
                           + max(0.0, self.noradrenaline - 0.5) * 0.1)
        self.anandamide = np.clip(
            self.anandamide * 0.98 + ananda_drive - ananda_suppress,
            0.0, 1.0)

        # --- Boredom ---
        if is_routine:
            self.boredom = min(1.0, self.boredom + 0.005)
        else:
            self.boredom *= 0.8

    # ------------------------------------------------------------------
    # Event triggers
    # ------------------------------------------------------------------

    def trigger_social_bonding(self) -> None:
        self.oxytocin              = min(1.0, self.oxytocin + 0.10)
        self.dopamine_mesolimbic   = min(1.0, self.dopamine_mesolimbic + 0.05)
        self.cortisol              = max(0.0, self.cortisol - 0.02)
        self.vasopressin           = max(0.0, self.vasopressin - 0.01)
        self.prolactin             = max(0.0, self.prolactin  - 0.05)

    def trigger_endorphin_rush(self) -> None:
        self.endorphins            = 1.0
        self.dopamine_mesolimbic   = 1.0
        self.dopamine_nigrostriatal= min(1.0, self.dopamine_nigrostriatal + 0.3)
        self.adrenaline            = 0.0
        self.noradrenaline        *= 0.2
        self.insulin               = min(1.0, self.insulin + 0.20)
        self.leptin                = min(1.0, self.leptin  + 0.10)

    # ------------------------------------------------------------------
    # Derived signals consumed by other subsystems
    # ------------------------------------------------------------------

    def get_plasticity_multiplier(self) -> float:
        """STDP scale: dopamine/serotonin gate willingness; ACh gates precision.

        Uses mesocortical DA for cortical STDP gating — this pathway projects
        to PFC and association cortices where long-term memory consolidation
        occurs (Seamans & Yang 2004).

        Acute cortisol: the first 30 min after a cortisol spike briefly enhances
        encoding of threat-relevant memories (McEwen 2001). Only chronic elevation
        suppresses LTP via dendritic atrophy (Cerqueira 2007).

        Oxytocin→DA: PVN→VTA projection gates reward-learning in social contexts
        (Insel 1992). Represented as a multiplier on total plasticity capacity.
        """
        # Acute-vs-chronic cortisol: a fresh spike enhances; sustained level suppresses
        cortisol_delta = max(0.0, self.cortisol - self._cortisol_prev)
        acute_boost    = 1.0 + cortisol_delta * 5.0
        learning_capacity = max(0.0, 1.0 - self.cortisol) * acute_boost

        # PVN→VTA oxytocin amplifies social-context learning (Insel 1992)
        oxyto_boost = 1.0 + self.oxytocin * 0.2

        return (
            (self.dopamine_mesocortical * 1.5 + self.serotonin * 0.5)
            * learning_capacity
            * (1.0 + self.endorphins)
            * self.acetylcholine
            * oxyto_boost
        )

    @property
    def dopamine(self) -> float:
        """Backward-compatible dopamine accessor — returns mesolimbic (VTA→NAc)."""
        return self.dopamine_mesolimbic

    @dopamine.setter
    def dopamine(self, value: float) -> None:
        """Setting .dopamine updates all three pathways proportionally."""
        ratio = float(np.clip(value / (self.dopamine_mesolimbic + 1e-9), 0.01, 3.0))
        self.dopamine_mesolimbic    = float(np.clip(value, 0.0, 1.0))
        self.dopamine_mesocortical  = float(np.clip(self.dopamine_mesocortical  * ratio, 0.0, 1.0))
        self.dopamine_nigrostriatal = float(np.clip(self.dopamine_nigrostriatal * ratio, 0.0, 1.0))

    @property
    def effective_pain_signal(self) -> float:
        """Substance-P-sensitised pain for injection into sensory pain neurons."""
        return self._effective_pain

    @property
    def hippocampal_salience_gate(self) -> float:
        """Multiplicative scale [0,1] on emotional valence before encoding.

        Anandamide (CB1 at CA1) reduces LTP magnitude during routine states.
        Applied as a multiplier on the salience passed to encode_episode(),
        not as a probabilistic skip — matching the synaptic mechanism.
        """
        return max(0.0, 1.0 - self.anandamide * 0.8)

    @property
    def effective_hunger_signal(self) -> float:
        """Ghrelin-amplified hunger drive for sensory neuron injection."""
        return min(1.0, self.ghrelin * 1.5 + 0.05)

    @property
    def mirror_neuron_amplifier(self) -> float:
        """Oxytocin + moderate noradrenaline modulate mirror-neuron sensitivity.

        Biological basis:
        - Oxytocin (PVN → CeA → STS pathway): increases social salience
          and attention to conspecific actions (Domes 2007; Groppe 2013).
          This is the primary driver of mirror neuron up-regulation.
        - Noradrenaline (LC → premotor/prefrontal cortex α₁ receptors):
          sharpens attention to socially relevant stimuli at moderate levels;
          at high levels (panic/fight-or-flight) suppresses social processing.
          Inverted-U relationship (Arnsten 2009).

        Correction from previous version: prolactin has no direct pathway to
        premotor mirror neurons.  Prolactin's social effects are restricted to
        parental/nursing contexts (nucleus accumbens DA suppression) and are
        not generalisable to conspecific action observation.

        Returns a multiplier in [0.5, 1.5].
        """
        # Oxytocin: direct social attention enhancement (PVN→CeA→STS)
        oxy_gate = self.oxytocin * 0.7
        # NA inverted-U: peak around 0.4–0.5; suppressed at high arousal (panic)
        na_gate  = max(0.0, 1.0 - abs(self.noradrenaline - 0.45) * 2.2) * 0.3
        # Insula/amygdala emotional gate (Singer 2004; Carr 2003):
        # Emotional context (fear, pain, joy observed in another) amplifies mirror
        # response via CeA→premotor pathway. Substance-P encodes felt distress;
        # high anandamide (CB1) attenuates amygdala reactivity (Bhattacharyya 2009).
        # We use substance_p as the emotional salience proxy (already models
        # pain-empathy wind-up) and dampen with anandamide.
        emotional_gate = float(
            np.clip(self.substance_p * (1.0 - self.anandamide * 0.5), 0.0, 0.5))
        return float(np.clip(0.5 + oxy_gate + na_gate + emotional_gate, 0.5, 2.0))

    def derive_mood(self) -> str:
        """Derive mood label from neuromodulator balance — no attractor math.

        Biological basis: mood states emerge from the balance of monoamines
        and stress hormones (Schildkraut 1965 catecholamine hypothesis;
        Cowen 2008 serotonin-depression review).

        Rules grounded in clinical neurochemistry:
          depression : low DA + low 5-HT + high cortisol
          anxiety    : low DA + high NA + high cortisol + low 5-HT
          mania      : very high DA + high NA + low cortisol
          calm       : balanced DA/5-HT + low cortisol + low NA
          stressed   : high cortisol + moderate NA
          content    : default balanced state
        """
        da  = self.dopamine_mesolimbic if hasattr(self, 'dopamine_mesolimbic') else self.dopamine
        ht  = self.serotonin
        na  = self.noradrenaline
        cor = self.cortisol

        if da < 0.3 and ht < 0.3 and cor > 0.5:
            return 'depression'
        if da < 0.4 and na > 0.6 and cor > 0.4:
            return 'anxiety'
        if da > 0.8 and na > 0.6 and cor < 0.3:
            return 'mania'
        if cor > 0.6 and na > 0.5:
            return 'stressed'
        if da > 0.55 and ht > 0.4 and cor < 0.3:
            return 'calm'
        return 'content'
