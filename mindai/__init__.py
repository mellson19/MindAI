"""mindai — biologically-inspired artificial mind.

Learning is exclusively Hebbian/STDP. No gradient descent. No reward
functions. Behavior emerges from physiology gated by neuromodulators.

Quick start::

    from mindai import Brain
    from mindai.worlds.agent_world import AgentWorld
    from mindai.neurochemistry.neuromodulators import EndocrineSystem

    world = AgentWorld(text_corpus='corpus.txt', interactive=True)

    brain = Brain(
        num_neurons    = 500_000,
        sensory_layout = world.sensory_layout,
        motor_layout   = world.motor_layout,
        synapse_density= 0.001,
    )
    brain.attach(EndocrineSystem())
    brain.run(world, headless=True)
"""

from mindai.brain import Brain
from mindai.layout import SensoryLayout
from mindai.feels import FeelingSystem, Feel, curves

__all__ = ['Brain', 'SensoryLayout', 'FeelingSystem', 'Feel', 'curves']
__version__ = '0.3.0'
