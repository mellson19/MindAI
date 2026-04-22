import time
import random
import numpy as np
import os
import yaml
import pickle
import torch
import scipy.sparse as sp
from src.environment.world_2d import AdvancedWorld2D
from src.environment.ui_renderer import GameUI
from src.environment.hearing_system import Cochlea
from src.engine.spatial_topology_3d import BrainGeometry
from src.engine.plasticity_core import StructuralPlasticity
from src.engine.temporal_windows import HusserlianTime
from src.neurochemistry.neuromodulators import EndocrineSystem
from src.neurochemistry.attractor_dynamics import MoodAttractors
from src.architecture.predictive_hierarchy import PredictiveMicrocircuits
from src.architecture.thalamocortical_core import Thalamus
from src.architecture.hippocampus_buffer import Hippocampus
from src.architecture.prefrontal_cortex import PrefrontalCortex
from src.architecture.semantic_memory import SemanticMemory
from src.consciousness.global_workspace import PhaseCoupledWorkspace
from src.consciousness.self_model_ego import EgoModel
from src.consciousness.volition_and_agency import FreeWillEngine, BasalGanglia
from src.consciousness.qualia_space_iit import QualiaSpace
from src.lifecycle.sleep_consolidation import SleepCycle
from src.lifecycle.circadian_rhythm import BiologicalClock
SAVE_FILE = 'savegame.pkl'

def load_config():
    with open('config/default_sim.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_god_mode_agi():
    config = load_config()
    base_energy = config['biology']['base_energy']
    target_world_type = config['ENV_CONFIG'].get('world_type', 'advanced')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'>>> АППАРАТНАЯ СИСТЕМА: Сознание загружается на {device}')
    torch.set_grad_enabled(False)
    load_saved_game = False
    saved_state = None
    if os.path.exists(SAVE_FILE):
        ans = input('Найден файл. Продолжить эволюцию? (y/n): ').strip().lower()
        if ans == 'y':
            try:
                with open(SAVE_FILE, 'rb') as f:
                    saved_state = pickle.load(f)
                load_saved_game = True
            except Exception as e:
                print(f'Ошибка загрузки: {e}')
    final_num_nodes = config['hardware']['num_nodes']
    if load_saved_game and saved_state is not None:
        final_num_nodes = saved_state['weights'].shape[0]
    world = AdvancedWorld2D(size=config['ENV_CONFIG']['world_size'])
    ui = GameUI(world_size=config['ENV_CONFIG']['world_size'])
    geometry = BrainGeometry(final_num_nodes)
    plasticity = StructuralPlasticity(final_num_nodes)
    time_perception = HusserlianTime(final_num_nodes, window_size=3)
    predictor = PredictiveMicrocircuits(final_num_nodes)
    workspace = PhaseCoupledWorkspace(final_num_nodes, device)
    thalamus = Thalamus(final_num_nodes, device)
    chemistry = EndocrineSystem()
    attractors = MoodAttractors()
    ego = EgoModel(expected_baseline=base_energy)
    clock = BiologicalClock(cycle_length_ticks=1500)
    sleep_system = SleepCycle()
    hippocampus = Hippocampus()
    free_will = FreeWillEngine(delay_ticks=3)
    pfc = PrefrontalCortex(final_num_nodes)
    semantics = SemanticMemory()
    qualia = QualiaSpace(final_num_nodes)
    motor_cortex_size = 100
    basal_ganglia = BasalGanglia(motor_cortex_size=motor_cortex_size, num_actions=5)
    ear = Cochlea(num_bands=32)
    activity_vector = torch.zeros(final_num_nodes, device=device)
    agent_energy = base_energy
    agent_water = base_energy
    agent_pain = 0.0
    tick = 0
    mood = 'calm'
    surprise = 0.0
    is_daydreaming = False
    expected_reward_prev = 0.0
    prev_human_pos = list(world.human_pos)
    world.last_agent_vocalization = np.zeros(32)
    world.isolation_ticks = 0
    if load_saved_game and saved_state is not None:
        print('>>> Перенос синапсов в VRAM...')
        scipy_w = saved_state['weights'].tocoo()
        scipy_i = saved_state['integrity'].tocoo()
        plasticity.indices = torch.tensor(np.vstack((scipy_w.row, scipy_w.col)), dtype=torch.long, device=device)
        plasticity.weights_values = torch.tensor(scipy_w.data, dtype=torch.float32, device=device)
        plasticity.integrity_values = torch.tensor(scipy_i.data, dtype=torch.float32, device=device)
        world.agent_pos = saved_state['world_pos']
        world.objects = saved_state['world_objects']
        world.natural_walls = saved_state.get('world_walls', [])
        world.human_pos = saved_state.get('world_human', [1, 1])
        world.inventory = saved_state['world_inv']
        world.human_inventory = saved_state.get('world_human_inv', 'empty')
        world.equipment = saved_state.get('world_equipment', {'pickaxe': 0, 'weapon': 0, 'armor': 0})
        world.chests_data = saved_state.get('world_chests', {})
        world.world_tick = saved_state['world_tick']
        tick = saved_state['tick']
    speeds = [1.0, 0.2, 0.1, 0.04, 0.016, 0.0]
    speed_names = ['1 FPS', '5 FPS', '10 FPS', '25 FPS', '60 FPS', 'MAX']
    speed_idx = 4
    print('>>> ЗАПУСК AGI НА GPU...')
    try:
        while True:
            tick += 1
            keys_pressed, quit_sim, speed_change = ui.handle_events()
            if speed_change == 1:
                speed_idx = min(5, speed_idx + 1)
            if speed_change == -1:
                speed_idx = max(0, speed_idx - 1)
            if quit_sim or keys_pressed.get('q', False):
                break
            if sleep_system.is_sleeping:
                world.process_human_input(keys_pressed)
                if tick % 50 == 0:
                    semantics.extract_concept_during_sleep(workspace.history_buffer, plasticity)
                cortisol_level = getattr(chemistry, 'cortisol', 0.0)
                still_sleeping = sleep_system.process_sleep_tick(hippocampus, plasticity, cortisol_level, activity_vector.cpu().numpy())
                if not still_sleeping:
                    clock.is_awake = True
                    clock.adenosine = 0.0
            else:
                energy_deficit = max(0, base_energy * 0.5 - agent_energy)
                water_deficit = max(0, base_energy * 0.5 - agent_water)
                stomach_limit = base_energy * 1.2
                gastric_stretch_energy = max(0, agent_energy - stomach_limit)
                gastric_stretch_water = max(0, agent_water - stomach_limit)
                gastric_pain = (gastric_stretch_energy + gastric_stretch_water) * 0.05
                agent_pain = agent_pain * 0.95 + gastric_pain
                agent_energy -= gastric_stretch_energy * 0.005
                agent_water -= gastric_stretch_water * 0.005
                h_ratio = min(1.0, agent_energy / base_energy)
                w_ratio = min(1.0, agent_water / base_energy)
                raw_sensory_cpu = np.zeros(final_num_nodes)
                retina_data = world.get_sensory_retina(final_num_nodes)
                raw_sensory_cpu[:len(retina_data)] = retina_data
                h_sig = min(1.0, energy_deficit / (base_energy * 0.5))
                w_sig = min(1.0, water_deficit / (base_energy * 0.5))
                p_sig = min(1.0, agent_pain / 100.0)
                raw_sensory_cpu[1010:1055] = h_sig
                raw_sensory_cpu[1055:1100] = w_sig
                raw_sensory_cpu[1100:1220] = p_sig
                ear.is_listening = keys_pressed.get('v', False)
                mic_audio_spectrum = ear.get_auditory_nerve_signal()
                world_sound = getattr(world, 'pop_world_sound', lambda: np.zeros(32))()
                raw_sensory_cpu[1300:1332] = mic_audio_spectrum + world.last_agent_vocalization + world_sound
                world.process_human_input(keys_pressed)
                human_dy = world.human_pos[0] - prev_human_pos[0]
                human_dx = world.human_pos[1] - prev_human_pos[1]
                prev_human_pos = list(world.human_pos)
                human_interacted = keys_pressed.get('e', False)
                mirror_strength = 0.8
                if human_dy < 0:
                    raw_sensory_cpu[1000] += mirror_strength
                if human_dy > 0:
                    raw_sensory_cpu[1001] += mirror_strength
                if human_dx < 0:
                    raw_sensory_cpu[1002] += mirror_strength
                if human_dx > 0:
                    raw_sensory_cpu[1003] += mirror_strength
                if human_interacted:
                    raw_sensory_cpu[1004] += mirror_strength
                    chemistry.trigger_social_bonding()
                if human_interacted and tick % 3 == 0:
                    world.human_interact()
                is_daydreaming = chemistry.boredom > 0.8
                if len(hippocampus.episodic_memory) > 0:
                    memory_flash = random.choice(hippocampus.episodic_memory)['pattern']
                    raw_sensory_cpu += memory_flash * 0.15
                raw_sensory_cpu = np.clip(raw_sensory_cpu, 0.0, 1.0)
                raw_sensory = torch.tensor(raw_sensory_cpu, dtype=torch.float32, device=device)
                sparse_brain = plasticity.get_sparse_weights()
                internal_thoughts = torch.sparse.mm(sparse_brain, activity_vector.unsqueeze(1)).squeeze(1)
                combined_signal = torch.clamp(raw_sensory + internal_thoughts * 0.1, 0.0, 1.0)
                surprise_tensor, fep_updated_state = predictor.process_inference_step(combined_signal, activity_vector, plasticity_rate=0.5)
                surprise = surprise_tensor.item()
                activity_vector = time_perception.create_conscious_now(combined_signal, fep_updated_state)
                activity_vector = torch.clamp(activity_vector, 0.0, 1.0)
                salient_signal = thalamus.filter_attention(activity_vector, chemistry.noradrenaline, chemistry.boredom)
                if salient_signal.any():
                    activity_vector = workspace.broadcast_via_synchrony(salient_signal, activity_vector)
                    if surprise > 5.0 or p_sig > 0.2 or np.sum(mic_audio_spectrum) > 1.0:
                        hippocampus.encode_episode(activity_vector.cpu().numpy(), chemistry.dopamine - p_sig)
                act_cpu = activity_vector.cpu().numpy()
                if tick % 10 == 0:
                    active_indices = np.where(act_cpu > 0.5)[0]
                    if len(active_indices) > 3:
                        sub_weights = plasticity.weights_values.cpu().numpy()
                        sub_indices = plasticity.indices.cpu().numpy()
                        csr_brain_cpu = sp.coo_matrix((sub_weights, (sub_indices[0], sub_indices[1])), shape=(final_num_nodes, final_num_nodes)).tocsr()
                        raw_qualia = qualia.calculate_qualia_shape(act_cpu, csr_brain_cpu)
                        qualia_shape = np.abs(raw_qualia).tolist() if isinstance(raw_qualia, np.ndarray) else [0.33, 0.33, 0.33]
                    else:
                        qualia_shape = [0.33, 0.33, 0.33]
                chemistry.update_state(global_arousal=np.mean(act_cpu), layer23_error_spikes=surprise, raw_pain_signal=p_sig, energy_ratio=h_ratio, water_ratio=w_ratio, auditory_spikes=mic_audio_spectrum)
                mood, _, _ = attractors.apply_attractor_pull(current_energy=agent_energy, current_stress=agent_pain, dopamine=chemistry.dopamine, base_energy=base_energy)
                if hasattr(plasticity, 'apply_cortisol_damage'):
                    plasticity.apply_cortisol_damage(getattr(chemistry, 'cortisol', 0.0))
                motor_signals = act_cpu[1000:1000 + motor_cortex_size]
                vocal_cords = act_cpu[1250:1282]
                if np.sum(vocal_cords) > 3.0:
                    world.last_agent_vocalization = vocal_cords.copy()
                    if hasattr(world, 'add_sound'):
                        world.add_sound(world.agent_pos, vocal_cords.copy() * 0.5)
                else:
                    world.last_agent_vocalization = np.zeros(32)
                spasm_intensity = np.mean(motor_signals)
                spasm_energy_cost = spasm_intensity * 5.0
                agent_energy -= spasm_energy_cost
                if spasm_intensity > 0.5:
                    agent_pain += spasm_intensity * 2.0
                motor_potentials = basal_ganglia.map_to_action_potentials(motor_signals)
                subconscious_action = free_will.unconscious_decision_making(motor_potentials, chemistry.noradrenaline)
                final_action = free_will.conscious_veto_and_awareness()
                results = {'energy': 0, 'water': 0, 'stress': 0}
                if final_action is not None and final_action < 5 and (not is_daydreaming):
                    results = world.execute_action(final_action)
                    agent_energy += results['energy']
                    agent_water += results['water']
                    agent_pain += results['stress']
                    dopamine_burst = 0.0
                    pain_burst = 0.0
                    if results['energy'] > 0 or results['water'] > 0:
                        if hasattr(chemistry, 'trigger_endorphin_rush'):
                            chemistry.trigger_endorphin_rush()
                        dopamine_burst = 1.0
                    if results['stress'] > 10.0:
                        free_will.update_somatic_markers(final_action, results['stress'])
                        pain_burst = min(1.0, results['stress'] / 100.0)
                    basal_ganglia.reinforce_learning(final_action, dopamine=chemistry.dopamine, pain=pain_burst)
                agent_energy -= 0.5
                agent_water -= 0.8
                ego.evaluate_self(agent_energy, agent_pain)
                if agent_energy <= 0 or agent_water <= 0:
                    print(f'\n>>> СМЕРТЬ ОРГАНИЗМА (Отказ метаболизма).')
                    if os.path.exists(SAVE_FILE):
                        os.remove(SAVE_FILE)
                    break
                pain_suppression = max(0.0, 1.0 - agent_pain / 100.0)
                plasticity_rate = chemistry.get_plasticity_multiplier() * pain_suppression
                plasticity.apply_stdp_learning(activity_vector, plasticity_rate)
                plasticity.synaptogenesis_and_pruning(activity_vector, agent_energy)
                if tick % 100 == 0:
                    plasticity.maintain_homeostasis()
                clock.update_clock(energy_spent=1.0 + spasm_energy_cost)
                if not clock.is_awake:
                    sleep_system.is_sleeping = True
            if tick % 2 == 0:
                stats = {'tick': tick, 'energy': agent_energy, 'water': agent_water, 'base_energy': base_energy, 'nodes': plasticity.active_limit, 'vfe': surprise, 'mood': 'ГРЁЗЫ (DMN)' if is_daydreaming else mood, 'phi': workspace.calculate_integration_metric(activity_vector), 'speed_name': speed_names[speed_idx], 'cortisol': getattr(chemistry, 'cortisol', 0.0), 'oxytocin': getattr(chemistry, 'oxytocin', 0.0), 'adrenaline': getattr(chemistry, 'adrenaline', 0.0), 'endorphins': getattr(chemistry, 'endorphins', 0.0), 'boredom': getattr(chemistry, 'boredom', 0.0), 'qualia': qualia_shape if 'qualia_shape' in locals() else [0.33, 0.33, 0.33], 'agency': ego.sense_of_agency}
                ui.render(world, stats)
            if speeds[speed_idx] > 0.0:
                time.sleep(speeds[speed_idx])
    except KeyboardInterrupt:
        print('\n>>> Экстренное прерывание.')
    finally:
        import pygame
        pygame.quit()
        if hasattr(ear, 'stream') and ear.stream is not None:
            ear.stream.stop()
        if agent_energy > 0 and agent_water > 0:
            try:
                print('>>> Извлечение мозга из VRAM для сохранения...')
                cpu_ind = plasticity.indices.cpu().numpy()
                cpu_val_w = plasticity.weights_values.cpu().numpy()
                cpu_val_i = plasticity.integrity_values.cpu().numpy()
                save_weights = sp.coo_matrix((cpu_val_w, (cpu_ind[0], cpu_ind[1])), shape=(final_num_nodes, final_num_nodes))
                save_integrity = sp.coo_matrix((cpu_val_i, (cpu_ind[0], cpu_ind[1])), shape=(final_num_nodes, final_num_nodes))
                state_to_save = {'world_type': target_world_type, 'tick': tick, 'energy': agent_energy, 'water': agent_water, 'weights': save_weights, 'integrity': save_integrity, 'active_limit': plasticity.active_limit, 'is_inhibitory': plasticity.is_inhibitory, 'world_pos': world.agent_pos, 'world_objects': world.objects, 'world_inv': world.inventory, 'world_tick': world.world_tick, 'world_walls': world.natural_walls, 'world_human': world.human_pos, 'world_human_inv': world.human_inventory, 'world_equipment': world.equipment, 'world_chests': world.chests_data}
                with open(SAVE_FILE, 'wb') as f:
                    pickle.dump(state_to_save, f)
                print('>>> Состояние мозга успешно сохранено.')
            except Exception as e:
                print(f'>>> Ошибка при сохранении: {e}')
if __name__ == '__main__':
    run_god_mode_agi()