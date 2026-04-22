import numpy as np
import random
from src.environment.vision_system import RetinaUniversal

class AdvancedWorld2D:

    def __init__(self, size=60):
        self.size = size
        self.agent_pos = [size // 2, size // 2]
        self.human_pos = [size // 2 - 1, size // 2]
        self.eye_offset = [0, 0]
        self.objects = {'food': [], 'water': [], 'poison': [], 'herb': [], 'stone': [], 'tree': [], 'copper_ore': [], 'iron_ore': [], 'meat': [], 'cooked_meat': [], 'leather': [], 'seeds': [], 'crops': [], 'wolves': [], 'rabbits': [], 'campfires': [], 'built_walls': [], 'forges': [], 'chests': [], 'tracks': [], 'boulders': [], 'switches': [], 'doors': [], 'ice': []}
        self.natural_walls = []
        self.logic_gates = []
        self.inventory = 'empty'
        self.human_inventory = 'empty'
        self.chests_data = {}
        self.equipment = {'pickaxe': 0, 'weapon': 0, 'armor': 0}
        self.object_health = {}
        self.meat_age = {}
        self.world_sound_buffer = np.zeros(32)
        self.world_tick = 0
        self.day = 1
        self.is_night = False
        self.wolves_spawned = False
        self.season = 'summer'
        self.vision = RetinaUniversal()
        self._generate_terrain()

    def add_sound(self, source_pos, sound_vector):
        dist = abs(self.agent_pos[0] - source_pos[0]) + abs(self.agent_pos[1] - source_pos[1])
        if dist < 15:
            volume = max(0.0, 1.0 - dist / 15.0)
            self.world_sound_buffer += np.array(sound_vector) * volume

    def pop_world_sound(self):
        sound = np.clip(self.world_sound_buffer, 0.0, 1.0)
        self.world_sound_buffer = np.zeros(32)
        return sound

    def human_interact(self):
        if self.human_inventory == 'wood':
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = (self.human_pos[0] + dy, self.human_pos[1] + dx)
                if 0 <= ny < self.size and 0 <= nx < self.size:
                    if not self._is_solid([ny, nx]):
                        self.objects['built_walls'].append([ny, nx])
                        self.human_inventory = 'empty'
                        build_snd = np.zeros(32)
                        build_snd[10:15] = 0.5
                        self.add_sound(self.human_pos, build_snd)
                        print('👨 Человек построил стену!')
                        return
        elif self.human_inventory == 'empty':
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                check_pos = [self.human_pos[0] + dy, self.human_pos[1] + dx]
                if check_pos in self.objects['tree']:
                    self.objects['tree'].remove(check_pos)
                    self.human_inventory = 'wood'
                    crack = np.zeros(32)
                    crack[20:30] = 0.9
                    self.add_sound(self.human_pos, crack)
                    print('👨 Человек срубил дерево!')
                    return

    def _generate_terrain(self):
        for y in range(self.size):
            self.natural_walls.append([y, 0])
            self.natural_walls.append([y, self.size - 1])
        for x in range(self.size):
            self.natural_walls.append([0, x])
            self.natural_walls.append([self.size - 1, x])
        for _ in range(150):
            self._spawn('natural_walls')
        for _ in range(6):
            cy, cx = (random.randint(5, self.size - 6), random.randint(5, self.size - 6))
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if abs(dy) + abs(dx) <= 3 and random.random() > 0.3:
                        p = [cy + dy, cx + dx]
                        if p not in self.natural_walls:
                            self.objects['water'].append(p)
        for _ in range(40):
            self._spawn('food')
        for _ in range(20):
            self._spawn('poison')
        for _ in range(15):
            self._spawn('herb')
        for _ in range(40):
            self._spawn('tree')
        for _ in range(30):
            self._spawn('stone')
        for _ in range(15):
            self._spawn('copper_ore')
        for _ in range(8):
            self._spawn('iron_ore')
        for _ in range(15):
            self._spawn('rabbits')

    def _get_free_pos(self):
        while True:
            p = [random.randint(1, self.size - 2), random.randint(1, self.size - 2)]
            if p != self.agent_pos and p != self.human_pos and (not self._is_solid(p)) and (p not in self.objects['water']):
                return p

    def _spawn(self, obj_type):
        self.objects.get(obj_type, self.natural_walls).append(self._get_free_pos())

    def process_human_input(self, keys_pressed):
        self.world_tick += 1
        if self.world_tick % 3 != 0:
            return
        new_pos = list(self.human_pos)
        moved = False
        if keys_pressed['w']:
            new_pos[0] -= 1
            moved = True
        elif keys_pressed['s']:
            new_pos[0] += 1
            moved = True
        elif keys_pressed['a']:
            new_pos[1] -= 1
            moved = True
        elif keys_pressed['d']:
            new_pos[1] += 1
            moved = True
        if moved and 0 <= new_pos[0] < self.size and (0 <= new_pos[1] < self.size):
            if not self._is_solid(new_pos):
                self.human_pos = new_pos
                step_snd = np.zeros(32)
                step_snd[5:8] = 0.5
                self.add_sound(self.human_pos, step_snd)

    def _is_solid(self, pos):
        if pos[0] < 0 or pos[0] >= self.size or pos[1] < 0 or (pos[1] >= self.size):
            return True
        return pos in self.natural_walls or pos in self.objects['built_walls'] or pos in self.objects['forges'] or (pos in self.objects['chests']) or (pos in self.objects['ice'])

    def _god_hand_rescue(self):
        if not self._is_solid(self.agent_pos):
            return
        for radius in range(1, 15):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = (self.agent_pos[0] + dy, self.agent_pos[1] + dx)
                    if 0 <= ny < self.size and 0 <= nx < self.size:
                        if not self._is_solid([ny, nx]) and [ny, nx] not in self.objects['water']:
                            self.agent_pos = [ny, nx]
                            return

    def _safe_place_object(self, obj_list_key):
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ny, nx = (self.agent_pos[0] + dy, self.agent_pos[1] + dx)
            if 0 <= ny < self.size and 0 <= nx < self.size:
                if not self._is_solid([ny, nx]):
                    self.objects[obj_list_key].append(list(self.agent_pos))
                    self.agent_pos = [ny, nx]
                    self.inventory = 'empty'
                    return True
        return False

    def _simulate_npc(self):
        self.day = self.world_tick // 1000 + 1
        self.is_night = self.world_tick % 1000 > 500
        year_cycle = self.world_tick % 6000
        is_winter_now = year_cycle > 3000
        if is_winter_now and self.season == 'summer':
            self.season = 'winter'
            print('>>> НАСТУПИЛА ЗИМА! Трава погибла, вода замерзла.')
            self.objects['ice'] = list(self.objects['water'])
            self.objects['water'].clear()
            self.objects['food'].clear()
            self.objects['herb'].clear()
        elif not is_winter_now and self.season == 'winter':
            self.season = 'summer'
            print('>>> НАСТУПИЛО ЛЕТО! Лед растаял.')
            self.objects['water'] = list(self.objects['ice'])
            self.objects['ice'].clear()
            for _ in range(40):
                self._spawn('food')
            for _ in range(15):
                self._spawn('herb')
        if self.day >= 3 and (not self.wolves_spawned):
            for _ in range(3):
                self._spawn('wolves')
            self.wolves_spawned = True
        if self.world_tick % 5 == 0:
            for meat_pos in list(self.objects['meat']):
                mt_pos_t = tuple(meat_pos)
                self.meat_age[mt_pos_t] = self.meat_age.get(mt_pos_t, 0) + 1
                if self.meat_age[mt_pos_t] > 100:
                    if meat_pos in self.objects['meat']:
                        self.objects['meat'].remove(meat_pos)
                        self.objects['poison'].append(meat_pos)
                    del self.meat_age[mt_pos_t]
        if self.world_tick % 20 == 0:
            new_fires = []
            for fire in self.objects['campfires']:
                fire_snd = np.zeros(32)
                fire_snd[2:5] = 0.4
                self.add_sound(fire, fire_snd)
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = [fire[0] + dy, fire[1] + dx]
                    if neighbor in self.objects['tree'] and random.random() < 0.02:
                        self.objects['tree'].remove(neighbor)
                        new_fires.append(neighbor)
                        print('    [ЭКОЛОГИЯ] Лесной пожар!')
            self.objects['campfires'].extend(new_fires)
        wolf_speed = 3 if self.is_night or self.season == 'winter' else 6
        if self.world_tick % wolf_speed == 0:
            for i, wolf in enumerate(self.objects['wolves']):
                self.objects['tracks'].append(list(wolf))
                if random.random() < 0.05:
                    howl_snd = np.zeros(32)
                    howl_snd[10:15] = 0.8
                    self.add_sound(wolf, howl_snd)
                near_fire = any((abs(c[0] - wolf[0]) + abs(c[1] - wolf[1]) <= 3 for c in self.objects['campfires']))
                if near_fire:
                    new_w = [wolf[0] + random.choice([-1, 1]), wolf[1] + random.choice([-1, 1])]
                else:
                    targets = [self.agent_pos] + self.objects['rabbits']
                    closest = min(targets, key=lambda t: abs(t[0] - wolf[0]) + abs(t[1] - wolf[1]))
                    new_w = list(wolf)
                    new_w[0] += 1 if closest[0] > wolf[0] else -1
                    new_w[1] += 1 if closest[1] > wolf[1] else -1
                if 0 <= new_w[0] < self.size and 0 <= new_w[1] < self.size and (not self._is_solid(new_w)) and (new_w not in self.objects['water']):
                    self.objects['wolves'][i] = new_w
        if self.world_tick % 2 == 0:
            for i, r in enumerate(self.objects['rabbits']):
                m = random.choice([[-1, 0], [1, 0], [0, -1], [0, 1]])
                new_r = [r[0] + m[0], r[1] + m[1]]
                if 0 <= new_r[0] < self.size and 0 <= new_r[1] < self.size and (not self._is_solid(new_r)) and (new_r not in self.objects['water']):
                    self.objects['rabbits'][i] = new_r

    def get_sensory_retina(self, num_nodes) -> np.ndarray:
        return self.vision.get_visual_array(self, num_nodes)

    def simulate_action(self, pos, action_idx, inventory):
        sim_pos = list(pos)
        if action_idx == 0:
            sim_pos[0] -= 1
        elif action_idx == 1:
            sim_pos[0] += 1
        elif action_idx == 2:
            sim_pos[1] -= 1
        elif action_idx == 3:
            sim_pos[1] += 1
        if self._is_solid(sim_pos):
            return 50.0
        if sim_pos in self.objects['wolves']:
            return 1000.0 if self.equipment['weapon'] == 0 else 100.0
        if sim_pos in self.objects['poison']:
            return 200.0
        if sim_pos in self.objects['campfires']:
            return 500.0
        return 0.0

    def _mine_object(self, obj_list_key, required_tool_level, hit_sound, break_sound, base_health, yield_item):
        if self.agent_pos in self.objects[obj_list_key] and self.inventory == 'empty':
            pos_t = tuple(self.agent_pos)
            damage = 1 + self.equipment['pickaxe']
            self.object_health[pos_t] = self.object_health.get(pos_t, base_health) - damage
            if self.object_health[pos_t] <= 0:
                if self.equipment['pickaxe'] >= required_tool_level:
                    self.inventory = yield_item
                self.objects[obj_list_key].remove(self.agent_pos)
                del self.object_health[pos_t]
                if obj_list_key in ['tree', 'stone']:
                    self._spawn(obj_list_key)
                self.add_sound(self.agent_pos, break_sound)
                return True
            else:
                self.add_sound(self.agent_pos, hit_sound)
                return False
        return False

    def execute_action(self, motor_idx: int) -> dict:
        self._god_hand_rescue()
        self._simulate_npc()
        energy_change, water_change, stress_change = (-0.1, -0.1, 0.0)
        if motor_idx in [0, 1, 2, 3]:
            energy_change -= 0.5 if self.season == 'winter' else 0.2
            new_pos = list(self.agent_pos)
            if motor_idx == 0:
                new_pos[0] -= 1
            elif motor_idx == 1:
                new_pos[0] += 1
            elif motor_idx == 2:
                new_pos[1] -= 1
            elif motor_idx == 3:
                new_pos[1] += 1
            if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
                if not self._is_solid(new_pos):
                    self.agent_pos = new_pos
                    walk_snd = np.zeros(32)
                    walk_snd[0:3] = 0.2
                    self.add_sound(self.agent_pos, walk_snd)
                elif new_pos in self.natural_walls:
                    stress_change += 5.0
                    wall_hit = np.zeros(32)
                    wall_hit[1:4] = 0.8
                    self.add_sound(self.agent_pos, wall_hit)
        elif motor_idx in [5, 6, 7, 8]:
            energy_change -= 0.05
            if motor_idx == 5:
                self.eye_offset[0] = max(-5, self.eye_offset[0] - 1)
            elif motor_idx == 6:
                self.eye_offset[0] = min(5, self.eye_offset[0] + 1)
            elif motor_idx == 7:
                self.eye_offset[1] = max(-5, self.eye_offset[1] - 1)
            elif motor_idx == 8:
                self.eye_offset[1] = min(5, self.eye_offset[1] + 1)
        elif motor_idx == 4:
            energy_change -= 1.0
            pos_t = tuple(self.agent_pos)
            thud = np.zeros(32)
            thud[1:5] = 0.7
            crack = np.zeros(32)
            crack[20:30] = 0.9
            clink = np.zeros(32)
            clink[25:31] = 0.9
            nearby_forge = None
            nearby_campfire = None
            near_chest_pos = None
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    check_p = [self.agent_pos[0] + dy, self.agent_pos[1] + dx]
                    if check_p in self.objects['forges']:
                        nearby_forge = check_p
                    if check_p in self.objects['campfires']:
                        nearby_campfire = check_p
                    if check_p in self.objects['chests']:
                        near_chest_pos = tuple(check_p)
            if nearby_campfire and self.inventory == 'meat':
                self.inventory = 'cooked_meat'
                print('    [КУЛЬТУРА] Мясо пожарено!')
                sizzle = np.zeros(32)
                sizzle[20:25] = 1.0
                self.add_sound(self.agent_pos, sizzle)
            elif near_chest_pos:
                stored = self.chests_data.get(near_chest_pos, 'empty')
                self.chests_data[near_chest_pos] = self.inventory
                self.inventory = stored
            elif nearby_forge:
                if self.inventory == 'wood':
                    self.equipment['pickaxe'] = max(self.equipment['pickaxe'], 1)
                    self.inventory = 'empty'
                elif self.inventory == 'copper_ore':
                    self.equipment['pickaxe'] = max(self.equipment['pickaxe'], 2)
                    self.inventory = 'empty'
                elif self.inventory == 'iron_ore':
                    self.equipment['weapon'] = 1
                    self.inventory = 'empty'
                elif self.inventory == 'leather':
                    self.equipment['armor'] = 1
                    self.inventory = 'empty'
            elif self.agent_pos in self.objects['water']:
                water_change += 250.0
                splash = np.zeros(32)
                splash[15:18] = 0.5
                self.add_sound(self.agent_pos, splash)
            elif self.agent_pos in self.objects['food']:
                energy_change += 250.0
                self.objects['food'].remove(self.agent_pos)
                if random.random() < 0.2 and self.inventory == 'empty':
                    self.inventory = 'seeds'
                self._spawn('food')
            elif self._mine_object('tree', 0, thud, crack, base_health=4, yield_item='wood'):
                pass
            elif self._mine_object('stone', 0, thud, clink, base_health=6, yield_item='stone'):
                pass
            elif self._mine_object('copper_ore', 1, thud, clink, base_health=10, yield_item='copper_ore'):
                pass
            elif self._mine_object('iron_ore', 2, thud, clink, base_health=15, yield_item='iron_ore'):
                pass
            elif self.agent_pos in self.objects['leather'] and self.inventory == 'empty':
                self.inventory = 'leather'
                self.objects['leather'].remove(self.agent_pos)
            elif self.agent_pos in self.objects['meat'] and self.inventory == 'empty':
                self.inventory = 'meat'
                self.objects['meat'].remove(self.agent_pos)
            elif self.agent_pos in self.objects['cooked_meat'] and self.inventory == 'empty':
                self.inventory = 'cooked_meat'
                self.objects['cooked_meat'].remove(self.agent_pos)
            elif self.inventory == 'wood':
                if self.is_night or self.season == 'winter':
                    self.objects['campfires'].append(list(self.agent_pos))
                    self.inventory = 'empty'
                else:
                    self._safe_place_object('built_walls')
            elif self.inventory == 'stone':
                self._safe_place_object('forges')
            elif self.inventory == 'copper_ore':
                if self._safe_place_object('chests'):
                    self.chests_data[tuple(self.objects['chests'][-1])] = 'empty'
            elif self.inventory == 'seeds':
                self.objects['crops'].append(list(self.agent_pos))
                self.inventory = 'empty'
            elif self.inventory == 'meat':
                energy_change += 200.0
                if random.random() < 0.3:
                    stress_change += 50.0
                self.inventory = 'empty'
            elif self.inventory == 'cooked_meat':
                energy_change += 800.0
                self.inventory = 'empty'
            elif self.inventory != 'empty':
                if self.inventory in self.objects:
                    self.objects[self.inventory].append(list(self.agent_pos))
                    self.inventory = 'empty'
        if self.agent_pos in self.objects['herb']:
            energy_change += 50.0
            stress_change -= 100.0
            self.objects['herb'].remove(self.agent_pos)
            self._spawn('herb')
        if self.agent_pos in self.objects['poison']:
            energy_change -= 100.0
            stress_change += 150.0
            self.objects['poison'].remove(self.agent_pos)
            self._spawn('poison')
        if self.agent_pos in self.objects['campfires']:
            stress_change += 100.0
        if self.season == 'winter':
            energy_change -= 1.5
            if nearby_campfire:
                energy_change += 1.0
        for w in self.objects['wolves']:
            if abs(w[0] - self.agent_pos[0]) + abs(w[1] - self.agent_pos[1]) <= 1:
                dmg = 30.0 if self.equipment['armor'] == 1 else 150.0
                energy_change -= dmg
                stress_change += dmg
                bite_snd = np.zeros(32)
                bite_snd[0:5] = 1.0
                self.add_sound(self.agent_pos, bite_snd)
                if self.equipment['weapon'] == 1:
                    self.objects['wolves'].remove(w)
                    self._spawn('wolves')
                    self.objects['leather'].append(list(self.agent_pos))
                    self.objects['meat'].append(list(self.agent_pos))
        return {'energy': energy_change, 'water': water_change, 'stress': stress_change}

    def get_render_string(self) -> str:
        return ''