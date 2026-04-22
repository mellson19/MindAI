import numpy as np

class RetinaUniversal:

    def __init__(self, vision_radius=5):
        self.vision_radius = vision_radius
        self.grid_size = vision_radius * 2 + 1
        self.channels = 5
        self.max_visual_nodes = self.grid_size ** 2 * self.channels
        self.prev_frame_luma = np.zeros((self.grid_size, self.grid_size))
        self.physical_materials = {'wolves': [0.4, 0.4, 0.4, 0.0], 'rabbits': [0.9, 0.9, 0.9, 0.0], 'human': [0.8, 0.6, 0.5, 0.0], 'food': [0.9, 0.1, 0.1, 0.0], 'poison': [0.6, 0.1, 0.8, 0.0], 'water': [0.1, 0.4, 0.9, 0.0], 'campfires': [1.0, 0.5, 0.0, 1.0], 'forges': [0.3, 0.3, 0.3, 0.4], 'natural_walls': [0.5, 0.5, 0.5, 0.0], 'built_walls': [0.6, 0.4, 0.2, 0.0], 'tree': [0.1, 0.5, 0.1, 0.0], 'stone': [0.6, 0.6, 0.6, 0.0], 'herb': [0.2, 0.7, 0.2, 0.0], 'meat': [0.8, 0.3, 0.3, 0.0]}
        self.render_priority = ['wolves', 'human', 'rabbits', 'poison', 'campfires', 'forges', 'water', 'meat', 'food', 'herb', 'stone', 'tree', 'built_walls']

    def get_visual_array(self, world, num_nodes) -> np.ndarray:
        retina = np.zeros(self.max_visual_nodes)
        current_frame_luma = np.zeros((self.grid_size, self.grid_size))
        focus_y = world.agent_pos[0] + world.eye_offset[0]
        focus_x = world.agent_pos[1] + world.eye_offset[1]
        near_light = False
        light_sources = world.objects.get('campfires', []) + world.objects.get('forges', [])
        for src in light_sources:
            if abs(src[0] - world.agent_pos[0]) + abs(src[1] - world.agent_pos[1]) <= 4:
                near_light = True
                break
        ambient_light = 1.0 if not world.is_night else 0.1
        if near_light:
            ambient_light = 1.0
        for dy in range(-self.vision_radius, self.vision_radius + 1):
            for dx in range(-self.vision_radius, self.vision_radius + 1):
                if abs(dy) + abs(dx) > self.vision_radius + 1:
                    continue
                ry = dy + self.vision_radius
                rx = dx + self.vision_radius
                idx_base = (ry * self.grid_size + rx) * self.channels
                if idx_base >= self.max_visual_nodes:
                    continue
                y, x = (focus_y + dy, focus_x + dx)
                pos = [y, x]
                found_obj = None
                if not (0 <= y < world.size and 0 <= x < world.size):
                    found_obj = 'natural_walls'
                elif pos in world.natural_walls:
                    found_obj = 'natural_walls'
                elif hasattr(world, 'human_pos') and pos == world.human_pos:
                    found_obj = 'human'
                else:
                    for obj_name in self.render_priority:
                        if obj_name in world.objects and pos in world.objects[obj_name]:
                            found_obj = obj_name
                            break
                if found_obj and found_obj in self.physical_materials:
                    mat = self.physical_materials[found_obj]
                    r, g, b = (mat[0], mat[1], mat[2])
                    emission = mat[3]
                    final_luma = min(1.0, emission + ambient_light)
                    r_final = r * final_luma
                    g_final = g * final_luma
                    b_final = b * final_luma
                    luma_gray = 0.299 * r_final + 0.587 * g_final + 0.114 * b_final
                    current_frame_luma[ry, rx] = max(luma_gray, emission)
                    motion_signal = abs(current_frame_luma[ry, rx] - self.prev_frame_luma[ry, rx])
                    motion_signal = min(1.0, motion_signal * 3.0)
                    retina[idx_base:idx_base + self.channels] = [r_final, g_final, b_final, current_frame_luma[ry, rx], motion_signal]
        self.prev_frame_luma = current_frame_luma.copy()
        full_sensory = np.zeros(num_nodes)
        full_sensory[:len(retina)] = retina
        return full_sensory