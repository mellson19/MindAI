import pygame
import numpy as np
import math

class GameUI:

    def __init__(self, world_size=40, tile_size=20):
        pygame.init()
        self.tile_size = tile_size
        self.world_size = world_size
        self.map_width = world_size * tile_size
        self.ui_width = 350
        self.width = self.map_width + self.ui_width
        self.height = world_size * tile_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('AGI Evolution - Active Inference Sandbox')
        self.font = pygame.font.SysFont('Consolas', 16, bold=True)
        self.title_font = pygame.font.SysFont('Consolas', 22, bold=True)
        self.colors = {'bg': (30, 30, 30), 'grid': (40, 40, 40), 'ui_bg': (20, 20, 25), 'agent': (0, 255, 255), 'eye': (255, 255, 0), 'human': (255, 100, 100), 'wall': (100, 100, 100), 'water': (50, 150, 255), 'food': (50, 200, 50), 'herb': (100, 255, 150), 'poison': (150, 0, 150), 'tree': (34, 139, 34), 'stone': (169, 169, 169), 'copper': (210, 105, 30), 'iron': (220, 220, 220), 'wolf': (200, 0, 0), 'rabbit': (255, 255, 255), 'meat': (255, 120, 120), 'leather': (139, 69, 19), 'campfire': (255, 140, 0), 'forge': (80, 80, 80), 'chest': (160, 82, 45), 'boulder': (139, 115, 85), 'track': (80, 0, 0), 'switch': (200, 200, 0), 'door': (180, 130, 50), 'built_wall': (120, 90, 90)}

    def render(self, world, stats):
        self.screen.fill(self.colors['bg'])
        for y in range(self.world_size):
            for x in range(self.world_size):
                rect = (x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size)
                pygame.draw.rect(self.screen, self.colors['grid'], rect, 1)

        def draw_objs(obj_list, color_key, radius=1.0, is_circle=False):
            for p in obj_list:
                cx = p[1] * self.tile_size + self.tile_size // 2
                cy = p[0] * self.tile_size + self.tile_size // 2
                r = int(self.tile_size // 2 * radius)
                if is_circle:
                    pygame.draw.circle(self.screen, self.colors[color_key], (cx, cy), r)
                else:
                    rect = (p[1] * self.tile_size + 2, p[0] * self.tile_size + 2, self.tile_size - 4, self.tile_size - 4)
                    pygame.draw.rect(self.screen, self.colors[color_key], rect)
        draw_objs(world.objects.get('water', []), 'water')
        draw_objs(world.objects.get('tracks', []), 'track', 0.4, True)
        draw_objs(world.objects.get('switches', []), 'switch', 0.8)
        if hasattr(world, 'logic_gates'):
            for g in world.logic_gates:
                if not g['is_open']:
                    draw_objs([g['door']], 'door')
        draw_objs(world.natural_walls, 'wall')
        draw_objs(world.objects.get('built_walls', []), 'built_wall')
        draw_objs(world.objects.get('food', []), 'food', 0.6, True)
        draw_objs(world.objects.get('herb', []), 'herb', 0.5, True)
        draw_objs(world.objects.get('poison', []), 'poison', 0.6, True)
        draw_objs(world.objects.get('tree', []), 'tree')
        draw_objs(world.objects.get('stone', []), 'stone', 0.8)
        draw_objs(world.objects.get('copper_ore', []), 'copper', 0.7)
        draw_objs(world.objects.get('iron_ore', []), 'iron', 0.7)
        draw_objs(world.objects.get('meat', []), 'meat', 0.5)
        draw_objs(world.objects.get('leather', []), 'leather', 0.5)
        draw_objs(world.objects.get('boulders', []), 'boulder', 0.9, True)
        draw_objs(world.objects.get('campfires', []), 'campfire')
        draw_objs(world.objects.get('forges', []), 'forge')
        draw_objs(world.objects.get('chests', []), 'chest')
        draw_objs(world.objects.get('wolves', []), 'wolf', 0.8, True)
        draw_objs(world.objects.get('rabbits', []), 'rabbit', 0.6, True)
        draw_objs([world.human_pos], 'human', 0.8, True)
        draw_objs([world.agent_pos], 'agent', 0.9, True)
        eye_y, eye_x = (world.agent_pos[0] + world.eye_offset[0], world.agent_pos[1] + world.eye_offset[1])
        pygame.draw.rect(self.screen, self.colors['eye'], (eye_x * self.tile_size, eye_y * self.tile_size, self.tile_size, self.tile_size), 2)
        if world.is_night:
            overlay = pygame.Surface((self.map_width, self.height))
            overlay.set_alpha(150)
            overlay.fill((0, 0, 10))
            self.screen.blit(overlay, (0, 0))
        pygame.draw.rect(self.screen, self.colors['ui_bg'], (self.map_width, 0, self.ui_width, self.height))
        y_offset = 15

        def draw_text(text, color=(255, 255, 255), is_title=False):
            nonlocal y_offset
            f = self.title_font if is_title else self.font
            surface = f.render(text, True, color)
            self.screen.blit(surface, (self.map_width + 15, y_offset))
            y_offset += 25

        def draw_bar(label, value, max_val, color):
            nonlocal y_offset
            draw_text(f'{label}: {int(value)}/{int(max_val)}')
            bar_w = 300
            pygame.draw.rect(self.screen, (50, 50, 50), (self.map_width + 15, y_offset, bar_w, 15))
            fill_w = max(0, min(bar_w, int(value / max_val * bar_w)))
            pygame.draw.rect(self.screen, color, (self.map_width + 15, y_offset, fill_w, 15))
            y_offset += 20
        draw_text('ИИ с сознанием', (0, 255, 255), True)
        draw_text(f'ДЕНЬ: {world.day} | {('НОЧЬ' if world.is_night else 'ДЕНЬ')} | Тик: {stats['tick']}', (255, 200, 100))
        y_offset += 5
        draw_text('--- ВНУТРЕННИЙ МИР (QUALIA) ---', (150, 150, 150))
        center_x = self.map_width + self.ui_width // 2
        center_y = y_offset + 50
        max_radius = 40
        agency = stats.get('agency', 1.0)
        ego_color = (int(255 * agency), int(255 * agency), int(255 * agency))
        ego_radius = int(max_radius * 1.3)
        pygame.draw.circle(self.screen, ego_color, (center_x, center_y), ego_radius, max(1, int(agency * 3)))
        qualia_shape = stats.get('qualia', [0.33, 0.33, 0.33])
        if np.sum(qualia_shape) > 0:
            angles = [0, 2 * math.pi / 3, 4 * math.pi / 3]
            points = []
            for i in range(3):
                r = max_radius * (qualia_shape[i] * 2.0)
                px = center_x + int(r * math.sin(angles[i]))
                py = center_y - int(r * math.cos(angles[i]))
                points.append((px, py))
            adr = stats.get('adrenaline', 0)
            oxt = stats.get('oxytocin', 0)
            end = stats.get('endorphins', 0)
            q_color = (min(255, int(adr * 255) + 50), min(255, int(oxt * 255) + 50), min(255, int(end * 255) + 150))
            if 'ГРЁЗЫ' in stats['mood']:
                q_color = (150, 50, 200)
            pygame.draw.polygon(self.screen, q_color, points)
            pygame.draw.polygon(self.screen, (255, 255, 255), points, 1)
        y_offset += 105
        draw_text(f'Эго (Агентность): {agency:.2f}', ego_color)
        draw_text(f'Настроение:       {stats['mood']}')
        y_offset += 10
        draw_text('--- ЭНДОКРИНОЛОГИЯ ---', (150, 150, 150))
        draw_text(f'Окситоцин (Доверие): {stats.get('oxytocin', 0):.2f}', (100, 255, 100))
        draw_text(f'Кортизол (Травма):   {stats.get('cortisol', 0):.2f}', (150, 150, 50))
        draw_text(f'Адреналин (Ярость):  {stats.get('adrenaline', 0):.2f}', (255, 100, 100))
        draw_text(f'Эндорфины (Экстаз):  {stats.get('endorphins', 0):.2f}', (100, 200, 255))
        vfe_color = (255, 100, 100) if stats['vfe'] > 5.0 else (100, 255, 100)
        draw_text(f'Тревога (VFE Ошибка): {stats['vfe']:.2f}%', vfe_color)
        y_offset += 10
        draw_text('--- ФИЗИОЛОГИЯ ---', (150, 150, 150))
        draw_bar('Энергия', stats['energy'], stats['base_energy'], (50, 255, 50))
        draw_bar('Вода', stats['water'], stats['base_energy'], (50, 150, 255))
        draw_text('--- ИНВЕНТАРЬ ---', (150, 150, 150))
        draw_text(f'ИИ держит: {world.inventory.upper()}', (0, 255, 255))
        p_lvl = ['Кулаки', 'Каменная', 'Медная', 'Железная'][world.equipment['pickaxe']]
        draw_text(f'Кирка ИИ:  {p_lvl}')
        draw_text(f'ВЫ держите: {world.human_inventory.upper()}', (255, 100, 100))
        p_lvl = ['Кулаки', 'Каменная', 'Медная', 'Железная'][world.equipment['pickaxe']]
        draw_text(f'Кирка: {p_lvl}')
        pygame.draw.rect(self.screen, (40, 40, 40), (self.map_width, self.height - 40, self.ui_width, 40))
        font_small = pygame.font.SysFont('Consolas', 12)
        ctrl_surf = font_small.render('[Q] Выход | [W/A/S/D] Ходить | [V] Голос', True, (150, 150, 150))
        self.screen.blit(ctrl_surf, (self.map_width + 10, self.height - 25))
        pygame.display.flip()

    def handle_events(self):
        keys_pressed = {'w': False, 'a': False, 's': False, 'd': False, 'q': False, 'v': False}
        quit_sim = False
        speed_change = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_sim = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    speed_change = 1
                elif event.key == pygame.K_DOWN:
                    speed_change = -1
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            keys_pressed['w'] = True
        if keys[pygame.K_s]:
            keys_pressed['s'] = True
        if keys[pygame.K_a]:
            keys_pressed['a'] = True
        if keys[pygame.K_d]:
            keys_pressed['d'] = True
        if keys[pygame.K_e]:
            keys_pressed['e'] = True
        if keys[pygame.K_q]:
            keys_pressed['q'] = True
        if keys[pygame.K_v]:
            keys_pressed['v'] = True
        return (keys_pressed, quit_sim, speed_change)