import pygame
import numpy as np
import os
from src.config import CONFIG

class Renderer:
    def __init__(self, env):
        self.env = env
        pygame.init()
        pygame.display.set_caption("AI Railway Traffic Controller")
        self.screen = pygame.display.set_mode((CONFIG["SCREEN_WIDTH"], CONFIG["SCREEN_HEIGHT"]))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 22)
        self.font_medium = pygame.font.Font(None, 28)

        # Camera attributes
        self.camera_offset = np.array([0.0, 0.0])
        self.zoom_level = 1.0
        self.dragging = False
        self.drag_start_mouse_pos = np.array([0.0, 0.0])
        self.drag_start_camera_offset = np.array([0.0, 0.0])

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False # Signal to exit
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button for dragging
                    self.dragging = True
                    self.drag_start_mouse_pos = np.array(event.pos)
                    self.drag_start_camera_offset = self.camera_offset.copy()
                elif event.button == 4:  # Scroll up to zoom in
                    self.zoom_level *= 1.1
                elif event.button == 5:  # Scroll down to zoom out
                    self.zoom_level /= 1.1
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    mouse_pos = np.array(event.pos)
                    delta = mouse_pos - self.drag_start_mouse_pos
                    self.camera_offset = self.drag_start_camera_offset + delta
        return True

    def _world_to_screen(self, world_pos):
        screen_center = np.array([CONFIG["SCREEN_WIDTH"] / 2, CONFIG["SCREEN_HEIGHT"] / 2])
        return (np.array(world_pos, dtype=float) - screen_center) * self.zoom_level + screen_center + self.camera_offset

    def render(self):
        self.screen.fill(CONFIG["COLOR_BACKGROUND"])
        self._draw_world_objects()
        self._draw_ui_overlay() # Draw UI on top of the world
        pygame.display.flip()
        self.clock.tick(self.env.metadata["render_fps"])

    def _draw_world_objects(self):
        # Draw tracks
        for edge in CONFIG["EDGES"]:
            start_pos = self._world_to_screen(CONFIG["NODES"][edge[0]])
            end_pos = self._world_to_screen(CONFIG["NODES"][edge[1]])
            
            # --- FIX: Ensure direction vector is float ---
            direction = end_pos.astype(float) - start_pos.astype(float)
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction /= distance # Now safe to do in-place
                normal = np.array([-direction[1], direction[0]])
                offset = normal * 3 * self.zoom_level
                pygame.draw.line(self.screen, CONFIG["COLOR_TRACK"], start_pos - offset, end_pos - offset, int(2 * self.zoom_level))
                pygame.draw.line(self.screen, CONFIG["COLOR_TRACK"], start_pos + offset, end_pos + offset, int(2 * self.zoom_level))
                
                num_sleepers = int(distance / (15 * self.zoom_level))
                if num_sleepers > 0:
                    for i in range(num_sleepers + 1):
                        pos = start_pos + (direction * i / num_sleepers)
                        pygame.draw.line(self.screen, CONFIG["COLOR_TRACK"], pos - offset*1.5, pos + offset*1.5, int(3 * self.zoom_level))

        # Draw nodes
        for node_name, pos in CONFIG["NODES"].items():
            screen_pos = self._world_to_screen(pos)
            color = CONFIG["COLOR_STATION"] if "STATION" in node_name else CONFIG["COLOR_TRACK"]
            pygame.draw.circle(self.screen, color, screen_pos, int(10 * self.zoom_level))

        # Draw signals
        for node_name in CONFIG["SIGNAL_NODES"]:
            screen_pos = self._world_to_screen(CONFIG["NODES"][node_name])
            color = CONFIG["COLOR_SIGNAL_GREEN"] if self.env.signals[node_name] == "GREEN" else CONFIG["COLOR_SIGNAL_RED"]
            pygame.draw.circle(self.screen, color, screen_pos, int(15 * self.zoom_level))

        # --- NEW: Simplified, Unified Train Drawing Logic ---
        # This new approach draws all trains using the same logic to ensure they are
        # always perfectly on their high-resolution path.
        for train in self.env.trains.values():
            if train.status == "FINISHED":
                continue
            
            # The train's current_pos is now always on the high-res path.
            # We draw it exactly there, with no complex offsets.
            self._draw_train_rect(train)


    def _draw_train_rect(self, train):
        """Draws a single train rectangle, perfectly aligned to its path."""
        
        # Calculate direction for rotation based on the high-res path
        direction = np.array([0., -1.]) # Default up
        if train.path_progress_idx < len(train.high_res_path) - 1:
            # Vector from current point to next point on the smooth path
            p1 = train.high_res_path[train.path_progress_idx]
            p2 = train.high_res_path[train.path_progress_idx + 1]
            vec = p2 - p1
            dist = np.linalg.norm(vec)
            if dist > 0:
                direction = vec / dist
        
        angle = np.degrees(np.arctan2(-direction[1], direction[0]))

        # Create rotated rectangle
        w, h = int(15 * self.zoom_level), int(30 * self.zoom_level)
        box_points = np.array([[-h/2, -w/2], [h/2, -w/2], [h/2, w/2], [-h/2, w/2]])
        rotation_matrix = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))], 
                                    [np.sin(np.radians(angle)),  np.cos(np.radians(angle))]])
        rotated_points = box_points @ rotation_matrix.T
        
        # Position the rectangle exactly at the train's current position
        screen_pos = self._world_to_screen(train.current_pos)
        screen_points = rotated_points + screen_pos

        # Draw the train and its ID
        pygame.draw.polygon(self.screen, train.color, screen_points)
        pygame.draw.polygon(self.screen, (200,200,200), screen_points, 1)
        id_text = self.font_small.render(train.id, True, CONFIG["COLOR_TEXT"])
        self.screen.blit(id_text, screen_pos + np.array([15, -15]))

    def _draw_ui_overlay(self):
        """Draws the UI panels for decisions and schedule."""
        # --- LATEST DECISION & STATS PANEL ---
        panel_height = 160
        panel_rect = pygame.Rect(20, self.screen.get_height() - panel_height - 20, 500, panel_height)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, (80, 80, 80), panel_rect, 2, border_radius=10)
        
        y_offset = panel_rect.y + 12
        x_offset = panel_rect.x + 15 # Added margin
        max_text_width = panel_rect.width - 30
        
        title_text = self.font_medium.render("LATEST AI DECISION", True, (220, 220, 220))
        self.screen.blit(title_text, (x_offset, y_offset))
        y_offset += 30

        action_str = f"ACTION: {self.env.last_decision['action']}"
        action_text = self.font_small.render(action_str, True, CONFIG["COLOR_TRAIN_EXPRESS"])
        self.screen.blit(action_text, (x_offset, y_offset))
        y_offset += 22

        rationale_str = f"RATIONALE: {self.env.last_decision['rationale']}"
        y_offset = self._blit_wrapped_text(rationale_str, self.font_small, CONFIG["COLOR_TEXT"], x_offset, y_offset, max_text_width)

        # --- NEW: Add Throughput to UI ---
        y_offset += 10 # Spacer
        pygame.draw.line(self.screen, (80, 80, 80), (x_offset, y_offset), (panel_rect.right - 15, y_offset), 1)
        y_offset += 10

        gain = ((self.env.current_throughput / self.env.cfg['BASELINE_THROUGHPUT']) - 1) * 100 if self.env.cfg['BASELINE_THROUGHPUT'] > 0 else 0
        throughput_str = f"Throughput: {self.env.current_throughput:.1f} Trains/Hour ({gain:+.0f}%)"
        throughput_text = self.font_small.render(throughput_str, True, (200, 200, 200))
        self.screen.blit(throughput_text, (x_offset, y_offset))


        # --- UPCOMING SCHEDULE PANEL ---
        schedule_panel_rect = pygame.Rect(self.screen.get_width() - 320, 10, 310, 210)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), schedule_panel_rect, border_radius=10)

        y_offset = schedule_panel_rect.y + 15
        title_text = self.font_medium.render("UPCOMING TRAINS", True, (200, 200, 200))
        self.screen.blit(title_text, (schedule_panel_rect.x + 15, y_offset))
        y_offset += 35

        # Display next 5 trains from schedule
        for i, train in enumerate(self.env.schedule[:5]):
            dep_seconds = int(train.ideal_departure_time)
            hh = dep_seconds // 3600
            mm = (dep_seconds % 3600) // 60
            schedule_str = f"{train.id} ({train.type}) @ {hh:02d}:{mm:02d}"
            schedule_text = self.font_small.render(schedule_str, True, train.color)
            self.screen.blit(schedule_text, (schedule_panel_rect.x + 15, y_offset))
            y_offset += 25

    def _blit_wrapped_text(self, text, font, color, x, y, max_width):
        """Render multiline word-wrapped text. Returns new y after drawing."""
        words = text.split()
        line = ""
        for word in words:
            test_line = f"{line} {word}" if line else word
            if font.size(test_line)[0] <= max_width:
                line = test_line
            else:
                surf = font.render(line, True, color)
                self.screen.blit(surf, (x, y))
                y += surf.get_height() + 2
                line = word
        if line:
            surf = font.render(line, True, color)
            self.screen.blit(surf, (x, y))
            y += surf.get_height()
        return y

    def close(self):
        pygame.quit()
