import numpy as np
import gymnasium as gym
import cv2
from gymnasium import spaces
from gymnasium.utils import seeding
from collections import deque
from copy import deepcopy

from .forest import Forest
from .forest_env import STATE_H, STATE_W, T_HORIZON

# Building dimensions and damage constant
HOSPITAL_SIZE = (5, 5)
POWER_PLANT_SIZE = (4, 5)
APARTMENT_SIZE = (5, 6)
DAMAGE_PER_FIRE_CELL = 0.08

class EnergyGridForestFireEnv(gym.Env):
    """
    A forest fire environment with an energy grid.
    Buildings (hospitals, power plants, apartments) are placed on a grid.
    A power network connects these buildings, and rewards are computed based on their status.
    """

    def __init__(self, **env_kwargs):
        self.seed(env_kwargs.pop("seed"))
        self.reward = 0
        self.state = None
        self.t = 0

        self.render_modes = ("human", "rgb_array")
        self.num_hospitals = env_kwargs.pop("num_hospitals", 4)
        self.num_power_plants = env_kwargs.pop("num_power_plants", 2)
        self.num_apartments = env_kwargs.pop("num_apartments", 4)
        self.render_mode_ = env_kwargs.pop("render_mode_", None)
        self._resize = True

        self.forest = Forest(**env_kwargs)

        self._best_actions = []
        self._best_rewards = []
        
        # Building positions, health, and apartment color settings
        self.hospitals = []
        self.hospitals_health = []
        self.power_plants = []
        self.power_plants_health = []
        self.apartments = []
        self.apartments_health = []
        self.apartment_colors = []
        self.apartment_color_options = [
            [102, 178, 255],
            [200, 150, 230],
            [150, 255, 150],
            [255, 200, 150]
        ]

        # Precomputed damage bounds for each building type
        self.damage_bounds = {
            "hospitals": [],
            "power_plants": [],
            "apartments": []
        }
        
        # List of power network edges (each edge is a tuple of two building centers)
        self.power_network_edges = []
        
        # Initial building placement
        self._place_structures()
        if self.num_power_plants > 0:
            self._compute_power_network()

        # Action & observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=10, shape=(STATE_H, STATE_W), dtype=np.uint8)
        self._max_episode_steps = T_HORIZON

    # ---------------------------------------------------------------------
    #  Initialization: seeding, building placement, damage bounds, power net
    # ---------------------------------------------------------------------
    def seed(self, seed=None):
        """Seed the random number generator."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _compute_damage_bounds(self, building_pos, building_size):
        """
        Compute and return the damage bounds (x0, y0, x1, y1) for a building.
        These bounds represent the area around the building that we check for fire.
        """
        x, y = building_pos
        w, h = building_size
        x0 = max(0, x - 1)
        y0 = max(0, y - 1)
        x1 = min(STATE_W, x + w + 1)
        y1 = min(STATE_H, y + h + 1)
        return (x0, y0, x1, y1)

    def _place_building(self, available_positions, building_size, building_list):
        """
        Place a building of a given size at a random available position.
        Remove the occupied positions from available_positions.
        """
        while available_positions:
            x, y = self.np_random.choice(list(available_positions))
            if x + building_size[0] <= STATE_W and y + building_size[1] <= STATE_H:
                if all((x + dx, y + dy) in available_positions
                       for dx in range(building_size[0])
                       for dy in range(building_size[1])):
                    for dx in range(building_size[0]):
                        for dy in range(building_size[1]):
                            available_positions.discard((x + dx, y + dy))
                    building_list.append((x, y))
                    break

    def _place_structures(self):
        """
        Place hospitals, power plants, and apartments without overlap.
        Initialize each building's health to 1.0 and precompute their damage bounds.
        """
        available_positions = {(x, y) for x in range(STATE_W) for y in range(STATE_H)}
        self.hospitals = []
        self.hospitals_health = []
        self.power_plants = []
        self.power_plants_health = []
        self.apartments = []
        self.apartments_health = []
        self.apartment_colors = []

        self.damage_bounds["hospitals"] = []
        self.damage_bounds["power_plants"] = []
        self.damage_bounds["apartments"] = []
        
        # Hospitals
        for _ in range(self.num_hospitals):
            prev_count = len(self.hospitals)
            self._place_building(available_positions, HOSPITAL_SIZE, self.hospitals)
            if len(self.hospitals) > prev_count:
                self.hospitals_health.append(1.0)
                bounds = self._compute_damage_bounds(self.hospitals[-1], HOSPITAL_SIZE)
                self.damage_bounds["hospitals"].append(bounds)
        
        # Power plants
        for _ in range(self.num_power_plants):
            prev_count = len(self.power_plants)
            self._place_building(available_positions, POWER_PLANT_SIZE, self.power_plants)
            if len(self.power_plants) > prev_count:
                self.power_plants_health.append(1.0)
                bounds = self._compute_damage_bounds(self.power_plants[-1], POWER_PLANT_SIZE)
                self.damage_bounds["power_plants"].append(bounds)

        # Apartments
        for _ in range(self.num_apartments):
            prev_count = len(self.apartments)
            self._place_building(available_positions, APARTMENT_SIZE, self.apartments)
            if len(self.apartments) > prev_count:
                self.apartments_health.append(1.0)
                bounds = self._compute_damage_bounds(self.apartments[-1], APARTMENT_SIZE)
                self.damage_bounds["apartments"].append(bounds)
                color = self.np_random.choice(self.apartment_color_options)
                self.apartment_colors.append(color)

    def _compute_power_network(self):
        """
        Connect consumers (hospitals/apartments) to the nearest node in the set of power plants.
        """
        self.power_network_edges = []
        power_plant_centers = [self._get_building_center(pos, POWER_PLANT_SIZE) for pos in self.power_plants]
        hospital_centers = [self._get_building_center(pos, HOSPITAL_SIZE) for pos in self.hospitals]
        apartment_centers = [self._get_building_center(pos, APARTMENT_SIZE) for pos in self.apartments]
        
        distribution_edges = []
        network_nodes = power_plant_centers.copy()
        consumers = hospital_centers + apartment_centers
        while consumers:
            best_edge = None
            best_distance = float('inf')
            best_consumer = None
            for consumer in consumers:
                for node in network_nodes:
                    d = abs(consumer[0] - node[0]) + abs(consumer[1] - node[1])
                    if d < best_distance:
                        best_distance = d
                        best_edge = (node, consumer)
                        best_consumer = consumer
            distribution_edges.append(best_edge)
            network_nodes.append(best_consumer)
            consumers.remove(best_consumer)
        
        self.power_network_edges = distribution_edges

    def _get_building_center(self, top_left, size):
        """Return the (x, y) center coordinate of a building."""
        return (top_left[0] + size[0] // 2, top_left[1] + size[1] // 2)

    # ---------------------------------------------------------------------
    #  Step / Reset
    # ---------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """
        Reset the forest and buildings.
        Remove trees within building footprints, compute damage bounds, and compute the power network.
        """
        self.seed(seed)
        self.forest.reset()
        self.reward = 0
        self.t = 0
        self._place_structures()
        self._clear_trees_in_buildings()
        if self.num_power_plants > 0:
            self._compute_power_network()
        state = self.forest.world
        if state.shape != (STATE_H, STATE_W):
            state = self._scale(state, STATE_H, STATE_W)
        self.state = np.array(state) / self.forest.FIRE_CELL
        return self.state, {}

    @property
    def truncated(self):
        return self.t >= T_HORIZON

    def step(self, action):
        """
        Advance the simulation one step.
        Update building health and compute reward from trees, buildings, hospitals, and power.
        Also compute a best-action for debugging.
        """
        best_action, best_reward = self.compute_best_action()
        self._best_actions.append(best_action)
        self._best_rewards.append(best_reward)

        # Agent's action
        aimed_fire, is_fire = self.forest.step(action)
        self.t += 1

        # Update building health
        self._update_building_health()

        # Compute reward
        self.reward = self._compute_reward(aimed_fire, is_fire)

        # Build new observation
        state = self.forest.world
        if state.shape != (STATE_H, STATE_W):
            state = self._scale(state, STATE_H, STATE_W)
        self.state = np.array(state) / self.forest.FIRE_CELL

        terminated = False

        og_render_mode_ = self.render_mode_
        self.render_mode_ = "rgb_array"
        self._resize = False
        im = self.render()
        self.render_mode_ = og_render_mode_
        self._resize = True

        return im, self.reward, terminated, self.truncated, {
            "best_action": best_action,
            "best_reward": best_reward
        }
        # return self.state, self.reward, terminated, self.truncated, {
        #     "best_action": best_action,
        #     "best_reward": best_reward
        # }

    # ---------------------------------------------------------------------
    #  Fire Handling: clearing trees in buildings, updating building health
    # ---------------------------------------------------------------------
    def _clear_trees_in_buildings(self):
        """Remove tree cells within all building footprints."""
        for (x, y) in self.hospitals:
            w, h = HOSPITAL_SIZE
            region = self.forest.world[y:y+h, x:x+w]
            self.forest.world[y:y+h, x:x+w] = np.where(region == self.forest.TREE_CELL, 0, region)
        for (x, y) in self.power_plants:
            w, h = POWER_PLANT_SIZE
            region = self.forest.world[y:y+h, x:x+w]
            self.forest.world[y:y+h, x:x+w] = np.where(region == self.forest.TREE_CELL, 0, region)
        for (x, y) in self.apartments:
            w, h = APARTMENT_SIZE
            region = self.forest.world[y:y+h, x:x+w]
            self.forest.world[y:y+h, x:x+w] = np.where(region == self.forest.TREE_CELL, 0, region)

    def _compute_damage_for_building(self, damage_bounds, world):
        """Compute damage based on the number of fire cells in the precomputed damage region."""
        x0, y0, x1, y1 = damage_bounds
        region = world[y0:y1, x0:x1]
        fire_count = np.sum(region == self.forest.FIRE_CELL)
        return DAMAGE_PER_FIRE_CELL * fire_count

    def _update_building_health(self):
        """Reduce each building's health based on nearby fire using precomputed damage bounds."""
        world = self.forest.world
        for i in range(len(self.hospitals)):
            damage = self._compute_damage_for_building(self.damage_bounds["hospitals"][i], world)
            self.hospitals_health[i] = max(0.0, self.hospitals_health[i] - damage)
        for i in range(len(self.power_plants)):
            damage = self._compute_damage_for_building(self.damage_bounds["power_plants"][i], world)
            self.power_plants_health[i] = max(0.0, self.power_plants_health[i] - damage)
        for i in range(len(self.apartments)):
            damage = self._compute_damage_for_building(self.damage_bounds["apartments"][i], world)
            self.apartments_health[i] = max(0.0, self.apartments_health[i] - damage)

    def _update_building_health_copy(self, forest_copy, hospitals_health_copy,
                                     power_plants_health_copy, apartments_health_copy):
        """Like _update_building_health but for a forest copy used in candidate simulations."""
        world = forest_copy.world
        # Hospitals
        for i in range(len(self.hospitals)):
            damage = self._compute_damage_for_building(self.damage_bounds["hospitals"][i], world)
            hospitals_health_copy[i] = max(0.0, hospitals_health_copy[i] - damage)
        # Power Plants
        for i in range(len(self.power_plants)):
            damage = self._compute_damage_for_building(self.damage_bounds["power_plants"][i], world)
            power_plants_health_copy[i] = max(0.0, power_plants_health_copy[i] - damage)
        # Apartments
        for i in range(len(self.apartments)):
            damage = self._compute_damage_for_building(self.damage_bounds["apartments"][i], world)
            apartments_health_copy[i] = max(0.0, apartments_health_copy[i] - damage)
        return hospitals_health_copy, power_plants_health_copy, apartments_health_copy

    # ---------------------------------------------------------------------
    #  Best-Action Search: Brute Force & Conv
    # ---------------------------------------------------------------------
    def compute_best_action(self, alg="conv"):
        """Pick which method to compute the best action."""
        if alg == "brute":
            return self.compute_best_action_brute()
        elif alg == "conv":
            return self.compute_best_action_conv()
        else:
            raise AssertionError(f"Unknown algorithm: {alg}")

    def compute_best_action_brute(self):
        """
        Perform a brute-force one-step lookahead over interior cells only,
        skipping corners by a margin = extinguisher_size/2 in each dimension.
        """
        global_rng_state = np.random.get_state()

        # Copy the environment state
        health_copies = {
            'hospitals': list(self.hospitals_health),
            'power_plants': list(self.power_plants_health),
            'apartments': list(self.apartments_health)
        }

        # Figure out the margin
        w_ratio = self.forest.extinguisher_ratio
        margin_x = int((self.forest.world.shape[1] * w_ratio) / 2)
        margin_y = int((self.forest.world.shape[0] * w_ratio) / 2)

        best_reward = -np.inf
        best_action = None

        for y in range(margin_y, self.forest.world.shape[0] - margin_y):
            for x in range(margin_x, self.forest.world.shape[1] - margin_x):
                a = self.forest._xy_to_action(x, y)
                candidate_reward = self._simulate_candidate(a, health_copies, global_rng_state)
                if candidate_reward > best_reward:
                    best_reward = candidate_reward
                    best_action = a

        np.random.set_state(global_rng_state)
        return best_action, best_reward

    def compute_best_action_conv(self):
        """
        Find the best action by convolving a weighted 'fire map' with a kernel
        the size of the extinguisher area. Fires near buildings get extra weight
        by reusing self.damage_bounds, which already has a +1 margin.

        Steps:
        1) Build a weight_map where:
            - each burning cell gets +2
            - each neighbor of a burning cell gets +1
            - an extra +3 if that burning cell is within the damage_bounds region
            of any building (i.e. near that building)
        2) Convolve weight_map with a kernel of shape (extinguisher_h, extinguisher_w)
            in 'same' mode.
        3) The highest value in the interior region (skipping corners by a margin)
            is the best spot to extinguish.

        Returns:
        (best_action, best_value)
        """
        import cv2  # Ensure cv2 is installed and available

        rows, cols = self.forest.world.shape
        FIRE = self.forest.FIRE_CELL

        # 1) Build the weighted fire map
        w_map = np.zeros_like(self.forest.world, dtype=np.float32)

        # Mark cells that are on fire with +2
        on_fire = (self.forest.world == FIRE)
        w_map[on_fire] += 2.0

        for i in range(len(self.hospitals)):
            x0, y0, x1, y1 = self.damage_bounds["hospitals"][i]
            w_map[y0:y1, x0:x1] += (on_fire[y0:y1, x0:x1] * 3.0)

        for i in range(len(self.power_plants)):
            x0, y0, x1, y1 = self.damage_bounds["power_plants"][i]
            w_map[y0:y1, x0:x1] += (on_fire[y0:y1, x0:x1] * 3.0)

        for i in range(len(self.apartments)):
            x0, y0, x1, y1 = self.damage_bounds["apartments"][i]
            w_map[y0:y1, x0:x1] += (on_fire[y0:y1, x0:x1] * 3.0)

        # 2) Convolve in 'same' mode with a kernel sized to the extinguisher rectangle
        ext_w = int(self.forest.world.shape[1] * self.forest.extinguisher_ratio)
        ext_h = int(self.forest.world.shape[0] * self.forest.extinguisher_ratio)
        kernel = np.ones((ext_h, ext_w), dtype=np.float32)

        conv_map = cv2.filter2D(w_map, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # 3) Find the best location in the interior region (skip corners by margin)
        w_ratio = self.forest.extinguisher_ratio
        margin_x = int(cols * w_ratio / 2)
        margin_y = int(rows * w_ratio / 2)

        best_val = -1e9
        best_yx = (margin_y, margin_x)
        for y in range(margin_y, rows - margin_y):
            for x in range(margin_x, cols - margin_x):
                val = conv_map[y, x]
                if val > best_val:
                    best_val = val
                    best_yx = (y, x)

        # Convert that location back to an action
        best_y, best_x = best_yx
        best_action = self.forest._xy_to_action(best_x, best_y)


        # calculate best reward
        global_rng_state = np.random.get_state()
        health_copies = {
            'hospitals': list(self.hospitals_health),
            'power_plants': list(self.power_plants_health),
            'apartments': list(self.apartments_health)
        }
        best_reward = self._simulate_candidate(best_action, health_copies, global_rng_state)
        np.random.set_state(global_rng_state)

        return best_action, best_reward


    def _simulate_candidate(self, a, health_copies, rng_state):
        """
        Simulate a candidate action 'a' with a one-step lookahead on a copy of the environment.
        """
        np.random.set_state(rng_state)
        forest_copy = deepcopy(self.forest)
        hospitals_health_copy = health_copies['hospitals'].copy()
        power_plants_health_copy = health_copies['power_plants'].copy()
        apartments_health_copy = health_copies['apartments'].copy()

        aimed_fire, is_fire = forest_copy.step(a)
        hospitals_health_copy, power_plants_health_copy, apartments_health_copy = \
            self._update_building_health_copy(forest_copy,
                                              hospitals_health_copy,
                                              power_plants_health_copy,
                                              apartments_health_copy)
        return self._compute_reward(aimed_fire, is_fire)

    # ---------------------------------------------------------------------
    #  Reward Functions
    # ---------------------------------------------------------------------
    def _compute_reward(self, aimed_fire, is_fire):
        tree_reward = self._compute_tree_reward(aimed_fire, is_fire)
        building_reward = self._compute_building_reward()
        hospital_reward = self._compute_hospital_reward()
        power_reward = self._compute_power_reward()
        return tree_reward + building_reward + hospital_reward + power_reward

    def _compute_tree_reward(self, aimed_fire, is_fire):
        tree_reward = aimed_fire - is_fire
        if self.truncated:
            if np.mean(self.forest.world) > 0.5 * self.forest.p_init_tree:
                tree_reward += 100
            else:
                tree_reward -= 100
        return tree_reward

    def _compute_building_reward(self):
        if (self.num_apartments + self.num_power_plants) == 0:
            return 0
        total_building_health = sum(self.power_plants_health) + sum(self.apartments_health)
        building_reward = 5 * total_building_health
        if self.truncated:
            if total_building_health / (self.num_power_plants + self.num_apartments) > 0.8:
                building_reward += 200
            else:
                building_reward -= 200
        return building_reward

    def _compute_power_reward(self):
        if self.num_power_plants == 0:
            return 0
        powered_nodes, node_status = self._compute_powered_nodes()
        power_reward = sum(2 if node in powered_nodes else -2 for node in node_status)
        if self.truncated:
            total_buildings = self.num_power_plants + self.num_apartments + self.num_hospitals
            if len(powered_nodes) / total_buildings > 0.5:
                power_reward += 400
            else:
                power_reward -= 400
        return power_reward

    def _compute_hospital_reward(self):
        if self.num_hospitals == 0:
            return 0
        total_hospital_health = sum(self.hospitals_health)
        hospital_reward = 10 * total_hospital_health
        if self.truncated:
            if total_hospital_health / self.num_hospitals > 0.8:
                hospital_reward += 300
            else:
                hospital_reward -= 300
        return hospital_reward

    def _compute_powered_nodes(self):
        """
        Determine which buildings are powered based on their health and connectivity.
        Returns (powered_nodes, node_status).
        """
        node_status = {}
        # Power plants
        for i, pos in enumerate(self.power_plants):
            center = self._get_building_center(pos, POWER_PLANT_SIZE)
            node_status[center] = ('plant', self.power_plants_health[i])
        # Hospitals
        for i, pos in enumerate(self.hospitals):
            center = self._get_building_center(pos, HOSPITAL_SIZE)
            node_status[center] = ('hospital', self.hospitals_health[i])
        # Apartments
        for i, pos in enumerate(self.apartments):
            center = self._get_building_center(pos, APARTMENT_SIZE)
            node_status[center] = ('apartment', self.apartments_health[i])
        
        # Build adjacency list
        graph = {}
        for edge in self.power_network_edges:
            u, v = edge
            graph.setdefault(u, []).append(v)
            graph.setdefault(v, []).append(u)
        
        # BFS from any power plant with health > 0
        powered_nodes = set()
        queue = deque()
        for node, (btype, health) in node_status.items():
            if btype == 'plant' and health > 0:
                powered_nodes.add(node)
                queue.append(node)
        while queue:
            current = queue.popleft()
            for neighbor in graph.get(current, []):
                if neighbor not in powered_nodes and neighbor in node_status and node_status[neighbor][1] > 0:
                    powered_nodes.add(neighbor)
                    queue.append(neighbor)
        
        return powered_nodes, node_status

    # ---------------------------------------------------------------------
    #  Rendering
    # ---------------------------------------------------------------------
    def render(self):
        """
        Render the environment: draw trees, fire, buildings, and the power network.
        Uses translucent rectangles for agent's action & best action.
        """
        im = np.zeros((STATE_H, STATE_W, 3), dtype=np.uint8)
        im[self.forest.world == self.forest.TREE_CELL] = [34, 139, 34]   # Green for trees
        im[self.forest.world == self.forest.FIRE_CELL] = [0, 165, 255]  # Orange for fire

        # Draw buildings
        for (x, y), health in zip(self.hospitals, self.hospitals_health):
            self._draw_hospital(im, x, y, health)
        for (x, y), health in zip(self.power_plants, self.power_plants_health):
            self._draw_power_plant(im, x, y, health)
        for (x, y), health, base_color in zip(self.apartments, self.apartments_health, self.apartment_colors):
            self._draw_apartment(im, x, y, base_color, health)

        # Draw power network
        powered_nodes, node_status = self._compute_powered_nodes()
        for (start, end) in self.power_network_edges:
            color = [200, 0, 0] if (start in powered_nodes and end in powered_nodes) else [128, 128, 128]
            self._draw_manhattan_line(im, start, end, color)

        # Prepare an overlay for translucent rectangle outlines
        overlay = im.copy()

        # 1) Draw the agent's action rectangle (if any)
        if self.forest.action_rect is not None:
            (x1, y1), (x2, y2) = self.forest.action_rect
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)

        # 2) Draw the best action rectangle in a different color
        if len(self._best_actions) > 0:
            p1, p2 = self.forest._compute_action_rect(self._best_actions[-1])
            x1_b, y1_b = p1
            x2_b, y2_b = p2
            cv2.rectangle(overlay, (x1_b, y1_b), (x2_b, y2_b), (52, 255, 235), thickness=1)

        # Blend overlay with the original image to get translucent outlines
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)

        # Resize for clearer viewing
        if self._resize:
            im = cv2.resize(im, (640, 640), interpolation=cv2.INTER_NEAREST)

        if self.render_mode_ == 'human':
            cv2.imshow("EnergyGridForestFireEnv", im)
            cv2.waitKey(50)
        elif self.render_mode_ == 'rgb_array':
            return im

    def _draw_hospital(self, im, x, y, health):
        """Draw a hospital with a red '+' symbol and damage overlay."""
        height, width = HOSPITAL_SIZE[1], HOSPITAL_SIZE[0]
        building_img = np.zeros((height, width, 3), dtype=np.uint8)
        building_img[:] = [240, 240, 240]
        building_img[height // 2, 1:width - 1] = [0, 0, 220]   # horizontal '+'
        building_img[1:height - 1, width // 2] = [0, 0, 220]   # vertical '+'
        building_img = self._apply_damage_overlay(building_img, health)
        im[y:y + height, x:x + width] = building_img

    def _draw_power_plant(self, im, x, y, health):
        """Draw a power plant with a lightning bolt and damage overlay."""
        height, width = POWER_PLANT_SIZE[1], POWER_PLANT_SIZE[0]
        building_img = np.zeros((height, width, 3), dtype=np.uint8)
        building_img[:] = [70, 70, 70]
        if height > 3 and width > 2:
            building_img[3, 1] = [0, 255, 255]
            building_img[2, 1] = [0, 255, 255]
            building_img[2, 2] = [0, 255, 255]
            building_img[1, 2] = [0, 255, 255]
        building_img = self._apply_damage_overlay(building_img, health)
        im[y:y + height, x:x + width] = building_img

    def _draw_apartment(self, im, x, y, base_color, health):
        """Draw an apartment with a simple pixel-art style pattern and damage overlay."""
        height, width = APARTMENT_SIZE[1], APARTMENT_SIZE[0]
        building_img = np.zeros((height, width, 3), dtype=np.uint8)
        windows_color = [255, 255, 255]
        door_color = [42, 42, 165]
        apartment_art = [
            [base_color,           base_color,     base_color,      base_color,      base_color],
            [base_color,           windows_color,  base_color,      windows_color,   base_color],
            [base_color,           base_color,     base_color,      base_color,      base_color],
            [base_color,           windows_color,  base_color,      windows_color,   base_color],
            [base_color,           base_color,     base_color,      base_color,      base_color],
            [base_color,           base_color,     door_color,      base_color,      base_color],
        ]
        for dy in range(height):
            for dx in range(width):
                building_img[dy, dx] = apartment_art[dy][dx]
        building_img = self._apply_damage_overlay(building_img, health)
        im[y:y + height, x:x + width] = building_img

    def _apply_damage_overlay(self, building_img, health, overlay_color=[0, 0, 150], alpha=0.5):
        """Blend an overlay color to indicate damage (1-health) on top portion of the building."""
        height = building_img.shape[0]
        overlay_rows = int((1 - health) * height)
        for row in range(overlay_rows):
            building_img[row, :] = (
                (1 - alpha) * building_img[row, :].astype(np.float32) +
                alpha * np.array(overlay_color, dtype=np.float32)
            ).astype(np.uint8)
        return building_img

    def _draw_manhattan_line(self, im, start, end, color=[200, 0, 0]):
        """Draw an L-shaped line from start to end, skipping over building interiors."""
        sx, sy = start
        ex, ey = end
        step_x = 1 if ex >= sx else -1
        for x in range(sx, ex + step_x, step_x):
            if not self._inside_building(x, sy):
                im[sy, x] = color
        step_y = 1 if ey >= sy else -1
        for y in range(sy, ey + step_y, step_y):
            if not self._inside_building(ex, y):
                im[y, ex] = color

    def _inside_building(self, x, y):
        """Return True if (x, y) is inside any placed building."""
        # Hospitals
        for (hx, hy) in self.hospitals:
            if hx <= x < hx + HOSPITAL_SIZE[0] and hy <= y < hy + HOSPITAL_SIZE[1]:
                return True
        # Power Plants
        for (px, py) in self.power_plants:
            if px <= x < px + POWER_PLANT_SIZE[0] and py <= y < py + POWER_PLANT_SIZE[1]:
                return True
        # Apartments
        for (ax, ay) in self.apartments:
            if ax <= x < ax + APARTMENT_SIZE[0] and ay <= y < ay + APARTMENT_SIZE[1]:
                return True
        return False

    def _scale(self, im, height, width):
        """
        Scale the image (2D array) to the specified dimensions.
        """
        original_height, original_width = im.shape
        return [
            [
                im[int(original_height * r / height)][int(original_width * c / width)]
                for c in range(width)
            ]
            for r in range(height)
        ]
