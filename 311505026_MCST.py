import pygame
import math
import sys
import os 
import random
import itertools

pygame.init()

# Set environment variable to ensure the window opens at the center of the screen
os.environ['SDL_VIDEO_CENTERED'] = '1'

# Initialize Pygame
pygame.init()

# Get current screen resolution
screen_info = pygame.display.Info()
screen_width = screen_info.current_w
screen_height = screen_info.current_h

# Create a window
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Define colors and hexagon properties
BG_COLOR = (30, 30, 30)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
HEX_SIZE = 30
HEX_BORDER = 2
PIECE_RADIUS = int(HEX_SIZE * 0.8)

# Initialize font for text rendering
pygame.font.init()
font = pygame.font.SysFont(None, int(HEX_SIZE * 0.7))

# Initialize game state variables
hexagon_board = {}
selected_counts = {}
turn_ended = False
max_selected_counts = {}
initial_counts = {}  # Store initial counts for each label

def draw_player_turn_button(screen, button_text, message=""):
    """Draws a turn indicator button at the top right of the screen."""
    screen_width, screen_height = screen.get_size()
    font = pygame.font.SysFont(None, 36)  # Set the font size to 36
    button_width, button_height = 150, 50  # Width and height of the button
    button_x = screen_width - button_width - 10  # 10 pixels margin from the right edge
    button_y = 10  # 10 pixels margin from the top edge
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
    pygame.draw.rect(screen, (0, 0, 255), button_rect)  # Draw a blue button

    # Check if the button text is 'Game Over'
    if button_text == 'Game Over':
        text = button_text
    else:
        text = button_text + "'s turn"
    
    text_surf = font.render(text, True, pygame.Color('white'))
    screen.blit(text_surf, text_surf.get_rect(center=button_rect.center))

    return button_rect

def draw_hexagon(surface, x, y, size, border_color, fill_color, number=None, border_thickness=2, number_color=BLACK):
    """Draws a hexagon with optional number in its center."""
    angles_deg = [60 * i + 30 for i in range(6)]
    outer_points = [(x + (size + border_thickness) * math.cos(math.radians(angle)),
                     y + (size + border_thickness) * math.sin(math.radians(angle))) for angle in angles_deg]
    inner_points = [(x + size * math.cos(math.radians(angle)),
                     y + size * math.sin(math.radians(angle))) for angle in angles_deg]
    
    if fill_color == WHITE:
        border_color = GRAY  
        number_color = WHITE  
    elif fill_color == BLACK:
        border_color = WHITE  
        
    pygame.draw.polygon(surface, border_color, outer_points)
    pygame.draw.polygon(surface, fill_color, inner_points)

    if number is not None:
        text_surface = font.render(str(number), True, number_color)
        text_rect = text_surface.get_rect(center=(x, y))
        surface.blit(text_surface, text_rect)

def point_in_hex(x, y, hex_x, hex_y, size):
    """Check if the point (x, y) is inside the hexagon centered at (hex_x, hex_y)."""
    # Calculate relative position
    qx, qy = abs(x - hex_x) / size, abs(y - hex_y) / size

    # Check if the point is inside the hexagon
    if qx > 1.0 or qy > 0.866:
        return False
    return qx + 0.5 * qy <= 1.0

def draw_hex_shape_grid(surface, center_row, center_col, size):
    """Draws a grid of hexagons on the screen, labeled by distance from center."""
    global hexagon_board
    initial_counts.clear()  

    def get_hex_label(row, col, max_dist):
        dist_from_center = max(abs(row), abs(col), abs(row + col))
        if dist_from_center == 0:
            return 1
        label = 5
        
        corners = [(row, col) for row in (-max_dist, max_dist) for col in (-max_dist, max_dist)]
        corners.extend([(-max_dist, 0), (0, -max_dist), (max_dist, 0), (0, max_dist)])

        if (row, col) in corners:
            label = 6
        elif abs(row) == max_dist or abs(col) == max_dist or abs(row + col) == max_dist:
            label = 5
        elif abs(row) == max_dist - 1 or abs(col) == max_dist - 1 or abs(row + col) == max_dist - 1:
            label = 3
        elif abs(row) <= max_dist - 2 and abs(col) <= max_dist - 2 and abs(row + col) <= max_dist - 2:
            label = 2
       
        return label

    max_dist = center_row

    for row in range(-center_row, center_row + 1):
        for col in range(-center_col, center_col + 1):
            dist_from_center = max(abs(row), abs(col), abs(row + col))
            if dist_from_center <= center_row:
                x = WIDTH / 2 + (col + row / 2) * (math.sqrt(3) * (size + HEX_BORDER))
                y = HEIGHT / 2 + row * ((size + HEX_BORDER) * 1.5)
                label = get_hex_label(row, col, max_dist)
                
                # hexagon_board[(row, col)] = {'x': x, 'y': y, 'label': label, 'selected': False} 
                hexagon_board[(row, col)] = {
                                                'x': x,
                                                'y': y,
                                                'label': label,
                                                'selected': False,
                                                'owner': None  # Track which player has selected the hexagon
                                            }
                
                initial_counts[label] = initial_counts.get(label, 0) + 1

                draw_hexagon(surface, x, y, size, (255, 255, 255), (255, 228, 205), label)

def draw_piece(surface, center_x, center_y, color):
    """Draws a game piece at specified coordinates."""
    pygame.draw.circle(surface, color, (int(center_x), int(center_y)), PIECE_RADIUS)

def draw_end_turn_button(screen):
    # Draw a simple button on the screen and return its rect
    font = pygame.font.SysFont(None, 36)
    button_rect = pygame.Rect(650, 550, 150, 50)
    pygame.draw.rect(screen, (0, 0, 255), button_rect)  # Blue button
    text_surf = font.render('End Turn', True, pygame.Color('white'))
    screen.blit(text_surf, text_surf.get_rect(center=button_rect.center))
    return button_rect

def check_all_hexes_selected():
    """Checks if all hexes on the board have been selected."""
    return all(hex_info['selected'] for hex_info in hexagon_board.values())

def calculate_connected_areas(owner):
    """Calculates the largest connected area of hexes of the specified owner."""
    def dfs(row, col, visited):
        if (row, col) in visited or not (row, col) in hexagon_board or hexagon_board[(row, col)]['owner'] != owner:
            return 0
        visited.add((row, col))
        count = 1
        # 相鄰的六邊形的相對座標
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
        for dr, dc in directions:
            next_row, next_col = row + dr, col + dc
            count += dfs(next_row, next_col, visited)
        return count

    visited = set()
    max_area = 0
    for (row, col), info in hexagon_board.items():
        if (row, col) not in visited and info['owner'] == owner:
            area = dfs(row, col, visited)
            if area > max_area:
                max_area = area
    return max_area

def display_connected_areas():
    """Displays the size of the largest connected areas for black and white pieces."""
    black_area = calculate_connected_areas('black')  # 黑子的擁有者標籤為 'black'
    white_area = calculate_connected_areas('white')  # 白子的擁有者標籤為 'white'
    black_text = font.render(f"Black Area: {black_area}", True, pygame.Color('green'))
    white_text = font.render(f"White Area: {white_area}", True, WHITE)
    screen.blit(black_text, (10, 10))
    screen.blit(white_text, (10, 30))
    pygame.display.update()
    # 顯示結果
    print(f"Game Over: Black Area: {black_area}, White Area: {white_area}")


def update_selected_hexes(selected_hexes):
    """Updates the state and visual representation of selected hexes."""
    global current_turn, selected_counts
    for hex_info in selected_hexes:
        hex_info['selected'] = True
        hex_info['booked'] = True
        hex_info['owner'] = current_turn  # Update the owner when a hex is selected
        selected_counts[hex_info['label']] = selected_counts.get(hex_info['label'], 0) + 1
        fill_color = BLACK if current_turn == 'black' else WHITE
        draw_hexagon(screen, hex_info['x'], hex_info['y'], HEX_SIZE, (128, 128, 128), fill_color, hex_info['label'])
        pygame.display.flip()

def process_selections_by_round(x, y, current_round):
    """Processes player selections on the board based on the current round number."""
    global current_label, required_selections
    selected_hexes = []
    
    for (hx, hy), hex_info in hexagon_board.items():
        if point_in_hex(x, y, hex_info['x'], hex_info['y'], HEX_SIZE) and not hex_info.get('booked', False):
            if current_round == 1 and hex_info['label'] == 2 and selected_counts.get(2, 0) < 1:
                selected_hexes.append(hex_info)
                break  # In the first round, only one hexagon labeled as 2 is allowed to be selected.
            elif current_round > 1:
                if current_label is None:
                    current_label = hex_info['label']
                    required_selections = hex_info['label']
                if current_label == hex_info['label'] and selected_counts.get(current_label, 0) < required_selections:
                    selected_hexes.append(hex_info)
                if selected_counts.get(current_label, 0) >= required_selections:
                    break  # Stop selecting once the required number is reached.

    return selected_hexes

def display_remaining_hexes():
    """Calculates and displays the remaining number of hexes for each label on the terminal."""
    remaining_counts = {1: 0, 2: 0, 3: 0, 5: 0, 6: 0} # Initialize counters
    
    # Traverse the hexagon_board to update the remaining count for each label
    for hex_info in hexagon_board.values():
        if not hex_info['selected']:  # If the hex is not selected
            label = hex_info['label']
            if label in remaining_counts:
                remaining_counts[label] += 1
    
    # Output the remaining count for each label
    all_selected = True
    # print("Remaining hexes by label:")
    for label, count in sorted(remaining_counts.items()):
        # print(f"Label {label}: {count}")
        if count > 0:
            all_selected = False
            # break
    return all_selected
            
def auto_select_remaining_hexes(label, required_selections):
    """Automatically selects the remaining hexes for a label if the turn timer expires."""
    hexes_by_label = {}
    # Collect all unbooked hexes of the current label
    for pos, info in hexagon_board.items():
        if not info.get('booked', False) and info['label'] == label:
            hexes_by_label.setdefault(label, []).append(info)
    
    remaining_hexes = hexes_by_label.get(label, [])
    number_to_select = min(len(remaining_hexes), required_selections - selected_counts.get(label, 0))

    selected_hexes = random.sample(remaining_hexes, number_to_select) if number_to_select > 0 else []
    # Process selected hexes
    update_selected_hexes(selected_hexes)

    print(f"AI selected {number_to_select} hexes automatically due to timeout.")     

def Count_Hexagons_by_Owner():
    owner_count = {'None': 0, 'white': 0, 'black': 0}
    for hex_info in hexagon_board.values():
        owner = hex_info['owner']
        if owner is None:
            owner_count['None'] += 1
        else:
            owner_count[owner] += 1
    
    print(owner_count)


def end_current_round():
    """Ends the current round, checking if the conditions for round completion are met."""
    global current_label, current_turn, turn_ended, current_round, selected_counts, hexagon_board
       
    if current_label is None:
        if current_round != 1:
            print("No selections have been made this round. You must select at least one hex.")
            return  # Do not end the round if no selections have been made
    # # Before ending the round, calculate the remaining and selected counts
    remaining_hexes = len([info for info in hexagon_board.values() if info['label'] == current_label and not info.get('booked', False)])
    selected_hexes_count = selected_counts.get(current_label, 0)

    if (remaining_hexes == 0) and (selected_hexes_count + 1) == current_label:
        
        print(f"Ending round with sufficient selections for label: {current_label}.")
        display_remaining_hexes()
        reset_round_state()    
        return
    
    # Check if the round can be ended
    elif current_label is not None:
        
        if selected_hexes_count != current_label:
            
            print(f"You have not selected enough hexes of label: {current_label}. Needed: {current_label}, selected: {selected_hexes_count}.")
            return 
         
        elif remaining_hexes < current_label:
            if selected_hexes_count < remaining_hexes:
                print(f"Not enough hexes left for label: {current_label}. You need to select all {remaining_hexes} available hexes.")
                # Additional steps might be needed to cancel the originally booked hexes
                return  # Do not end the round if not all available hexes are selected when fewer than needed
        # All conditions are met to end the round
        elif selected_hexes_count < current_label:
            # Additional steps might be needed to cancel the originally booked hexes
            print(f"You have not selected enough hexes of label: {current_label}. Needed: {current_label}, selected: {selected_hexes_count}.")
    
    # All conditions for round completion are satisfied
    display_remaining_hexes()
    print(f"Ending round with sufficient selections for label: {current_label}.")
    reset_round_state()


def reset_round_state():
    """Resets the state of the round, preparing for the next one."""
    global current_label, current_turn, turn_ended, current_round, selected_counts
    current_label = None
    selected_counts = {}
    print("Round ended, it's now " + current_turn + "'s turn.")
    current_turn = 'white' if current_turn == 'black' else 'black'
    turn_ended = True
    draw_player_turn_button(screen, current_turn)
    current_round += 1
    pygame.display.flip()

def select_hexes_by_random(hexes_by_label, current_round):
    """Selects hexes randomly based on the current round and label availability."""
    selected_hexes = []
    if current_round == 1:
        # Special handling for the first round: only select one hexagon labeled as 2.
        if 2 in hexes_by_label and any(not hex_info['selected'] for _, hex_info in hexes_by_label[2]):
            available_hexes = [(pos, hex_info) for pos, hex_info in hexes_by_label[2] if not hex_info['selected']]
            if available_hexes:
                selected_hexes.append(random.choice(available_hexes))
    else:
        # Calculate remaining unselected hexes for each label and choose only those labels with remaining hexes
        available_labels = {
            label: hexes
            for label, hexes in hexes_by_label.items()
            if any(not hex_info['selected'] for _, hex_info in hexes)
        }
        if available_labels:
            selected_label = random.choice(list(available_labels.keys()))
            available_hexes = [(pos, hex_info) for pos, hex_info in available_labels[selected_label] if not hex_info['selected']]
            n = selected_label  # Determine the number of hexagons to select based on their labels.
        
            # Randomly select n hexes, select all remaining hexes if fewer than n are available
            if len(available_hexes) > n:
                selected_hexes.extend(random.sample(available_hexes, n))
            else:
                selected_hexes.extend(available_hexes)

        print(current_round)
    return selected_hexes

    
def check_timeout_and_autocomplete():
    """Checks for turn timeout and auto-completes selection if necessary."""
    global start_time
    current_time = pygame.time.get_ticks()
    if ((current_turn == 'black' and black_player_type == "human") or
        (current_turn == 'white' and white_player_type == "human")):
        if (current_time - start_time) > 100000:  # If the count exceeds 30, automatically complete the move.
            auto_select_remaining_hexes(current_label, required_selections)
            end_current_round()
            start_time = pygame.time.get_ticks()  # 重置起始時間    

def get_around(pos):
    enemy = 0
    frind = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
        x, y = pos
        x += dr
        y += dc
        if (x,y) in hexagon_board:
            if 'owner' in hexagon_board[(x,y)]:
                if hexagon_board[(x,y)]['owner'] != current_turn:
                    enemy += 1
                else:
                    frind += 1
    return enemy * 2 + frind

def get_liberty(pos):
    cnt = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
        x, y = pos
        x += dr
        y += dc
        if (x,y) in hexagon_board:
            if hexagon_board[(x,y)]['selected'] == False:
                cnt += 1
        if cnt == 6:
            return 4
        elif cnt == 5:
            return 3.5
        elif cnt == 4:
            return 3
        elif cnt == 3:
            return 2.5
        elif cnt == 2:
            return 2
        elif cnt == 1:
            return 1
        elif cnt == 0:
            return 0
        else:
            raise KeyError("get liberty error")

def get_label_point(pos):
    label = hexagon_board[pos]['label']
    if label == 1:
        return 5.5
    elif label == 2:
        return 5
    elif label == 3:
        return 4
    elif label == 5:
        return 2
    elif label == 6:
        return 1
    else:
        raise KeyError("get label point has weong label")

def get_near_point(pos):
    label = hexagon_board[pos]['label']
    if label == 5 or label == 6:
        return 2
    elif label == 3:
        return 1
    elif label == 2 or label == 1:
        return 0
    else:
        raise KeyError("get near point has wrong label")

def get_point(pos):
    point = 0.0
    point += get_label_point(pos)
    point += get_liberty(pos)
    point += get_around(pos)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
        x, y = pos
        x += dr
        y += dc
        if (x,y) in hexagon_board:
            point += get_near_point((x,y))
    return point

class TreeNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_possible_moves())

    def best_child(self, exploration_weight=1.4):
        choices_weights = [
            (c.reward / c.visits) + exploration_weight * math.sqrt((2 * math.log(self.visits) / c.visits))
            for c in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def most_visited_child(self):
        return max(self.children, key=lambda c: c.visits)

class MCTS:
    def __init__(self, iterations):
        self.iterations = iterations

    def search(self, initial_state):
        root = TreeNode(initial_state)

        for _ in range(self.iterations):
            node = self._select(root)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)

        return root.most_visited_child().state

    def _select(self, node):
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return self._expand(node)
            else:
                node = node.best_child()
        return node

    def _expand(self, node):
        tried_children_states = [child.state for child in node.children]
        new_state = random.choice([
            move for move in node.state.get_possible_moves()
            if move not in tried_children_states
        ])
        child_node = TreeNode(new_state, parent=node)
        node.children.append(child_node)
        return child_node

    def _simulate(self, state):
        while not state.is_terminal():
            state = random.choice(state.get_possible_moves())
        return state.get_reward()

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

# Example game state class
class HexagonBoardState:
    def __init__(self, board, color, depth):
        self.board = board
        self.color = color
        self.depth = depth

    def get_possible_moves(self):
        hexes_by_label = {}
        for pos, info in self.board.items():
            if not info.get('booked', False):
                label = info['label']
                if label in hexes_by_label:
                    hexes_by_label[label].append((pos, info))
                else:
                    hexes_by_label[label] = [(pos, info)]
        available_labels = {
            label: hexes
            for label, hexes in hexes_by_label.items()
            if any(not hex_info['selected'] for _, hex_info in hexes)
        }
        
        possible_moves = []
        for selected_label in available_labels.keys():
            combinations = []
            available_hexes = [(pos, hex_info) for pos, hex_info in available_labels[selected_label] if not hex_info['selected']]
            print(available_hexes)
            if (len(available_hexes) < selected_label):
                selected_hexes = []
                selected_hexes.extend(available_hexes)
                combinations.append(selected_hexes)
            else:
                combinations = itertools.combinations(available_hexes, selected_label)
            for selected_hex in combinations:
                new_board = self.board.copy()
                for pos, hex_info in selected_hex:
                    new_board[pos] = hex_info.copy()
                    new_board[pos]['selected'] = True
                    new_board[pos]['booked'] = True
                    new_board[pos]['owner'] = self.color
                next_color = 'black' if self.color == 'white' else 'white'
                new_state = HexagonBoardState(new_board, next_color, self.depth - 1)
                
                possible_moves.append(new_state)
                # possible_moves.append(HexagonBoardState(new_board, next_color, self.depth - 1))
                # print("Possible Move:")
                # for pos, hex_info in new_state.board.items():
                #     print(f"Position: {pos}, Info: {hex_info}")
                # print("---- End of Possible Move ----")
        return possible_moves

    def is_terminal(self):
        return self.depth == 0 or display_remaining_hexes()

    def get_reward(self):
        if self.color == 'black':
            return calculate_connected_areas('black')
        else:
            return calculate_connected_areas('white')

def select_hexes_by_AI(hexes_by_label, current_round):
    selected_hexes = []
    flag = 0
    if current_round == 1:
        if 2 in hexes_by_label and any(not hex_info['selected'] for _, hex_info in hexes_by_label[2]):
            available_hexes = [(pos, hex_info) for pos, hex_info in hexes_by_label[2] if not hex_info['selected']]
            max_point = 0
            select = None
            for hex in available_hexes:
                pos = hex[0]
                new = get_point(pos)
                if new > max_point:
                    max_point = new
                    select = hex
            selected_hexes.append(select)

    else:
        available_labels = {
            label: hexes
            for label, hexes in hexes_by_label.items()
            if any(not hex_info['selected'] for _, hex_info in hexes)
        }
        false_count_1 = 1
        false_count_2 = 1
        false_count_3 = 1
        false_count_5 = 1
        if 1 in available_labels.keys():
            false_count_1 = sum(1 for label in available_labels[1] if available_labels[1][0][1]['selected'] is False)
        if 2 in available_labels.keys():
            false_count_2 = sum(1 for label in available_labels[2] if available_labels[2][0][1]['selected'] is False)
        if 3 in available_labels.keys():
            false_count_3 = sum(1 for label in available_labels[3] if available_labels[3][0][1]['selected'] is False)
        if 5 in available_labels.keys():
            false_count_5 = sum(1 for label in available_labels[5] if available_labels[5][0][1]['selected'] is False)
        total = false_count_1 * math.comb(false_count_2, 2) * math.comb(false_count_3, 3) * math.comb(false_count_5, 5)
        if available_labels:
            if (total < 100):
                # Example usage
                initial_board_state = HexagonBoardState(hexagon_board, current_turn, 3)
                mcts = MCTS(iterations=1000)
                best_move_state = mcts.search(initial_board_state)
                best_move = best_move_state.board
                selected_hexes = best_move.items()
            elif 3 in available_labels.keys():
                available_hexes = [(pos, hex_info) for pos, hex_info in available_labels[3] if not hex_info['selected']]
                if len(available_hexes) > 9:
                    for _ in range(3):
                        max_point = 0
                        select = None
                        for hex in available_hexes:
                            pos = hex[0]
                            new = get_point(pos)
                            if new > max_point:
                                max_point = new
                                select = hex
                        selected_hexes.append(select)
                        available_hexes.remove(select)
                        pos = select[0]
                        hexagon_board[pos]['color'] = current_turn
                        # print(hexagon_board[pos])
                elif 5 in available_labels.keys():
                    available_hexes = [(pos, hex_info) for pos, hex_info in available_labels[5] if not hex_info['selected']]
                    if len(available_hexes) > 13:
                        for _ in range(5):
                            max_point = 0
                            select = None
                            for hex in available_hexes:
                                pos = hex[0]
                                new = get_point(pos)
                                if new > max_point:
                                    max_point = new
                                    select = hex
                            selected_hexes.append(select)
                            available_hexes.remove(select)
                            pos = select[0]
                            hexagon_board[pos]['color'] = current_turn
                    else:
                        max_point = 0
                        select = None
                        for selected_label in available_labels.keys():
                            available_hexes = [(pos, hex_info) for pos, hex_info in available_labels[selected_label] if not hex_info['selected']]
                            for hex in available_hexes:
                                pos = hex[0]
                                new = get_point(pos)
                                if new > max_point:
                                    max_point = new
                                    select = hex
                                    flag = 1
                else:
                    max_point = 0
                    select = None
                    for selected_label in available_labels.keys():
                        available_hexes = [(pos, hex_info) for pos, hex_info in available_labels[selected_label] if not hex_info['selected']]
                        for hex in available_hexes:
                            pos = hex[0]
                            new = get_point(pos)
                            if new > max_point:
                                max_point = new
                                select = hex
                                flag = 1
            else:
                max_point = 0
                select = None
                for selected_label in available_labels.keys():
                    available_hexes = [(pos, hex_info) for pos, hex_info in available_labels[selected_label] if not hex_info['selected']]
                    for hex in available_hexes:
                        pos = hex[0]
                        new = get_point(pos)
                        if new > max_point:
                            max_point = new
                            select = hex
                            flag = 1
            if flag == 1:
                pos = select[0]
                label = hexagon_board[pos]['label']
                available_hexes = [(pos, hex_info) for pos, hex_info in available_labels[label] if not hex_info['selected']]
                if (len(available_hexes) >= label):
                    for _ in range(label):
                        max_point = 0
                        select = None
                        for hex in available_hexes:
                            pos = hex[0]
                            new = get_point(pos)
                            if new > max_point:
                                max_point = new
                                select = hex
                        selected_hexes.append(select)
                        available_hexes.remove(select)
                        pos = select[0]
                        hexagon_board[pos]['color'] = current_turn
                else:
                    selected_hexes.extend(available_hexes)

    # print(available_hexes[0][0])
    # print(hexagon_board[(-2,0)])
    # print(hexagon_board[(0,0)])
    # print(get_grade(available_hexes[0]))
    return selected_hexes


def main(black_player, white_player):
    # Your existing implementation of the game logic here
    print(f"Player 1: {black_player}, Player 2: {white_player}")
    
    """Main game loop."""
    global selected_counts, current_label, current_turn, turn_ended, current_round, \
           start_time, required_selections, black_player_type, white_player_type, remaining_hexes, \
           hexagon_board, running  
    start_time = pygame.time.get_ticks()  # 記錄起始時間
    running = True
    current_round = 1
    current_label = None  # Track the label selected in the current round
    required_selections = 0  # Required selections based on the first selected label
    clock = pygame.time.Clock()
    black_player_type = black_player
    white_player_type = white_player
    current_turn = 'black'

    # Set the game window background color and update the display
    screen.fill(BG_COLOR)
    draw_hex_shape_grid(screen, 5, 5, HEX_SIZE)
    pygame.display.flip()

    # Draw the 'End Turn' button and update the display
    button_rect = draw_end_turn_button(screen)
    draw_player_turn_button(screen, current_turn)
    pygame.display.flip()
    
    # Define a custom event.
    KEEP_ALIVE_EVENT = pygame.USEREVENT + 1
    # Set up a timer to trigger a custom event every 1 second
    pygame.time.set_timer(KEEP_ALIVE_EVENT, 1000)
    
    
    # Start the main game loop
    while running:
        # check_timeout_and_autocomplete()
        # Handle all events in the game
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                # Exit the game if the window is closed
                pygame.quit()
                sys.exit()
            # elif event.type == KEEP_ALIVE_EVENT:
            #     print("Keep alive event triggered.")
                
            # Handle mouse click events for players set as "Human"
            if event.type == pygame.MOUSEBUTTONUP:
                x, y = pygame.mouse.get_pos()
                    
                if (current_turn == 'black' and black_player_type == "human") or \
                   (current_turn == 'white' and white_player_type == "human"):
                   
                    # 手動選擇座標
                    selected_hexes = process_selections_by_round(x, y, current_round)
                    # 更新座標
                    update_selected_hexes(selected_hexes)
                    
                # Handle the 'End Turn' button click
                if button_rect.collidepoint(event.pos):
                    # Count_Hexagons_by_Owner()
                    end_current_round()
                        
            # Handle random selections for players set as "Random"
            if not turn_ended and ((current_turn == 'black' and black_player_type == "random") or \
                                   (current_turn == 'white' and white_player_type == "random")):
                # Filter unbooked hexes from the hexagon_board and categorize them by label
                hexes_by_label = {}
                for pos, info in hexagon_board.items():
                    if not info.get('booked', False):
                        label = info['label']
                        if label in hexes_by_label:
                            hexes_by_label[label].append((pos, info))
                        else:
                            hexes_by_label[label] = [(pos, info)]
                    
                # Randomly select a specified number of hexes from those filtered by label
                selected_hexes = select_hexes_by_random(hexes_by_label, current_round)
                selected_hexes = select_hexes_by_AI(hexes_by_label, current_round)
                print(selected_hexes)
                # Process the selected hexes
                for pos, hex_info in selected_hexes:
                    hex_info['selected'] = True
                    hex_info['booked'] = True
                    hex_info['owner'] = current_turn
                    fill_color = BLACK if current_turn == 'black' else WHITE
                    draw_hexagon(screen, hex_info['x'], hex_info['y'], HEX_SIZE, (128, 128, 128), fill_color, hex_info['label'])
                
                pygame.display.flip()
                pygame.time.wait(100)  # Wait a second to let players see AI's choice
                # Count_Hexagons_by_Owner()    
                # time.sleep(1)
                current_turn = 'white' if current_turn == 'black' else 'black'
                turn_ended = True
                draw_player_turn_button(screen, current_turn)
                pygame.display.flip()
            
                display_remaining_hexes()
                # Update the round number
                current_round += 1
            
            turn_ended = False
        
        if display_remaining_hexes():
            print("Game Over: All hexes have been selected.")
            draw_player_turn_button(screen, "Game Over")
            display_connected_areas()
            running = False  # Exit the main loop
            pygame.display.flip()
            clock.tick(30)
            
        # Update the display and control the game update rate
        pygame.display.flip()
        clock.tick(30)
          
            
          
if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python main.py [player1_type] [player2_type]")
    #     print("player1_type and player2_type should be 'human' or 'random'")
    #     sys.exit(1)  # Exit the script with an error code

    # main(sys.argv[1], sys.argv[2])
    main('random', 'human')