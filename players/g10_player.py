import math
import operator
import numpy as np
import logging
from typing import Callable, Dict, List, Tuple, Union
from collections import defaultdict


class Player:
    def __init__(self, flavor_preference: List[int], rng: np.random.Generator, logger: logging.Logger) -> None:
        """Initialise the player with given preference.

        Args: flavor_preference (List[int]): flavor preference, most flavored flavor is first element in the list and
        last element is least preferred flavor rng (np.random.Generator): numpy random number generator, use this for
        same player behavior across run logger (logging.Logger): logger use this like logger.info("message")
        """
        self.flavor_preference = flavor_preference
        self.rng = rng
        self.logger = logger
        self.state = None
        self.curr_turn = 0
        self.num_units_in_turn = 0

    def round_score(self, num):
        return round(num, 4)

    # function to evaluate the similarity between two preference lists
    def preference_distance(self, pref1, pref2):
        pref1_dict = {flavor: index for index, flavor in enumerate(pref1)}
        pref2_dict = {flavor: index for index, flavor in enumerate(pref2)}
        distance = 0
        for key in pref1_dict.keys():
            distance += abs(pref1_dict[key] - pref2_dict[key])
        return distance

    # calculate the total score that a scoop gets based on the given preference list
    def calc_flavor_points(self, flavors_scooped, flavor_preference):
        total = 0
        for flavor_cell in flavors_scooped:
            preference_idx = flavor_preference.index(flavor_cell)
            preference_score = len(self.flavor_preference) - preference_idx
            total += preference_score
        return total

    # return a list of the cells that we scoop up, position (i,j) being the top left corner
    def get_flavor_cells_from_scoop(self, i, j, curr_level, top_layer):
        max_level = max(curr_level[i, j], curr_level[i, j + 1], curr_level[i + 1, j], curr_level[i + 1, j + 1])
        flavor_cells = []
        for i_offset in range(2):
            for j_offset in range(2):
                current_level = curr_level[i + i_offset, j + j_offset]
                if current_level >= 0 and current_level == max_level:
                    flavor_cells.append(top_layer[i + i_offset, j + j_offset])
        return flavor_cells

    # composing the two functions above and return the total score of the scoop
    def calc_scoop_points(self, i, j, curr_level, top_layer, flavor_preference):
        if i >= len(curr_level - 1) or j >= len(curr_level[0] - 1):
            return 0
        flavor_cells = self.get_flavor_cells_from_scoop(i, j, curr_level, top_layer)
        return self.calc_flavor_points(flavor_cells, flavor_preference), len(flavor_cells)

    # return a dictionary of the totals flavors we can see in the bowls+on the board
    def get_all_flavors_seen(self, top_layer, get_served):
        total_each_flavor = defaultdict(int)
        # how many of what flavors we see in each player's bowl
        for team in get_served():
            for flavor in team:
                total_each_flavor[flavor] += team[flavor]
        # added by what we can see on the top layer
        for i in range(len(top_layer)):
            for j in range(len(top_layer[0])):
                if top_layer[i][j] != -1:
                    total_each_flavor[top_layer[i][j]] += 1
        return total_each_flavor

    # return a dictionary of the total flavors that we are expected to see below the top layer
    def flavors_left(self, total_each_favor, get_flavors):
        flav_left = defaultdict(int)
        num_flavors = len(get_flavors())
        amount_of_each_flavor = (24 * 15 * 8) // num_flavors
        for flavor in total_each_favor:
            flav_left[flavor] = amount_of_each_flavor - total_each_favor[flavor]
        return flav_left

    # composing the two functions above and return the total flavors that we are expected to see below the top layer
    def get_flavors_left_underneath(self, top_layer, get_served, get_flavors):
        total_each_flavor = self.get_all_flavors_seen(top_layer, get_served)
        return self.flavors_left(total_each_flavor, get_flavors)

    # function to find the scoop in hope to maximize our score in the game
    def find_max_scoop(self, top_layer, curr_level, flavor_preference, max_scoop_size, divide_by_scoop_size=True):
        max_scoop_loc = (0, 0)
        max_scoop_points_per_unit = 0
        max_scoop_points = 0
        for i in range(len(top_layer) - 1):
            for j in range(len(top_layer[0]) - 1):
                scoop_points, scoop_size = self.calc_scoop_points(i, j, curr_level, top_layer, flavor_preference)
                if 0 < scoop_size <= max_scoop_size:
                    if divide_by_scoop_size:
                        scoop_points_per_unit = self.round_score(scoop_points / scoop_size)

                        if scoop_points_per_unit == max_scoop_points_per_unit:
                            if scoop_points > max_scoop_points:
                                max_scoop_loc = (i, j)
                                max_scoop_points = scoop_points
                        elif scoop_points_per_unit > max_scoop_points_per_unit:
                            max_scoop_loc = (i, j)
                            max_scoop_points = scoop_points
                            max_scoop_points_per_unit = scoop_points_per_unit
                    else:
                        if scoop_points > max_scoop_points:
                            max_scoop_loc = (i, j)
                            max_scoop_points = scoop_points

        return max_scoop_loc, max_scoop_points

    # calculate a player's score based on the list of top layer flavor counts and their own preferences
    def get_player_score(self, top_layer_flavour_count, player_preference):
        score = 0
        flavor_preference = {}
        for i in range(len(player_preference)):
            (key, val) = player_preference[i]
            flavor_preference[key] = i
        for i in range(len(top_layer_flavour_count)):
            score += top_layer_flavour_count[i] * flavor_preference[i + 1]
        return score

    # helper function to select player to pass to, taking into account what they can get from the current board
    # and how different/similar their preferences are with ours
    def get_player_preferences(self, top_layer, served, available_players, pass_to) -> List[List[int]]:
        # get player preference lists based on their bowls
        player_preferences = [sorted(d.items(), key=operator.itemgetter(1)) for d in served]
        # estimation of scores that each available player can get from the top layer that we are passing to them
        estimated_score = []
        max_score = 0
        # "magic" but tested, determining what is a close decision if we choose to pass to a member with different preferences
        magic_percentage = 0.07
        if pass_to == 1:
            # "magic" but tested, determining what is a close decision if we choose to pass to a member with similar preferences
            magic_percentage = 0.2

        # get a list a flavors that are presented on the top layer at the moment we want to pass
        top_layer_flavour_count = self.get_top_layer_flavour_count(top_layer)

        for i in range(len(player_preferences)):
            if i in available_players:
                # calculate the scores of available players based on the current board
                score = self.get_player_score(top_layer_flavour_count, player_preferences[i])
                estimated_score.append(score)
                if score > max_score:
                    # the highest score in estimated score
                    max_score = score

        # change max_score to be cut for "close decision"
        max_score *= (1 - magic_percentage)
        # select eligible players based on whether they make it to "close decision"
        select_players = []
        for i in range(len(estimated_score)):
            if estimated_score[i] >= max_score:
                select_players.append(available_players[i])

        # select a player from select_player, currently no one
        select = -1

        same_preference_estimate = -math.inf
        if pass_to == 1:
            same_preference_estimate = math.inf
        # calculate how similar or how different the preference list of each eligible player who makes the
        # "close decision" cut are to our preference list
        for sp in select_players:
            sp_flavour_preferences = []
            for (flavour, count) in player_preferences[sp]:
                sp_flavour_preferences.append(flavour)
            # since in player_preferences most preferred flavour is last
            sp_flavour_preferences = sp_flavour_preferences[::-1]
            temp = self.preference_distance(self.flavor_preference, sp_flavour_preferences)
            if pass_to == 0 and temp > same_preference_estimate:
                # select the most different
                select = sp
                same_preference_estimate = temp
            elif pass_to == 1 and temp < same_preference_estimate:
                # select the most similar
                select = sp
                same_preference_estimate = temp

        return select

    # count cell by cell, how many of what flavors are presented on the top layer
    # return a list where the index indicates flavor, value indicates counts
    def get_top_layer_flavour_count(self, top_layer: np.ndarray) -> List[int]:
        top_layer_flavour_count = [0 for x in self.flavor_preference]
        m, n = top_layer.shape
        for i in range(m):
            for j in range(n):
                if top_layer[i][j] >= 1:
                    top_layer_flavour_count[top_layer[i][j] - 1] += 1

        return top_layer_flavour_count

    # function to determine which player to pass to
    def return_optimal_pass(self, get_turns_received, get_player_count, get_served, player_idx, top_layer, get_flavors):
        action = "pass"
        action = "pass"
        turns_received = get_turns_received()
        # get the current turn that we are in
        curr_iteration = turns_received[player_idx]

        # calculate the available players for this turn
        available_players = [i for i in range(len(turns_received)) if turns_received[i] < curr_iteration]

        # if we have a "big" family, pass to members with similar preferences
        # else pass to members with different preferences
        pass_to = 0  # different
        if get_player_count() >= 6:
            pass_to = 1  # same

        # if we are the last to go in this round, we can pass to anyone including ourselves
        if len(available_players) == 0:
            available_players = [i for i in range(len(turns_received))]
            values = self.get_player_preferences(top_layer, get_served(), available_players, pass_to)
        # if there is only one person who we can pass to, just pass it to them
        elif len(available_players) == 1:
            values = available_players[0]
        # do the evaluation
        else:
            values = self.get_player_preferences(top_layer, get_served(), available_players, pass_to)

        return action, values

    def serve(self, top_layer: np.ndarray, curr_level: np.ndarray, player_idx: int, get_flavors: Callable[[], List[int]], get_player_count: Callable[[], int],
              get_served: Callable[[], List[Dict[int, int]]], get_turns_received: Callable[[], List[int]]) -> Dict[str, Union[Tuple[int], int]]:
        """Request what to scoop or whom to pass in the given step of the turn. In each turn the simulator calls this
        serve function multiple times for each step for a single player, until the player has scooped 24 units of
        ice-cream or asked to pass to next player or made an invalid request. If you have scooped 24 units of
        ice-cream in a turn then you get one last step in that turn where you can specify to pass to a player.

        Args:
            top_layer (np.ndarray): Numpy 2d array of size (24, 15) containing flavor at each cell location
            curr_level (np.ndarray): Numpy 2d array of size (24, 15) containing current level at each cell location from
                8 to 0, where 8 is highest level at start and 0 means no icecream left at this level
            player_idx (int): index of your player, 0-indexed
            get_flavors (Callable[[], List[int]]): method which returns a list of all possible flavors
            get_player_count(Callable[[], int]): method which returns number of total players
            get_served (Callable[[], List[Dict[int, int]]]): method which returns a list of dictionaries corresponding
                to each player, each dictionary at index i tells how units of a flavor are present in the bowl of the
                player with index i. E.g. lets say the fourth element is {1: 0, 2: 8...} means the corresponding player
                with index 4 has 0 units of flavor 1 and 8 units of flavor
            get_turns_received (Callable[[], List[int]]): method which returns a list of integers corresponding to each player,
                each element at index i tells how many turns a player withindex i has played so far.

        Returns:
            Dict[str, Union[Tuple[int],int]]: Return a dictionary specifying what action to take in the next
        step. 2 possible return values {"action": "scoop",  "values" : (i,j)} stating to scoop the 4 cells with index
        (i,j), (i+1,j), (i,j+1), (i+1,j+1) {"action": "pass",  "values" : i} pass to next player with index i
        """
        # check if we are in a new turn and record any necessary state variables
        if get_turns_received()[player_idx] > self.curr_turn:
            self.num_units_in_turn = 0
            self.curr_turn = get_turns_received()[player_idx]

        # if we have scooped all 24 units, it's time to pass
        if self.num_units_in_turn >= 24:
            action, values = self.return_optimal_pass(get_turns_received, get_player_count, get_served, player_idx,
                                                      top_layer, get_flavors)
        # if not, scoop more units
        else:
            action = "scoop"
            values, points = self.find_max_scoop(top_layer, curr_level, self.flavor_preference,
                                                 24 - self.num_units_in_turn, divide_by_scoop_size=True)
            # but if no scoop was found, pass it
            if points == 0:
                action, values = self.return_optimal_pass(get_turns_received, get_player_count, get_served, player_idx,
                                                          top_layer, get_flavors)
            else:
                # record how many units we got in this turn to determine when we should pass
                self.num_units_in_turn += len(
                    self.get_flavor_cells_from_scoop(values[0], values[1], curr_level, top_layer))

        # release control to the program
        return {"action": action, "values": values}
