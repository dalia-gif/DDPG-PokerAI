# example observation:
#
# def _get_observation(self, player):
#     observation = np.zeros(self.observation_space.shape[0], dtype=np.float32)
#     observation[0] = player.identifier
#     valid_actions = self._get_valid_actions(player)
#     for action in valid_actions["actions_list"]:
#         observation[action.value + 1] = 1
#     observation[5] = valid_actions["bet_range"][0]
#     observation[6] = valid_actions["bet_range"][1]
#     observation[7] = player.position
#     observation[8] = Card.get_suit_int(player.cards[0])
#     observation[9] = Card.get_rank_int(player.cards[0])
#     observation[10] = Card.get_suit_int(player.cards[1])
#     observation[11] = Card.get_rank_int(player.cards[1])
#     observation[12] = player.stack
#     observation[13] = player.money_in_pot
#     observation[14] = player.bet_this_street
#     observation[15] = self.street
#     for i in range(len(self.cards)):
#         observation[16 + (i * 2)] = Card.get_suit_int(self.cards[i])
#         observation[17 + (i * 2)] = Card.get_rank_int(self.cards[i])
#     observation[20] = self.pot
#     observation[21] = self.bet_to_match
#     observation[22] = self.minimum_raise
#     others = [other for other in self.players if other is not player]
#     for i in range(len(others)):
#         observation[23 + i * 6] = others[i].position
#         observation[24 + i * 6] = others[i].state.value
#         observation[25 + i * 6] = others[i].stack
#         observation[26 + i * 6] = others[i].money_in_pot
#         observation[27 + i * 6] = others[i].bet_this_street
#         observation[28 + i * 6] = int(others[i].all_in)
#     return observation

#######################################################################

# exmpale input

# Game started at: 2016/11/29 15:25:37
# Game ID: 787027613 0.50/1 (PRR) Karkadann (Hold'em)
# Seat 7 is the button
# Seat 1: StephCurry (105.78).
# Seat 2: PANDAisEVIL (101).
# Seat 3: AironVega (103.20).
# Seat 4: IlxxxlI (43).
# Seat 5: pineapplesand (40).
# Seat 6: dankmann (206.52).
# Seat 7: ElvenEyes (115).
# Seat 8: gust (46).
# Seat 9: VegetablesArentYummy (273.59).
# Player gust has small blind (0.50)
# Player VegetablesArentYummy has big blind (1)
# Player gust received a card.
# Player gust received a card.
# Player VegetablesArentYummy received a card.
# Player VegetablesArentYummy received a card.
# Player StephCurry received a card.
# Player StephCurry received a card.
# Player PANDAisEVIL received a card.
# Player PANDAisEVIL received a card.
# Player AironVega received a card.
# Player AironVega received a card.
# Player IlxxxlI received card: [9d]
# Player IlxxxlI received card: [7d]
# Player pineapplesand received a card.
# Player pineapplesand received a card.
# Player dankmann received a card.
# Player dankmann received a card.
# Player ElvenEyes received a card.
# Player ElvenEyes received a card.
# Player StephCurry folds
# Player PANDAisEVIL folds
# Player AironVega folds
# Player IlxxxlI folds
# Player pineapplesand folds
# Player dankmann raises (3)
# Player ElvenEyes folds
# Player gust folds
# Player VegetablesArentYummy folds
# Uncalled bet (2) returned to dankmann
# Player dankmann mucks cards
# ------ Summary ------
# Pot: 2.50. Rake 0
# Player StephCurry does not show cards.Bets: 0. Collects: 0. Wins: 0.
# Player PANDAisEVIL does not show cards.Bets: 0. Collects: 0. Wins: 0.
# Player AironVega does not show cards.Bets: 0. Collects: 0. Wins: 0.
# Player IlxxxlI does not show cards.Bets: 0. Collects: 0. Wins: 0.
# Player pineapplesand does not show cards.Bets: 0. Collects: 0. Wins: 0.
# *Player dankmann mucks (does not show cards). Bets: 1. Collects: 2.50. Wins: 1.50.
# Player ElvenEyes does not show cards.Bets: 0. Collects: 0. Wins: 0.
# Player gust does not show cards.Bets: 0.50. Collects: 0. Loses: 0.50.
# Player VegetablesArentYummy does not show cards.Bets: 1. Collects: 0. Loses: 1.
# Game ended at: 2016/11/29 15:26:36


def split_log(log):
    # split log by "Game ended at" keyword
    logs = log.split("Game ended at")
    return logs


def parsing_log(log):
    # split log by "\n"
    log = log.split("\n")
    # get game id
    game_id = log[1].split(" ")[2]
    # get player list
    player_list = []
    for i in range(3, 12):
        player = log[i].split(":")[1].split(" ")[1]
        player_list.append(player)
    # get action list
    action_list = []
    for i in range(12, len(log) - 1):
        action = log[i].split(" ")[1:]
        action_list.append(action)
    return game_id, player_list, action_list


def file_processing(file):
    with open(file, "r") as f:
        log = f.read()
    logs = split_log(log)
    for log in logs:
        game_id, player_list, action_list = parsing_log(log)
        print(game_id, player_list, action_list)
        c = input("Continue? (y/n): ")
        if c.lower() == "n":
            break

    return 0
