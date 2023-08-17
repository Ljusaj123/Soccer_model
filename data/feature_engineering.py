import numpy as np
import pandas as pd
import time

start = time.time()


matches_df = pd.read_csv('./data/data_cleaned.csv')

#changing from date to datetime
matches_df['date'] = pd.to_datetime(matches_df.date)

def get_rank(x, team, delta_year):
    full_season_df = matches_df[(matches_df.season == (x.season - delta_year))]

    # for every home team calculate the sum of the h_match_points, home_score and away_score
    full_home_df = full_season_df.groupby(['home_team']).sum()[['h_match_points', 'home_score', 'away_score']].reset_index()

    # for every away team calculate the sum of the a_match_points, away_score and home_score
    full_away_df = full_season_df.groupby(['away_team']).sum()[['a_match_points', 'away_score', 'home_score']].reset_index()

    # renaming columns
    full_home_df.columns = ['team', 'points', 'goals', 'goals_sf']
    full_away_df.columns = ['team', 'points', 'goals', 'goals_sf']


    #concat together away and home team df
    rank_df = pd.concat([full_home_df, full_away_df], ignore_index = True)

    #calculate difference between goals that team scored and team suffered
    rank_df['goals_df'] = rank_df.goals - rank_df.goals_sf

    #group together teams (one team was home somewhere and away somewhere else)
    rank_df = rank_df.groupby(['team']).sum().reset_index()

    #sort ascending
    rank_df = rank_df.sort_values(by = ['points', 'goals_df', 'goals'], ascending = False)

    #this is function that ranks teams by points they have made
    rank_df['rank'] = rank_df.points.rank(method = 'first', ascending = False).astype(int)
        
    # getting rank for the team we send into this function
    team_rank = rank_df[rank_df.team == team].min()['rank']
        
    return team_rank


def get_match_stats(x, team):
    
    #home team df filter    
    home_df = matches_df[(matches_df.home_team == team) & (matches_df.date < x.date) & (matches_df.season == x.season)]

    #away team df filter
    away_df = matches_df[(matches_df.away_team == team) & (matches_df.date < x.date) & (matches_df.season == x.season)]

    #for every home team calculate the sum of the h_match_points, home_score and away_score
    home_table = home_df.groupby(['date']).sum()[['h_match_points', 'home_score', 'away_score']].reset_index()

    #for every away team calculate the sum of the a_match_points, away_score and home_score
    away_table = away_df.groupby(['date']).sum()[['a_match_points', 'away_score', 'home_score']].reset_index()

    #rename columns
    home_table.columns = ['date', 'points', 'goals', 'goals_sf']
    away_table.columns = ['date', 'points', 'goals', 'goals_sf']

    #calculate difference between goals that team scored and team suffered
    home_table['goals_df'] = home_table.goals - home_table.goals_sf
    away_table['goals_df'] = away_table.goals - away_table.goals_sf

    #new column host
    home_table['host'] = 'home'
    away_table['host'] = 'away'

    #combine two tables and sort by date
    full_table = pd.concat([home_table, away_table], ignore_index = True)
    full_table = full_table.sort_values('date', ascending = True) #KRIVO SORTIRA

    #sum of home_points, home_goals and home_goals_sf
    home_points = home_table.points.sum()
    home_goals = home_table.goals.sum()
    home_goals_sf = home_table.goals_sf.sum()

    #sum of away_points, away_goals and away_goals_sf
    away_points = away_table.points.sum()
    away_goals = away_table.goals.sum()
    away_goals_sf = away_table.goals_sf.sum()


    #calculating home wins, draws and loses
    home_wins = len(home_table[home_table.points == 2])
    home_draws = len(home_table[home_table.points == 1])
    home_losses = len(home_table[home_table.points == 0])

    #calculating away wins, draws and loses
    away_wins = len(away_table[away_table.points == 2])
    away_draws = len(away_table[away_table.points == 1])
    away_losses = len(away_table[away_table.points == 0])


    #total points stats
    total_points = home_points + away_points
    total_goals = home_goals + away_goals
    total_goals_sf = home_goals_sf + away_goals_sf
    total_wins = home_wins + away_wins
    total_draws = home_draws + away_draws
    total_losses = home_losses + away_losses

    #get streaks of wins
    full_table['start_of_streak'] = full_table.points.ne(full_table.points.shift())
    full_table['streak_id'] = full_table['start_of_streak'].cumsum()
    full_table['streak_counter'] = full_table.groupby('streak_id').cumcount() + 1

    streak_table = full_table[full_table.date == full_table.date.max()]

    if streak_table.points.min() == 2:
        win_streak = streak_table.streak_counter.sum()
        loss_streak = 0
        draw_streak = 0
    elif streak_table.points.min() == 0:
        win_streak = 0
        loss_streak = streak_table.streak_counter.sum()
        draw_streak = 0
    else:
        win_streak = 0
        loss_streak = 0
        draw_streak = streak_table.streak_counter.sum()
    

    #get last 3 games
    full_table_delta = full_table[full_table.date.isin(full_table.date[-3:])]

    home_l_points = full_table_delta[full_table_delta.host == 'home'].points.sum()
    away_l_points = full_table_delta[full_table_delta.host == 'away'].points.sum()

    #total metric in given delta averaged
    total_l_points = (home_l_points + away_l_points)/3
    total_l_goals = (home_goals + away_goals)/3
    total_l_goals_sf = (home_goals_sf + away_goals)/3

    return total_points, total_l_points, total_goals, total_l_goals, total_goals_sf, total_l_goals_sf, total_wins, total_draws, total_losses, win_streak, loss_streak, draw_streak

def get_days_ls_match(x, team):

    #filtering the last game of the team and getting date
    last_date = matches_df[(matches_df.season == x.season) & (matches_df.date < x.date) & (matches_df.match_name.str.contains(team))].date.max()
        
    #calculating the number of days since the last match
    days = (x.date - last_date)/np.timedelta64(1,'D')
    
    return days

def get_last_match_winner(x):
    
    temp_df = matches_df[(matches_df.date < x.date) & (matches_df.match_name.str.contains(x.home_team)) & (matches_df.match_name.str.contains(x.away_team))] #get all matches between two teams
    temp_df = temp_df[temp_df.date == temp_df.date.max()] #get the last match between two teams

    if len(temp_df) == 0:
        result = None
    else: 
        result = np.where(temp_df['winner'] == 'DRAW', "DRAW",                           #if the result of last match was draw
                 np.where(temp_df['winner'] == 'AWAY_TEAM', 'AWAY_TEAM', 'HOME_TEAM'))   #if the result of last match was home win or away win
    
        result = result[0] #getting just the string
    
    return result

def create_main_cols(x, team): #x is every row and team is every row home team or away team name

    #get current and last year rank
    team_rank = get_rank(x, team, 0)
    last_year_team_rank = get_rank(x, team, 1)

    # #get main match stats
    total_points, total_l_points, total_goals, total_l_goals, total_goals_sf, total_l_goals_sf, total_wins, total_draws, total_losses, win_streak, loss_streak, draw_streak = get_match_stats(x, team)

    # #get days since the last match
    days = get_days_ls_match(x, team)    

    return team_rank, last_year_team_rank, days, total_points, total_l_points, total_goals, total_l_goals, total_goals_sf, total_l_goals_sf, total_wins, total_draws, total_losses, win_streak, loss_streak, draw_streak

######################################
######################################


cols = ['_rank', '_ls_rank', '_days_ls_match', 
        '_points', '_l_points', '_goals', '_l_goals'
        , '_goals_sf', '_l_goals_sf', '_wins', '_draws', '_losses', 
        '_win_streak', '_loss_streak', '_draw_streak']

home_team_cols = ['ht' + col for col in cols]
away_team_cols = ['at' + col for col in cols]

#calculates statistics for home team
matches_df[home_team_cols] = pd.DataFrame(
    matches_df.apply(
        lambda x: create_main_cols(x, x.home_team), axis = 1).to_list(), index = matches_df.index) 

#calculates statistics for away team
matches_df[away_team_cols] = pd.DataFrame(
    matches_df.apply(
        lambda x: create_main_cols(x, x.away_team), axis = 1).to_list(), index = matches_df.index)

#result between last game of the teams
matches_df['ls_winner'] = matches_df.apply(lambda x: get_last_match_winner(x), axis = 1)

#filling NAs
matches_df.fillna(-33, inplace = True)

print("Elapsed time:", time.time() - start)

#saving data
matches_df.to_csv('./data/matches.csv', index = False)