import numpy as np
import pandas as pd
import time

start = time.time()

matches_df = pd.read_csv('./data/data.csv')

#setting season to be the start year of a season
matches_df['season'] = matches_df.season.str.split('/').str[0] # 2020/ 2021 -> 2020


#creating home and away teams
matches_df[['home_team', 'away_team']] = matches_df.match_name.str.split(' - ', expand = True) # Arsenal - Brighton -> Arsenal, Brighton


#creating home and away score
matches_df[['home_score', 'away_score']] = matches_df.result.str.split(':', expand = True) # 2:0 -> 2,0

#creating winner column
matches_df['winner'] = np.where(matches_df.home_score > matches_df.away_score, 'HOME_TEAM', np.where(matches_df.away_score > matches_df.home_score, 'AWAY_TEAM', 'DRAW'))

#droping result column
matches_df.drop(columns = 'result', inplace = True)

matches_df['home_score'] = matches_df['home_score'].str.replace('-', '0')
matches_df['away_score'] = matches_df['away_score'].str.replace('-', '0')


#turning columns into integers
matches_df['season'] = matches_df['season'].astype(int)
matches_df['home_score'] = matches_df['home_score'].astype(int)
matches_df['away_score'] = matches_df['away_score'].astype(int)

#cleaning up columns with missing number data
matches_df['a_odd'] = matches_df['a_odd'].str.replace('-', '0')
matches_df['d_odd'] = matches_df['d_odd'].str.replace('-', '0')
matches_df['h_odd'] = matches_df['h_odd'].str.replace('-', '0')


#turning columns into floats
matches_df['a_odd'] = matches_df['a_odd'].astype(float)
matches_df['d_odd'] = matches_df['d_odd'].astype(float)
matches_df['h_odd'] = matches_df['h_odd'].astype(float)


#home team points made in each match
matches_df['h_match_points'] = np.where(matches_df['winner'] == 'HOME_TEAM', 2 , np.where(matches_df['winner'] == 'DRAW',1, 0))

#away team points made in each match
matches_df['a_match_points'] = np.where(matches_df['winner'] == 'AWAY_TEAM', 2 , np.where(matches_df['winner'] == 'DRAW',1, 0))

#changing columns order
cols_order = ['season', 'date', 'match_name', 'home_team', 'away_team', 'winner', 'home_score', 'away_score',
                'h_odd', 'd_odd', 'a_odd', 'h_match_points', 'a_match_points']

matches_df = matches_df[cols_order]

matches_df.to_csv('./data/data_cleaned.csv', index = False)

print("Elapsed time:", time.time() - start)




