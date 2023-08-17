import numpy as np
import pandas as pd


matches_df = pd.read_csv('./data/matches.csv')

cols_to_drop = ['season', 'match_name','date', 'home_team', 'away_team', 'home_score', 'away_score',
                'h_match_points', 'a_match_points']

num_cols = matches_df.dtypes[matches_df.dtypes != 'object'].index.tolist()

corr_cols = list(set(num_cols) - set(cols_to_drop))

matches_df['winner_h'] = np.where(matches_df.winner == 'HOME_TEAM', 1, 0)
matches_df['winner_a'] = np.where(matches_df.winner == 'AWAY_TEAM', 1, 0)
matches_df['winner_d'] = np.where(matches_df.winner == 'DRAW', 1, 0)


matches_df[corr_cols + ['winner_h']].corr()['winner_h'].sort_values(ascending = False).reset_index()

matches_df[corr_cols + ['winner_a']].corr()['winner_a'].sort_values(ascending = False).reset_index()

matches_df[corr_cols + ['winner_d']].corr()['winner_d'].sort_values(ascending = False).reset_index()

