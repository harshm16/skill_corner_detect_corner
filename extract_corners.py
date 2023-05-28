# pip install kloppy
import pandas as pd
import kloppy
#2269, 3442 game potentially has some issue
from kloppy import skillcorner
import load_corner
import json 
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
from highlight_text import ax_text
import numpy as np
import scipy.signal as signal



match_ids = [2068, 2269, 2417, 2440, 2841, 3442, 3518, 3749, 4039] 

for matches in match_ids:
    matchdata_file = 'C:\\Users\\mishr\\Desktop\\fbref\\tracking data\\opendata-master\\data\\matches\\' + str(matches) + '\\match_data.json'
    tracking_file = 'C:\\Users\\mishr\\Desktop\\fbref\\tracking data\\opendata-master\\data\\matches\\' + str(matches) + '\\structured_data.json'


    dataset = load_corner.custom_load_skill_corner(meta_data=matchdata_file,
                            raw_data=tracking_file)


    df = dataset.to_df()

    f = open(matchdata_file)

    matchdata = json.load(f) 

    metadata = dataset.metadata
    home_team, away_team = metadata.teams


    def format_time(timestamp):
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        formatted_time = f"{minutes:02d}:{seconds:02d}"
        return formatted_time


    df['timestamp'] = df['timestamp'].astype(int)

    # Create a new column with the formatted time
    df['formatted_time'] = df['timestamp'].apply(format_time)

    df = df.drop(columns=[col for col in df.columns if '_anon_' in col])


    for column in df.columns:
        if '_x' in column:
            df[column] = df[column] * 105
        elif '_y' in column:
            df[column] = df[column] * 68


    def get_goalkeeper(matchdata):

        home_team_id = matchdata['home_team']['id']
        away_team_id = matchdata['away_team']['id']

        home_gk = None
        away_gk = None
        
        for player in matchdata['players']:
            if (player['player_role']['name'] == "Goalkeeper") and (player["start_time"] == "00:00:00"):
                
                if player['team_id'] == home_team_id:
                    home_gk = ("home_" + str(player['number']))
                
                else:
                    away_gk = ("away_" + str(player['number']))
            
        return home_gk, away_gk



    def find_playing_direction(dataframe, matchdata):
        
        home_gk, away_gk = get_goalkeeper(matchdata)
        # print(home_gk + "_x", away_gk + "_x")
        
        if ((dataframe[(dataframe['period_id'] == 1) & (dataframe[home_gk + "_x"].notna())][home_gk + "_x"].mean()) < 50) | ((dataframe[(dataframe['period_id'] == 1) & (dataframe[away_gk + "_x"].notna())][away_gk + "_x"].mean()) > 55):
        # +ve is left->right, -ve is right->left
        # home team attacking from left to right in the first half
            return 1
        
        else:
            return -1


    def to_single_playing_direction(dataframe,first_half_direction_flag):
        '''
        Flip coordinates in second half so that each team always shoots in the same direction through the match.
        '''

        dataframe_copy = dataframe.copy()
        columns_to_flip = [c for c in dataframe_copy.columns if c[-1].lower() in ['x','y']]

        if first_half_direction_flag == 1:
            conditions = (dataframe_copy['period_id'] == 2)

        else:
            conditions = (dataframe_copy['period_id'] == 1)

        for column in columns_to_flip:
            # conditions &= (~dataframe_copy[column].isna())

            if column[-1] == "x":
                dataframe_copy.loc[conditions, column] = 105 - dataframe_copy.loc[conditions, column]

            else:
                dataframe_copy.loc[conditions, column] = 68 - dataframe_copy.loc[conditions, column]


        return dataframe_copy


    new_Df = to_single_playing_direction(df, find_playing_direction(df, matchdata))


    def calc_player_velocities(team, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12):
        """ calc_player_velocities( tracking_data )
        
        Calculate player velocities in x & y direciton, and total player speed at each timestamp of the tracking data
        
        Parameters
        -----------
            team: the tracking DataFrame for home or away team
            smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
            filter: type of filter to use when smoothing the velocities. Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
            window: smoothing window size in # of frames
            polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity, so gradient is the acceleration
            maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN. 
            
        Returrns
        -----------
        team : the tracking DataFrame with columns for speed in the x & y direction and total speed added

        """
        # remove any velocity data already in the dataframe
        team = remove_player_velocities(team)
        
        # Get the player ids
        player_ids = np.unique( [ c[:-2] for c in team.columns if c[:4] in ['home','away'] ] )

        player_ids_plus_ball = np.append(player_ids, "ball")
        # frame id : if difference between consecutive is 1: then diff is 10 milisecond or whatever 
        # if frame id more, time same as time stamp: diff again from 0, take that frame to be first frame: as base


        team['diff'] = team['frame_id'].diff()

        dt = team['diff'].apply(lambda x: 0.1 if x == 1.0 else 0)


        # index of first frame in second half
        second_half_idx = team[team['period_id'] == 2].index[0]
        
        # estimate velocities for players in team
        for player in player_ids_plus_ball: # cycle through players individually
            # difference player positions in timestep dt to get unsmoothed estimate of velicity
            vx = team[player+"_x"].diff() / dt
            vy = team[player+"_y"].diff() / dt


            
            if maxspeed>0:
                # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
                raw_speed = np.sqrt( vx**2 + vy**2 )
                vx[ raw_speed>maxspeed ] = np.nan
                vy[ raw_speed>maxspeed ] = np.nan
                
                
            if smoothing:
                if filter_=='Savitzky-Golay':
                    # calculate first half velocity
                    vx.loc[:second_half_idx] = signal.savgol_filter(vx.loc[:second_half_idx],window_length=window,polyorder=polyorder, mode='constant')
                    vy.loc[:second_half_idx] = signal.savgol_filter(vy.loc[:second_half_idx],window_length=window,polyorder=polyorder, mode='constant')        
                    # calculate second half velocity
                    vx.loc[second_half_idx:] = signal.savgol_filter(vx.loc[second_half_idx:],window_length=window,polyorder=polyorder, mode='constant')
                    vy.loc[second_half_idx:] = signal.savgol_filter(vy.loc[second_half_idx:],window_length=window,polyorder=polyorder, mode='constant')

                    # print("hedre")
                    
                elif filter_=='moving average':
                    ma_window = np.ones( window ) / window 
                    # calculate first half velocity
                    vx.loc[:second_half_idx] = np.convolve( vx.loc[:second_half_idx] , ma_window, mode='same' ) 
                    vy.loc[:second_half_idx] = np.convolve( vy.loc[:second_half_idx] , ma_window, mode='same' )      
                    # calculate second half velocity
                    vx.loc[second_half_idx:] = np.convolve( vx.loc[second_half_idx:] , ma_window, mode='same' ) 
                    vy.loc[second_half_idx:] = np.convolve( vy.loc[second_half_idx:] , ma_window, mode='same' ) 
                    
            
            # put player speed in x,y direction, and total speed back in the data frame
            team[player + "_vx"] = vx
            team[player + "_vy"] = vy
            team[player + "_speed"] = np.sqrt( vx**2 + vy**2 )

        return team

    def remove_player_velocities(team):
        # remove player velocoties and direction measures that are already in the dataframe
        columns = [c for c in team.columns if c.split('_')[-1] in ['d', 's']] # Get the player ids
        team = team.drop(columns=columns)

        new_c = [col for col in team.columns if '_anon_' in col]
        team = team.drop(columns=new_c)

        return team
        

    df_with_speed = calc_player_velocities(new_Df)


    def is_within_radius(x, y, corner_x, corner_y, radius):
        distance = np.sqrt((x - corner_x)**2 + (y - corner_y)**2)
        return np.isclose(distance, radius, atol=1)

        # on_the_arc works better
        # def is_within_radius(x, y, corner_x, corner_y, radius):
        #     distance = np.sqrt((x - corner_x)**2 + (y - corner_y)**2)
        #     return np.isclose(distance, radius, atol=0.5)


    def extract_near_corner_situations(dataframe, home_flag):
        radius = 1  # 1-meter radius

        mask_nan = dataframe[["ball_x", "ball_y"]].isnull().any(axis=1)  # Mask for rows with NaN in ball_x or ball_y

        if home_flag:

            mask_on_perimeter = (is_within_radius(dataframe["ball_x"], dataframe["ball_y"], 105, 0, radius) | is_within_radius(dataframe["ball_x"], dataframe["ball_y"], 105, 68, radius))

        else:
            mask_on_perimeter = (is_within_radius(dataframe["ball_x"], dataframe["ball_y"], 0, 0, radius) | is_within_radius(dataframe["ball_x"], dataframe["ball_y"], 0, 68, radius))
            
            
        corner_situations = dataframe[~(mask_nan | ~mask_on_perimeter)]

        return corner_situations


    home_corner_situations = extract_near_corner_situations(df_with_speed, True)

    away_corner_situations = extract_near_corner_situations(df_with_speed, False)


    def extract_corners (dataframe, potential_corner_subset_df, home_team_flag):
        
        
        list_corner_frames = []
        all_corners = []   
        team_string = None
        final_third_line = None
        
        if len(potential_corner_subset_df) == 0:
            return all_corners
        
        potential_corner_subset_df['diff'] = potential_corner_subset_df['frame_id'].diff()

        starting_frames = [potential_corner_subset_df.iloc[0]['frame_id']]

        starting_frames = starting_frames + list(potential_corner_subset_df[potential_corner_subset_df['diff'].apply(lambda x: (x >= 100))]['frame_id'].values)
        
        #add frames upto 10 seconds from the time the ball was found near a corner arc
        for frames in starting_frames:
            index = dataframe[dataframe['frame_id'] == frames].index[0]
            list_corner_frames.append(dataframe.iloc[index : index + 100])


        for corners in list_corner_frames:
            
            if home_team_flag:
                team_string = 'home_'
                final_third_line = 88.5
            
            else:
                team_string = 'away_'
                final_third_line = 16.5

            x_cols = [col for col in corners.columns if col.startswith(team_string) and col.endswith('_x')]
            y_cols = [col for col in corners.columns if col.startswith(team_string) and col.endswith('_y')]

            # at least 4 attacking players tracked & at least 4 players in the final 3rd & at least 3 players in the penalty area  - 1 corner taker

            #at least 4 players in the final 3rd worked for most - not city - they had a throwin extra
            if home_team_flag:
                if (corners[x_cols].notna().sum() > 0).sum() >= 4 and ((corners[x_cols] > final_third_line).sum() > 0).sum() >= 4 and (((corners[y_cols] <= 54) & (14 <= corners[y_cols])).sum() > 0).sum() >= 4:
                    
                    all_corners.append(corners)

            else:
                # print(final_third_line)
                # 
                if (corners[x_cols].notna().sum() > 0).sum() >= 4 and ((corners[x_cols] < final_third_line).sum() > 0).sum() >= 4 and (((corners[y_cols] <= 54) & (14 <= corners[y_cols])).sum() > 0).sum() >= 4:
                    
                    all_corners.append(corners)
                

        return all_corners


    home_corners = extract_corners(df_with_speed, home_corner_situations, True )

    away_corners = extract_corners(df_with_speed, away_corner_situations, False )


    for i in range(len(home_corners)):
        game_id = str(matchdata_file.split("matches\\")[1].split("\\")[0])
        home_corners[i].to_csv(game_id + "_home_" + str(i) + "_.csv")


    for i in range(len(away_corners)):
        game_id = str(matchdata_file.split("matches\\")[1].split("\\")[0])
        away_corners[i].to_csv(game_id + "_away_" + str(i) + "_.csv")