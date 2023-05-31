import numpy as np
import scipy.signal as signal

def format_time(timestamp):
    """
    Converts the given timestamp back into MM:SS format.
    """   
    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    formatted_time = f"{minutes:02d}:{seconds:02d}"
    return formatted_time


def get_goalkeeper(matchdata):
    """
    get_goalkeeper(matchdata)
    
    extracts the Goalkeeper jersey number from Skillcorner's match_data json file & returns it in the format "home_#jersey" & "away_#jersey"
    
    Parameters
    -----------
    
    matchdata (json): loaded json
    
    Returns
    -----------
    
    home_gk (str): "home_#jersey"
    away_gk (str):  "away_#jersey"
    
    """  
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
    """
    find_playing_direction(dataframe, matchdata)
    
    Checks the average position of goalkeepers of both teams & finds the direction of attack for the home side.
    Home team is assumed to be attacking from left to right in the first half.
    function returns 1 if they are attacking left to right in the first half, -1 otherwise.
    
    Parameters
    -----------
    
    dataframe (pandas dataframe): dataframe in the kloppy format
    matchdata (json): loaded match_data json for the same game as the dataframe.
    
    Returns
    -----------
    
     (int): 1:If the home team is attacking left to right in the first half, -1 if the home team is attacking left to right in the first half
    
    """  

    home_gk, away_gk = get_goalkeeper(matchdata)

    if ((dataframe[(dataframe['period_id'] == 1) & (dataframe[home_gk + "_x"].notna())][home_gk + "_x"].mean()) < 50) | ((dataframe[(dataframe['period_id'] == 1) & (dataframe[away_gk + "_x"].notna())][away_gk + "_x"].mean()) > 55):

        return 1

    else:
        return -1

def to_single_playing_direction(dataframe,first_half_direction_flag):
    """
    to_single_playing_direction(dataframe,first_half_direction_flag)
    
    Flip coordinates in second half so that each team always shoots in the same direction through the match.
    Home team is assumed to be always attacking from left to right.
    
    Parameters
    -----------
    
    dataframe (pandas dataframe): dataframe in the kloppy format
    first_half_direction_flag (int): possible values, -1 or 1, as returned by the function find_playing_direction.
    
    Returns
    -----------
    
    dataframe_copy (pandas dataframe): a copy of the dataframe paramter, with the attacking directions for the second half flipped.
    
    """  

    dataframe_copy = dataframe.copy()
    columns_to_flip = [c for c in dataframe_copy.columns if c[-1].lower() in ['x','y']]

    if first_half_direction_flag == 1:
        conditions = (dataframe_copy['period_id'] == 2)

    else:
        conditions = (dataframe_copy['period_id'] == 1)

    for column in columns_to_flip:

        if column[-1] == "x":
            dataframe_copy.loc[conditions, column] = 105 - dataframe_copy.loc[conditions, column]

        else:
            dataframe_copy.loc[conditions, column] = 68 - dataframe_copy.loc[conditions, column]


    return dataframe_copy

def remove_player_velocities(dataframe):
    """
    remove_player_velocities(dataframe)
    
    Remove information about all '_anon_' "objects" tracked. Also remove the 'd', 's' columns from the kloppy dataframe, as they are None for skillcorner data.
 
    Parameters
    -----------
    
    dataframe (pandas dataframe): dataframe in the kloppy format
      
    Returns
    -----------
    
    dataframe (pandas dataframe): dataframe in the kloppy format with '_anon_' columns and 'd' & 's' columns for players removed.
    
    """ 

    columns = [c for c in dataframe.columns if c.split('_')[-1] in ['d', 's']] # Get the player ids
    dataframe = dataframe.drop(columns=columns)

    new_c = [col for col in dataframe.columns if '_anon_' in col]
    dataframe = dataframe.drop(columns=new_c)

    return dataframe



#Function inherited from Laurie Shaw's code for reading and working with Metrica's tracking and event data.
#Original code can be found here: https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Metrica_Velocities.py
def calc_player_velocities(dataframe, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12):
    """
    calc_player_velocities( tracking_data )
    
    Calculate player velocities in x & y direciton, and total player speed at each timestamp of the tracking data
    
    Parameters
    -----------
        dataframe (pandas dataframe): dataframe in the kloppy format
        smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
        filter: type of filter to use when smoothing the velocities. Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
        window: smoothing window size in # of frames
        polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity, so gradient is the acceleration
        maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN. 
        
    Returns
    -----------
    dataframe : the tracking DataFrame with columns for speed in the x & y direction and total speed added

    """
    # remove any velocity data already in the dataframe
    dataframe = remove_player_velocities(dataframe)
    
    # Get the player ids
    player_ids = np.unique( [ c[:-2] for c in dataframe.columns if c[:4] in ['home','away'] ] )

    player_ids_plus_ball = np.append(player_ids, "ball")

    # frame id : if difference between consecutive is 1: then the actual difference between the frames is 10 miliseconds 
    # But if the difference between frameids between successive rows is found to be more than 1, we assing the difference to be 0, 
    # so that the calculation of speeds can again start afresh. This is done since there are missing frames in between, when there was no player data captured.

    dataframe['diff'] = dataframe['frame_id'].diff()

    dt = dataframe['diff'].apply(lambda x: 0.1 if x == 1.0 else 0)


    # index of first frame in second half
    second_half_idx = dataframe[dataframe['period_id'] == 2].index[0]
    
    # estimate velocities for players in team
    for player in player_ids_plus_ball: # cycle through players individually
        # difference player positions in timestep dt to get unsmoothed estimate of velicity
        vx = dataframe[player+"_x"].diff() / dt
        vy = dataframe[player+"_y"].diff() / dt

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
        dataframe[player + "_vx"] = vx
        dataframe[player + "_vy"] = vy
        dataframe[player + "_speed"] = np.sqrt( vx**2 + vy**2 )

    return dataframe


def is_within_radius(x1, y1, x2, y2, radius):
    """
    is_within_radius(x, y, corner_x, corner_y, radius)
    
    Find if the distance between the given points (x, y) and (corner_x, corner_y) is within the chosen tolerance value of 'radius'.
    The tolerance value was chosen to be 1 as it helped extract a few extra corner situations, where the ball wasn't tracked when it was "on" the corner arc.
    
    Parameters
    -----------
        x1 (float): x coordinate
        y1 (float): y coordinate
        x2 (float): x coordinate 
        y2 (float): y coordinate
        radius (float): distance to be compared with
          
    Returns
    -----------
     (bool) : True if the points are within the 'radius + atol (tolerance)' of each other, False otherwise.
    """

    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return np.isclose(distance, radius, atol=1)


def extract_near_corner_situations(dataframe, home_team_flag):
    """
    extract_near_corner_situations(dataframe, home_team_flag)
    
    Home team is assumed to be always attacking from left to right.
    The function extracts all rows from the dataframe where if the ball location is found to be within 2 meters of the corner arc (1 meter actual distance of the corner 
    arc from the corner, + 1 meter tolerance), then there is a possibility of it being a corner situation.

    Although this logic does extract a few extra non-corner situations where the ball went out for a throw-in or just went over/near the corner arc without going out.

    Another logic that was tried was to extract situations when the ball went outside the pitch behind the goal line, but in some situations the ball wasn't tracked going out.
    This logic extracted a fewer number of corner situations. 
    
    Parameters
    -----------
    dataframe (pandas dataframe): dataframe in the kloppy format
    home_team_flag (bool): True denotes that the function needs to extract corner situations for the home team, false otherwise.
    
    Returns
    -----------
    
    corner_situations (pandas dataframe): all rows from the dataframe parameter where the ball location is within 1 meter (+ tolerance) of the corner arc
    
    """

    radius = 1  # 1-meter radius

    mask_nan = dataframe[["ball_x", "ball_y"]].isnull().any(axis=1)  # Mask for rows with NaN in ball_x or ball_y

    if home_team_flag:

        mask_on_perimeter = (is_within_radius(dataframe["ball_x"], dataframe["ball_y"], 105, 0, radius) | is_within_radius(dataframe["ball_x"], dataframe["ball_y"], 105, 68, radius))

    else:
        mask_on_perimeter = (is_within_radius(dataframe["ball_x"], dataframe["ball_y"], 0, 0, radius) | is_within_radius(dataframe["ball_x"], dataframe["ball_y"], 0, 68, radius))
        
        
    corner_situations = dataframe[~(mask_nan | ~mask_on_perimeter)]

    return corner_situations




def extract_corners (dataframe, potential_corner_subset_df, home_team_flag, min_attacking_players):
    """
    extract_corners (dataframe, potential_corner_subset_df, home_team_flag, min_attacking_players)
    
    Using the subset of rows extracted by the function "extract_near_corner_situations", to extract approximate corner situations.
    
    Creates a list of dataframes consisting of frames till 10 seconds (100 skillcorner frames) from when the ball was first found within a certain distance
    of the corners of the pitch & at least 'min_attacking_players' players were found inside the attacking penalty area within those 10 seconds.

    Parameters
    -----------
    dataframe (pandas dataframe): dataframe in the kloppy format
    potential_corner_subset_df (pandas dataframe): a subset of the above parameter as returned by the 'extract_near_corner_situations' function
    home_team_flag (bool): True denotes that the function needs to extract corner situations for the home team, false otherwise.
    min_attacking_players(int): Minimum number of attacking players to look for in the 10 second timeline.

    Returns
    -----------
    
    all_corners (list): list of dataframes of corner situations, each consisting of 100 skillcorner frames
    
    """
    
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

        # at least 4 attacking players tracked & at least 4 players in the final 3rd & at least 4 players in the penalty area

        #at least 4 players in the final 3rd worked for most - not city - they had an extra throwin where 4 attackers were making runs into the box.
        if home_team_flag:
            if (corners[x_cols].notna().sum() > 0).sum() >= min_attacking_players and ((corners[x_cols] > final_third_line).sum() > 0).sum() >= min_attacking_players and (((corners[y_cols] <= 54) & (14 <= corners[y_cols])).sum() > 0).sum() >= min_attacking_players:
                
                all_corners.append(corners)

        else:

            if (corners[x_cols].notna().sum() > 0).sum() >= min_attacking_players and ((corners[x_cols] < final_third_line).sum() > 0).sum() >= min_attacking_players and (((corners[y_cols] <= 54) & (14 <= corners[y_cols])).sum() > 0).sum() >= min_attacking_players:
                
                all_corners.append(corners)
            
    return all_corners
