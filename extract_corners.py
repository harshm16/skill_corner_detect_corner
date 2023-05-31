# pip install kloppy
import pandas as pd
import kloppy
from kloppy import skillcorner
#module with the modfied load data method by skillcorner
import load_corner
import json 
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
from highlight_text import ax_text
#module with all the helper functions
import helpers
import os

#loop over all matches in the skillcorner dataset - downloaded from: https://github.com/SkillCorner/opendata/tree/master/data/matches
match_ids = [2068, 2269, 2417, 2440, 2841, 3442, 3518, 3749, 4039] 


for matches in match_ids:

    #path to the skillcorner data
    path_to_files = 'C:\\Users\\mishr\\Desktop\\fbref\\tracking data\\opendata-master\\data\\matches'

    tracking_file = os.path.join(path_to_files, str(id) + '\\structured_data.json')
    matchdata_file = os.path.join(path_to_files, str(id) + '\\match_data.json')
    
    #Made some changes to the load data function provided by kloppy.
    #The issue in #2269 is due to a referee not having a trackable_object.
    #For #3442 it is a new group name "balls" being used for a few seconds for the actual ball, thus not having a tracking_object again.
    #Changes can be found in load_corner module, inherited from kloppy.
    dataset = load_corner.custom_load_skill_corner(meta_data=matchdata_file,
                            raw_data=tracking_file)

    #dataframe in the kloppy format
    df = dataset.to_df()

    #load match_data json separately to extract the information about starting keepers for either teams
    f = open(matchdata_file)
    matchdata = json.load(f) 

    metadata = dataset.metadata
    home_team, away_team = metadata.teams

    df['timestamp'] = df['timestamp'].astype(int)
    # Create a new column with the formatted time
    df['formatted_time'] = df['timestamp'].apply(helpers.format_time)

    #remove all columns with '_anon_' - anonymous objects tracked
    df = df.drop(columns=[col for col in df.columns if '_anon_' in col])

    #logic to find when the ball goes outside the goal line. - not used as it's hard to figure out whether the ball was last touched by the defending team or not
    #as the 'ball possession team' marker doesn't quite switch when the ball deflects away or ricochet's off defenders.
    # Sometimes the ball is also not tracked as going out of the pitch, as the camera pans out to something else.
    # out_of_pitch = df[((df["ball_x"] < 0) | (df["ball_x"] > 105)) & ~(((df["ball_y"].between(30.34, 37.66)) & (df["ball_z"] < 2.44)) | (df["ball_z"].isna()))]

    #adjust the coordinates of the data to a 105x68 sized pitch.
    for column in df.columns:
        if '_x' in column:
            df[column] = df[column] * 105
        elif '_y' in column:
            df[column] = df[column] * 68

    #a new dataframe where the home teams always attacks from left to right side throughout the game, away teams the opposite
    df_single_direction = helpers.to_single_playing_direction(df, helpers.find_playing_direction(df, matchdata))

    #a new dataframe with player and ball velocities in the x,y directions added.
    df_with_speed = helpers.calc_player_velocities(df_single_direction)

    '''
    Logic for extracting potential corner situations - Extract all rows from the game where the ball location is within 1 meter of the corner arc
    (1 meter is the actual distance of the corner taking arc from the corner). We also use a tolerance value (which can be adjusted, 1 meter currently).
    So, if the ball is within 2 meters of the corners of the pitch, we mark that row as a potential corner situation.

    More explanation in the report.
    '''
    #A dataframe with rows representing all the potential corner situations for the home team
    home_corner_situations = helpers.extract_near_corner_situations(df_with_speed, True)

    #A dataframe with rows representing all the potential corner situations for the home team
    away_corner_situations = helpers.extract_near_corner_situations(df_with_speed, False)

    #A list of dataframes consisting of frames till 10 seconds (100 skillcorner frames) from when the ball was first found within 2 meters of the corners of the pitch
    # & at least a chosen number of attacking players (4 in our case) were found inside the attacking penalty area within those 10 seconds.
    min_attacking_players = 4

    home_corners = helpers.extract_corners(df_with_speed, home_corner_situations, True, min_attacking_players)
    away_corners = helpers.extract_corners(df_with_speed, away_corner_situations, False, min_attacking_players)


    #save all the corner situations that were found within a game for the home & the away team.
    for i in range(len(home_corners)):
        game_id = str(matchdata_file.split("matches\\")[1].split("\\")[0])
        home_corners[i].to_csv(game_id + "_home_" + str(i) + "_.csv")

    for i in range(len(away_corners)):
        game_id = str(matchdata_file.split("matches\\")[1].split("\\")[0])
        away_corners[i].to_csv(game_id + "_away_" + str(i) + "_.csv")