FILE DESCRIPTION:


extract_corners.py - This file extracts potential corner situations from all the 9 skillcorner games and stores frames till 10 seconds of when a corner situation was detected.

load_corner.py - inherited from Kloppy, used to load the skill corner data. 2 extra if statements have been added to take care of 2 games which had issues working with existing kloppy code.

helpers.py - helpers module contaning the functions/logic used by extract_corners.py to extract corner situations.

corners (folder) - a folder with all the extracted corner situations stored in a csv format. files are named in the format - "game_id" + team taking the corner (home/away) +  corner number

analysis_source_code.ipynb - Jupyter notebook to analyze the extracted corner situations and create visualizations.

Report_Harsh_Mishra - Report descirbing the methodology and other details about the submission.

Opposition Corners Analysis.pdf - Presentation for the coaching staff
