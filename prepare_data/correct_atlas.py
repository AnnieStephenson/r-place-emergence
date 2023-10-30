import shutil, os

year = '2022'

# Read in the file
with open(os.path.join('../data',year,'atlas.json'), 'r') as file :
  filedata = file.read()

# Replace the target string
filedata = filedata.replace('{\"id\": \"tykicd\", \"name\": \"Snoo\", \"description\": \"Outside the r/place canvas, There were two Snoo (Reddit\'s mascot) and a dog looking off the edge of the map while on a sled. While not part of the canvas itself, They can be seen in the r/place timelapse.\", \"links\": {\"website\": [\"https://www.reddit.com/r/place/?cx=3&cy=0&px=6&ts=1649112460185\"], \"subreddit\": [\"place\"]}, \"path\": {\"1-164\": [[-2, -3], [4, -3], [4, 0], [-2, 0]]}, \"center\": {\"1-164\": [0, 0]}},', '')

filedata = filedata.replace('\"T\"', '\"T:0-1\"')

# Write the file out again
with open(os.path.join('../data',year,'atlas.json'), 'w') as file:
  file.write(filedata)
