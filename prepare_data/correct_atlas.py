import shutil

# Read in the file
with open('../data/atlas.json', 'r') as file :
  filedata = file.read()

# Replace the target string
filedata = filedata.replace('\"2b2tplace\"]}, \"path\": {\"1-165, T:0-1\": [[913, 421], [909, 482], [959, 480], [958, 421]]',
                            '\"2b2tplace\"]}, \"path\": {\"1-165, T:0-1\": [[912, 421], [912, 481], [960, 481], [960, 421]]')

filedata = filedata.replace('{\"id\": \"tykicd\", \"name\": \"Snoo\", \"description\": \"Outside the r/place canvas, There were two Snoo (Reddit\'s mascot) and a dog looking off the edge of the map while on a sled. While not part of the canvas itself, They can be seen in the r/place timelapse.\", \"links\": {\"website\": [\"https://www.reddit.com/r/place/?cx=3&cy=0&px=6&ts=1649112460185\"], \"subreddit\": [\"place\"]}, \"path\": {\"1-164\": [[-2, -3], [4, -3], [4, 0], [-2, 0]]}, \"center\": {\"1-164\": [0, 0]}},', '')

filedata = filedata.replace('\"T\"', '\"T:0-1\"')

filedata = filedata.replace('Java Discord server. discord.gg/java\", \"links\": {}, \"path\": {\"56-165, T:0-1\": [[1215, 870], [1224, 870], [1224, 879], [1215, 879]]',
                            'Java Discord server. discord.gg/java\", \"links\": {}, \"path\": {\"56-165, T:0-1\": [[1215, 870], [1226, 870], [1226, 880], [1215, 880]]')

filedata = filedata.replace('xQc before being reinstated.", "links": {"website": ["https://en.wikipedia.org/wiki/Star_Wars_(film)"], "subreddit": ["starwars_place"]}, "path": {"1-166, T:0-1": [[570, 698], [671, 698], [671, 844], [570, 844]]}',
                            'xQc before being reinstated.", "links": {"website": ["https://en.wikipedia.org/wiki/Star_Wars_(film)"], "subreddit": ["starwars_place"]}, "path": {"1-166, T:0-1": [[571, 699], [670, 699], [670, 843], [571, 843]]}')
# Write the file out again
with open('../data/atlas.json', 'w') as file:
  file.write(filedata)
