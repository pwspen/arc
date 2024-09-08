import json
base_path='data/'
def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

training_challenges =  load_json(base_path +'arc-agi_training_challenges.json')

print(type(training_challenges))
#print(training_challenges)