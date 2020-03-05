import json
from utils.direction import Direction

def export_q_table(q_dict, file_name):
    q_dict_json = {}
    for state in q_dict.keys():
        q_dict_json[str(state)] = {}
        for action in q_dict[state].keys():
            q_dict_json[str(state)][str(action)] = q_dict[state][action]
    with open(file_name + ".json","w") as fp:
        json.dump(q_dict_json,fp, skipkeys=True)
    print("[!] Dumped Q-table to JSON file")

def import_q_table(q_table_path):
    with open(q_table_path,"r") as fp:
        q_dict_json = json.load(fp)
        q_dict = {}
        for str_state in q_dict_json.keys():
            q_dict[int(str_state)] = {}
            for str_action in q_dict_json[str_state].keys():
                suffix = str_action.split(".")[1]
                if suffix in ["UP", "RIGHT", "DOWN", "LEFT"]:
                    q_dict[int(str_state)][Direction.UP if suffix == "UP" else Direction.RIGHT if suffix == "RIGHT" else Direction.DOWN if suffix == "DOWN" else Direction.LEFT] = q_dict_json[str_state][str_action]
                else:
                    raise KeyError("JSON action key not recognized : "+str_action)
        return q_dict
