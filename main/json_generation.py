import json
import os.path as osp
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import platform


# 1. N - Normal
# 2. V - PVC (Premature ventricular contraction)
# 3. \ - PAB (Paced beat)
# 4. R - RBB (Right bundle branch)
# 5. L - LBB (Left bundle branch)
# 6. A - APB (Atrial premature beat)
# 7. ! - AFW (Ventricular flutter wave)
# 8. E - VEB (Ventricular escape beat)


def get_slash():
    match (platform.system()):
        case "Linux":
            return '/'
        case "Windows":
            return "\\"


slash = get_slash()
CLASSES = ['N', 'V', slash, 'R', 'L', 'A', '!', 'E']
LEAD = "MLII"
extension = "npy"  # png for 2D
DATA_PATH = osp.abspath(f"../data/*/*/*/*/*.{extension}")
VAL_SIZE = 0.1  # [0, 1] for 2D
OUTPUT_PATH = '/'.join(DATA_PATH.split(slash)[:-5])
RANDOM_STATE = 7


def main():
    dataset = list()
    files = glob(DATA_PATH)
    for file in glob(DATA_PATH):
        # * - unpacking operator,  _, _, _, _, _, will be equivalent for *_
        *_, name, lead, label, filename = file.split("/")
        dataset.append(
            {
                "name": name,
                "lead": lead,
                "label": label,
                "filename": filename,
                "path": file
            }
        )

    data = pd.DataFrame(dataset)
    data = data[data["lead"] == LEAD]
    data = data[data["label"].isin(CLASSES)]
    data = data.sample(frac=1, random_state=RANDOM_STATE)

    train, valid = train_test_split(data, test_size=VAL_SIZE)

    # vals = list()
    # for c in CLASSES:
    #     vals.append(
    #         data[data["label"] == c]
    #         .sample(frac=VAL_SIZE, random_state=RANDOM_STATE)
    #         .index
    #     )
    #
    # valid = data.loc[vals, :]
    # train = data[~data.index.isin(valid.index)]

    train.to_json(osp.join(OUTPUT_PATH, "train.json"), orient="records")
    valid.to_json(osp.join(OUTPUT_PATH, "validation.json"), orient="records")

    label_dict = dict()
    for label in train.label.unique():
        label_dict[label] = len(label_dict)

    with open(osp.join(OUTPUT_PATH, "class-mapper.json"), "w") as file:
        file.write(json.dumps(label_dict, indent=1))


if __name__ == "__main__":
    main()
