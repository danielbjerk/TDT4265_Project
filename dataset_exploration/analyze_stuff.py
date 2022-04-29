from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def analyze_something(dataloader, cfg):
    # Key: class
    # Value: tuple (sum, count)
    box_size_per_label = {0: [0,0], 1: [0,0], 2: [0,0], 3: [0,0], 4: [0,0], 5: [0,0], 6: [0,0], 7: [0,0], 8: [0,0]}    
    all_box_size_per_label = [[] for _ in range(10)]

    dimension_per_label = {0: ([], []), 1: ([], []), 2: ([], []), 3: ([], []), 4: ([], []), 5: ([], []), 6: ([], []), 7: ([], []), 8: ([], [])}

    for batch in tqdm(dataloader):

        for (label, box) in zip(batch["labels"][0].tolist(), batch["boxes"][0].tolist()):
            area = (box[2] - box[0]) * (box[3] - box[1])
            box_size_per_label[label][0] += area
            box_size_per_label[label][1] += 1
            all_box_size_per_label[label].append(area)
            width = box[2] - box[0]
            height = box[3] - box[1]
            dimension_per_label[label][0].append(width * 1024)
            dimension_per_label[label][1].append(height * 128)
    
    average_area = {}

    for (key, entry) in box_size_per_label.items():
        if(entry[1] == 0):
            average_area[key] = 0
        else:
            average_area[key] = entry[0] / entry[1]

    counts = []
    for key, value in box_size_per_label.items():
        counts.append(value[1])


    plt.rcParams["figure.figsize"] = (10, 5)


    classes = ['bakgrunn', 'bil', 'lastebil', 'buss', 'motorsykkel', 'sykkel', 'sparkesykkel', 'person', 'syklist']
    """
    plt.bar(classes, counts)
    plt.xlabel("Klasse")
    plt.ylabel("Antall")
    plt.title("Frekvens av forskjellige klasser")
    plt.savefig("./figures/task1-1-histogram.png")
    plt.show()
    """


    for (i, x) in enumerate(classes):
        plt.figure()
        plt.grid()
        plt.scatter(dimension_per_label[i][0], dimension_per_label[i][1])
        plt.xlabel("bredde")
        plt.ylabel("høyde")
        plt.title(f"Dimensjon av {x}")
        # if(len(dimension_per_label[i][0])):
        #     m, b = np.polyfit(dimension_per_label[i][0], dimension_per_label[i][1], 1)
        #     regression_text = "Regresjon: " + str(m)+ "x + " + str(b) 
        #     plt.plot(dimension_per_label[i][0], m*dimension_per_label[i][0]) + b, label=regression_text)

        # plt.legend()
        plt.savefig(f"./figures/task1-1-index_{i}_bounding_boxes.png")


    """
    plt.bar(classes, average_area.values())
    plt.xlabel("Klasse")
    plt.ylabel("Gjennomsnittlig areal")
    plt.title("Gjennomsnittlig areal")
    plt.savefig("./figures/task1-1-area.png")
    plt.show()
    """

def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == '__main__':
    main()
