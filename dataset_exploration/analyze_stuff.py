from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import matplotlib.pyplot as plt


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

    for batch in tqdm(dataloader):

        for (label, box) in zip(batch["labels"][0].tolist(), batch["boxes"][0].tolist()):
            area = (box[2] - box[0]) * (box[3] - box[1])
            box_size_per_label[label][0] += area
            box_size_per_label[label][1] += 1
            all_box_size_per_label[label].append(area)
    
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
    plt.bar(classes, counts)
    plt.xlabel("Klasse")
    plt.ylabel("Antall")
    plt.title("Frekvens av forskjellige klasser")
    plt.savefig("./figures/task1-1-histogram.png")
    plt.show()

    plt.bar(classes, average_area.values())
    plt.xlabel("Klasse")
    plt.ylabel("Gjennomsnittlig areal")
    plt.title("Gjennomsnittlig areal")
    plt.savefig("./figures/task1-1-area.png")
    plt.show()

def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == '__main__':
    main()
