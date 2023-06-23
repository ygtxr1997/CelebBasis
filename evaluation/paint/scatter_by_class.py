import matplotlib.pyplot as plt
import numpy as np


def read_txt(txt_path: str, split_code: str = '\t'):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        row = line.strip().split(split_code)
        row = [float(x) for x in row]
        data.append(row)
    data_np = np.array(data)
    print(f'[Read txt] from {txt_path}', data_np.shape)
    return data_np


def split_data(data: np.ndarray, task_per_cls: int = 4):
    data = np.split(data, task_per_cls)
    return data


def paint_a_cls(data_a_cls: np.ndarray,
                markers: list = None,
                edge_color: str = 'blue',
                s: int = 400,
                ):
    rows, cols = data_a_cls.shape
    pxs = []
    for i in range(rows):
        x = data_a_cls[i, 0]
        y = data_a_cls[i, 1]
        s = s
        marker = markers[i] if markers is not None else 'o'
        px = plt.scatter(x, y,
                         s=s,
                         marker=marker,
                         facecolors='none',
                         edgecolors=edge_color,
                         linewidths=3,
                         )
        pxs.append(px)

    x_mean, x_std = data_a_cls[:, 0].mean(), data_a_cls[:, 0].std()
    y_mean, y_std = data_a_cls[:, 1].mean(), data_a_cls[:, 1].std()
    px = plt.scatter(x_mean, y_mean,
                     s=s * 0.9,
                     marker='o',
                     facecolors=edge_color,
                     edgecolors=edge_color,
                     linewidths=3,
                     )
    plt.errorbar(x_mean, y_mean,
                 xerr=x_std, yerr=y_std,
                 fmt='o',
                 color=edge_color,
                 ecolor=edge_color,
                 elinewidth=3,
                 capsize=20,
                 capthick=3,
                 )
    pxs.append(px)
    return pxs


def start_gen(txt_path: str):
    plt.figure(figsize=(12, 9))
    ax = plt.subplot()
    plt.subplots_adjust(top=0.86)

    plt.xlabel('Identity Similarity', fontsize=30)
    plt.ylabel('Prompt Similarity', fontsize=30)
    plt.yticks(size=25)
    plt.xticks(size=25)

    data = read_txt(txt_path)
    datas_by_cls = split_data(data)
    colors_by_cls = ['blue', 'black', 'green', 'red']
    means_by_cls = []

    for idx, data_a_cls in enumerate(datas_by_cls):
        pxs = paint_a_cls(data_a_cls,
                          markers=['o', 'v', 's', '*'],
                          edge_color=colors_by_cls[idx],
                          )
        means_by_cls.append(pxs[-1])

        if idx == 1:
            legend1 = ax.legend(pxs, ['Style', 'Single', 'With Celeb', 'Double'],
                                fontsize=25, loc='lower center',
                                ncols=2,
                                )

        if idx == len(datas_by_cls) - 1:
            ax.legend(means_by_cls, ['Textual Inversion', 'DreamBooth', 'Custom Diffusion', 'Ours'],
                      bbox_to_anchor=(0., 1.07, 1., .102),
                      loc='upper center',
                      borderaxespad=0.,
                      ncols=2,
                      fontsize=25,
                      markerscale=0.75,
                      edgecolor='none'
                      )
            plt.gca().add_artist(legend1)

    plt.savefig(txt_path.replace('txt', 'pdf'), bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":
    start_gen('./exp_id-txt_sota.txt')

