import os
import shutil
import argparse

from evaluation.parse_args import parser_eval


def copy_to(eval_root: str = './exp_eval/ours',
            max_img: int = 2000,
            ):
    goto_folder = os.path.join(eval_root, 'all')
    os.makedirs(goto_folder, exist_ok=True)

    idx = len(os.listdir(goto_folder))

    eval_project_folders = os.listdir(eval_root)
    for i in range(len(eval_project_folders)):
        eval_project_folder = os.path.join(eval_root, eval_project_folders[i])

        if 'all' in eval_project_folder:
            continue

        eval_time_folders = os.listdir(eval_project_folder)
        for j in range(len(eval_time_folders)):
            eval_time_folder = os.path.join(eval_project_folder, eval_time_folders[j])

            if not os.path.exists(os.path.join(eval_time_folder, 'imgs')):
                continue

            img_folders = os.listdir(os.path.join(eval_time_folder, 'imgs'))
            for k in range(len(img_folders)):
                img_folder = os.path.join(eval_time_folder, 'imgs', img_folders[k])
                imgs = os.listdir(img_folder)
                for x in range(len(imgs)):
                    img_path = os.path.join(img_folder, imgs[x])

                    if idx >= max_img:
                        return idx, goto_folder

                    to_file = os.path.join(goto_folder, f"{idx:06d}.jpg")
                    shutil.copyfile(img_path, to_file)
                    print(idx, img_path)

                    idx += 1

    return idx, goto_folder


if __name__ == "__main__":
    """
    Usage:
    export PYTHONPATH=/gavin/code/TextualInversion/
    python evaluation/fid/copy_together.py --eval_out_dir ./exp_eval/db
    """
    parser = argparse.ArgumentParser()
    parser = parser_eval(parser)
    opt = parser.parse_args()

    r_idx, r_goto = copy_to(opt.eval_out_dir, max_img=10000)
    print(f'[FINISHED] {r_idx} files copied to {r_goto}.')
