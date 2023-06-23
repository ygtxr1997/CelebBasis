import csv
import numpy as np


def read_csv(csv_path: str,
             questions: int = 140,
             start_line: int = 22,
             cycle: int = 7,
             cnt_per_method: int = 35,
             ):
    lines = []
    with open(csv_path, mode='rt', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            lines.append(line)

    quality = []
    prompt = []
    face = []

    idx = start_line
    for q in range(questions):
        q1 = lines[idx + 2]
        q2 = lines[idx + 3]
        q3 = lines[idx + 4]

        score, people = parse_line(q1)
        if people > 0:
            avg_score = score / people
            quality.append(avg_score)
        else:
            quality.append(None)

        score, people = parse_line(q2)
        if people > 0:
            avg_score = score / people
            prompt.append(avg_score)
        else:
            quality.append(None)

        score, people = parse_line(q3)
        if people > 0:
            avg_score = score / people
            face.append(avg_score)
        else:
            quality.append(None)

        idx += cycle

        if q in (65, 104):
            idx += 2


    lo, hi = 0, cnt_per_method
    methods = questions // cnt_per_method
    for m in range(methods):
        a_quality = list(filter(None, quality[lo: hi]))
        a_prompt = list(filter(None, prompt[lo: hi]))
        a_face = list(filter(None, face[lo: hi]))

        lo = hi
        hi += cnt_per_method

        print('-' * 10, 'method:', m, '-' * 10)
        print('quality:', np.array(a_quality).mean())
        print('prompt:', np.array(a_prompt).mean())
        print('face:', np.array(a_face).mean())


def parse_line(line: list):
    n = len(line)
    score = 0
    people = 0
    for i in range(n):
        if i == 0:
            continue
        if len(line[i]) < 1:
            continue
        score += i * int(line[i])
        people += int(line[i])
    return score, people


if __name__ == "__main__":
    csv_name = './survey.csv'
    read_csv(csv_name)
