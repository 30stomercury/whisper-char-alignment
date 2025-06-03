import re 

def parse_segment(phns, times, text):
    # all lists
    
    seg_length = [len(seg) for seg in times]
    time_points = [0]
    sum_l = time_points[0]
    for l in seg_length:
        sum_l += l* 0.01
        time_points.append(sum_l)
    print(time_points)

    word_index = 0
    word_boundaries = []
    words = text
    for i, entry in enumerate(phns):
        if entry.endswith('_S'):
            word_boundaries.append((words[word_index], time_points[i], time_points[i+1]))
            word_index += 1

        if entry.endswith('_B'):
            start_time = time_points[i]
        elif entry.endswith('_E'):
            # Ending the current word
            end_time = time_points[i+1]
            word_boundaries.append((words[word_index], start_time, end_time))
            word_index += 1
    assert word_index == len(words)
    return word_boundaries


# Example data
ftext = open('/home/htang2/ami_ihm_eval_tri4a/text')

texts = {}
for i, line in enumerate(ftext.readlines()):
    line_ = line.split()
    texts[line_[0]] = line_[1:]

data = {}
for i in range(1, 2):
    fali = open(f'/home/htang2/ami_ihm_eval_tri4a/ali.{i}.txt')
    for i, line in enumerate(fali.readlines()):
        if i % 3 == 0:
            time_segments = []
            for match in re.findall(r'\[([0-9\s]+)\]', line):
                time_segments.append(list(map(int, match.split())))
        elif i % 3 == 1:
            phns = line.split()
            assert len(phns[1:]) == len(time_segments)
            data[phns[0]] = {'phns': phns[1:], 'segs': time_segments, 'text': texts[phns[0]]}
    fali.close()
        
results = {}
for k in data.keys():
    phns = data[k]['phns']
    times = data[k]['segs']
    text = data[k]['text']
    results[k] = parse_segment(phns, times, text)

# Display the result
print(results)
