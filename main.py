import numpy as np

import matplotlib.pyplot as plt
from utils import create_barplot, create_sub_barplot, create_sub_heatmap, aprint, load_json, load_all_tasks, show_id_list
from viz import Visualizer

tasklist = ['007bbfb7', '017c7c7b']

#show_id_list(tasklist)

# v = Visualizer(['007bbfb7', '017c7c7b'])

base_path='/arc/data/'

training_challenges, training_solutions, evaluation_challenges, evaluation_solutions = load_all_tasks()

if 0:
    fig, axs = plt.subplots(2, 4, figsize=(10, 6))
    for i, challenges, solutions in zip([0, 1], [training_challenges, evaluation_challenges], [training_solutions, evaluation_solutions]):
        train_pairs = [challenges[c]['train'] for c in challenges] # list of: [{'input': in1, 'output': out1}, {'input': in2, 'output': out2}, ..]
        test_inputs = [challenges[c]['test'] for c in challenges] # list of: [{'input': in1}, {'input': in2}, ..] (usually just 1)

        dataset = 'eval' if i else 'train'

        create_sub_barplot(axs[i, 0], [len(p) for p in train_pairs], 'Training inputs dist.', 'Number of inputs', f'Tasks ({dataset} set)')
        create_sub_barplot(axs[i, 1], [len(p) for p in test_inputs], 'Test inputs dist.', 'Number of inputs', f'Tasks ({dataset} set)')

        train_shapes = [np.shape(input_dict['input'])
                        for task_inputs in train_pairs 
                        for input_dict in task_inputs]
        create_sub_heatmap(axs[i, 2], train_shapes, 'Training inputs shape', 'Rows', f'Columns ({dataset} set)')
        
        # For test inputs heatmap
        test_shapes = [np.shape(input_dict['input'])
                    for task_inputs in test_inputs 
                    for input_dict in task_inputs]
        create_sub_heatmap(axs[i, 3], test_shapes, 'Test inputs shape', 'Rows', f'Columns ({dataset} set)')

        plt.tight_layout()
        plt.show()

def find_mappings(i, j, k, a, b):
    def find_single_mapping(inputs, output):
        input_names = ['i', 'j', 'k']
        for idx, x in enumerate(inputs):
            # Special Case 1: y = x
            if np.allclose(x, output, rtol=1e-5):
                return lambda x: x, input_names[idx]

        for idx, x in enumerate(inputs): # Swapping 2 and 3 swaps a task for another
            # Special Case 2: y = c
            if np.allclose(output, output[0], rtol=1e-5):
                return lambda x: np.full_like(x, output[0]), input_names[idx]

        for idx, x in enumerate(inputs):
            # Special Case 3: y = m * x
            m = output / x
            if np.allclose(m, m[0], rtol=1e-5):
                return lambda x, m=m[0]: m * x, input_names[idx]

        for idx, x in enumerate(inputs):
            # Special Case 4: y = x^n (n=0.5, 2)
            for n in [0.5, 2]:
                m = output / (x ** n)
                if np.allclose(m, m[0], rtol=1e-5):
                    return lambda x, m=m[0], n=n: m * (x ** n), input_names[idx]

        # Algorithm: y = m * x^n + c
        for idx, x in enumerate(inputs):
            for n in [0.5, 1, 2]:
                for c in np.arange(-20, 20.1, 0.1):
                    m = (output - c) / (x ** n)
                    if np.allclose(m, m[0], rtol=1e-5):
                        return lambda x, m=m[0], n=n, c=c: m * (x ** n) + c, input_names[idx]

        return None, None

    if np.allclose(i, a, rtol=1e-5) and np.allclose(j, b, rtol=1e-5): # if inputdims=outputdims (encountered issue with 3f23242b)
        return (lambda x: x, 'i'), (lambda x: x, 'j')

    map1, input1 = find_single_mapping([i, j, k], a)
    map2, input2 = find_single_mapping([i, j, k], b)

    return (map1, input1), (map2, input2)

def apply_mapping(mapping_info, i, j, k):
    if mapping_info is None or mapping_info[0] is None:
        return None
    mapping, input_var = mapping_info
    input_dict = {'i': i, 'j': j, 'k': k}
    return mapping(input_dict[input_var])

challenges, solutions = evaluation_challenges, evaluation_solutions

correct = 0
incorrect = []
singles = ['93b4f4b3']
v = Visualizer(singles)
for i in range(len(challenges)):
    t=list(challenges)[i] # t is 'hash' of the task (8 char string)
    print(t)

    task=challenges[t] # {'train':[..], 'test':[..]}
    ans=solutions[t] # ordered array of solution(s) (typically 1)

    train = task['train'] # [{'input': in1, 'output': out1}, {'input': in2, 'output': out2}, ..]
    test = task['test'] # [{'input': in1}, {'input': in2}, ..] (usually just 1)
  
    assert len(test) == len(ans)

    # test[0]['input'] corresponds to ans[0]

    if t in singles:
        pass

    train_input_shapes = np.array([np.shape(pair['input']) for pair in train])
    # num_colors = np.array([np.unique(np.array([pair['input']])) for pair in train])
    num_colors = np.array([len(np.unique(pair['input'])) for pair in train])
    xdimin, ydimin = train_input_shapes[:, 0], train_input_shapes[:, 1]

    print(xdimin, ydimin, num_colors)

    train_output_shapes = np.array([np.shape(pair['output']) for pair in train])
    xdimout, ydimout = train_output_shapes[:, 0], train_output_shapes[:, 1]

    test_input_shapes = np.array([np.shape(pair['input']) for pair in test])
    num_colors_test = np.array([len(np.unique(pair['input'])) for pair in test])
    xdimtest, ydimtest = test_input_shapes[:, 0], test_input_shapes[:, 1]

    xmap, ymap = find_mappings(xdimin, ydimin, num_colors, xdimout, ydimout)


    if xmap[0] is None or ymap[0] is None:
        incorrect.append(t)
        continue
    else:
        xdimguess = apply_mapping(xmap, xdimtest, ydimtest, num_colors_test)
        ydimguess = apply_mapping(ymap, xdimtest, ydimtest, num_colors_test)
        guess = np.array((xdimguess, ydimguess), dtype=int).T

        test_output_shapes = np.array([np.shape(a) for a in ans])

        if np.all(np.equal(guess, test_output_shapes)):
            print('woo!')
            correct += 1
        else:
            incorrect.append(t)
        # print(ratio)
        # print(same_ratio)

print(f'Correct: {correct}/400')
print(f'Incorrect: {len(incorrect)}/400')
if incorrect:
    v = Visualizer(incorrect)
# task = challenges['007bbfb7'] # Below output should be ['test', 'train'], 5 training pairs, 1 test pair
# print(task.keys())

# n_train_pairs = len(task['train'])
# n_test_pairs = len(task['test'])

# print(f'task contains {n_train_pairs} training pairs')
# print(f'task contains {n_test_pairs} test pairs')