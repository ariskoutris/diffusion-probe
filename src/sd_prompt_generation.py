# %%
from nltk.corpus import wordnet as wn
import networkx as nx
import os

# %%
WORK_DIR = "D:/documents/prog/wordnet_diffusion"
DATA_DIR = f'{WORK_DIR}/data'
IMG_DIR = f'{DATA_DIR}/images'
VEC_DIR = f'{DATA_DIR}/vectors'
SCRIPT_DIR = f'{DATA_DIR}/sd_scripts'

# %%
hyper = lambda s: s.hypernyms()
hypo = lambda s: s.hyponyms()
get_name = lambda x: x.lemmas()[0].name().replace("_"," ").title()

def create_hypotree(root_synset, max_depth=1, frequency_threshold=0):
    
    def create_tree_aux(synset, graph, depth):
        graph.add_node(synset.name(), name=get_name(synset), frequency=synset.lemmas()[0].count(), definition=synset.definition())
        if depth < max_depth:
            for hyponym in synset.hyponyms():
                hyponym_freq = hyponym.lemmas()[0].count()
                if hyponym_freq >= frequency_threshold:
                    create_tree_aux(hyponym, G, depth+1)
                    G.add_edge(synset.name(), hyponym.name(), similarity=synset.wup_similarity(hyponym))

    G = nx.Graph()
    G.add_node(root_synset.name(), name=get_name(root_synset), frequency=root_synset.lemmas()[0].count(), definition=root_synset.definition())
    for hyponym in root_synset.hyponyms():
        hyponym_freq = hyponym.lemmas()[0].count()
        if hyponym_freq >= frequency_threshold:
            create_tree_aux(hyponym, G, 1)
            G.add_edge(root_synset.name(), hyponym.name(), similarity=root_synset.wup_similarity(hyponym))
    return G

def get_node_name(G,node):
    return G.nodes[node]['name']
    
def add_prompt_weights(words, parentheses=False):
    if parentheses:
        weighted_words = [f'({words[0]})']
    else:
        weighted_words = [f'{words[0]}']
    for i,word in enumerate(words[1:]):
        weight = 1/(i+3)
        if parentheses:
            weighted_words.append(f'({word} :{weight:.2f})')
        else:
            weighted_words.append(f'{word} :{weight:.2f}')
    return weighted_words

# %%
root_name = 'building'
root_synset = wn.synsets(root_name)[0]
G = create_hypotree(root_synset, max_depth=2, frequency_threshold=0)
root_node = list(G.nodes)[0]
root_category = get_node_name(G,root_node)

# %% [markdown]
# # Leaves Only

# %%
tree = nx.bfs_tree(G, root_node)
leaves = [x for x in tree.nodes() if tree.out_degree(x)==0 and tree.in_degree(x)==1]
leave_paths = [list(nx.dfs_tree(tree.reverse(), source=x).nodes) for x in leaves]
leave_name_paths = [[get_node_name(G,n) for n in path] for path in leave_paths]
weighted_words = [add_prompt_weights(words) for words in leave_name_paths]
prompts = [' AND a '.join([f'a {word_list[0]}, photograph'] + word_list[1:]) for word_list in weighted_words]
prompts = [f'{prompt} AND a {root_category}, {G.nodes[word]["definition"]} :0.2' for prompt,word in zip(prompts,leaves)]

# %%
results_dir = f"{SCRIPT_DIR}/{root_category}"
sd_params = f'--steps 35 --batch_size 8 --n_iter 3 --cfg_scale 5.5 --sampler_name "LMS"'

with open(f'{results_dir}/leaves.txt','w') as f:
    for subcategory, prompt in zip(leaves, prompts):
        outpath_samples = f'{IMG_DIR}/{root_category}/{get_node_name(G,subcategory)}'
        if not os.path.exists(outpath_samples):
            os.makedirs(outpath_samples)
        command = f'--prompt "{prompt}" '
        command += sd_params
        command += f' --outpath_samples "{outpath_samples}"'
        f.write(command + '\n')

# %% [markdown]
# ## Custom (dog)

# %%
tree = nx.bfs_tree(G, root_node)
leaves = [x for x in tree.nodes() if tree.out_degree(x)==0 and tree.in_degree(x)==1]
leave_paths = [list(nx.dfs_tree(tree.reverse(), source=x).nodes) for x in leaves]
leave_name_paths = [[get_node_name(G,n) for n in path] for path in leave_paths]
weighted_words = [add_prompt_weights(words) for words in leave_name_paths]
prompts = [' AND a '.join([f'a dog of breed {word_list[0]}, photograph'] + word_list[1:]) for word_list in weighted_words]
prompts = [f'{prompt} AND a {root_category}, {G.nodes[word]["definition"]} :0.2' for prompt,word in zip(prompts,leaves)]

# %%
if root_category == 'Dog':
    results_dir = f"{SCRIPT_DIR}/{root_category}"
    sd_params = f'--steps 35 --batch_size 8 --n_iter 3 --cfg_scale 5.5 --sampler_name "LMS"'

    with open(f'{results_dir}/leaves_alt.txt','w') as f:
        for subcategory, prompt in zip(leaves, prompts):
            outpath_samples = f'{IMG_DIR}/{root_category}/{get_node_name(G,subcategory)}'
            if not os.path.exists(outpath_samples):
                os.makedirs(outpath_samples)
            command = f'--prompt "{prompt}" '
            command += sd_params
            command += f' --outpath_samples "{outpath_samples}"'
            f.write(command + '\n')

# %% [markdown]
# ## Custom (building)

# %%
tree = nx.bfs_tree(G, root_node)
leaves = [x for x in tree.nodes() if tree.out_degree(x)==0 and tree.in_degree(x)==1]
leave_paths = [list(nx.dfs_tree(tree.reverse(), source=x).nodes) for x in leaves]
leave_name_paths = [[get_node_name(G,n) for n in path] for path in leave_paths]
weighted_words = [add_prompt_weights(words) for words in leave_name_paths]
prompts = [' AND a '.join([f'a building of type {word_list[0]}, photograph'] + word_list[1:]) for word_list in weighted_words]
prompts = [f'{prompt} AND a {root_category}, {G.nodes[word]["definition"]} :0.2' for prompt,word in zip(prompts,leaves)]

# %%
if root_category == 'Building':
    results_dir = f"{SCRIPT_DIR}/{root_category}"
    sd_params = f'--steps 35 --batch_size 8 --n_iter 3 --cfg_scale 5.5 --sampler_name "LMS"'

    with open(f'{results_dir}/leaves_alt.txt','w') as f:
        for subcategory, prompt in zip(leaves, prompts):
            outpath_samples = f'{IMG_DIR}/{root_category}/{get_node_name(G,subcategory)}'
            if not os.path.exists(outpath_samples):
                os.makedirs(outpath_samples)
            command = f'--prompt "{prompt}" '
            command += sd_params
            command += f' --outpath_samples "{outpath_samples}"'
            f.write(command + '\n')

# %% [markdown]
# # Whole Graph

# %%
nodes = list(nx.single_source_shortest_path(G, root_synset.name()).keys())
paths = list(path[::-1] for path in nx.single_source_shortest_path(G, root_synset.name()).values())
name_paths = [[get_node_name(G,n) for n in path] for path in paths]
weighted_words = [add_prompt_weights(words) for words in name_paths]
prompts = [' AND a '.join([f'a {word_list[0]}, photograph'] + word_list[1:]) for word_list in weighted_words]
prompts = [f'{prompt} AND a {root_category}, {G.nodes[word]["definition"]} :0.2' for prompt,word in zip(prompts,nodes)]

# %%
results_dir = f"{SCRIPT_DIR}/{root_category}"
sd_params = f'--steps 35 --batch_size 8 --n_iter 3 --cfg_scale 5.5 --sampler_name "LMS"'

with open(f'{results_dir}/graph.txt','w') as f:
    for subcategory, prompt in zip(nodes, prompts):
        outpath_samples = f'{IMG_DIR}/{root_category}/{get_node_name(G,subcategory)}'
        if not os.path.exists(outpath_samples):
            os.makedirs(outpath_samples)
        command = f'--prompt "{prompt}" '
        command += sd_params
        command += f' --outpath_samples "{outpath_samples}"'
        f.write(command + '\n')

# %% [markdown]
# ## Custom (dog)

# %%
nodes = list(nx.single_source_shortest_path(G, root_synset.name()).keys())
paths = list(path[::-1] for path in nx.single_source_shortest_path(G, root_synset.name()).values())
name_paths = [[get_node_name(G,n) for n in path] for path in paths]
weighted_words = [add_prompt_weights(words) for words in name_paths]
prompts = [' AND a '.join([f'a dog of breed {word_list[0]}, photograph'] + word_list[1:]) for word_list in weighted_words]
prompts = [f'{prompt} AND a {root_category}, {G.nodes[word]["definition"]} :0.2' for prompt,word in zip(prompts,nodes)]

# %%
if root_category == 'Dog':
    results_dir = f"{SCRIPT_DIR}/{root_category}"
    sd_params = f'--steps 35 --batch_size 8 --n_iter 3 --cfg_scale 5.5 --sampler_name "LMS"'

    with open(f'{results_dir}/graph_alt.txt','w') as f:
        for subcategory, prompt in zip(nodes, prompts):
            outpath_samples = f'{IMG_DIR}/{root_category}/{get_node_name(G,subcategory)}'
            if not os.path.exists(outpath_samples):
                os.makedirs(outpath_samples)
            command = f'--prompt "{prompt}" '
            command += sd_params
            command += f' --outpath_samples "{outpath_samples}"'
            f.write(command + '\n')

# %% [markdown]
# ## Custom (building)

# %%
nodes = list(nx.single_source_shortest_path(G, root_synset.name()).keys())
paths = list(path[::-1] for path in nx.single_source_shortest_path(G, root_synset.name()).values())
name_paths = [[get_node_name(G,n) for n in path] for path in paths]
weighted_words = [add_prompt_weights(words) for words in name_paths]
prompts = [' AND a '.join([f'a building of type {word_list[0]}, photograph'] + word_list[1:]) for word_list in weighted_words]
prompts = [f'{prompt} AND a {root_category}, {G.nodes[word]["definition"]} :0.2' for prompt,word in zip(prompts,nodes)]

# %%
if root_category == 'Building':
    results_dir = f"{SCRIPT_DIR}/{root_category}"
    sd_params = f'--steps 35 --batch_size 8 --n_iter 3 --cfg_scale 5.5 --sampler_name "LMS"'

    with open(f'{results_dir}/graph_alt.txt','w') as f:
        for subcategory, prompt in zip(nodes, prompts):
            outpath_samples = f'{IMG_DIR}/{root_category}/{get_node_name(G,subcategory)}'
            if not os.path.exists(outpath_samples):
                os.makedirs(outpath_samples)
            command = f'--prompt "{prompt}" '
            command += sd_params
            command += f' --outpath_samples "{outpath_samples}"'
            f.write(command + '\n')

# %% [markdown]
# # Parents of Leaves

# %% [markdown]
# ## Custom (dog)

# %%
tree = nx.bfs_tree(G, root_node)
leaves = [x for x in tree.nodes() if tree.out_degree(x)==0 and tree.in_degree(x)==1]
nodes = list(nx.single_source_shortest_path(G, root_synset.name()).keys())
pleaves = set(nodes).difference(set(leaves))
pleave_paths = [list(nx.dfs_tree(tree.reverse(), source=x).nodes) for x in pleaves]
pleave_name_paths = [[get_node_name(G,n) for n in path] for path in pleave_paths]
weighted_words = [add_prompt_weights(words) for words in pleave_name_paths]
prompts = [' AND a '.join([f'a dog of breed {word_list[0]}, photograph'] + word_list[1:]) for word_list in weighted_words]
prompts = [f'{prompt} AND a {root_category}, {G.nodes[word]["definition"]} :0.2' for prompt,word in zip(prompts,pleaves)]

# %%
if root_category == 'Dog':
    results_dir = f"{SCRIPT_DIR}/{root_category}"
    sd_params = f'--steps 35 --batch_size 8 --n_iter 10 --cfg_scale 5.5 --sampler_name "LMS"'

    with open(f'{results_dir}/pleaves.txt','w') as f:
        for subcategory, prompt in zip(pleaves, prompts):
            outpath_samples = f'{IMG_DIR}/{root_category}/{get_node_name(G,subcategory)}'
            if not os.path.exists(outpath_samples):
                os.makedirs(outpath_samples)
            command = f'--prompt "{prompt}" '
            command += sd_params
            command += f' --outpath_samples "{outpath_samples}"'
            f.write(command + '\n')

# %% [markdown]
# ## Custom (building)

# %%
tree = nx.bfs_tree(G, root_node)
leaves = [x for x in tree.nodes() if tree.out_degree(x)==0 and tree.in_degree(x)==1]
nodes = list(nx.single_source_shortest_path(G, root_synset.name()).keys())
pleaves = set(nodes).difference(set(leaves))
pleave_paths = [list(nx.dfs_tree(tree.reverse(), source=x).nodes) for x in pleaves]
pleave_name_paths = [[get_node_name(G,n) for n in path] for path in pleave_paths]
weighted_words = [add_prompt_weights(words) for words in pleave_name_paths]
prompts = [' AND a '.join([f'a building of type {word_list[0]}, photograph'] + word_list[1:]) for word_list in weighted_words]
prompts = [f'{prompt} AND a {root_category}, {G.nodes[word]["definition"]} :0.2' for prompt,word in zip(prompts,pleaves)]

# %%
if root_category == 'Building':
    results_dir = f"{SCRIPT_DIR}/{root_category}"
    sd_params = f'--steps 35 --batch_size 8 --n_iter 10 --cfg_scale 5.5 --sampler_name "LMS"'

    with open(f'{results_dir}/pleaves.txt','w') as f:
        for subcategory, prompt in zip(pleaves, prompts):
            outpath_samples = f'{IMG_DIR}/{root_category}/{get_node_name(G,subcategory)}'
            if not os.path.exists(outpath_samples):
                os.makedirs(outpath_samples)
            command = f'--prompt "{prompt}" '
            command += sd_params
            command += f' --outpath_samples "{outpath_samples}"'
            f.write(command + '\n')


