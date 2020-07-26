from collections import defaultdict

_lattice_edge_keys = ['from_id', 'to_id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'token_id']
_conll_node_keys = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']


class LatticeEdge:

    def __init__(self, values: dict):
        self._values = values

    def __getitem__(self, key):
        return self._values[key]

    def __str__(self):
        return str(self._values)

    def __repr__(self):
        return self.__str__()
    
    @property
    def form(self) -> str:
        return self['form']

    @property
    def lemma(self) -> str:
        return self['lemma']

    @property
    def postag(self) -> str:
        return self['postag']

    @property
    def cpostag(self) -> str:
        return self['cpostag']
    
    @property
    def features(self) -> dict:
        result = {}
        l = [f for feats in str(self['feats']).split("|") for f in feats.split("=")]
        for feat, value in zip(l[0::2], l[1::2]):
            result.setdefault(feat,[]).append(value)
        return result

    @property
    def from_id(self) -> int:
        return int(self['from_id'])

    @property
    def to_id(self) -> int:
        return int(self['to_id'])

    @property
    def token_id(self) -> int:
        return int(self['token_id'])


class LatticeNode:

    def __init__(self, id: int, token_id: int, edges: list):
        self.id = id
        self.token_id = token_id
        self._edges = edges

    def __getitem__(self, key) -> LatticeEdge:
        return self._edges[key]

    def __iter__(self):
        return iter(self._edges)

    def __str__(self):
        return str(self._edges)

    def __repr__(self):
        return self.__str__()


class LatticeGraph:

    def __init__(self, edges: list, nodes: dict, token_paths: dict):
        self._edges = edges
        self._nodes = nodes
        self._token_paths = token_paths

    def __iter__(self):
        return iter(self._nodes)

    def __getitem__(self, id) -> LatticeNode:
        return self._nodes[id]

    def __str__(self):
        return str(self._nodes)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self._nodes)

    def token_node_id(self, token_id: int) -> int:
        return self._token_paths[token_id][0][0].from_id


def _get_token_paths(token_id: int, node_id: int, nodes: dict) -> list:
    node = nodes[node_id]
    paths = []
    for edge in node:
        edge_paths = []
        _dfs_token_paths(edge, token_id, [], edge_paths, nodes)
        paths.extend(edge_paths)
    return paths


def _dfs_token_paths(edge: LatticeEdge, token_id: int, cur_path: list, paths: list, nodes: dict):
    if edge.token_id != token_id:
        return
    cur_path.append(edge)
    if edge.to_id not in nodes:
        paths.append(cur_path)
    else:
        next_node = nodes[edge.to_id]
        for next_edge in next_node:
            edge_path = cur_path.copy()
            # print(edge)
            _dfs_token_paths(next_edge, token_id, edge_path, paths, nodes)
        if next_node.token_id != token_id:
            paths.append(cur_path)


class DependencyNode:

    def __init__(self, values: dict):
        self._values = values

    def __getitem__(self, key):
        return self._values[key]

    def __str__(self):
        return str(self._values)

    def __repr__(self):
        return self.__str__()


class DependencyTree:
    
    def __init__(self, nodes: dict):
        self._nodes = nodes

    def __iter__(self):
        return iter(self._nodes)

    def __getitem__(self, id) -> DependencyNode:
        return self._nodes[id]

    def __str__(self):
        return str(self._nodes)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self._nodes)


def read_tokens(lines: list) -> list:
    return [line for line in lines]


def read_lattice(lines: list) -> LatticeGraph:
    edges = []
    lattice_nodes = defaultdict(list)
    lattice_token_nodes = defaultdict(list)
    for line in lines:
        parts = line.split('\t')
        edge = LatticeEdge(dict(zip(_lattice_edge_keys, parts)))
        edges.append(edge)
        lattice_nodes[edge.from_id].append(edge)
        lattice_token_nodes[edge.token_id].append(edge.from_id)
    nodes = {}
    token_nodes = defaultdict(list)
    for node_id, node_edges in lattice_nodes.items():
        token_id = node_edges[0].token_id
        node = LatticeNode(node_id, token_id, node_edges)
        nodes[node_id] = node
        token_nodes[token_id].append(node)
    token_paths = {}
    for token_id, token_nodes in token_nodes.items():
        sorted_token_nodes = sorted([t.id for t in token_nodes])
        node_id = sorted_token_nodes[0]
        paths = _get_token_paths(token_id, node_id, nodes)
        token_paths[token_id] = paths
    return LatticeGraph(edges, nodes, token_paths)


def read_dep_tree(lines: list) -> DependencyTree:
    nodes = defaultdict(list)
    for line in lines:
        parts = line.split('\t')
        node = DependencyNode(dict(zip(_conll_node_keys, parts)))
        nodes[int(node['id'])].append(node)
    return DependencyTree(nodes)


def format_edge(edge: LatticeEdge):
    return '\t'.join([str(edge.from_id), str(edge.to_id), edge.form, edge.lemma, edge.postag, edge.cpostag, str(edge['feats']), str(edge.token_id)])


def write_lattice(g: LatticeGraph, f):
    for node in g:
        for edge in g[node]:
            line = format_edge(edge)
            f.write(line)
            f.write('\n')
