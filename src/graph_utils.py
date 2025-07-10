"""Утилиты для работы с графами молекул для современных GNN моделей."""

import torch
from rdkit import Chem
from torch_geometric.data import Data

from src.logging_config import get_logger

logger = get_logger(__name__)


def smiles_to_graph(smiles: str) -> Data | None:
    """Преобразует SMILES в граф для PyTorch Geometric.

    Args:
        smiles: SMILES строка

    Returns:
        Data объект с узлами и ребрами или None если ошибка
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Получаем атомы
        atom_features = []
        for atom in mol.GetAtoms():
            features = get_atom_features(atom)
            atom_features.append(features)

        # Получаем связи
        edge_indices = []
        edge_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Добавляем ребро в оба направления (undirected graph)
            edge_indices.extend([[i, j], [j, i]])

            # Получаем признаки связи
            bond_features = get_bond_features(bond)
            edge_features.extend([bond_features, bond_features])

        # Преобразуем в тензоры
        x = torch.tensor(atom_features, dtype=torch.float)

        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            # Если нет связей, создаем пустые тензоры
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float)  # 4 признака связи

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    except Exception as e:
        logger.warning(f"Ошибка при преобразовании SMILES {smiles}: {e}")
        return None


def get_atom_features(atom) -> list[float]:
    """Получает признаки атома.

    Args:
        atom: RDKit атом

    Returns:
        Список признаков атома
    """
    features = []

    # Атомный номер (one-hot encoding для основных элементов)
    atomic_nums = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]  # C, N, O, F, P, S, Cl, Br, I, Other
    atomic_num = atom.GetAtomicNum()
    atom_type = [0] * len(atomic_nums)
    if atomic_num in atomic_nums:
        atom_type[atomic_nums.index(atomic_num)] = 1
    else:
        atom_type[-1] = 1  # Other
    features.extend(atom_type)

    # Степень
    degree = atom.GetDegree()
    degree_features = [0] * 6  # degrees 0-5
    if degree < 6:
        degree_features[degree] = 1
    features.extend(degree_features)

    # Формальный заряд
    formal_charge = atom.GetFormalCharge()
    charge_features = [0] * 5  # charges -2, -1, 0, +1, +2
    charges = [-2, -1, 0, 1, 2]
    if formal_charge in charges:
        charge_features[charges.index(formal_charge)] = 1
    features.extend(charge_features)

    # Число неспаренных электронов
    radical_electrons = atom.GetNumRadicalElectrons()
    radical_features = [0] * 5  # 0-4 радикалов
    if radical_electrons < 5:
        radical_features[radical_electrons] = 1
    features.extend(radical_features)

    # Гибридизация
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    hybridization = atom.GetHybridization()
    hybrid_features = [0] * (len(hybridizations) + 1)
    if hybridization in hybridizations:
        hybrid_features[hybridizations.index(hybridization)] = 1
    else:
        hybrid_features[-1] = 1  # Other
    features.extend(hybrid_features)

    # Ароматичность
    features.append(float(atom.GetIsAromatic()))

    # В кольце
    features.append(float(atom.IsInRing()))

    # Число водородов
    num_hs = atom.GetTotalNumHs()
    h_features = [0] * 5  # 0-4 водорода
    if num_hs < 5:
        h_features[num_hs] = 1
    features.extend(h_features)

    return features


def get_bond_features(bond) -> list[float]:
    """Получает признаки связи.

    Args:
        bond: RDKit связь

    Returns:
        Список признаков связи
    """
    features = []

    # Тип связи
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    bond_type = bond.GetBondType()
    bond_type_features = [0] * len(bond_types)
    if bond_type in bond_types:
        bond_type_features[bond_types.index(bond_type)] = 1
    features.extend(bond_type_features)

    return features


def create_graphs_from_smiles(smiles_list: list[str]) -> list[Data]:
    """Создает графы из списка SMILES.

    Args:
        smiles_list: Список SMILES строк

    Returns:
        Список Data объектов
    """
    graphs = []
    failed_count = 0

    logger.info(f"Создание графов для {len(smiles_list)} молекул...")

    for i, smiles in enumerate(smiles_list):
        if i % 1000 == 0:
            logger.info(f"Обработано {i}/{len(smiles_list)} молекул...")

        graph = smiles_to_graph(smiles)
        if graph is not None:
            graphs.append(graph)
        else:
            failed_count += 1
            # Создаем пустой граф для сохранения порядка
            graphs.append(create_dummy_graph())

    logger.info(f"Создано {len(graphs)} графов, неудачных: {failed_count}")
    return graphs


def create_dummy_graph() -> Data:
    """Создает пустой граф для молекул, которые не удалось обработать.

    Returns:
        Data объект с одним узлом
    """
    # Создаем граф с одним узлом (все признаки = 0)
    x = torch.zeros((1, 74), dtype=torch.float)  # 74 признака атома
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 4), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def graphs_to_descriptor_tensor(graphs: list[Data]) -> torch.Tensor:
    """Преобразует графы в тензор дескрипторов для моделей, которые работают с дескрипторами.

    Args:
        graphs: Список графов

    Returns:
        Тензор дескрипторов
    """
    descriptors = []

    for graph in graphs:
        # Берем среднее по узлам графа как дескриптор молекулы
        if graph.x.shape[0] > 0:
            desc = torch.mean(graph.x, dim=0)
        else:
            desc = torch.zeros(74)  # Пустой дескриптор

        descriptors.append(desc)

    return torch.stack(descriptors)


def prepare_graph_data(data, smiles_column, target_column, train_indices, test_indices):
    """Подготавливает данные для обучения графовых нейронных сетей.

    Args:
        data: DataFrame с данными
        smiles_column: имя колонки с SMILES
        target_column: имя колонки с целевыми значениями
        train_indices: индексы для обучающей выборки
        test_indices: индексы для тестовой выборки

    Returns:
        X_train_graphs, y_train_graphs, X_test_graphs, y_test_graphs
    """
    logger.info(f"Подготовка графовых данных из {len(data)} строк")

    # Получаем все графы из SMILES
    all_smiles = data[smiles_column].to_list()
    all_graphs = create_graphs_from_smiles(all_smiles)

    # Получаем все целевые значения
    all_targets = data[target_column].to_list()

    # Фильтруем индексы, где есть валидные графы и целевые значения
    valid_indices = []
    for i, (graph, target) in enumerate(zip(all_graphs, all_targets, strict=False)):
        if graph is not None and target is not None:
            valid_indices.append(i)

    logger.info(f"Найдено {len(valid_indices)} валидных образцов из {len(data)}")

    # Пересекаем train_indices и test_indices с валидными индексами
    valid_train_indices = [i for i in train_indices if i in valid_indices]
    valid_test_indices = [i for i in test_indices if i in valid_indices]

    logger.info(f"Валидные индексы - train: {len(valid_train_indices)}, test: {len(valid_test_indices)}")

    # Создаем обучающую выборку
    X_train_graphs = [all_graphs[i] for i in valid_train_indices]
    y_train_graphs = [all_targets[i] for i in valid_train_indices]

    # Создаем тестовую выборку
    X_test_graphs = [all_graphs[i] for i in valid_test_indices]
    y_test_graphs = [all_targets[i] for i in valid_test_indices]

    # Преобразуем целевые значения в тензоры
    y_train_graphs = torch.tensor(y_train_graphs, dtype=torch.float)
    y_test_graphs = torch.tensor(y_test_graphs, dtype=torch.float)

    logger.info(f"Подготовлено графов: train={len(X_train_graphs)}, test={len(X_test_graphs)}")

    return X_train_graphs, y_train_graphs, X_test_graphs, y_test_graphs


def add_graph_indices(graphs: list[Data]) -> list[Data]:
    """Добавляет индексы к графам для batch processing.

    Args:
        graphs: Список графов

    Returns:
        Графы с добавленными индексами
    """
    for i, graph in enumerate(graphs):
        graph.graph_idx = i

    return graphs
