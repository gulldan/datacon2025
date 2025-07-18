# Мини‑таск 2: Расчёт и отбор дескрипторов

В этом задании вы продолжаете работу с датасетом, подготовленным на предыдущем этапе, и переходите к **извлечению признаков**, которые будут использоваться для построения модели машинного обучения.

## Цель

Рассчитать и отобрать информативные **молекулярные дескрипторы** и/или **фингерпринты**, характеризующие химическую структуру соединений, — с целью дальнейшего построения модели предсказания биологической активности (мини‑таск 3).


## Описание

Молекулярные дескрипторы — это количественные характеристики молекулы, отражающие её структуру, физико-химические свойства и топологию. Они широко применяются в QSAR-моделировании и генеративном дизайне лекарств.

Для этого задания предполагается, что в вашем датасете (полученном из мини‑таска 1) есть колонка со структурой молекулы в формате **SMILES**.



## Поддерживаемые библиотеки

Вы можете использовать любую библиотеку для расчёта дескрипторов. Ниже приведены наиболее популярные инструменты:


| Библиотека     | Описание                                  | Пример дескрипторов           |
| -------------- | ----------------------------------------- | ----------------------------- |
| **RDKit**      | Стандартный инструментарий для химии      | MW, LogP, TPSA, HBA/HBD       |
| **Mordred**    | Расширенный набор (1800+ дескрипторов)    | Constitutional, Charge и др.  |
| **Padelpy**    | Python-обёртка над PaDEL-Descriptor       | CDK, PubChem, MACCS, ECFP     |

>  **Hint**: попробуйте сформировать несколько разных наборов признаков — это поможет на следующем этапе сравнить производительность моделей.



## Что нужно сделать

### 1. Загрузить очищенный датасет

Используйте результат мини‑таска 1 (датасет с валидными SMILES и активностями).

### 2. Рассчитать дескрипторы

- С помощью RDKit, Mordred, PaDEL или других библиотек извлеките **молекулярные признаки** для каждой молекулы.
- Рекомендуется сохранить `SMILES`, `standard_value` и рассчитанные дескрипторы в одну таблицу.

> В RDKit, например, можно получить 43 дескриптора:
> `rdkit.Chem.Descriptors`  
> В Mordred — более 1800, но часть из них может быть некорректной или пустой.

### 3. (Опционально) Рассчитать фингерпринты

- Сформируйте бинарные векторы (например, Morgan/ECFP, MACCS), которые можно использовать как альтернативу дескрипторам.
- Фингерпринты могут пригодиться для сравнения моделей и оценки важности признаков.

### 4. Отфильтровать и отобрать признаки

- Удалите дескрипторы с:
  - некорректными или отсутствующими значениями (`NaN`);
  - нулевой дисперсией (одинаковое значение для всех молекул);
  - высокой корреляцией (например, `r > 0.7` между признаками — можно оставить один из пары);
- Также рекомендуется исключить категориальные признаки, если они неинформативны или дублируют числовые.

>  Внимание: нормализация и стандартизация признаков не требуется на этом этапе — она будет выполнена в следующем мини‑таске.



## Что нужно сдать

- Jupyter Notebook (`.ipynb`) с кодом и комментариями.