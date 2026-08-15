# Machine Learning Learning Path

This folder has two content formats:

- `*.md` files for concept-first reading.
- `*.ipynb` files for hands-on notebook practice.

The content is intentionally paired by phase where available (not a duplicate error).

## Start Here

1. Open the notebook hub: [00-Start-Here.ipynb](./00-Start-Here.ipynb)
2. Then follow phases in order using the map below.

## Organized Curriculum Map

| Phase | Topic                                                 | Reading Page                                                                                     | Practice Notebook                                                            |
| ----- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| 0     | Prerequisites                                         | -                                                                                                | [Phase-0-Prerequisites.ipynb](./Phase-0-Prerequisites.ipynb)                 |
| 1     | ML Concepts                                           | [Phase-1-ML-Concepts.md](./Phase-1-ML-Concepts.md)                                               | [Phase-1-ML-Concepts.ipynb](./Phase-1-ML-Concepts.ipynb)                     |
| 2A    | Supervised Learning (Regression)                      | [Phase-2-Supervised-Learning.md](./Phase-2-Supervised-Learning.md)                               | [Phase-2-Regression.ipynb](./Phase-2-Regression.ipynb)                       |
| 2B    | Supervised Learning (Classification)                  | [Phase-2-Supervised-Learning-Classification.md](./Phase-2-Supervised-Learning-Classification.md) | [Phase-2-Classification.ipynb](./Phase-2-Classification.ipynb)               |
| 3     | Unsupervised Learning                                 | [Phase-3-Unsupervised-Learning.md](./Phase-3-Unsupervised-Learning.md)                           | [Phase-3-Unsupervised-Learning.ipynb](./Phase-3-Unsupervised-Learning.ipynb) |
| 4     | Feature Engineering                                   | [Phase-4-Feature-Engineering.md](./Phase-4-Feature-Engineering.md)                               | [Phase-4-Feature-Engineering.ipynb](./Phase-4-Feature-Engineering.ipynb)     |
| 5-9   | Evaluation, Ensembles, DL Intro, Production, Projects | [Phase-5-9-Advanced-ML.md](./Phase-5-9-Advanced-ML.md)                                           | [Phase-5-6-Advanced-ML.ipynb](./Phase-5-6-Advanced-ML.ipynb)                 |

## Navigation Sequence

- [Phase-0-Prerequisites.ipynb](./Phase-0-Prerequisites.ipynb)
- [Phase-1-ML-Concepts.md](./Phase-1-ML-Concepts.md)
- [Phase-2-Supervised-Learning.md](./Phase-2-Supervised-Learning.md)
- [Phase-2-Supervised-Learning-Classification.md](./Phase-2-Supervised-Learning-Classification.md)
- [Phase-3-Unsupervised-Learning.md](./Phase-3-Unsupervised-Learning.md)
- [Phase-4-Feature-Engineering.md](./Phase-4-Feature-Engineering.md)
- [Phase-5-9-Advanced-ML.md](./Phase-5-9-Advanced-ML.md)

## Environment Setup

```bash
python -m venv ml_env
source ml_env/bin/activate  # Windows: ml_env\\Scripts\\activate
pip install numpy pandas matplotlib seaborn scikit-learn jupyter xgboost lightgbm
```

## Notes on Cleanup

- Removed broken links to files that do not exist in this folder (old split files for Phases 5-9).
- Kept markdown + notebook pairs where both exist because they serve different learning styles.
