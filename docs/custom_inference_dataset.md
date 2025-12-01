# 推理场景下编写最小 Dataset（Option B）

本指南详细讲解 **Option B：仅推理的自定义 Dataset**，适用于你的数据集
不符合 ONCE 目录/标注格式，但想直接用 ONCE 训练好的模型做推理的场景。
核心思路是保持改动最少：写一个精简 Dataset 类、在 YAML 中引用它，再
用现有的 `tools/test.py` 跑推理。

## 1）实现精简版 Dataset 类
创建一个继承 `pcdet.datasets.dataset.DatasetTemplate` 的类，文件可以放在
`pcdet/datasets/my_inference_dataset.py` 等可被导入的位置。推理只需要
实现少量方法：

```python
# pcdet/datasets/my_inference_dataset.py
from pathlib import Path
import numpy as np
from pcdet.datasets.dataset import DatasetTemplate

class MyInferenceDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        self.root_path = Path(root_path)
        # 这里加载你的样本列表（文件名、元信息等），示例为从分割文件读取
        self.sample_ids = [x.strip() for x in open(self.root_path / 'ImageSets/test.txt')]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        # 读取该帧的原始点云（按你的数据格式替换路径/读取逻辑）
        points = np.fromfile(self.root_path / 'lidar' / f'{sample_id}.bin', dtype=np.float32).reshape(-1, 4)

        input_dict = {
            'points': points,      # [N, 4] (x, y, z, intensity)
            'frame_id': sample_id, # 用于输出的唯一 ID
            # 推理无需标签，gt_boxes/gt_names 可以省略
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
```

关键要点：
- **无需标签**：推理模式 (`training=False`) 会跳过监督相关逻辑，`gt_boxes`、
  `gt_names` 可以不提供。
- **坐标系与范围**：确保点云坐标系/单位与 YAML 中的 `POINT_CLOUD_RANGE` 保持
  一致，避免裁掉目标或落在网格之外。
- **输出格式化（可选）**：若需要自定义保存格式，可重写
  `generate_prediction_dicts`；否则基类默认实现即可把预测写到
  `output/.../preds/`。

## 2）让类可被构建器发现
若将类放在 `pcdet/datasets/` 下，需要在 `pcdet/datasets/__init__.py` 中导出，
便于 `build_dataloader` 通过类名找到它：

```python
from .my_inference_dataset import MyInferenceDataset
__all__ += ['MyInferenceDataset']
```

## 3）编写数据集 YAML
复制 `tools/cfgs/dataset_configs/once_dataset.yaml`，精简为推理配置，并修改：
- `DATASET: MyInferenceDataset`（与类名一致）；
- `DATA_PATH` 指向你的点云根目录；
- `POINT_CLOUD_RANGE`、voxel 设置贴合你的坐标范围；推理可关闭
  `DATA_AUGMENTOR`。

Example snippet:
```yaml
DATASET: MyInferenceDataset
DATA_PATH: ../data/my_dataset
POINT_CLOUD_RANGE: [-80, -80, -3, 80, 80, 3]
INFO_PATH:
  test: [ ]          # 自定义加载逻辑时可留空
DATA_SPLIT:
  test: test

POINT_FEATURE_ENCODING:
  encoding_type: absolute_coordinates_encoding
  used_feature_list: ['x', 'y', 'z', 'intensity']
  src_feature_list:  ['x', 'y', 'z', 'intensity']
```

## 4）让模型配置引用新数据集
在模型 YAML（可复制 `tools/cfgs/once_models/sup_models/centerpoint_pillar.yaml`
后修改）中：
- 将 `CLASS_NAMES` 改为你的类别；
- 确保 `DATA_CONFIG` 引用新建的数据集 YAML。

示例：
```yaml
CLASS_NAMES: ['Car', 'Pedestrian']
DATA_CONFIG: cfg.DATA_CONFIG
```
启动推理时通过 `--cfg_file path/to/your_model.yaml` 指定此模型配置。

## 5）运行推理
使用标准测试入口，加载 ONCE 训练好的权重：
```bash
python tools/test.py \
  --cfg_file path/to/your_model.yaml \
  --batch_size 1 \
  --ckpt path/to/once_trained_model.ckpt \
  --save_to_file
```
预测结果（框、分数）会保存在 `output/<exp>/preds/` 目录。

按以上步骤，你无需将数据转换成 ONCE 的 info 格式，就能用训练好的模型对
自定义点云数据做推理。
