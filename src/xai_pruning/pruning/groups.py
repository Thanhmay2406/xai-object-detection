"""Discovery of the 32 supported ResNet-50 bottleneck hidden-width groups."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn


def discover_resnet_bottleneck_pruning_groups(
    model: nn.Module,
    include_stages: Sequence[str] = ("layer1", "layer2", "layer3", "layer4"),
    include_convs: Sequence[str] = ("conv1", "conv2"),
    *,
    include_modules: bool = True,
) -> list[dict]:
    """Return only the local ``conv1/bn1/conv2`` and ``conv2/bn2/conv3`` groups.

    The default call intentionally requires all 32 ResNet-50 groups. Passing a
    restricted stage/conv subset is supported for Pipeline 03 analysis and does
    not broaden the set of prunable layers.
    """

    modules = dict(model.named_modules())
    groups = []
    prefix = "backbone.body."

    for stage in include_stages:
        stage_prefix = prefix + stage + "."
        block_indices = sorted(
            {
                int(name.split(".")[3])
                for name in modules
                if name.startswith(stage_prefix)
                and len(name.split(".")) >= 5
                and name.split(".")[3].isdigit()
            }
        )
        for block_idx in block_indices:
            block_prefix = f"{prefix}{stage}.{block_idx}"
            for conv_name in include_convs:
                producer_name = f"{block_prefix}.{conv_name}"
                if producer_name not in modules:
                    continue
                producer = modules[producer_name]
                if not isinstance(producer, nn.Conv2d):
                    continue

                if conv_name == "conv1":
                    norm_name = f"{block_prefix}.bn1"
                    consumer_name = f"{block_prefix}.conv2"
                    group_kind = "bottleneck_conv1_hidden"
                elif conv_name == "conv2":
                    norm_name = f"{block_prefix}.bn2"
                    consumer_name = f"{block_prefix}.conv3"
                    group_kind = "bottleneck_conv2_hidden"
                else:
                    raise ValueError(f"Unsupported conv group: {conv_name}")

                if norm_name not in modules or consumer_name not in modules:
                    raise RuntimeError(
                        "Expected local dependency module is missing: "
                        f"{producer_name} -> {norm_name} -> {consumer_name}"
                    )
                norm = modules[norm_name]
                consumer = modules[consumer_name]
                if not isinstance(norm, nn.BatchNorm2d):
                    raise TypeError(f"Normalization is not BatchNorm2d: {norm_name}")
                if not isinstance(consumer, nn.Conv2d):
                    raise TypeError(f"Consumer is not Conv2d: {consumer_name}")
                if producer.out_channels != norm.num_features:
                    raise RuntimeError(f"Producer/BatchNorm mismatch at {producer_name}")
                if producer.out_channels != consumer.in_channels:
                    raise RuntimeError(f"Dependency mismatch at {producer_name}")

                metadata = {
                    "group_id": f"{stage}.{block_idx}.{conv_name}",
                    "stage": stage,
                    "block": block_idx,
                    "group_kind": group_kind,
                    "producer_name": producer_name,
                    "norm_name": norm_name,
                    "consumer_name": consumer_name,
                    "channels": int(producer.out_channels),
                    "producer_out_channels": int(producer.out_channels),
                    "consumer_in_channels": int(consumer.in_channels),
                }
                if include_modules:
                    metadata.update(
                        {
                            "producer_module": producer,
                            "norm_module": norm,
                            "consumer_module": consumer,
                        }
                    )
                groups.append(metadata)

    if tuple(include_stages) == ("layer1", "layer2", "layer3", "layer4") and tuple(
        include_convs
    ) == ("conv1", "conv2") and len(groups) != 32:
        raise RuntimeError(f"Expected 32 ResNet-50 pruning groups, found {len(groups)}")
    return groups
