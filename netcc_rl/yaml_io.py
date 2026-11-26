"""Read user configuration without silently accepting duplicate keys."""

import yaml


class _ConfigLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = set()
        for key, _ in node.value:
            value = self.construct_object(key, deep=deep)
            if value in keys:
                raise ValueError(f"Duplicate YAML key {value!r}")
            keys.add(value)
        return super().construct_mapping(node, deep=deep)


def read_yaml(path):
    return yaml.load(path.read_text(), Loader=_ConfigLoader)
