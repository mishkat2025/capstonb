"""Run a Caffe network in PyTorch without installing Caffe.

A Caffe model is two files: a text file that lists the layers and how they are wired
(.prototxt), and a binary file with the trained numbers (.caffemodel). This module reads
both and rebuilds the network. It knows the five layer types OpenPose's body network
uses (Convolution, ReLU, PReLU, max Pooling, Concat) and refuses anything else.

Nothing in either file is executed: the text file is parsed as text, and from the binary
file only layer names and arrays of numbers are read.
"""
import re
import struct

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- the weights file
# It is in Google's "protocol buffer" format: a flat sequence of (field number, value) records, where a
# value can itself be such a sequence. Only the fields needed are named here; the rest are skipped.

def _varint(buf, pos):
    value, shift = 0, 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7


def _records(buf, pos, end):
    """Yield (field number, kind, value). kind 0: an integer. kind 2: (start, end) of a nested block. kinds 1 and 5: raw 8 or 4 bytes."""
    while pos < end:
        key, pos = _varint(buf, pos)
        field, kind = key >> 3, key & 7
        if kind == 0:
            value, pos = _varint(buf, pos)
        elif kind == 2:
            length, pos = _varint(buf, pos)
            value, pos = (pos, pos + length), pos + length
        elif kind in (1, 5):
            size = 8 if kind == 1 else 4
            value, pos = bytes(buf[pos:pos + size]), pos + size
        else:
            raise ValueError(f"unexpected record kind {kind} in the weights file")
        yield field, kind, value


def _array(buf, start, end):
    """One stored array ("blob"): its numbers and its shape."""
    shape, legacy, chunks = [], {}, []
    for field, kind, value in _records(buf, start, end):
        if field == 5 and kind == 2:                    # the numbers, packed as 32-bit floats
            chunks.append(np.frombuffer(buf[value[0]:value[1]], dtype="<f4"))
        elif field == 5 and kind == 5:                  # the numbers, one record each
            chunks.append(np.array(struct.unpack("<f", value), dtype="<f4"))
        elif field == 7 and kind == 2:                  # shape: a block holding the sizes
            for f, k, v in _records(buf, *value):
                if f == 1 and k == 0:
                    shape.append(v)
                elif f == 1 and k == 2:
                    pos = v[0]
                    while pos < v[1]:
                        size, pos = _varint(buf, pos)
                        shape.append(size)
        elif field in (1, 2, 3, 4) and kind == 0:       # older files give num, channels, height, width instead
            legacy[field] = value
    data = np.concatenate(chunks) if chunks else np.zeros(0, "<f4")
    if not shape and legacy:
        shape = [legacy.get(i, 1) for i in (1, 2, 3, 4)]
    return data.reshape(shape) if shape else data


def read_weights(path):
    """{layer name: [arrays]} for every layer of a .caffemodel that has trained numbers."""
    with open(path, "rb") as f:
        buf = memoryview(f.read())
    layouts = {100: (1, 7), 2: (4, 6)}      # top-level field of a layer -> (field of its name, field of its arrays); 2 is the older layout
    out = {}
    for field, kind, value in _records(buf, 0, len(buf)):
        if field in layouts and kind == 2:
            name_field, array_field = layouts[field]
            name, arrays = None, []
            for f, k, v in _records(buf, *value):
                if f == name_field and k == 2:
                    name = bytes(buf[v[0]:v[1]]).decode()
                elif f == array_field and k == 2:
                    arrays.append(_array(buf, *v))
            if name and arrays:
                out[name] = arrays
    return out


# ---------------------------------------------------------------- the definition file

def read_definition(path):
    """The .prototxt as nested dicts; every field is a list, because fields may repeat."""
    text = re.sub(r"#[^\n]*", "", open(path, encoding="utf-8").read())
    tokens = re.findall(r'"[^"]*"|[{}:]|[^\s{}:"]+', text)

    def block(i):
        fields = {}
        while i < len(tokens) and tokens[i] != "}":
            key = tokens[i]
            i += 1
            if tokens[i] == ":":
                i += 1
            if tokens[i] == "{":
                value, i = block(i + 1)
                i += 1                                  # the closing brace
            else:
                value = tokens[i].strip('"')
                i += 1
            fields.setdefault(key, []).append(value)
        return fields, i

    return block(0)[0]


# ---------------------------------------------------------------- the network

class CaffeNet(torch.nn.Module):
    def __init__(self, definition_path, weights_path):
        super().__init__()
        definition = read_definition(definition_path)
        weights = read_weights(weights_path)
        self.input = definition["input"][0]
        self.ops = torch.nn.ModuleDict()
        self.steps = []         # (layer type, layer name, input names, output name)
        for layer in definition["layer"]:
            kind, name = layer["type"][0], layer["name"][0]
            if kind == "Convolution":
                p = layer["convolution_param"][0]
                w, b = weights[name]
                conv = torch.nn.Conv2d(w.shape[1], w.shape[0], w.shape[2], stride=int(p.get("stride", [1])[0]), padding=int(p.get("pad", [0])[0]))
                if int(p["num_output"][0]) != w.shape[0] or int(p["kernel_size"][0]) != w.shape[2]:
                    raise ValueError(f"{name}: the weights do not fit the definition")
                conv.weight.data = torch.from_numpy(w.copy())
                conv.bias.data = torch.from_numpy(b.reshape(-1).copy())
                self.ops[name] = conv
            elif kind == "PReLU":
                slope = weights[name][0].reshape(-1)
                prelu = torch.nn.PReLU(len(slope))
                prelu.weight.data = torch.from_numpy(slope.copy())
                self.ops[name] = prelu
            elif kind == "Pooling":
                p = layer["pooling_param"][0]
                if p.get("pool", ["MAX"])[0] != "MAX":
                    raise ValueError(f"{name}: only max pooling is supported")
                self.ops[name] = torch.nn.MaxPool2d(int(p["kernel_size"][0]), int(p.get("stride", [1])[0]), ceil_mode=True)   # Caffe rounds sizes up
            elif kind not in ("ReLU", "Concat"):
                raise ValueError(f"{name}: layer type {kind} is not supported")
            self.steps.append((kind, name, layer["bottom"], layer["top"][0]))
        self.output = self.steps[-1][3]

    def forward(self, image):
        values = {self.input: image}
        for kind, name, inputs, output in self.steps:
            if kind == "ReLU":
                values[output] = F.relu(values[inputs[0]])
            elif kind == "Concat":
                values[output] = torch.cat([values[i] for i in inputs], dim=1)
            else:
                values[output] = self.ops[name](values[inputs[0]])
        return values[self.output]
