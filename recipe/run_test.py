import subprocess
import tempfile
from pathlib import Path

import onnx
import onnxslim
from onnx import TensorProto, helper


def make_model():
    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    zero = helper.make_tensor("zero", TensorProto.FLOAT, [1, 2], [0.0, 0.0])
    add = helper.make_node("Add", ["x", "zero"], ["y"], name="add_zero")
    graph = helper.make_graph(
        [add],
        "add-zero",
        [input_info],
        [output_info],
        initializer=[zero],
    )
    model = helper.make_model(graph, producer_name="conda-forge-test")
    model.opset_import[0].version = 13
    model.ir_version = 10
    return model


def check_model(model):
    onnx.checker.check_model(model)
    assert model.graph.name == "add-zero"
    assert model.graph.output[0].name == "y"


def test_python_api():
    model = make_model()
    check_model(model)

    slimmed = onnxslim.slim(model)
    check_model(slimmed)
    assert len(slimmed.graph.node) <= len(model.graph.node)


def test_cli_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        source = Path(tmpdir) / "source.onnx"
        target = Path(tmpdir) / "target.onnx"
        onnx.save(make_model(), source)

        subprocess.run(["onnxslim", str(source), str(target)], check=True)

        assert target.is_file()
        check_model(onnx.load(target))


if __name__ == "__main__":
    test_python_api()
    test_cli_round_trip()
