import numpy as np

from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    Float64Type,
    TensorType,
    f64,
)
from xdsl.ir import Dialect, OpResult
from xdsl.irdl import IRDLOperation, attr_def, irdl_op_definition, result_def


@irdl_op_definition
class LiteralOp(IRDLOperation):
    name = "gem.literal"

    value: DenseIntOrFPElementsAttr = attr_def(DenseIntOrFPElementsAttr)
    result: OpResult = result_def()

    @staticmethod
    def from_numpy(array: np.ndarray) -> "LiteralOp":
        tensor_type = TensorType(f64, list(array.shape))
        value = DenseIntOrFPElementsAttr.from_list(tensor_type, array.flatten().tolist())
        return LiteralOp.build(
            attributes={"value": value},
            result_types=[tensor_type],
        )


GEM = Dialect("gem", 
              [
                  LiteralOp,
               ])
