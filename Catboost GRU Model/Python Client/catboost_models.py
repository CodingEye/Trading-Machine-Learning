from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

from onnx.helper import get_attribute_value
from skl2onnx import convert_sklearn, update_registered_converter
from sklearn.pipeline import Pipeline
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
)  # noqa
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType,
    guess_tensor_type,
)
from skl2onnx._parse import _apply_zipmap, _get_sklearn_operator_name
from catboost.utils import convert_to_onnx_object

# Example initial data (X_initial, y_initial are your initial feature matrix and target)

class CatBoostClassifierModel():
    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None

    def train(self, iterations=100, depth=6, learning_rate=0.1, loss_function="CrossEntropy", use_best_model=True):
        # Initialize the CatBoost model

        params = {
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "loss_function": loss_function,
            "use_best_model": use_best_model
        }

        self.model = Pipeline([ # wrap a catboost classifier in sklearn pipeline | good practice (not necessary tho :))
            ("catboost", CatBoostClassifier(**params))
        ])

        # Testing the model
        
        self.model.fit(X=self.X_train, y=self.y_train, catboost__eval_set=(self.X_test, self.y_test))

        y_pred = self.model.predict(self.X_test)
        print("Model's accuracy on out-of-sample data = ",accuracy_score(self.y_test, y_pred))


    def skl2onnx_parser_castboost_classifier(self, scope, model, inputs, custom_parsers=None):
        
        options = scope.get_options(model, dict(zipmap=True))
        no_zipmap = isinstance(options["zipmap"], bool) and not options["zipmap"]

        alias = _get_sklearn_operator_name(type(model))
        this_operator = scope.declare_local_operator(alias, model)
        this_operator.inputs = inputs

        label_variable = scope.declare_local_variable("label", Int64TensorType())
        prob_dtype = guess_tensor_type(inputs[0].type)
        probability_tensor_variable = scope.declare_local_variable(
            "probabilities", prob_dtype
        )
        this_operator.outputs.append(label_variable)
        this_operator.outputs.append(probability_tensor_variable)
        probability_tensor = this_operator.outputs

        if no_zipmap:
            return probability_tensor

        return _apply_zipmap(
            options["zipmap"], scope, model, inputs[0].type, probability_tensor
        )


    def skl2onnx_convert_catboost(self, scope, operator, container):
        """
        CatBoost returns an ONNX graph with a single node.
        This function adds it to the main graph.
        """
        onx = convert_to_onnx_object(operator.raw_operator)
        opsets = {d.domain: d.version for d in onx.opset_import}
        if "" in opsets and opsets[""] >= container.target_opset:
            raise RuntimeError("CatBoost uses an opset more recent than the target one.")
        if len(onx.graph.initializer) > 0 or len(onx.graph.sparse_initializer) > 0:
            raise NotImplementedError(
                "CatBoost returns a model initializers. This option is not implemented yet."
            )
        if (
            len(onx.graph.node) not in (1, 2)
            or not onx.graph.node[0].op_type.startswith("TreeEnsemble")
            or (len(onx.graph.node) == 2 and onx.graph.node[1].op_type != "ZipMap")
        ):
            types = ", ".join(map(lambda n: n.op_type, onx.graph.node))
            raise NotImplementedError(
                f"CatBoost returns {len(onx.graph.node)} != 1 (types={types}). "
                f"This option is not implemented yet."
            )
        node = onx.graph.node[0]
        atts = {}
        for att in node.attribute:
            atts[att.name] = get_attribute_value(att)
        container.add_node(
            node.op_type,
            [operator.inputs[0].full_name],
            [operator.outputs[0].full_name, operator.outputs[1].full_name],
            op_domain=node.domain,
            op_version=opsets.get(node.domain, None),
            **atts,
        )

    # a function for saving the trained CatBoost model to ONNX format
    
    def to_onnx(self, model_name):

        update_registered_converter(
            CatBoostClassifier,
            "CatBoostCatBoostClassifier",
            calculate_linear_classifier_output_shapes,
            self.skl2onnx_convert_catboost,
            parser=self.skl2onnx_parser_castboost_classifier,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )

        model_onnx = convert_sklearn(
            self.model,
            "pipeline_catboost",
            [("input", FloatTensorType([None, self.X_train.shape[1]]))],
            target_opset={"": 12, "ai.onnx.ml": 2},
        )

        # And save.
        with open(model_name, "wb") as f:
            f.write(model_onnx.SerializeToString())
