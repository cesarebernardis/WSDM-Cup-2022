import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow_serving.apis import input_pb2

# The maximum number of documents per query in the dataset.
# Document lists are padded or truncated to this size.
_LIST_SIZE = 100

# The document relevance label.
_LABEL_FEATURE = "relevance"

# Learning rate for optimizer.
_LEARNING_RATE = 0.05

# Parameters to the scoring function.
_BATCH_SIZE = 32
_HIDDEN_LAYER_DIMS = ["64", "32", "16"]
_DROPOUT_RATE = 0.8
_GROUP_SIZE = 1    # Pointwise scoring.

_EMBEDDING_DIMENSION = 20


def context_feature_columns(df, feat_names):
    """Returns context feature names to column definitions."""
    features = {}
    for name in feat_names:
        features[name] = tf.feature_column.numeric_column(name, dtype=tf.float32, default_value=0)
    return features


def example_feature_columns(df, feat_names, examples_per_group=100):
    """Returns context feature names to column definitions."""
    features = {}
    for name in feat_names:
        features[name] = tf.feature_column.numeric_column(name, shape=(examples_per_group,),
                                                          dtype=tf.float32, default_value=0)
    return features


def input_fn(df, context_feat_names, example_feat_names, label=None, batch_size=32, examples_per_group=100, num_epochs=None):
    df = df.sort_values(by=['user'])
    context_feature_spec = tf.feature_column.make_parse_example_spec(
                context_feature_columns().values())
    example_feature_spec = tf.feature_column.make_parse_example_spec(
                example_feature_columns().values())
    context_df = {}
    for name in context_feat_names:
        context_df[name] = df[name].to_numpy()
    example_df = {}
    for name in example_feat_names:
        example_df[name] = df[name].to_numpy().reshape((-1, examples_per_group))
    if label is None:
        dataset = tf.data.Dataset.from_tensor_slices({**context_df, **example_df})
    else:
        dataset = tf.data.Dataset.from_tensor_slices(({**context_df, **example_df},
                                                      label.to_numpy().reshape((-1, examples_per_group))))
        dataset = dataset.batch(batch_size).repeat(num_epochs)
        dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    return dataset

def make_transform_fn():
    def _transform_fn(features, mode):
        """Defines transform_fn."""
        context_features, example_features = tfr.feature.encode_listwise_features(
                features=features,
                context_feature_columns=context_feature_columns(),
                example_feature_columns=example_feature_columns(),
                mode=mode,
                scope="transform_layer")
        return context_features, example_features
    return _transform_fn

def make_score_fn(hidden_layers, context_feat_names, example_feat_names, group_size):
    """Returns a scoring function to build `EstimatorSpec`."""

    def _score_fn(context_features, example_features, mode, params, config):
        """Defines the network to score a group of documents."""
        with tf.compat.v1.name_scope("input_layer"):
            context_input = [
                    tf.compat.v1.layers.flatten(context_features[name])
                    for name in sorted(context_feat_names)
            ]
            group_input = [
                    tf.compat.v1.layers.flatten(example_features[name])
                    for name in sorted(example_feat_names())
            ]
            input_layer = tf.concat(context_input + group_input, 1)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        cur_layer = input_layer
        cur_layer = tf.compat.v1.layers.batch_normalization(
            cur_layer,
            training=is_training,
            momentum=0.99)

        for i, layer_width in enumerate(int(d) for d in hidden_layers):
            cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
            cur_layer = tf.compat.v1.layers.batch_normalization(
                cur_layer,
                training=is_training,
                momentum=0.99)
            cur_layer = tf.nn.relu(cur_layer)
            cur_layer = tf.compat.v1.layers.dropout(
                    inputs=cur_layer, rate=_DROPOUT_RATE, training=is_training)
        logits = tf.compat.v1.layers.dense(cur_layer, units=group_size)
        return logits

    return _score_fn

def eval_metric_fns():
    metric_fns = {}
    metric_fns.update({
            f"metric/ndcg@{topn}": tfr.metrics.make_ranking_metric_fn(
                    tfr.metrics.RankingMetricKey.NDCG, topn=topn)
            for topn in [1, 3, 5, 10]
    })
    return metric_fns

loss_fn = tfr.losses.make_loss_fn(tfr.losses.RankingLossKey.APPROX_NDCG_LOSS)
optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=_LEARNING_RATE)


def _train_op_fn(loss):
    """Defines train op used in ranking head."""
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    minimize_op = optimizer.minimize(
            loss=loss, global_step=tf.compat.v1.train.get_global_step())
    train_op = tf.group([update_ops, minimize_op])
    return train_op


ranking_head = tfr.head.create_ranking_head(
            loss_fn=loss_fn,
            eval_metric_fns=eval_metric_fns(),
            train_op_fn=_train_op_fn)


model_fn = tfr.model.make_groupwise_ranking_fn(
                    group_score_fn=make_score_fn(),
                    transform_fn=make_transform_fn(),
                    group_size=_GROUP_SIZE,
                    ranking_head=ranking_head)

run_config = tf.estimator.RunConfig(save_checkpoints_steps=1000)
ranker = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=_MODEL_DIR,
        config=run_config)

train_input_fn = lambda: input_fn(_TRAIN_DATA_PATH)
eval_input_fn = lambda: input_fn(_TEST_DATA_PATH, num_epochs=1)

train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=_NUM_TRAIN_STEPS)
eval_spec = tf.estimator.EvalSpec(
        name="eval",
        input_fn=eval_input_fn,
        throttle_secs=15)

tf.estimator.train_and_evaluate(ranker, train_spec, eval_spec)
