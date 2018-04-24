#### Personal instructions

Let's say you have a directory of images: ```/data/photos/class1```, ```/data/photos/class2```,...

To create TFRecord from directory of images in /data/tf_record:

```
python3 convert_data.py --dataset_name=standard --dataset_dir=/data/
```

To train model from scratch using TFRecords:

```
python3 train_image_classifier.py --train_dir=/tmp/train_logs \
  --dataset_name=standard \
  --dataset_split_name=train \
  --dataset_dir=/data/tf_record \
  --model_name=inception_v3
```

Export graph:

```
python3 export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --output_file=/tmp/inception_v3_inf_graph.pb \
  --dataset_name=standard
```


## Freezing the exported Graph
If you then want to use the resulting model with your own or pretrained
checkpoints as part of a mobile model, you can run freeze_graph to get a graph
def with the variables inlined as constants using:

```shell
bazel build tensorflow/python/tools:freeze_graph

bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=/tmp/inception_v3_inf_graph.pb \
  --input_checkpoint=/checkpoints/model.ckpt-10000 \
  --input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
  --output_node_names=InceptionV3/Predictions/Reshape_1
```

## Creating Saved Model
```
python3 export_saved_model.py \
  --input_layer=input \
  --output_layer=InceptionV3/Predictions/Reshape_1 \
  --graph_file=/tmp/frozen_inception_v3.pb \
  --output_dir=/path/to/saved_model
```

