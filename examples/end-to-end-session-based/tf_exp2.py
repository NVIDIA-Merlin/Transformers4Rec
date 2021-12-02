import os
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import glob

# from nvtabular.loader.tensorflow import KerasSequenceLoader

# from transformers4rec import tf as tr
# from transformers4rec.tf.ranking_metric import NDCGAt, RecallAt
import cudf
import pandas as pd
import numpy as np

from merlin_standard_lib import Schema
SCHEMA_PATH = "schema_tf.pb"
schema = Schema().from_proto_text(SCHEMA_PATH)
# You can select a subset of features for training

x_cat_names, x_cont_names = ['category_list_trim', 'item_id_list_trim'], ['timestamp/age_days_list_trim', 'timestamp/weekday/sin_list_trim']

schema = schema.select_by_name(x_cat_names + x_cont_names)
# dictionary representing max sequence length for column
sparse_features_max = {
    fname: 20
    for fname in x_cat_names + x_cont_names
}

# inputs = tr.TabularSequenceFeatures.from_schema(
#         schema,
#         max_sequence_length=20,
#         continuous_projection=64,
#         d_output=100,
#         masking="mlm",
# )

# # Define XLNetConfig class and set default parameters for HF XLNet config  
# transformer_config = tr.XLNetConfig.build(
#     d_model=64, n_head=4, n_layer=2, total_seq_length=20
# )
# # Define the model block including: inputs, masking, projection and transformer block.
# body = tr.SequentialBlock(
#     [inputs, tr.MLPBlock([64]), tr.TransformerBlock(transformer_config, masking=inputs.masking)]
# )

# # Defines the evaluation top-N metrics and the cut-offs
# metrics = (
#     NDCGAt(top_ks=[1, 5, 20, 40], labels_onehot=True),  
#     RecallAt(top_ks=[1, 5, 20, 40], labels_onehot=True)
#           )

# # link task to body and generate the end-to-end keras model
# task = tr.NextItemPredictionTask(weight_tying=True, metrics=metrics)
 
# model = task.to_model(body=body)

# def get_dataloader(paths_or_dataset, batch_size=64):
#     dataloader = KerasSequenceLoader(
#         paths_or_dataset,
#         batch_size=batch_size,
#         label_names=None,
#         cat_names=x_cat_names,
#         cont_names=x_cont_names,
#         sparse_names=list(sparse_features_max.keys()),
#         sparse_max=sparse_features_max,
#         sparse_as_dense=True,
#     )
#     return dataloader.map(lambda X, y: (X, []))

# import tensorflow as tf
# # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
# #     initial_learning_rate=0.0005,
# #     decay_steps=1.5)
# #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# model.compile(optimizer="adam", run_eagerly=True)
# OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./sessions_by_day")

# import warnings
# warnings.filterwarnings('ignore')

# start_time_window_index = 1
# final_time_window_index = 2
# #Iterating over days of one week
# for time_index in range(start_time_window_index, final_time_window_index):
#     # Set data 
#     time_index_train = time_index
#     time_index_eval = time_index + 1
#     train_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_train}/train.parquet"))
#     eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))
#     print(train_paths)

#     # Train on day related to time_index 
#     print('*'*20)
#     print("Launch training for day %s are:" %time_index)
#     print('*'*20 + '\n')
#     train_loader = get_dataloader(train_paths) 
#     losses = model.fit(train_loader, epochs=1, verbose=0, )
#     model.reset_metrics()
#     print('finished')
#     # Evaluate on the following day
#     eval_loader = get_dataloader(eval_paths) 
#     eval_metrics = model.evaluate(eval_loader, return_dict=True)
#     print('*'*20)
#     print("Eval results for day %s are:\t" %time_index_eval)
#     print('\n' + '*'*20 + '\n')
#     for key in sorted(eval_metrics.keys()):
#         print(" %s = %s" % (key, str(eval_metrics[key])))

# # model.save('./tmp/tensorflow')
# # model = tf.keras.models.load_model('./tmp/tensorflow')

print("start serving the model")
import nvtabular as nvt
workflow = nvt.Workflow.load("workflow_etl")

# from nvtabular.inference.triton import export_tensorflow_ensemble
# export_tensorflow_ensemble(model, workflow, 'tf4rec', '/workspace/models/tf4rec/', [], sparse_max=sparse_features_max)

import tritonhttpclient
import nvtabular.inference.triton as nvt_triton
import tritonclient.grpc as grpcclient

try:
    triton_client = tritonhttpclient.InferenceServerClient(url="192.168.0.4:8000", verbose=True)
    print("client created.")
except Exception as e:
    print("channel creation failed: " + str(e))
print(triton_client.is_server_live())


NUM_ROWS = 100000
long_tailed_item_distribution = np.clip(np.random.lognormal(3., 1., NUM_ROWS).astype(np.int32), 1, 50000)

# generate random item interaction features 
df = pd.DataFrame(np.random.randint(70000, 71000, NUM_ROWS), columns=['session_id']).astype(np.int32)
df['item_id'] = long_tailed_item_distribution

# generate category mapping for each item-id
df['category'] = pd.cut(df['item_id'], bins=334, labels=np.arange(1, 335)).astype(np.int32)
df['timestamp/age_days'] = np.random.uniform(0, 1, NUM_ROWS).astype(np.float32)
df['timestamp/weekday/sin']= np.random.uniform(0, 1, NUM_ROWS).astype(np.float32)

# generate day mapping for each session 
map_day = dict(zip(df.session_id.unique(), np.random.randint(1, 10, size=(df.session_id.nunique()))))
df['day'] =  df.session_id.map(map_day).astype(np.int32)

batch = df[:300]
batch=cudf.DataFrame(batch)

# interactions_merged_df=cudf.read_parquet('./sessions_by_day/1/train.parquet')
# batch = interactions_merged_df[:30][x_cat_names + x_cont_names]
# print(batch.head(2))
modelName = "tf4rec_nvt"

# triton_client.load_model(model_name=modelName)
print(triton_client.is_server_live())
print(triton_client.is_server_ready())
print(triton_client.is_model_ready(modelName,"1"))

#triton_client.get_model_metadata(modelName)

inputs = nvt_triton.convert_df_to_triton_input(batch.columns, batch, grpcclient.InferInput)
#print(inputs[4]._get_tensor())

#print(workflow.output_node.output_columns.names)

output_names = workflow.output_node.output_columns.names

outputs = []
for col in output_names:
    outputs.append(grpcclient.InferRequestedOutput(col))

#print(outputs[4]._get_tensor())
#with grpcclient.InferenceServerClient("192.168.0.4:8001") as client:
    #response = client.infer(modelName, inputs)
    #print("---printing response now---")
    #print(col, ':\n', response.as_numpy(col))


with grpcclient.InferenceServerClient("192.168.0.4:8001") as client:
    response = client.infer(modelName, inputs, request_id="1", outputs=outputs)

for col in output_names:
    print(col, response.as_numpy(col), response.as_numpy(col).shape)
