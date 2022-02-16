# -*- coding: utf-8 -*-
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
import os,sys,re,copy,json,importlib,random,math
from sklearn.utils import resample as sklearnResample
import multiprocessing
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML
import condor_tensorflow as condor
import multiprocessing_functions
import news_retrieval_common_funcs
import newsgetdata,constructmodel
import mlflow.tensorflow

mlflow.tensorflow.autolog()
workingdir=os.getcwd()
savingpath=os.path.join(os.getcwd())
os.chdir(workingdir)
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# %%
if True:
  import json,os,shutil
  print('now copy kaggle api auth')
  os.makedirs('~/.kaggle/', exist_ok=True)
  os.makedirs('/root/.kaggle', exist_ok=True)
  shutil.copy('kaggle.json','~/.kaggle/kaggle.json')
  shutil.copy('kaggle.json','/root/.kaggle/kaggle.json')
  print('completed copying kaggle api auth')
  print('now downloading kaggle dataset')
  f = open('kaggle.json', encoding='utf-8')
  apikey = json.loads(f.read())
  f.close()
  print('apikey is {}'.format(apikey))
  os.environ['KAGGLE_USERNAME'] = apikey['username']
  os.environ['KAGGLE_KEY'] = apikey['key']
  import kaggle
  kaggle.api.authenticate()
  kaggle.api.dataset_download_files(apikey['username']+'/aidea2019', path=os.path.join(os.getcwd(),'temp'), unzip=True)
  for datasetfile in ["NC_1.csv",
    "NC_2.csv",
    "newsdf3.csv",
    "QS_1.csv",
    "QS_2.csv",
    "TD.csv",
    "template.csv",
    "url_to_title.json",
    "url2content.json"]:
    olddatasetfile = os.path.join(os.getcwd(),'temp',datasetfile)
    newdatasetfile = os.path.join(os.getcwd(),datasetfile)
    os.replace(olddatasetfile, newdatasetfile)
  print('finished downloading kaggle dataset')

# %%
instanceGetdata = newsgetdata.getdata()
instanceConstructmodel = constructmodel.constructmodel()
num_workers = multiprocessing.cpu_count()
scheduler = 'threads'

# %%
from transformers import (
  AutoConfig,
  BertTokenizerFast,
  TFAutoModelForMaskedLM,
)
ckipModelName = 'ckiplab/albert-base-chinese'
ckiptokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
ckipModelConfig = AutoConfig.from_pretrained(ckipModelName)
ckipModelConfig.output_hidden_states = True

# %%
try:
  print(os.environ['COLAB_TPU_ADDR'])
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
  tf.config.experimental_connect_to_cluster(tpu)
  tf.tpu.experimental.initialize_tpu_system(tpu)
  needed_strategy = tf.distribute.experimental.TPUStrategy(tpu)
  usingtpu = True
  #disable_eager_execution()
except Exception as tpuerror:
  needed_strategy = tf.distribute.MultiWorkerMirroredStrategy()
  print('Number of devices: {}'.format(needed_strategy.num_replicas_in_sync))
  usingtpu = False
  print("fail in using TPU for {}".format(tpuerror))
finally:
  print("usingtpu is {}".format(usingtpu))

# %% overall data prepare
with needed_strategy.scope():
  import importlib,newsgetdata,constructmodel
  import condor_tensorflow as condor
  import numpy as np
  from transformers import (
    AutoConfig,
    BertTokenizerFast,
    TFAutoModelForMaskedLM,
  )
  importlib.reload(newsgetdata)
  importlib.reload(constructmodel)
  importlib.reload(np)
  importlib.reload(condor)
  instanceGetdata = newsgetdata.getdata()
  candidate_negative_sampling_rate = 1.3125 #1.3125 fine for largest tpu
  needData = instanceGetdata.readsrc(candidate_negative_sampling_rate=candidate_negative_sampling_rate)

# %%
sliced_batch_size_train = 1#int(needData['sampleweight_train'].shape[0]) # 100 2.7 fine full would be 25.53G
sliced_batch_size_val = 1#int(needData['sampleweight_val'].shape[0]) # 2.0 2.03 safest 2.05 is  #2.1 2.20 2.25 fine #2.5 fine
print(f'sliced_batch_size_train is {sliced_batch_size_train} and sliced_batch_size_val is {sliced_batch_size_val}')
epochs = 200
batch_size = 200 #best for tpu
prefetchesNum = batch_size*1500

# %% train data
with needed_strategy.scope():
  modelInputTrain_args = {
    'src_X_all':needData['X_train'],
    'UniquedTokens':needData['UniquedTokens'],
    'src_y_all':needData['Y_train'],
    'sweights':needData['sampleweight_train'],
    'bsize':1,
    'returncallable':True,
    'returntensor':True
  }
  modelInputTrain = instanceGetdata.iterate_dataset(**{**modelInputTrain_args,**{'bsize':batch_size}})
  sampleModelInputTrainDataSpec = instanceGetdata.iterate_dataset(**{**modelInputTrain_args,**{'bsize':batch_size, 'returnspec':True}})
  datasetGeneratorTrain = tf.data.Dataset.from_generator(
      modelInputTrain,
      output_signature=sampleModelInputTrainDataSpec,
      name='datasetGeneratorTrain'
  ).prefetch(prefetchesNum)

# %% val data
with needed_strategy.scope():
  modelInputVal_args = {**modelInputTrain_args, **{
    'src_X_all':needData['X_val'], 
    'src_y_all':needData['Y_val'],
    'sweights': needData['sampleweight_val'],
    'bsize':sliced_batch_size_val,
  }}
  sampleModelInputValDataSpec = instanceGetdata.iterate_dataset(**{**modelInputVal_args, **{'bsize':batch_size, 'returnspec':True}})
  modelInputValidation = instanceGetdata.iterate_dataset(**{**modelInputVal_args, **{'bsize':batch_size}})
  datasetGeneratorVal = tf.data.Dataset.from_generator(
      modelInputValidation,
      output_signature=sampleModelInputValDataSpec,
      name='datasetGeneratorVal'
  ).prefetch(prefetchesNum)


"""
full data rows are 458677
compset data shape is (410022, 8)
X_train 348518
for batch_size in range(1,needData['compset'].shape[0],2000):
  modelInputTrain = instanceGetdata.iterate_dataset(needData['X_train'], needData['UniquedTokens'], needData['Y_train'], needData['sampleweight_train'], batch_size, returncallable=False, returntensor=True)
  sampleData = copy.deepcopy(next(iter(modelInputTrain)) )
  print('batch_size {} object size {}'.format(batch_size, sys.getsizeof(sampleData)))

dispatcher = tf.data.experimental.service.DispatchServer()
print(f'targets is {dispatcher.target} and dispatcher is {dispatcher}')
dispatcher_address = dispatcher.target.split("://")[1]
print(f'dispatcher_address is {dispatcher_address}')
#dispatcher_address = '{}://{}'.format(dispatcher.target.split("://")[0], os.environ['COLAB_TPU_ADDR'])
workers = [tf.data.experimental.service.WorkerServer(
        tf.data.experimental.service.WorkerConfig(
          dispatcher_address=dispatcher_address))
      for _ in range(2)]
.apply(tf.data.experimental.service.distribute(
      processing_mode="distributed_epoch", service=dispatcher.target)
    ).prefetch(prefetchesNum)
"""

# %% 
optimizer_initlr = 2e-4
optimizer_description = 'adam{}'.format(optimizer_initlr)
classifier_dropout_prob = 0.1085 #0.1075 fine
descriptive_model_name_prefix = 'condors_twotower_2x256-freezealbertbase-lrsched-sampleweightAndAugSamples-{}-elu_{}_dropout{}'.format(
      candidate_negative_sampling_rate, optimizer_description, classifier_dropout_prob
  )

# %%
with needed_strategy.scope():
  import importlib,newsgetdata,constructmodel
  import condor_tensorflow as condor
  from transformers import (
    AutoConfig,
    BertTokenizerFast,
    TFAutoModelForMaskedLM,
  )
  importlib.reload(news_retrieval_common_funcs)
  importlib.reload(newsgetdata)
  importlib.reload(constructmodel)
  importlib.reload(condor)
  instanceConstructmodel = constructmodel.constructmodel(classifier_dropout_prob=classifier_dropout_prob, ckipModelName=ckipModelName)
  steps_per_epoch = sliced_batch_size_train//batch_size
  steps_whole_training = sliced_batch_size_train//batch_size*epochs
  learningRate_schedule = constructmodel.MyLRSchedule(
      max_learning_rate=optimizer_initlr*100,
      globalSteps=steps_whole_training,
      warmupSteps=int(steps_per_epoch*5),
      first_decay_steps=int(steps_per_epoch*40),
      t_mul=1.0,
      m_mul=1.0)
  compilers = {
      'steps_per_execution':1, #16
      'optimizer': tf.keras.optimizers.Adam(learning_rate=optimizer_initlr),#, clipnorm=1.0 learningRate_schedule
      'run_eagerly':False,
      'metrics':[condor.OrdinalEarthMoversDistance(name='condorErrOrdinalMoversDist')],#condor.OrdinalMeanAbsoluteError(name='ordinalMAbsErr'),
      'loss':condor.CondorOrdinalCrossEntropy(reduction=tf.keras.losses.Reduction.SUM)#(reduction=tf.keras.losses.Reduction.NONE)#
  }
  compilers.pop('weighted_metrics',None)
  load_locally = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
  if True:
    model = instanceConstructmodel.generateFineTuneModel()
    #model.load_weights(
    #    filepath=os.path.join(workingdir,'ckpt','epoch025-val_loss497.266condors_twotower_2x256-freezealbertbase-lrsched-sampleweightAndAugSamples-1.3125-elu_adam2e-05_dropout0.1085-weights.hdf5'),
    #    options=load_locally,
    #)
  else:
    model = tf.keras.models.load_model(
        os.path.join(workingdir,'epoch220-val_loss35.470condors_twotower_2x256-freezetransformer-lrsched-sampleweightAndAugSamples-1.3125-elu_adam2e-05_dropout0.1085.hdf5'),
        custom_objects={
              'TFAlbertMainLayer':TFAutoModelForMaskedLM.from_pretrained(ckipModelName, config=ckipModelConfig, from_pt=True).layers[0],
              'CondorOrdinalCrossEntropy':condor.CondorOrdinalCrossEntropy(),
              'OrdinalEarthMoversDistance':condor.OrdinalEarthMoversDistance(name='condorErrOrdinalMoversDist'),
              #'OrdinalMeanAbsoluteError':condor.OrdinalMeanAbsoluteError(name='ordinalMAbsErr'),
            },
        options=load_locally,
        compile=False
    )
    #model.layers[17].rate = classifier_dropout_prob
    #model.layers[18].rate = classifier_dropout_prob
  model.layers[6].trainable = False
  model.compile(**compilers)

# %%
batch_size = batch_size #430 #500 with steps per execution 100 OK
modelSaveFilePathParentPath = os.path.join(savingpath, 'ckpt')
n_train = needData['X_train'][list(needData['X_train'].keys())[0]].shape[0]
n_valid = needData['X_val'][list(needData['X_val'].keys())[0]].shape[0]
steps_per_epoch = n_train // batch_size #n_train // batch_size#n_train / epochs // batch_size
steps_whole_training = steps_per_epoch * epochs
validation_steps = n_valid // batch_size #n_train // batch_size#n_train / epochs // batch_size
steps_whole_validation = validation_steps*epochs

# %%
with needed_strategy.scope():
  callbackReduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.3, patience=5, min_lr=1e-8)
  #training with TPU with this callback function would generate errors
  def callbackModelcheckpoint_with_filepath(prefix, **callbacksettings):
      metricNames = [m.name for m in compilers['metrics']]
      allmetricsStr = ['val_'+m+'{val_'+m+':.2f}' for m in metricNames]
      allmetricsStr = ['epoch{epoch:03d}', 'val_loss{val_loss:.3f}'] + allmetricsStr
      callbacksettings.setdefault('verbose',1)
      callbacksettings.setdefault('save_best_only',True)
      callbacksettings.setdefault('save_weights_only',False)
      callbacksettings.setdefault('monitor','val_loss')
      callbacksettings.setdefault('mode','min')
      callbacksettings.setdefault('save_freq','epoch')
      allmetricsStr = allmetricsStr+['weights'] if callbacksettings['save_weights_only'] else allmetricsStr
      allmetricsStr = '-'.join(allmetricsStr)+'.hdf5'
      needFilePath = os.path.join(prefix,allmetricsStr)
      #"epoch{epoch:03d}-val_loss{val_loss:.3f}-val_f1_score{val_f1_score:.2f}-val_AUC{val_AUC:.2f}-val_recall{val_recall:.2f}-val_precision{val_precision:.2f}-val_categorical_accuracy{val_categorical_accuracy:.2f}.hdf5"
      callbackModelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
          filepath=needFilePath,
          **callbacksettings
      )
      return callbackModelcheckpoint
  save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
  modelSaveFilePath = os.path.join(modelSaveFilePathParentPath,descriptive_model_name_prefix)
  callbackTensorboard = tf.keras.callbacks.TensorBoard(log_dir=modelSaveFilePath, update_freq=1)
  callbacksettings = {}
  callbacksettings.setdefault('verbose',1)
  callbacksettings.setdefault('save_best_only',True)
  callbacksettings.setdefault('save_weights_only',False)
  callbacksettings.setdefault('monitor','val_loss')
  callbacksettings.setdefault('mode','min')
  callbacksettings.setdefault('save_freq','epoch')
  callbacksInFitting=[
      #callbackReduce_lr,
      tf.keras.callbacks.ModelCheckpoint(
          filepath=os.path.join(savingpath,'epoch{epoch:03d}-val_loss{val_loss:.3f}'+descriptive_model_name_prefix+'.hdf5'), options=save_locally,
          **callbacksettings
      ),
      tf.keras.callbacks.ModelCheckpoint(
          filepath=os.path.join(savingpath,'epoch{epoch:03d}-val_loss{val_loss:.3f}'+descriptive_model_name_prefix+'-weights.hdf5'),
          **{**callbacksettings, **{'save_weights_only':True}}
      ),
      #callbackTensorboard
  ]

# %%
try:
    news_retrieval_common_funcs.displaymodelinf(model)
except Exception as e:
    print('error at display model information for {}'.format(e))
    pass
previousEpochs = 0
#datasetGeneratorTrain datasetGeneratorVal
fittingHistory = model.fit(
    x=modelInputTrain(),#datasetGeneratorTrain sampleModelInputTrainData[0],#,#tr_needData['X_train'], y=tr_needData['Y_train'],  #
    #y=sampleModelInputTrainData[1],
    validation_data=modelInputValidation(),#datasetGeneratorVal sampleModelInputValData,#,#(tr_needData['X_val'], tr_needData['Y_val'], tr_needData['sampleweight_val']), #,  modelInputValidation,#
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=epochs+300,
    callbacks=callbacksInFitting,
    initial_epoch=previousEpochs,
    max_queue_size=prefetchesNum,
    #batch_size=batch_size,
    #sample_weight=sampleModelInputTrainData[2]#tr_needData['sampleweight_train'],
    #workers=num_workers,
    #use_multiprocessing=True,
)



#tf.keras.models.save_model(model, filepath=os.path.join(savingpath,'{}.hdf5'.format(descriptive_model_name_prefix)), overwrite=True, include_optimizer=True)
#model.save_weights(os.path.join(savingpath,'{}-weights.hdf5'.format(descriptive_model_name_prefix)))
# %%
