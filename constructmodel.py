from transformers import (
    AutoConfig,
    BertTokenizerFast,
    TFAutoModelForMaskedLM,
)
import os,re
import pandas as pd
import news_retrieval_common_funcs
import multiprocessing_functions
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import resample as sklearnResample
from sklearn.utils import class_weight
import condor_tensorflow as condor
import multiprocessing


class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                **settings):
        super(MyLRSchedule, self).__init__()
        settings.setdefault('initial_learning_rate', 5e-5) #only used in linearWarmupLearningRateFunc
        settings.setdefault('max_learning_rate', 5e-3)
        settings.setdefault('globalSteps', 20000)
        settings.setdefault('warmupSteps', 1000)
        settings.setdefault('name', 'warmupcosinerestartdecay')
        settings.setdefault('first_decay_steps', settings['globalSteps']*0.5)
        settings.setdefault('t_mul', 1.0)
        settings.setdefault('m_mul', 0.5)
        settings.setdefault('alpha', 0.0)
        settings.setdefault('power', 1.0)
        self.settings = settings
        self.CosineDecayRestartsScheduleFunc = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=self.settings['max_learning_rate'], first_decay_steps=self.settings['first_decay_steps'], t_mul=self.settings['t_mul'], m_mul=self.settings['m_mul'], alpha=self.settings['alpha'],
            name='CosineDecayRestarts'
        )
        self.learningRateDecayScheduleFunc = lambda step: self.CosineDecayRestartsScheduleFunc( step-self.settings['warmupSteps'] )
    def linearWarmupLearningRateFunc(self, step):
        step_float = tf.cast(step, tf.float32)
        returnv = tf.math.subtract(self.max_learning_rate,self.settings['initial_learning_rate'])
        returnv = tf.math.divide(returnv,self.warmupSteps_float)
        returnv = tf.math.multiply(returnv,step_float)
        returnv = tf.math.add(returnv,self.settings['initial_learning_rate'])
        return returnv
    def __call__(self, step):
        with tf.name_scope(self.settings['name'] or 'WarmUp') as name:
            step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.settings['warmupSteps'], tf.float32)
            warmup_percent_done = tf.math.divide(step_float, warmup_steps_float)
            warmup_learning_rate = tf.math.multiply(  self.settings['max_learning_rate'], tf.math.pow(warmup_percent_done, self.settings['power']))
            return_learning_rate = tf.cond(tf.math.less_equal(step_float,warmup_steps_float), lambda: warmup_learning_rate, lambda: self.learningRateDecayScheduleFunc(step_float), name=name)
        return return_learning_rate
    def get_config(self):
        config = {
            'settings':self.settings,
        }
        #base_config = super(MyLRSchedule, self).get_config()
        return config

class constructmodel:
    def __init__(self, **modelsettings):
        modelsettings.setdefault('workingdir', os.getcwd())
        modelsettings.setdefault('ckipModelName', 'ckiplab/albert-tiny-chinese')
        modelsettings.setdefault('epochs', 500)
        modelsettings.setdefault('modelSaveFilePathParentPath', os.path.join(modelsettings['workingdir'], 'ckpt'))
        modelsettings.setdefault('batch_size', 32)
        modelsettings.setdefault('output_num_classes', 3)
        modelsettings.setdefault('classifier_dropout_prob', 0.1)
        modelsettings.setdefault('denseLayersAfterMeanPoolingUnits2', 256)
        modelsettings.setdefault('l2regularizerRate', 1e-5)
        modelsettings.setdefault('inputshape_sequencelen', 512)
        modelsettings.setdefault('max_learning_rate', 5e-4)
        self.modelsettings = modelsettings

    def generateCompiler(self, **compilerSettings):
        compilerSettings.setdefault('optimizer', tf.keras.optimizers.Adam(learning_rate=5e-4))
        compilerSettings.setdefault('loss', condor.CondorOrdinalCrossEntropy())
        compilerSettings.setdefault('metrics', [
            condor.OrdinalEarthMoversDistance(name='condorErrOrdinalMoversDist'),
            condor.OrdinalMeanAbsoluteError(name='ordinalMAbsErr')
            ])
        compilerSettings.setdefault('weighted_metrics', compilerSettings['metrics'])
        #optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        #optimizer = tfa.optimizers.AdamW(learning_rate=learningRate_schedule, weight_decay=weightDecay_schedule)
        #optimizer = optimizer
        #lossfunction = tfa.losses.WeightedKappaLoss(num_classes=output_num_classes) not that good
        #srcmodel.compile(srcoptimizer, metrics=srcmetrics, loss=lossfunction)
        return compilerSettings

    def generateFineTuneModel(self,modelWeightsFile=None,compileOptions=None):
        """
        每篇文章簡單只取前面幾個詞的雙塔模型
        """
        simulateX_trainItems = ['Query_input_ids','Query_token_type_ids','Query_attention_mask','Doc_input_ids','Doc_token_type_ids','Doc_attention_mask']
        inputshape_sequencelen = self.modelsettings['inputshape_sequencelen']
        ckipModelConfig = AutoConfig.from_pretrained(self.modelsettings['ckipModelName'])
        ckipModelConfig.output_hidden_states = True
        ckipmodel = TFAutoModelForMaskedLM.from_pretrained(self.modelsettings['ckipModelName'], config=ckipModelConfig, from_pt=True)
        inputs = {key:tf.keras.Input(shape=(inputshape_sequencelen,), dtype=tf.int32, name=key) for i,key in enumerate(simulateX_trainItems)}
        extracted_TFAlbertMainLayer = ckipmodel.layers[0]
        extracted_TFAlbertMainLayer.trainable = False
        albertMainLayer_output = {}
        meanpooledLayer = {}
        denseLayersAfterMeanPooling1 = {}
        denseLayersAfterMeanPooling2 = {}
        denseLayersAfterMeanPooling3 = {}
        dropoutLayer = {}
        luLayer1 = {}
        luLayer2 = {}
        luLayer3 = {}
        luLayerNamePrefix = 'elu'
        for contenttype in ['Query','Doc']: #compset_X.keys()
            neededInputLayers = {key.replace(contenttype+'_',''):value for key,value in inputs.items() if key.find(contenttype)!=-1}
            albertMainLayer_output[contenttype] = extracted_TFAlbertMainLayer(**neededInputLayers, return_dict=True)
            meanpooledLayer[contenttype] = tf.keras.layers.Lambda(lambda x, lib:news_retrieval_common_funcs.mean_pooling(**x, lib=lib), arguments={'lib':'tf'}, name='mean_pooling_with_mask'+contenttype)({
                'token_embeddings':albertMainLayer_output[contenttype]['last_hidden_state'], 'attention_mask':neededInputLayers['attention_mask']
            })
            denseLayersAfterMeanPooling1[contenttype] = tf.keras.layers.Dense(units=ckipModelConfig.hidden_size, name='denseAfterPooling1'+contenttype)(meanpooledLayer[contenttype])
            luLayer1[contenttype] = tf.keras.layers.ELU(name='{}1{}'.format(luLayerNamePrefix,contenttype))(denseLayersAfterMeanPooling1[contenttype])
            denseLayersAfterMeanPooling2[contenttype] = tf.keras.layers.Dense(
                units=self.modelsettings['denseLayersAfterMeanPoolingUnits2'],
                #activation=tf.keras.activations.relu,
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.modelsettings['l2regularizerRate']),
                bias_regularizer=tf.keras.regularizers.L2(self.modelsettings['l2regularizerRate']),
                activity_regularizer=tf.keras.regularizers.L2(self.modelsettings['l2regularizerRate']),
                name='denseAfterPooling2WithL2'+contenttype)(luLayer1[contenttype])
            luLayer2[contenttype] = tf.keras.layers.ELU(name='{}2{}'.format(luLayerNamePrefix, contenttype))(denseLayersAfterMeanPooling2[contenttype])
            """
            denseLayersAfterMeanPooling3[contenttype] = tf.keras.layers.Dense(
                units=self.modelsettings['denseLayersAfterMeanPoolingUnits2'],
                #activation=tf.keras.activations.relu,
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.modelsettings['l2regularizerRate']),
                bias_regularizer=tf.keras.regularizers.L2(self.modelsettings['l2regularizerRate']),
                activity_regularizer=tf.keras.regularizers.L2(self.modelsettings['l2regularizerRate']),
                name='denseAfterPooling3WithL2'+contenttype)(luLayer2[contenttype])
            luLayer3[contenttype] = tf.keras.layers.ELU(name='{}3{}'.format(luLayerNamePrefix, contenttype))(denseLayersAfterMeanPooling3[contenttype])
            """
            dropoutLayer[contenttype] = tf.keras.layers.Dropout(rate=self.modelsettings['classifier_dropout_prob'], name='dropout'+contenttype)(luLayer2[contenttype])
        cosineSimilarityLayer = tf.keras.layers.Dot(axes=(1,1), normalize=True)(list(dropoutLayer.values()))
        classifierLayer = tf.keras.layers.Dense(
            units=self.modelsettings['output_num_classes'], name="classifierLogits"
        )(cosineSimilarityLayer)
        finetunemodel = tf.keras.Model(inputs=inputs,outputs=classifierLayer)
        if modelWeightsFile is not None:
            finetunemodel.load_weights(modelWeightsFile)
        if compileOptions is not None:
            finetunemodel.compile(**compileOptions)
        return finetunemodel

    def generateTwoTowerModels(self, **settings):
        settings.setdefault('modelWeightsFile',None)
        trainingModel = self.generateFineTuneModel()
        if settings['modelWeightsFile'] is not None:
            trainingModel.load_weights(settings['modelWeightsFile'])
        evaluateSimilarityModelQueryPart = tf.keras.Model(inputs=trainingModel.inputs[3:], outputs=trainingModel.layers[17].output)
        evaluateSimilarityModelDocPart = tf.keras.Model(inputs=trainingModel.inputs[:3], outputs=trainingModel.layers[18].output)
        rankingModelInput = [tf.keras.Input(shape=(self.modelsettings['denseLayersAfterMeanPoolingUnits2'],), name=key) for key in ['queryEmbedding','docEmbedding']]
        cosineSimilarityLayer = tf.keras.layers.Dot(axes=(1,1), normalize=True)(rankingModelInput)
        classifierLogitsLayer = trainingModel.layers[20](cosineSimilarityLayer)
        rankingModel = tf.keras.Model(inputs=rankingModelInput,outputs=classifierLogitsLayer)
        return {
            'trainingModel':trainingModel,
            'rankingModel':rankingModel,
            'queryModel':evaluateSimilarityModelQueryPart,
            'docModel':evaluateSimilarityModelDocPart
        }


if __name__ == '__main__':
    from IPython.display import display, HTML
    test = constructmodel()
    testdata = test.generateTwoTowerModels()
    for k,m in testdata.items():
        news_retrieval_common_funcs.displaymodelinf(m)