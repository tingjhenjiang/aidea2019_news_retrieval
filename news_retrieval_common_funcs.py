import pandas as pd
import numpy as np
import tensorflow as tf
#import nltk
import re,os
from IPython.display import display

#nltk.download('punkt')
#from nltk import tokenize as nltktokenize
class SentenceBreaker:
    ##regular expression for tokenize document to sentences
    def __init__(self):
        self.__to_sents_rule = re.compile(u'([^\n]+)[\n]*')

        ## 中文字元
        self.__ch=u'[\u4E00-\u9fa5]'

        ## 英文字元
        self.__eng=u'[a-zA-Z0-9_ａ-ｚＡ-Ｚ０-９－＿]'

        ## 英文字元+符號
        self.__eng_sym=u'[a-zA-Z0-9_ａ-ｚＡ-Ｚ０-９－＿\\/,.&\-]'

        ## 其他字元
        self.__other=u'[^\w\u4E00-\u9fa5\s]'

        ## 
        self.__to_tokens_rule = re.compile(
            self.__ch +
            u"|"+ self.__eng+self.__eng_sym+u"*" +
            u"|"+ self.__other
        )

    def ToSentsFirstRun(self,doc):
        '''split a document into sentences'''
        doc = re.sub('([。！？\?])([^」”’])', r"\1\n\2", doc)
        doc = re.sub('(\.{6})([^」”’])', r"\1\n\2", doc)
        doc = re.sub('(\…{2})([^」”’])', r"\1\n\2", doc)
        doc = re.sub('([。！？\?][」”’])([^，。！？\?])', r'\1\n\2', doc)
        sents=[m.group(0).strip(" \t\r\n") for m in re.finditer(self.__to_sents_rule, doc)]
        return sents

    def ToSents(self,doc):
        doc = re.sub('([a-zA-Z0-9_ａ-ｚＡ-Ｚ０-９－＿]{4,}\.)([a-zA-Z0-9_ａ-ｚＡ-Ｚ０-９－＿]{4,})',r'\1 \2',doc)
        breakedInChiDoc = self.ToSentsFirstRun(doc) #nltktokenize.sent_tokenize(doc)
        newdoc = []
        for sentence in breakedInChiDoc:
            if re.search(self.__eng, sentence)==None:
                newdoc.append(sentence)
            else:
                try:
                    newdoc.extend(nltktokenize.sent_tokenize(sentence))
                except Exception as e:
                    print("error in handling {} for {}, append instead.".format(sentence,e))
                    newdoc.append(sentence)
        return newdoc

    def ToTokens(self,sent):
        '''split a sentence into Chinese chars or English words'''
        tokens = [data for data in self.__to_tokens_rule.findall(sent)]
        return tokens

#sentencebreaker = SentenceBreaker()
#sentencebreaker = sentencebreaker.ToSents
#display(sentencebreaker.ToSents(testtargetstr.replace('&#39;','\'').replace('audiojs.events.ready(function(){var as = audiojs.createAll();});',' ')))

def is_nan(x):
    return (x != x)
def antijoin(TableA,TableB):
    outer_join = TableA.merge(TableB, how = 'outer', indicator = True)
    anti_join = outer_join[~(outer_join._merge == 'both')].drop('_merge', axis = 1)
    return anti_join
def displaymodelinf(dispmodel, expand_nested=False, savingpath='./'):
    display(dispmodel.summary())
    try:
        display(tf.keras.utils.plot_model(
        dispmodel, to_file=os.path.join(savingpath, 'model.png'), show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=expand_nested, dpi=96, show_dtype=True
        ))
    except Exception as e:
        print('error in displaying model structure in image for {}'.format(e))

def remove_all_zeros_rows(srctensor):
    boolean_mask = tf.cast(tf.reduce_sum(tf.abs(srctensor), 1), dtype=tf.bool)
    no_zeros = tf.boolean_mask(srctensor, boolean_mask, axis=0)
    return no_zeros

def mean_pooling(**args):
    args.setdefault('lib', 'tf')
    if args['lib']=='pt':
        import torch
        input_mask_expanded = args['attention_mask'].unsqueeze(-1).expand(args['token_embeddings'].size()).float()
        sum_embeddings = torch.sum(args['token_embeddings'] * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    else:
        input_mask_expanded = tf.expand_dims(args['attention_mask'], axis=-1)
        input_mask_expanded = tf.broadcast_to(input_mask_expanded, tf.shape(args['token_embeddings']))
        input_mask_expanded = tf.cast(input_mask_expanded, dtype=tf.float32)
        sum_embeddings = tf.math.reduce_sum(args['token_embeddings'] * input_mask_expanded, axis=1)
        sum_mask = tf.math.reduce_sum(input_mask_expanded, axis=1)
        sum_mask = tf.clip_by_value(sum_mask, clip_value_min=1e-9, clip_value_max=9999^9)
    returnres = sum_embeddings / sum_mask
    del input_mask_expanded,sum_embeddings,sum_mask
    return returnres

def strs_to_a_indexed_df(srclist):
    targetdf = pd.DataFrame.from_records(
        list({v:i for i,v in enumerate(srclist)}.items()), index=0
    ).drop(columns=0)
    targetdf.columns = ['index']
    return targetdf



"""

  learningRate_schedule = MyLRSchedule(
      initial_learning_rate=5e-5,
      max_learning_rate=5e-3,
      globalSteps=steps_whole_training,
      warmupSteps=int(steps_per_epoch*1.5))
  weightDecay_schedule = MyLRSchedule(
      initial_learning_rate=5e-7,
      max_learning_rate=5e-5,
      globalSteps=steps_whole_training,
      warmupSteps=int(steps_per_epoch*1.5))
  #plt.plot(learningRate_schedule(tf.range(steps_whole_training, dtype=tf.float32)))
  #optimizer = tfa.optimizers.AdamW(weight_decay=weightDecay_schedule, learning_rate=learningRate_schedule)#learningRate_schedule
  optimizer_description = 'SGDdec5e-5Mom1e-6'
  #optimizer = tf.keras.optimizers.SGD(learningRate_schedule, momentum=1e-6, nesterov=True, name=optimizer_description)
  optimizer_description = 'adam5e-5'
  optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
  plt.plot(tf.vectorized_map(learningRate_schedule, tf.range(steps_whole_training)))
  plt.title("learning rate schedule")
  plt.show()
"""