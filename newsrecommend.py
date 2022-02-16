# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import importlib,newsgetdata,constructmodel,os
import tensorflow as tf
from IPython.display import display, HTML
import faiss
import pandas as pd
import numpy as np

# %%
class recommend:
    def __init__(self, **settings):
        settings.setdefault('workingdir', os.getcwd())
        settings.setdefault('size_nbits_per_idx', 8)
        settings.setdefault('nlist', 10)
        settings.setdefault('nprobe', 10)
        settings.setdefault('sizeM_divider', 20)
        settings.setdefault('docEmbeddings',None)
        settings.setdefault('topk',200)
        if settings['docEmbeddings']!=None:
            self.generateRecommendModel(settings['docEmbeddings'])
        self.settings = settings

    # %%
    def findfactor(self, targetnum, divider):
        divider = int(divider)
        while True:
            remainder_d = targetnum % divider
            if remainder_d==0:
                return divider
            else:
                divider -= 1
            if divider==0:
                return None
    # %%
    def generateRecommendModel(self, docEmbeddings):
        dim = docEmbeddings.shape[1]
        quantizer = faiss.IndexFlatL2(dim)
        nlist = self.settings['nlist']
        size_M = self.findfactor(dim,self.settings['sizeM_divider'])
        size_nbits_per_idx = self.settings['size_nbits_per_idx']
        index = faiss.IndexIVFPQ(
            quantizer,
            dim,
            nlist,
            size_M,
            size_nbits_per_idx,
            faiss.METRIC_INNER_PRODUCT)
        #print(index.is_trained)
        index.nprobe = self.settings['nprobe']
        index.train(docEmbeddings)
        index.add(docEmbeddings)
        self.index = index
        return index

    # %%
    def returnSimilarResultsOnSingleQuery(self, queryEmbedding, **kwargs):
        """
        
        Returns:
            dict(
                'queryEmbedding': input query matrix in (n_query, embedding_size)
                'searchRes' : result matrix contains position index in (n_query, topk)
                'searchResDf': a paired search result dataframe in (topk, pairedDFcols)
            )
        """
        kwargs.setdefault('topk',self.settings['topk'])
        kwargs.setdefault('pairedDF',None)
        kwargs.setdefault('index',self.index)
        if queryEmbedding.ndim==1:
            queryEmbedding = np.expand_dims(queryEmbedding, axis=0)
        searchQuery, searchRes = kwargs['index'].search(queryEmbedding, kwargs['topk'])
        returnres = {'queryEmbedding':searchQuery, 'searchRes':searchRes}
        if isinstance(kwargs['pairedDF'], pd.DataFrame):
            returnres['searchResDf'] = kwargs['pairedDF'].loc[searchRes[0]]#.reset_index(drop=True)
        return returnres


if __name__ == '__main__':
    queryEmbeddings = models['queryModel'].predict(
        x={'Query_'+key:tf.convert_to_tensor( value ) for key, value in needData['compset_X_QueryUniquedTokens'].items()},
        batch_size=32 )
    queryEmbeddings = tf.math.l2_normalize(queryEmbeddings).numpy()#.astype('float16')
